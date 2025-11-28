import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table


console = Console()


# Env loading
load_dotenv()  # noop if none


# API endpoints
BSCSCAN_API = "https://api.bscscan.com/api"
MONDAY_API = "https://api.monday.com/v2"
PPLX_API = "https://api.perplexity.ai/chat/completions"


def get_env(name: str, fallback: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name)
    if val:
        return val
    return fallback


# Support both naming variants for BscScan key
BSCSCAN_KEY = get_env("BSCSCAN_KEY", get_env("BSCSCAN_API_KEY"))
PPLX_KEY = get_env("PPLX_KEY")
MONDAY_TOKEN = get_env("MONDAY_TOKEN")
GITHUB_TOKEN = get_env("GITHUB_TOKEN")


def domain_from_url(url: str) -> Optional[str]:
    try:
        m = re.match(r"https?://([^/]+)/?", url)
        if not m:
            return None
        host = m.group(1).lower()
        # strip common subdomains
        for prefix in ["www.", "app.", "docs."]:
            if host.startswith(prefix):
                host = host[len(prefix) :]
        return host
    except Exception:
        return None


def extract_urls(text: str) -> List[str]:
    if not text:
        return []
    urls = re.findall(r"https?://[^\s,;\]\)]+", str(text))
    # sanitize
    cleaned: List[str] = []
    for u in urls:
        u = u.strip().strip("'\"")
        # drop trailing punctuation
        u = re.sub(r"[\.,;:]+$", "", u)
        cleaned.append(u)
    # de-dup preserve order
    seen = set()
    uniq = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def bscscan_official_links(client: httpx.Client, token_addr: str) -> List[str]:
    if not token_addr:
        return []
    if not BSCSCAN_KEY:
        return []
    try:
        params = {
            "module": "contract",
            "action": "getsourcecode",
            "address": token_addr,
            "apikey": BSCSCAN_KEY,
        }
        r = client.get(BSCSCAN_API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        result = (data or {}).get("result") or []
        if not result:
            return []
        rec = result[0]
        links: List[str] = []
        for key in ["Website", "Twitter", "Github", "Telegram", "Comments", "CompilerVersion", "LicenseType"]:
            links.extend(extract_urls(rec.get(key)))
        # Some projects encode multiple URLs with semicolons or newlines in Website2..Website10 fields
        for k, v in rec.items():
            if isinstance(v, str) and (k.lower().startswith("website") or k.lower().endswith("url")):
                links.extend(extract_urls(v))
        # Keep only http(s)
        links = [u for u in links if u.startswith("http://") or u.startswith("https://")]
        # de-dup preserving order
        seen = set()
        uniq: List[str] = []
        for u in links:
            if u not in seen:
                seen.add(u)
                uniq.append(u)
        return uniq[:12]
    except Exception:
        return []


def coingecko_official_links(client: httpx.Client, token_addr: str) -> List[str]:
    if not token_addr:
        return []
    # Coingecko contract endpoint (no key required, rate-limited)
    url = f"https://api.coingecko.com/api/v3/coins/binance-smart-chain/contract/{token_addr}"
    try:
        r = client.get(url, timeout=30)
        if r.status_code == 429:
            # Backoff once
            time.sleep(1.0)
            r = client.get(url, timeout=30)
        if r.status_code != 200:
            return []
        data = r.json() or {}
        links = []
        l = data.get("links") or {}
        # Collect common official links
        for hp in (l.get("homepage") or []):
            links.extend(extract_urls(hp))
        for d in (l.get("documentation") or []):
            links.extend(extract_urls(d))
        for gh in (l.get("repos_url") or {}).get("github", []) or []:
            links.extend(extract_urls(gh))
        for tw in (l.get("twitter_screen_name") or [] if isinstance(l.get("twitter_screen_name"), list) else [l.get("twitter_screen_name")]):
            if tw:
                links.append(f"https://twitter.com/{tw}")
        # Also scan description for URLs, limited
        desc = ((data.get("description") or {}).get("en") or "")
        links.extend(extract_urls(desc))
        # Deduplicate and keep only http(s)
        links = [u for u in links if isinstance(u, str) and (u.startswith("http://") or u.startswith("https://"))]
        seen = set()
        uniq: List[str] = []
        for u in links:
            if u not in seen:
                seen.add(u)
                uniq.append(u)
        return uniq[:12]
    except Exception:
        return []


def pplx_extract(client: httpx.Client, urls: List[str]) -> Dict:
    if not urls or not PPLX_KEY:
        return {}
    prompt = (
        "You classify a crypto project STRICTLY from the given URLs. "
        "Return ONLY JSON with keys: "
        "category(one of: defi,desoc,depin,meme,other), "
        "primary_chain(one of: bnb, ethereum, solana, base, arbitrum, optimism, ton, sui, aptos, other), "
        "description, has_audit(true/false), github_main(url or null), "
        "bnb_launch_date_iso(ISO date or null). "
        "If unsure, set category='other', primary_chain='other', has_audit=false. "
        "Use exact chain names above. Do not invent URLs."
    )
    body = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "user",
                "content": prompt + "\nURLS:\n" + "\n".join(urls),
            }
        ],
        "return_images": False,
    }
    try:
        r = client.post(
            PPLX_API,
            headers={
                "Authorization": f"Bearer {PPLX_KEY}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=60,
        )
        r.raise_for_status()
        payload = r.json()
        txt = (
            (((payload.get("choices") or [{}])[0] or {}).get("message") or {}).get(
                "content"
            )
            or ""
        )
        # Try strict JSON first
        try:
            return json.loads(txt)
        except Exception:
            # Extract first JSON object from the text if any
            m = re.search(r"\{[\s\S]*\}", txt)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return {}
            return {}
    except Exception:
        return {}


def github_last_commit(client: httpx.Client, repo_url: Optional[str]) -> Optional[str]:
    if not repo_url or "github.com" not in repo_url:
        return None
    m = re.search(r"github\.com/([^/]+)/([^/#?]+)", repo_url)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2).replace(".git", "")
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "binance-alpha/1.0"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    try:
        r = client.get(url, headers=headers, params={"per_page": 1}, timeout=30)
        if r.status_code != 200:
            return None
        arr = r.json()
        if not arr:
            return None
        return arr[0]["commit"]["committer"]["date"]
    except Exception:
        return None


def monday_find_item(client: httpx.Client, name: str, domain: Optional[str], board_id: Optional[int]) -> Optional[dict]:
    if not MONDAY_TOKEN or not board_id or not name:
        return None
    query = (
        "query($bid: [ID!]!, $term: String!) { "
        "boards(ids: $bid) { items_page(query: $term, limit: 10) { items { id name created_at column_values { id text } } } } }"
    )
    headers = {
        "Authorization": MONDAY_TOKEN,
        "Content-Type": "application/json",
    }
    variables = {"bid": [int(board_id)], "term": name}
    try:
        r = client.post(
            MONDAY_API, json={"query": query, "variables": variables}, headers=headers, timeout=30
        )
        if r.status_code != 200:
            return None
        boards = ((r.json().get("data") or {}).get("boards") or [])
        if not boards:
            return None
        board = boards[0] or {}
        items = ((board.get("items_page") or {}).get("items")) or []

        # Prefer matching by Lead column text when present, else by item name, with fuzzy threshold
        # Fetch columns metadata
        cols_meta = (board.get("columns") or []) if board.get("columns") else []
        if not cols_meta:
            # fetch columns via separate query when not available
            try:
                r2 = client.post(
                    MONDAY_API,
                    headers=headers,
                    json={
                        "query": "query($bid:[ID!]!){ boards(ids:$bid){ columns{ id title } } }",
                        "variables": {"bid": [int(board_id)]},
                    },
                    timeout=20,
                )
                cols_meta = (((r2.json() or {}).get("data") or {}).get("boards") or [{}])[0].get("columns") or []
            except Exception:
                cols_meta = []
        lead_col_id = None
        owner_col_id = None
        website_col_id = None
        for c in cols_meta:
            title_l = (c.get("title") or "").strip().lower()
            if title_l == "lead":
                lead_col_id = c.get("id")
            if title_l in {"deal owner", "owner"}:
                owner_col_id = c.get("id")
            if title_l in {"website", "url", "project website"}:
                website_col_id = c.get("id")

        def get_lead_text(it: dict) -> str:
            if not lead_col_id:
                return ""
            for cv in it.get("column_values", []) or []:
                if cv.get("id") == lead_col_id:
                    return (cv.get("text") or "").strip()
            return ""

        def get_owner_text(it: dict) -> str:
            if not owner_col_id:
                return ""
            for cv in it.get("column_values", []) or []:
                if cv.get("id") == owner_col_id:
                    return (cv.get("text") or "").strip()
            return ""

        from difflib import SequenceMatcher

        target = (name or "").lower()

        def get_website_domain(it: dict) -> str:
            if not website_col_id:
                return ""
            for cv in it.get("column_values", []) or []:
                if cv.get("id") == website_col_id:
                    txt = (cv.get("text") or "").strip()
                    urls = extract_urls(txt)
                    if urls:
                        d = domain_from_url(urls[0] or "") or ""
                        return d
            return ""

        def score_item(it: dict) -> float:
            lead_text = (get_lead_text(it) or "").lower()
            item_name = (it.get("name") or "").lower()
            site_domain = (get_website_domain(it) or "").lower()
            s1 = SequenceMatcher(a=target, b=lead_text).ratio() if lead_text else 0.0
            s2 = SequenceMatcher(a=target, b=item_name).ratio()
            boost = 0.0
            penalty = 0.0
            if domain:
                if domain in site_domain:
                    boost += 0.25
                elif domain in item_name or domain in lead_text:
                    boost += 0.10
                else:
                    penalty += 0.20
            return max(s1, s2) + boost - penalty

        if not items:
            return None
        # Focus pool by domain when available
        pool = items
        if domain:
            pool = [it for it in items if (get_website_domain(it) or "").lower().find(domain) >= 0] or items
        ranked = sorted(pool, key=score_item, reverse=True)
        best = ranked[0]
        if score_item(best) < 0.60:
            return None
        # Attach convenience field for downstream use
        try:
            best["__deal_owner"] = get_owner_text(best)
        except Exception:
            pass
        return best
    except Exception:
        return None


def decide_bd_attribution(monday_item: Optional[dict], bnb_launch_date_iso: Optional[str]) -> str:
    if not monday_item:
        return "unknown"
    created = monday_item.get("created_at")
    try:
        created_dt = datetime.fromisoformat((created or "").replace("Z", ""))
    except Exception:
        created_dt = None
    launch_dt: Optional[datetime] = None
    if bnb_launch_date_iso:
        try:
            launch_dt = datetime.fromisoformat(bnb_launch_date_iso.replace("Z", ""))
        except Exception:
            launch_dt = None
    if created_dt and launch_dt and created_dt <= (launch_dt - timedelta(days=30)):
        return "bd_sourced"
    if created_dt and launch_dt and abs((created_dt - launch_dt).days) <= 14:
        return "co_sourced"
    return "unknown"


def quick_category_flag(urls: List[str]) -> Optional[str]:
    txt = " ".join(urls).lower()
    parts = txt.split()
    has_docs = any(("/docs" in u) or ("docs." in u) or ("whitepaper" in u) for u in parts)
    has_audit_kw = any(
        ("audit" in u) or ("audits" in u) or ("certik" in u) or ("slowmist" in u) or ("peckshield" in u)
        for u in parts
    )
    if not has_docs and not has_audit_kw:
        return "meme_quick"
    return None


def compute_confidence(docs_present: bool, has_audit: bool, gh_recent: bool, monday_hit: bool) -> float:
    conf = 0.5
    if docs_present:
        conf += 0.2
    if has_audit:
        conf += 0.2
    if gh_recent:
        conf += 0.1
    if monday_hit:
        conf += 0.1
    return min(1.0, conf)


@dataclass
class Overrides:
    by_name: Dict[str, Dict[str, str]]


def load_overrides(path: Optional[str]) -> Overrides:
    mapping: Dict[str, Dict[str, str]] = {}
    if not path:
        return Overrides(mapping)
    p = Path(path)
    if not p.exists():
        return Overrides(mapping)
    try:
        with p.open() as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                mapping[name.lower()] = {
                    "domain": (row.get("domain") or "").strip().lower(),
                    "bnb_launch_date": (row.get("bnb_launch_date") or "").strip(),
                }
    except Exception:
        pass
    return Overrides(mapping)


def decide_primary_chain(llm_primary: str, urls: List[str]) -> str:
    p = (llm_primary or "other").lower()
    if p in {"bnb", "ethereum", "solana", "base", "arbitrum", "optimism", "ton", "sui", "aptos", "other"}:
        return p
    # Fallback: detect hints
    joined = " ".join(urls).lower()
    if any(k in joined for k in ["binance smart chain", "bnb chain", "bsc.", "bscscan."]):
        return "bnb"
    return "other"


def run(
    input_csv: str,
    out_dir: str,
    board_id: Optional[int] = None,
    overrides_csv: Optional[str] = None,
    limit: Optional[int] = None,
    sleep_ms: int = 300,
    disable_bscscan: bool = False,
) -> Tuple[int, Path]:
    inp = Path(input_csv)
    if not inp.exists():
        console.print(f"[red]Input CSV not found:[/] {inp}")
        return 1, Path(out_dir)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_csv = out_path / "classified_tokens.csv"

    overrides = load_overrides(overrides_csv)

    out_rows: List[Dict[str, str]] = []

    with httpx.Client(follow_redirects=True) as client, inp.open() as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        if limit is not None and limit > 0:
            rows = rows[:limit]
        for row in rows:
            addr = (row.get("address") or "").strip()
            name = (row.get("name") or row.get("symbol") or "").strip()

            links = [] if disable_bscscan else bscscan_official_links(client, addr)
            # Fallbacks: CoinGecko, then Binance Alpha page
            if (not links) and addr:
                cg_links = coingecko_official_links(client, addr)
                if cg_links:
                    links = cg_links
            if (not links) and addr:
                links = [f"https://www.binance.com/en/alpha/bsc/{addr}"]

            # Prepare overrides
            override = overrides.by_name.get(name.lower()) if name else None
            override_domain = (override or {}).get("domain") or None
            override_launch = (override or {}).get("bnb_launch_date") or None

            llm = pplx_extract(client, links)
            category = (llm.get("category") or "other").lower() if isinstance(llm, dict) else "other"
            primary = decide_primary_chain((llm.get("primary_chain") if isinstance(llm, dict) else None) or "other", links)
            gh_url = (llm.get("github_main") if isinstance(llm, dict) else None) or None
            audit = bool(llm.get("has_audit")) if isinstance(llm, dict) else False
            bnb_launch = (llm.get("bnb_launch_date_iso") if isinstance(llm, dict) else None) or None
            if not bnb_launch and override_launch:
                bnb_launch = override_launch

            gh_last = github_last_commit(client, gh_url)
            gh_recent = False
            if gh_last:
                try:
                    gh_recent = (datetime.utcnow() - datetime.fromisoformat(gh_last.replace("Z", ""))) <= timedelta(days=90)
                except Exception:
                    gh_recent = False

            qflag = quick_category_flag(links)
            docs_present = any(("/docs" in u) or ("docs." in u) for u in links)

            # Monday matching
            domain_hint = override_domain or None
            if not domain_hint:
                # derive from links
                domains = [domain_from_url(u) for u in links]
                domains = [d for d in domains if d]
                domain_hint = domains[0] if domains else None
            monday_item = monday_find_item(client, name, domain_hint, board_id)
            bd_attr = decide_bd_attribution(monday_item, bnb_launch)

            # Final category decision
            if qflag == "meme_quick" and category in ["other", "meme"]:
                final_cat = "meme"
            elif category in ["defi", "desoc", "depin", "meme"]:
                final_cat = category
            else:
                final_cat = "other"

            is_bnb_project = str(primary == "bnb").lower()
            conf = compute_confidence(docs_present, audit, gh_recent, monday_item is not None)

            out_rows.append(
                {
                    "address": addr,
                    "name": name,
                    "primary_chain": primary,
                    "category": final_cat,
                    "is_bnb_project": is_bnb_project,
                    "bd_attribution": bd_attr,
                    "confidence": f"{conf:.2f}",
                    "github_last_commit_iso": gh_last or "",
                    "bnb_launch_date_guess": bnb_launch or "",
                    "evidence_urls": " | ".join(links)[:2000],
                    "deal_owner": (monday_item or {}).get("__deal_owner", "") if monday_item else "",
                }
            )

            # polite pacing
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

    # Write CSV
    with out_csv.open("w", newline="") as f:
        cols = [
            "address",
            "name",
            "primary_chain",
            "category",
            "is_bnb_project",
            "bd_attribution",
            "confidence",
            "github_last_commit_iso",
            "bnb_launch_date_guess",
            "evidence_urls",
            "deal_owner",
        ]
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(out_rows)

    # Print quick summary
    total = len(out_rows)
    bnb_count = sum(1 for r in out_rows if r["is_bnb_project"] == "true")
    non_bnb = total - bnb_count
    meme_count = sum(1 for r in out_rows if r["category"] == "meme")
    unknown_count = sum(1 for r in out_rows if r["category"] == "other")

    console.print(f"[green]Wrote[/] {out_csv}  ([cyan]{total} rows[/])")

    table = Table(title="Classification Summary", show_lines=False)
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    table.add_row("BNB primary", str(bnb_count))
    table.add_row("Non-BNB primary", str(non_bnb))
    table.add_row("Meme", str(meme_count))
    table.add_row("Unknown (other)", str(unknown_count))
    console.print(table)

    # Totals by primary_chain, category, bd_attribution
    def count_by(key: str) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for r in out_rows:
            k = r.get(key) or ""
            c[k] = c.get(k, 0) + 1
        return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))

    for dim in ["primary_chain", "category", "bd_attribution"]:
        subt = Table(title=f"by {dim}")
        subt.add_column(dim)
        subt.add_column("count", justify="right")
        for k, v in count_by(dim).items():
            subt.add_row(k or "(blank)", str(v))
        console.print(subt)

    return 0, out_csv


def parse_args(argv: List[str]):
    import argparse

    parser = argparse.ArgumentParser(description="Classify tokens as BNB/other, category, and BD attribution")
    parser.add_argument("--input", type=str, default="./data/alpha/alpha_tokens.csv", help="Path to alpha_tokens.csv")
    parser.add_argument("--out", type=str, default="./data/alpha", help="Output directory for classified_tokens.csv")
    parser.add_argument("--board-id", type=int, default=None, help="Monday.com board id for matching (optional)")
    parser.add_argument("--overrides", type=str, default=None, help="Optional CSV with columns: name,domain,bnb_launch_date")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to process")
    parser.add_argument("--sleep-ms", type=int, default=300, help="Sleep between rows to respect API rate limits")
    parser.add_argument("--disable-bscscan", action="store_true", help="Do not query BscScan for links; use Binance Alpha page URL fallback only")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    code, _ = run(
        input_csv=args.input,
        out_dir=args.out,
        board_id=args.board_id,
        overrides_csv=args.overrides,
        limit=args.limit,
        sleep_ms=args.sleep_ms,
        disable_bscscan=args.disable_bscscan,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())


