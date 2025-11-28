import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from difflib import SequenceMatcher


console = Console()
load_dotenv()

MONDAY_API = "https://api.monday.com/v2"
MONDAY_TOKEN = os.getenv("MONDAY_TOKEN")


def require_monday_token() -> str:
    if not MONDAY_TOKEN:
        raise SystemExit("Missing MONDAY_TOKEN in environment or .env file")
    return MONDAY_TOKEN


def monday_headers() -> Dict[str, str]:
    return {"Authorization": require_monday_token(), "Content-Type": "application/json"}


def monday_get_board_columns(client: httpx.Client, board_id: int) -> List[dict]:
    q = """
    query($bid: [ID!]!) {
      boards(ids: $bid) {
        id
        name
        columns { id title type }
      }
    }
    """
    r = client.post(MONDAY_API, headers=monday_headers(), json={"query": q, "variables": {"bid": [int(board_id)]}}, timeout=30)
    r.raise_for_status()
    data = r.json()
    boards = (data.get("data", {}).get("boards") or [])
    if not boards:
        return []
    return boards[0].get("columns") or []


def monday_fetch_all_items(client: httpx.Client, board_id: int, max_items: int = 1000, page_size: int = 200) -> List[dict]:
    q = """
    query($bid: [ID!]!, $limit: Int!, $cursor: String) {
      boards(ids: $bid) {
        items_page(limit: $limit, cursor: $cursor) {
          cursor
          items { id name created_at column_values { id text value } }
        }
      }
    }
    """
    items: List[dict] = []
    cursor: Optional[str] = None
    while len(items) < max_items:
        r = client.post(
            MONDAY_API,
            headers=monday_headers(),
            json={"query": q, "variables": {"bid": [int(board_id)], "limit": int(page_size), "cursor": cursor}},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        boards = (data.get("data", {}).get("boards") or [])
        if not boards:
            break
        page = (boards[0].get("items_page") or {})
        new_items = page.get("items") or []
        if not new_items:
            break
        items.extend(new_items)
        cursor = page.get("cursor")
        if not cursor:
            break
    return items[:max_items]


def monday_search_items_simple(client: httpx.Client, board_id: int, term: str, limit: int = 20) -> List[dict]:
    q = """
    query($bid: [ID!]!, $term: String!, $limit: Int!) {
      boards(ids: $bid) {
        items_page(query: $term, limit: $limit) {
          items { id name created_at column_values { id text value } }
        }
      }
    }
    """
    r = client.post(
        MONDAY_API,
        headers=monday_headers(),
        json={"query": q, "variables": {"bid": [int(board_id)], "term": term, "limit": int(limit)}},
        timeout=40,
    )
    r.raise_for_status()
    data = r.json()
    boards = (data.get("data", {}).get("boards") or [])
    if not boards:
        return []
    return ((boards[0].get("items_page") or {}).get("items") or [])


def best_name_match(term: str, items: List[dict], columns: List[dict]) -> Optional[dict]:
    if not items:
        return None
    lead_col_id = None
    for c in columns or []:
        if (c.get("title") or "").strip().lower() == "lead":
            lead_col_id = c.get("id")
            break

    def get_lead_text(it: dict) -> str:
        if not lead_col_id:
            return ""
        for cv in it.get("column_values", []) or []:
            if cv.get("id") == lead_col_id:
                return (cv.get("text") or "").strip()
        return ""

    def score_item(it: dict) -> float:
        target = (term or "").lower()
        lead_text = (get_lead_text(it) or "").lower()
        item_name = (it.get("name") or "").lower()
        s1 = SequenceMatcher(a=target, b=lead_text).ratio() if lead_text else 0.0
        s2 = SequenceMatcher(a=target, b=item_name).ratio()
        return max(s1, s2)

    ranked = sorted(items, key=score_item, reverse=True)
    top = ranked[0]
    if score_item(top) < 0.55:
        return None
    return top


def value_for_title(columns: List[dict], item: dict, title: str) -> Optional[str]:
    # map title -> id
    id_by_title = {c.get("title"): c.get("id") for c in columns}
    col_id = id_by_title.get(title)
    if not col_id:
        return None
    for cv in item.get("column_values", []) or []:
        if cv.get("id") == col_id:
            return cv.get("text") or None
    return None


def run(board_id: int, input_csv: str, out_csv: str, limit: Optional[int], term: Optional[str] = None, domain_filter: Optional[str] = None) -> Tuple[int, Path]:
    inp = Path(input_csv)
    if not inp.exists():
        console.print(f"[red]Input CSV not found:[/] {inp}")
        return 1, Path(out_csv)
    outp = Path(out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(follow_redirects=True) as client, inp.open() as f, outp.open("w", newline="") as w:
        cols_meta = monday_get_board_columns(client, board_id)
        board_items: List[dict] = []
        if term:
            board_items = monday_search_items_simple(client, board_id, term, limit=50)
        else:
            board_items = monday_fetch_all_items(client, board_id, max_items=3000, page_size=200)

        def website_matches_domain(it: dict, dom: str) -> bool:
            dom = (dom or "").lower().strip()
            if not dom:
                return True
            for cv in it.get("column_values", []) or []:
                if (cv.get("text") or "") and isinstance(cv.get("text"), str):
                    if cv.get("text").lower().find(dom) >= 0:
                        return True
            return False

        rdr = csv.DictReader(f)
        rows = list(rdr)
        if limit and limit > 0:
            rows = rows[:limit]

        writer = csv.writer(w)
        writer.writerow([
            "address", "name", "item_id", "matched_name", "created_at", "website", "stage", "category", "subcategory", "deal_owner", "priority"
        ])

        matches = 0
        for row in rows:
            name = (row.get("name") or row.get("symbol") or "").strip()
            addr = (row.get("address") or "").strip()
            if not name:
                continue
            # Apply optional domain filter to reduce false positives
            search_pool = [it for it in board_items if website_matches_domain(it, domain_filter)] if domain_filter else board_items
            best = best_name_match(name, search_pool, cols_meta)
            if not best:
                writer.writerow([addr, name, "", "", "", "", "", "", "", "", ""])
                continue
            matches += 1
            website = value_for_title(cols_meta, best, "Website") or ""
            stage = value_for_title(cols_meta, best, "Stage") or ""
            category = value_for_title(cols_meta, best, "Category") or ""
            subcategory = value_for_title(cols_meta, best, "Subcategory") or ""
            owner = value_for_title(cols_meta, best, "Deal Owner") or value_for_title(cols_meta, best, "Owner") or ""
            priority = value_for_title(cols_meta, best, "Priority") or ""

            writer.writerow([
                addr,
                name,
                best.get("id") or "",
                best.get("name") or "",
                best.get("created_at") or "",
                website,
                stage,
                category,
                subcategory,
                owner,
                priority,
            ])

    # Print quick summary
    table = Table(title="Monday matches")
    table.add_column("metric")
    table.add_column("count", justify="right")
    table.add_row("rows processed", str(len(rows)))
    table.add_row("name matches", str(matches))
    console.print(table)
    console.print(f"[green]Wrote[/] {outp}")
    return 0, outp


def parse_args(argv: List[str]):
    import argparse
    p = argparse.ArgumentParser(description="Search Monday board by project name from alpha_tokens.csv")
    p.add_argument("--board-id", type=int, required=True)
    p.add_argument("--input", type=str, default="./data/alpha/alpha_tokens.csv")
    p.add_argument("--out", type=str, default="./data/alpha/monday_matches.csv")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--term", type=str, default=None, help="Optional search term to pre-filter Monday items")
    p.add_argument("--domain-filter", type=str, default=None, help="Domain substring to filter Website column by (e.g. talex.world)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    code, _ = run(args.board_id, args.input, args.out, args.limit, term=args.term, domain_filter=args.domain_filter)
    return code


if __name__ == "__main__":
    raise SystemExit(main())


