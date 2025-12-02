import asyncio
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from rich.console import Console

console = Console()

TOKEN_LIST_URL = "https://www.binance.com/bapi/defi/v1/public/wallet-direct/buw/wallet/cex/alpha/all/token/list"
PAIR_KLINES_URL = "https://www.binance.com/bapi/defi/v1/public/alpha-trade/klines"
PAIR_INTERVAL = "1m"
PAIR_QUOTE = "USDT"
PAGE_TIMEOUT = 30
CONCURRENCY = 8

SUPPORTED_CHAINS = {
    "bsc": "BSC",
    "ethereum": "Ethereum",
    "solana": "Solana",
    "base": "Base",
}


def to_iso(ts_ms: int) -> str:
    try:
        return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()
    except Exception:
        return ""


async def fetch_alpha_index(client: httpx.AsyncClient) -> List[Dict]:
    resp = await client.get(TOKEN_LIST_URL, timeout=PAGE_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("data") or []


async def fetch_listing(token: Dict, client: httpx.AsyncClient) -> Optional[Tuple[int, float]]:
    alpha_id = token.get("alphaId") or token.get("tokenId")
    if not alpha_id:
        return None
    if not alpha_id.startswith("ALPHA_"):
        alpha_id = f"ALPHA_{alpha_id}"
    symbol = f"{alpha_id}{PAIR_QUOTE}"
    params = {
        "symbol": symbol,
        "interval": PAIR_INTERVAL,
        "limit": 1,
        "startTime": 0,
        "endTime": int(datetime.utcnow().timestamp() * 1000),
    }
    resp = await client.get(PAIR_KLINES_URL, params=params, timeout=PAGE_TIMEOUT)
    if resp.status_code != 200:
        return None
    payload = resp.json()
    if not payload.get("success"):
        return None
    rows = payload.get("data") or []
    if not rows:
        return None
    row = rows[0]
    try:
        ts_ms = int(row[0])
        open_price = float(row[1])
    except Exception:
        return None
    return ts_ms, open_price


async def run(out_csv: Path, chains: List[str]) -> int:
    wanted = {chain: SUPPORTED_CHAINS[chain] for chain in chains}
    results: List[Dict[str, str]] = []
    sema = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(timeout=PAGE_TIMEOUT) as client:
        tokens = await fetch_alpha_index(client)
        filtered = [
            t for t in tokens if (t.get("chainName") or "").lower() in {name.lower() for name in wanted.values()}
        ]
        console.print(f"Processing {len(filtered)} tokens across {', '.join(wanted.values())}")

        async def process(idx: int, token: Dict) -> None:
            chain_name = token.get("chainName") or "Unknown"
            addr = (token.get("contractAddress") or "").lower()
            symbol = token.get("symbol") or ""
            name = token.get("name") or ""
            async with sema:
                try:
                    info = await fetch_listing(token, client)
                except Exception as e:
                    console.print(f"warn {addr}: {e}")
                    info = None
            if info:
                ts, price = info
                results.append({
                    "chain": chain_name,
                    "chain_id": str(token.get("chainId") or ""),
                    "address": addr,
                    "symbol": symbol,
                    "name": name,
                    "listing_timestamp_ms": str(ts),
                    "listing_date_alpha": to_iso(ts),
                    "listing_price_usdt": f"{price:.10g}",
                    "alpha_pair": f"{token.get('alphaId') or token.get('tokenId')}{PAIR_QUOTE}",
                })

        await asyncio.gather(*(process(idx, token) for idx, token in enumerate(filtered)))

    fieldnames = [
        "chain",
        "chain_id",
        "address",
        "symbol",
        "name",
        "listing_timestamp_ms",
        "listing_date_alpha",
        "listing_price_usdt",
        "alpha_pair",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    console.print(f"Saved {len(results)} rows -> {out_csv}")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Alpha listings for multiple chains via USDT pair klines")
    parser.add_argument("--out", default="./data/alpha/cross_chain_listings.csv")
    parser.add_argument(
        "--chains",
        default="bsc,ethereum,solana,base",
        help="Comma-separated list of chains (bsc,ethereum,solana,base)",
    )
    args = parser.parse_args()

    selected = []
    for chain in (part.strip().lower() for part in args.chains.split(",")):
        if chain in SUPPORTED_CHAINS:
            selected.append(chain)
        else:
            console.print(f"[yellow]Unknown chain '{chain}', skipping[/]")
    if not selected:
        console.print("[red]No valid chains provided[/]")
        raise SystemExit(1)

    raise SystemExit(asyncio.run(run(Path(args.out), selected)))

