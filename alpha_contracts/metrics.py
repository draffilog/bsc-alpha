import asyncio
import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from aiolimiter import AsyncLimiter
from rich.console import Console

console = Console()
DEMO_BASE = "https://api.coingecko.com/api/v3"
PRO_BASE = "https://pro-api.coingecko.com/api/v3"


async def fetch_token_core(client: httpx.AsyncClient, rate: AsyncLimiter, address: str, params: Dict, log) -> Tuple[Optional[Dict], Optional[int]]:
    url = f"/coins/binance-smart-chain/contract/{address}"
    async with rate:
        r = await client.get(url, params=params, timeout=60)
    return (r.json() if r.status_code == 200 else None, r.status_code)


async def fetch_coin_rank_core(client: httpx.AsyncClient, rate: AsyncLimiter, cg_id: str, params: Dict, log) -> Optional[int]:
    """Fetch global market_cap_rank from /coins/{id}."""
    url = f"/coins/{cg_id}"
    q = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "false",
        "developer_data": "false",
        "sparkline": "false",
    }
    # carry API key param if present
    if "x_cg_pro_api_key" in params:
        q["x_cg_pro_api_key"] = params["x_cg_pro_api_key"]
    try:
        async with rate:
            r = await client.get(url, params=q, timeout=60)
        if r.status_code != 200:
            log(f"rank fetch {cg_id} -> {r.status_code}")
            return None
        data = r.json()
        rank = data.get("market_cap_rank")
        try:
            return int(rank) if rank is not None else None
        except Exception:
            return None
    except Exception as e:
        log(f"rank fetch error {cg_id}: {e}")
        return None


async def fetch_onchain_fallback(client: httpx.AsyncClient, rate: AsyncLimiter, address: str, params: Dict, log) -> Optional[Dict]:
    # Best-effort fallback using Onchain Token Info endpoint
    # GET /onchain/networks/bsc/tokens/{address}
    url = f"/onchain/networks/bsc/tokens/{address}"
    try:
        async with rate:
            r = await client.get(url, params=params, timeout=60)
        if r.status_code != 200:
            log(f"Onchain fallback {address} -> {r.status_code}")
            return None
        return r.json()
    except Exception as e:
        log(f"Onchain fallback error {address}: {e}")
        return None


async def fetch_token(client: httpx.AsyncClient, rate: AsyncLimiter, address: str, params: Dict, log) -> Optional[Dict]:
    for attempt in range(5):
        try:
            data, code = await fetch_token_core(client, rate, address, params, log)
            if code == 200:
                log(f"OK {address}")
                return data
            if code == 404:
                log(f"404 Not Found {address}")
                # try onchain fallback
                return await fetch_onchain_fallback(client, rate, address, params, log)
            if code == 400:
                log(f"400 Bad Request {address}")
                return await fetch_onchain_fallback(client, rate, address, params, log)
            if code == 429:
                wait = 1.5 * (attempt + 1) + random.random()
                log(f"429 Too Many Requests {address}, retry in {wait:.1f}s")
                await asyncio.sleep(wait)
                continue
            log(f"{code} {address}")
            return None
        except Exception as e:
            if attempt < 4:
                log(f"ERR {address}: {e}, retry {attempt+1}")
                await asyncio.sleep(0.8 * (attempt + 1))
                continue
            log(f"SKIP {address}: {e}")
            return None


async def run(in_tokens_csv: Path, out_metrics_csv: Path, concurrency_rps: int = 5, api_key: Optional[str] = None) -> int:
    # Prepare logging
    log_path = out_metrics_csv.parent / "metrics_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    def log(msg: str) -> None:
        with log_path.open("a") as lf:
            lf.write(msg + "\n")
        console.print(msg)

    try:
        if log_path.exists():
            log_path.unlink()
    except Exception:
        pass

    # Load addresses
    addresses: List[str] = []
    with in_tokens_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            addr = (row.get("address") or "").lower()
            if addr.startswith("0x") and len(addr) == 42:
                addresses.append(addr)
    if not addresses:
        console.print("[red]No addresses found in input CSV[/]")
        return 1

    # Configure client
    headers = {"Accept": "application/json"}
    params_base: Dict[str, str] = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "false",
        "developer_data": "false",
        "sparkline": "false",
    }
    api_key = api_key or os.getenv("COINGECKO_API_KEY") or os.getenv("CG_API_KEY")
    base_url = DEMO_BASE
    if api_key:
        base_url = PRO_BASE
        params_base["x_cg_pro_api_key"] = api_key
        headers["x-cg-pro-api-key"] = api_key
        log("Using CoinGecko Pro API base and key")

    rate = AsyncLimiter(concurrency_rps, 1)
    async with httpx.AsyncClient(base_url=base_url, headers=headers, follow_redirects=True) as client:
        results: List[Dict] = []
        for idx, addr in enumerate(addresses, 1):
            log(f"[{idx}/{len(addresses)}] GET {addr}")
            data = await fetch_token(client, rate, addr, params_base, log)
            if not data:
                continue
            # Normalize from either coins or onchain format
            md = ((data.get("market_data") or {}) if isinstance(data, dict) and "market_data" in data else {})
            cur = (md.get("current_price") or {}).get("usd") if md else None
            ath = (md.get("ath") or {}).get("usd") if md else None
            ath_date = (md.get("ath_date") or {}).get("usd") if md else None
            mcap = (md.get("market_cap") or {}).get("usd") if md else None
            cg_id = data.get("id") if isinstance(data, dict) else None
            global_rank = data.get("market_cap_rank") if isinstance(data, dict) else None
            # If rank missing, query /coins/{id}
            if cg_id and not global_rank:
                rk = await fetch_coin_rank_core(client, rate, cg_id, params_base, log)
                if rk is not None:
                    global_rank = rk
            name = data.get("name") or ""
            symbol = (data.get("symbol") or "").upper()
            # Onchain fallback mapping
            if not cur and isinstance(data, dict) and "data" in data:
                try:
                    attrs = (data.get("data") or {}).get("attributes") or {}
                    cur = attrs.get("usd_price") or attrs.get("price_usd")
                    name = attrs.get("name", name)
                    symbol = (attrs.get("symbol") or symbol or "").upper()
                except Exception:
                    pass
            results.append({
                "address": addr,
                "symbol": symbol,
                "name": name,
                "price_usd": cur,
                "ath_price_usd": ath,
                "ath_date": ath_date,
                "market_cap_usd": mcap,
                "cg_id": cg_id,
                "global_rank": int(global_rank) if isinstance(global_rank, (int, float, str)) and str(global_rank).isdigit() else None,
            })

    out_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_metrics_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "address",
                "symbol",
                "name",
                "price_usd",
                "ath_price_usd",
                "ath_date",
                "market_cap_usd",
                "cg_id",
                "global_rank",
            ],
        )
        w.writeheader()
        for row in results:
            w.writerow(row)

    log(f"Saved {len(results)} rows to {out_metrics_csv}")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch current and ATH metrics for BSC contracts via CoinGecko")
    parser.add_argument("--in", dest="in_csv", default="./data/alpha/alpha_tokens.csv")
    parser.add_argument("--out", dest="out_csv", default="./data/alpha/metrics.csv")
    parser.add_argument("--rps", dest="rps", type=int, default=5)
    parser.add_argument("--api-key", dest="api_key", type=str, default=None, help="CoinGecko Pro API key or set COINGECKO_API_KEY")
    args = parser.parse_args()

    raise SystemExit(asyncio.run(run(Path(args.in_csv), Path(args.out_csv), args.rps, args.api_key)))
