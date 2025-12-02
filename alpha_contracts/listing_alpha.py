import asyncio
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import httpx
from rich.console import Console

console = Console()

TOKEN_LIST_URL = "https://www.binance.com/bapi/defi/v1/public/wallet-direct/buw/wallet/cex/alpha/all/token/list"
PAIR_KLINES_URL = "https://www.binance.com/bapi/defi/v1/public/alpha-trade/klines"
PAIR_INTERVAL = "1m"
PAIR_QUOTE = "USDT"
PAGE_TIMEOUT = 30
CONCURRENCY = 8

Number = Union[int, float]


def to_iso(ts_ms: Number) -> str:
	try:
		return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc).isoformat()
	except Exception:
		return ""


async def fetch_alpha_index(client: httpx.AsyncClient) -> Dict[str, Dict]:
	resp = await client.get(TOKEN_LIST_URL, timeout=PAGE_TIMEOUT)
	resp.raise_for_status()
	data = resp.json().get("data") or []
	return {
		(item.get("contractAddress") or "").lower(): item
		for item in data
		if str(item.get("chainId")).lower() in {"56", "bsc"}
	}


async def fetch_listing_for_address(
	address: str,
	client: httpx.AsyncClient,
	alpha_index: Dict[str, Dict],
) -> Optional[Tuple[int, float]]:
	info = alpha_index.get(address.lower())
	if not info:
		return None
	alpha_id = info.get("alphaId") or info.get("tokenId")
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


async def run(in_tokens_csv: Path, out_csv: Path, limit: int = 0) -> int:
	if not in_tokens_csv.exists():
		console.print(f"[red]Missing input tokens CSV {in_tokens_csv}[/]")
		return 2
	entries: List[Dict[str, str]] = []
	with in_tokens_csv.open(newline="") as f:
		r = csv.DictReader(f)
		for row in r:
			addr = (row.get("address") or "").strip()
			if addr:
				entries.append({
					"address": addr,
					"symbol": (row.get("symbol") or "").strip(),
					"name": (row.get("name") or "").strip(),
				})
	if limit and limit > 0:
		entries = entries[:limit]

	results: List[Optional[Dict[str, str]]] = [None] * len(entries)
	sema = asyncio.Semaphore(CONCURRENCY)

	async with httpx.AsyncClient(timeout=PAGE_TIMEOUT) as client:
		alpha_index = await fetch_alpha_index(client)
		console.print(f"Loaded {len(alpha_index)} Alpha tokens from index")

		async def process(idx: int, item: Dict[str, str]) -> None:
			addr = item["address"]
			console.print(f"[{idx + 1}/{len(entries)}] {addr}")
			async with sema:
				try:
					result = await fetch_listing_for_address(addr, client, alpha_index)
				except Exception as e:
					console.print(f"warn {addr}: {e}")
					result = None
			if result:
				ts, price = result
				results[idx] = {
					"address": addr,
					"symbol": item.get("symbol", ""),
					"name": item.get("name", ""),
					"listing_timestamp_ms": str(ts),
					"listing_date_alpha": to_iso(ts),
					"listing_price_quote": f"{price:.10g}",
					"listing_quote": PAIR_QUOTE,
					"alpha_pair": "",
				}
			else:
				results[idx] = {
					"address": addr,
					"symbol": item.get("symbol", ""),
					"name": item.get("name", ""),
					"listing_timestamp_ms": "",
					"listing_date_alpha": "",
					"listing_price_quote": "",
					"listing_quote": PAIR_QUOTE,
					"alpha_pair": "",
				}

		await asyncio.gather(*(process(idx, item) for idx, item in enumerate(entries)))

	rows = [r for r in results if r is not None]

	out_csv.parent.mkdir(parents=True, exist_ok=True)
	with out_csv.open("w", newline="") as f:
		fieldnames = [
			"address",
			"symbol",
			"name",
			"listing_timestamp_ms",
			"listing_date_alpha",
			"listing_price_quote",
			"listing_quote",
			"alpha_pair",
		]
		w = csv.DictWriter(f, fieldnames=fieldnames)
		w.writeheader()
		for r in rows:
			w.writerow(r)
	console.print(f"Saved {len(rows)} rows -> {out_csv}")
	return 0


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Fetch listing date from Binance Alpha public klines API")
	parser.add_argument("--in", dest="in_csv", default="./data/alpha/alpha_tokens.csv")
	parser.add_argument("--out", dest="out_csv", default="./data/alpha/binance_listings.csv")
	parser.add_argument("--limit", type=int, default=0)
	args = parser.parse_args()

	raise SystemExit(asyncio.run(run(Path(args.in_csv), Path(args.out_csv), args.limit)))
