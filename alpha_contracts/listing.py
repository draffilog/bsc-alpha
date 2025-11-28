import asyncio
import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from aiolimiter import AsyncLimiter
from rich.console import Console

console = Console()
COINGECKO_BASE_DEMO = "https://api.coingecko.com/api/v3"
COINGECKO_BASE_PRO = "https://pro-api.coingecko.com/api/v3"


async def fetch_market_chart(client: httpx.AsyncClient, rate: AsyncLimiter, cg_id: str, log) -> Optional[Dict]:
	params = {"vs_currency": "usd", "days": "max", "interval": "daily"}
	async with rate:
		resp = await client.get(f"/coins/{cg_id}/market_chart", params=params, timeout=60)
	if resp.status_code != 200:
		log(f"chart {cg_id} -> {resp.status_code}")
		return None
	return resp.json()


def extract_first_price(chart: Dict) -> Optional[Dict]:
	try:
		prices = chart.get("prices") or []
		if not prices:
			return None
		# each item: [timestamp_ms, price]
		ts_ms, px = prices[0]
		iso = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
		return {"listing_date": iso, "listing_price_usd": float(px)}
	except Exception:
		return None


async def run(metrics_csv: Path, out_csv: Path, rps: int = 5, api_key: Optional[str] = None) -> int:
	if not metrics_csv.exists():
		console.print(f"[red]Missing metrics file {metrics_csv}[/]")
		return 2
	rows: List[Dict] = []
	with metrics_csv.open(newline="") as f:
		r = csv.DictReader(f)
		for row in r:
			rows.append(row)
	# set up client
	headers = {"Accept": "application/json"}
	base = COINGECKO_BASE_DEMO
	api_key = api_key or os.getenv("COINGECKO_API_KEY") or os.getenv("CG_API_KEY")
	params: Dict[str, str] = {}
	if api_key:
		base = COINGECKO_BASE_PRO
		headers["x-cg-pro-api-key"] = api_key
		params["x_cg_pro_api_key"] = api_key

	rate = AsyncLimiter(rps, 1)
	async with httpx.AsyncClient(base_url=base, headers=headers, follow_redirects=True) as client:
		out_rows: List[Dict] = []
		for row in rows:
			cg_id = (row.get("cg_id") or "").strip()
			addr = row.get("address")
			if not cg_id:
				continue
			try:
				chart = await fetch_market_chart(client, rate, cg_id, console.print)
				if not chart:
					continue
				info = extract_first_price(chart)
				if not info:
					continue
				out_rows.append({"address": addr, **info})
			except Exception as e:
				console.print(f"warn listing {cg_id}: {e}")

	out_csv.parent.mkdir(parents=True, exist_ok=True)
	with out_csv.open("w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=["address", "listing_date", "listing_price_usd"])
		w.writeheader()
		for r in out_rows:
			w.writerow(r)
	console.print(f"Saved {len(out_rows)} rows to {out_csv}")
	return 0


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Fetch listing date/price from CoinGecko market chart")
	parser.add_argument("--metrics", default="./data/alpha/metrics.csv")
	parser.add_argument("--out", default="./data/alpha/listings.csv")
	parser.add_argument("--rps", type=int, default=5)
	parser.add_argument("--api-key", dest="api_key", type=str, default=None)
	args = parser.parse_args()

	raise SystemExit(asyncio.run(run(Path(args.metrics), Path(args.out), args.rps, args.api_key)))
