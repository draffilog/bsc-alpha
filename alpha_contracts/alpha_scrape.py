import asyncio
import json
from pathlib import Path
from typing import List, Set

from rich.console import Console

console = Console()

BINANCE_ALPHA_DEFAULT = "https://www.binance.com/en/alpha/bsc/0xe6df05ce8c8301223373cf5b969afcb1498c5528"


def find_addresses_in_text(text: str) -> Set[str]:
	import re
	return set(m.group(0).lower() for m in re.finditer(r"0x[a-fA-F0-9]{40}", text or ""))


async def extract_addresses_from_alpha(url: str) -> List[str]:
	from playwright.async_api import async_playwright

	addresses: Set[str] = set()

	async with async_playwright() as p:
		browser = await p.chromium.launch(headless=False, args=["--disable-blink-features=AutomationControlled"])  # headed for stability
		context = await browser.new_context(
			user_agent=(
				"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
				"AppleWebKit/537.36 (KHTML, like Gecko) "
				"Chrome/129.0.0.0 Safari/537.36"
			),
			viewport={"width": 1400, "height": 900},
			locale="en-US",
		)
		page = await context.new_page()
		page.set_default_timeout(60000)

		await page.goto(url, wait_until="load", timeout=90000)
		await page.wait_for_timeout(1500)

		# Handle cookie consent if shown
		for sel in [
			"text=/^Allow All$/",
			"text=/^Confirm My Choice$/",
			"role=button[name=/^Allow All$/i]",
			"role=button[name=/^Confirm My Choice$/i]",
		]:
			try:
				loc = page.locator(sel)
				if await loc.count() > 0:
					await loc.first.click(timeout=2000)
					await page.wait_for_timeout(800)
					break
			except Exception:
				pass

		# Ensure BSC tab selected
		for selector in [
			"role=button[name=/^\\s*BSC\\s*$/i]",
			"button:has-text('BSC')",
			"text=/^\\s*BSC\\s*$/i",
		]:
			try:
				loc = page.locator(selector)
				if await loc.count() > 0:
					await loc.first.click(timeout=1500)
					await page.wait_for_timeout(600)
					break
			except Exception:
				continue

		# Target the ReactVirtualized list container
		grid_selector = "div.ReactVirtualized__Grid.ReactVirtualized__List[role='grid']"
		inner_selector = "div.ReactVirtualized__Grid__innerScrollContainer"
		try:
			await page.wait_for_selector(grid_selector, timeout=20000)
		except Exception:
			grid_selector = None

		row_anchor_selector = (
			"div.cursor-pointer.hover\\:bg-Input.px-\\[16px\\] a[href*='/alpha/bsc/0x']"
		)

		async def collect_from_dom() -> int:
			# Prefer grid-specific anchors, but also fall back to page-wide alpha bsc links
			all_hrefs = []
			try:
				if grid_selector:
					all_hrefs += await page.eval_on_selector_all(
						f"{grid_selector} {row_anchor_selector}",
						"els => els.map(e => e.getAttribute('href'))",
					)
			except Exception:
				pass
			try:
				all_hrefs += await page.eval_on_selector_all(
					"a[href*='/alpha/bsc/0x']",
					"els => els.map(e => e.getAttribute('href'))",
				)
			except Exception:
				pass
			before = len(addresses)
			for href in all_hrefs or []:
				if not href:
					continue
				for addr in find_addresses_in_text(href):
					addresses.add(addr)
			return len(addresses) - before

		stagnant = 0
		for _ in range(2400):
			added1 = await collect_from_dom()
			# Scroll grid and inner container; fallback to window
			try:
				if grid_selector and await page.locator(grid_selector).count() > 0:
					await page.eval_on_selector(grid_selector, "el => { el.scrollTop = Math.min(el.scrollTop + 1200, el.scrollHeight); }")
					await page.wait_for_timeout(120)
					if await page.locator(inner_selector).count() > 0:
						await page.eval_on_selector(inner_selector, "el => { el.scrollTop = Math.min(el.scrollTop + 1200, el.scrollHeight); }")
				else:
					await page.evaluate("window.scrollBy(0, 1200)")
			except Exception:
				pass
			await page.wait_for_timeout(200)
			added2 = await collect_from_dom()
			if (added1 + added2) == 0:
				stagnant += 1
				if stagnant >= 30:
					break
			else:
				stagnant = 0

		await context.close()
		await browser.close()

	return sorted(addresses)


async def enrich_with_tickers(addresses: List[str]) -> List[dict]:
	from .fetch_contracts import gather_tokens

	tokens_map = await gather_tokens(["pancakeswap", "coingecko"])  # BSC only
	enriched = []
	for addr in addresses:
		t = tokens_map.get(addr.lower())
		enriched.append({
			"address": addr.lower(),
			"symbol": (t or {}).get("symbol", ""),
			"name": (t or {}).get("name", ""),
		})
	return enriched


async def run(url: str, out_dir: str) -> int:
	out = Path(out_dir)
	out.mkdir(parents=True, exist_ok=True)

	addresses = await extract_addresses_from_alpha(url)
	enriched = await enrich_with_tickers(addresses)

	# Save outputs
	(out / "alpha_tokens.json").write_text(json.dumps(enriched, indent=2))
	import csv
	with (out / "alpha_tokens.csv").open("w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["address", "symbol", "name"])
		for row in enriched:
			w.writerow([row["address"], row["symbol"], row["name"]])

	console.print(f"Saved {len(enriched)} alpha tokens to {out / 'alpha_tokens.csv'}")
	return 0


def main():
	import argparse

	parser = argparse.ArgumentParser(description="Scrape Binance Alpha BSC token list and export CSV/JSON")
	parser.add_argument("--url", type=str, default=BINANCE_ALPHA_DEFAULT)
	parser.add_argument("--out", type=str, default="./data/alpha")
	args = parser.parse_args()

	return asyncio.run(run(args.url, args.out))


if __name__ == "__main__":
	raise SystemExit(main())
