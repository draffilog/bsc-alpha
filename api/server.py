from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from alpha_contracts.alpha_klines import extract_earliest_candle_from_json


app = FastAPI(title="Alpha Programmatic API", version="1.0.0")


class ListingResponse(BaseModel):
	chainId: int
	contract: str
	listing_timestamp_ms: int
	listing_iso: str
	first_pair: Optional[str] = None
	initial_price_quote: Optional[float] = None
	initial_price_usd: Optional[float] = None
	data_source: str = "alpha_ui_network"
	notes: Optional[str] = None


@app.get("/alpha/v1/listings/{chainId}/{contractAddress}", response_model=ListingResponse)
async def get_listing_metadata(
	chainId: int,
	contractAddress: str,
	quote: Optional[str] = Query(default=None),
):
	"""
	Return listing metadata derived from the earliest candle in Binance Alpha UI network responses.
	We load the Binance Alpha token page headlessly and inspect JSON responses to locate kline arrays.
	"""
	# Lazy import to avoid Playwright import cost on module import
	from playwright.async_api import async_playwright
	from alpha_contracts.listing_alpha import BINANCE_ALPHA_URL
	from alpha_contracts.alpha_klines import extract_earliest_candle_from_json

	# Only BSC (56) supported for now
	if chainId != 56:
		raise HTTPException(status_code=400, detail="Only chainId 56 (BSC) supported currently")

	target_url = BINANCE_ALPHA_URL.format(contractAddress.lower())
	earliest_candle = None

	async with async_playwright() as p:
		browser = await p.chromium.launch(headless=True)
		context = await browser.new_context()
		page = await context.new_page()

		async def on_response(resp):
			nonlocal earliest_candle
			try:
				ct = (resp.headers or {}).get("content-type", "")
				if "application/json" not in ct:
					return
				data = await resp.json()
			except Exception:
				return
			c = extract_earliest_candle_from_json(data)
			if c:
				if (earliest_candle is None) or (c.open_time_ms < earliest_candle.open_time_ms):
					earliest_candle = c

		page.on("response", on_response)
		await page.goto(target_url, wait_until="networkidle", timeout=90000)
		# Small scrolls to trigger lazy charts if any
		for _ in range(20):
			await page.mouse.wheel(0, 1200)
			await page.wait_for_timeout(150)
		await context.close()
		await browser.close()

	if earliest_candle is None:
		raise HTTPException(status_code=404, detail="Unable to locate earliest candle from Alpha UI")

	# The Alpha chart quote will typically be the base quote of the default pair (e.g., USDT)
	# We do not force-translate quotes here; return open as quote price if quote is unknown.
	initial_price_quote = float(earliest_candle.open)

	return ListingResponse(
		chainId=chainId,
		contract=contractAddress.lower(),
		listing_timestamp_ms=earliest_candle.open_time_ms,
		listing_iso=earliest_candle.open_time_iso,
		first_pair=None,
		initial_price_quote=initial_price_quote,
		initial_price_usd=None,  # Not available without FX normalization; can be added if present
		data_source="alpha_ui_network",
		notes="Derived from first visible kline in Alpha UI network responses",
	)


@app.get("/alpha/v1/klines")
async def get_earliest_kline(
	chainId: int = Query(...),
	contract: str = Query(...),
	interval: str = Query("1m"),
	quote: Optional[str] = Query(None),
	sort: str = Query("asc"),
	limit: int = Query(1),
):
	"""
	Return earliest candle as array: [openTime, open, high, low, close, volume, quote, pair_id]
	We only honor sort=asc&limit=1; other combos are not implemented here.
	"""
	if chainId != 56:
		raise HTTPException(status_code=400, detail="Only chainId 56 (BSC) supported currently")
	if sort != "asc" or limit != 1:
		raise HTTPException(status_code=400, detail="Only sort=asc&limit=1 supported for earliest candle")

	# Reuse the same mechanism as listing endpoint to capture the earliest candle from network responses.
	from playwright.async_api import async_playwright
	from alpha_contracts.listing_alpha import BINANCE_ALPHA_URL

	target_url = BINANCE_ALPHA_URL.format(contract.lower())
	earliest_candle = None

	async with async_playwright() as p:
		browser = await p.chromium.launch(headless=True)
		context = await browser.new_context()
		page = await context.new_page()

		async def on_response(resp):
			nonlocal earliest_candle
			try:
				ct = (resp.headers or {}).get("content-type", "")
				if "application/json" not in ct:
					return
				data = await resp.json()
			except Exception:
				return
			c = extract_earliest_candle_from_json(data)
			if c:
				if (earliest_candle is None) or (c.open_time_ms < earliest_candle.open_time_ms):
					earliest_candle = c

		page.on("response", on_response)
		await page.goto(target_url, wait_until="networkidle", timeout=90000)
		for _ in range(20):
			await page.mouse.wheel(0, 1200)
			await page.wait_for_timeout(150)
		await context.close()
		await browser.close()

	if earliest_candle is None:
		raise HTTPException(status_code=404, detail="Unable to locate earliest candle from Alpha UI")

	# Attach provided quote if any; otherwise None
	earliest_candle.quote = quote

	return [earliest_candle.as_list()]





