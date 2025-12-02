import io
import csv
import asyncio
import sys
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import httpx

# Ensure project root is on sys.path so we can import alpha_contracts
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

TOKENS_CSV = Path("./data/alpha/alpha_tokens.csv")
METRICS_CSV = Path("./data/alpha/metrics.csv")
LISTINGS_CSV = Path("./data/alpha/listings.csv")
BINANCE_LISTINGS_CSV = Path("./data/alpha/binance_listings.csv")
BINANCE_LISTINGS_COPY_CSV = Path("./data/alpha/binance_listings copy.csv")
CROSS_CHAIN_LISTINGS_CSV = Path("./data/alpha/cross_chain_listings.csv")
LOG_PATH = Path("./data/alpha/metrics_log.txt")

st.set_page_config(page_title="BSC Alpha Tokens Dashboard", layout="wide")
st.title("BSC Alpha Tokens — Performance Dashboard")

# Propagate secrets to env (for Streamlit Cloud deploys)
if "COINGECKO_API_KEY" in st.secrets:
	os.environ["COINGECKO_API_KEY"] = st.secrets["COINGECKO_API_KEY"]
if "BSCSCAN_API_KEY" in st.secrets:
	os.environ["BSCSCAN_API_KEY"] = st.secrets["BSCSCAN_API_KEY"]

# Removed instructional intro to keep presentation-ready layout

# Prefer Binance Alpha listings as primary dataset if present
primary_from_alpha = False
if BINANCE_LISTINGS_CSV.exists():
	primary_from_alpha = True
	primary_path = BINANCE_LISTINGS_CSV
elif BINANCE_LISTINGS_COPY_CSV.exists():
	primary_from_alpha = True
	primary_path = BINANCE_LISTINGS_COPY_CSV
else:
	primary_path = TOKENS_CSV

# Load tokens
if not primary_path.exists():
	st.error(f"Missing {primary_path}")
	st.stop()

raw_df = pd.read_csv(primary_path)
if primary_from_alpha:
	# Normalize columns and expose listing_date
	keep = [c for c in raw_df.columns if c in {"address","symbol","name","listing_date_alpha","listing_timestamp_ms","listing_price_quote","listing_quote","alpha_pair"}]
	tokens_df = raw_df[keep].copy()
	if "listing_date_alpha" in tokens_df.columns:
		tokens_df["listing_date"] = tokens_df["listing_date_alpha"]
else:
	tokens_df = raw_df.copy()

cross_chain_df = None
if CROSS_CHAIN_LISTINGS_CSV.exists():
	try:
		cross_chain_df = pd.read_csv(CROSS_CHAIN_LISTINGS_CSV)
	except Exception:
		cross_chain_df = None

# Helper: ensure metrics exist and are recent enough

def need_refresh(metrics_path: Path, expected_rows: int) -> bool:
	if not metrics_path.exists():
		return True
	try:
		m_df = pd.read_csv(metrics_path)
		if len(m_df) < max(5, int(0.7 * expected_rows)):
			return True
	except Exception:
		return True
	mtime = datetime.fromtimestamp(metrics_path.stat().st_mtime)
	return (datetime.now() - mtime) > timedelta(days=1)

# Controls (kept minimal)
right = st.container()
with right:
	colA, colB = st.columns(2)
	with colA:
		force_refresh = st.button("Refresh metrics now")
	with colB:
		rps = st.number_input("Requests/sec", min_value=1, max_value=50, value=8, step=1)
	force_listing = st.button("Refresh listings now (CoinGecko)")

# Live log area
log_area = st.empty()

# Auto/forced refresh using subprocess so Streamlit UI stays responsive
if force_refresh or need_refresh(METRICS_CSV, len(tokens_df)):
	st.info("Fetching metrics from CoinGecko... logs streaming below")
	# Clear old log file
	try:
		if LOG_PATH.exists():
			LOG_PATH.unlink()
	except Exception:
		pass
	cmd = [
		sys.executable,
		"-m",
		"alpha_contracts.metrics",
		"--in",
		str(TOKENS_CSV),
		"--out",
		str(METRICS_CSV),
		"--rps",
		str(int(rps)),
	]
	env = os.environ.copy()
	proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env)
	# Tail log while process runs
	while proc.poll() is None:
		if LOG_PATH.exists():
			try:
				text = LOG_PATH.read_text()
				lines = text.strip().splitlines()
				log_area.code("\n".join(lines[-200:]))
			except Exception:
				pass
		time.sleep(0.8)
	# Final read
	if LOG_PATH.exists():
		try:
			text = LOG_PATH.read_text()
			lines = text.strip().splitlines()
			log_area.code("\n".join(lines[-200:]))
		except Exception:
			pass
	st.toast(f"Metrics fetch finished with exit {proc.returncode}", icon="✅" if proc.returncode == 0 else "⚠️")

# Optional: fetch listings (CoinGecko fallback)
if force_listing:
	st.info("Fetching listing date/price from CoinGecko market charts (first data point)")
	cmd = [
		sys.executable,
		"-m",
		"alpha_contracts.listing",
		"--metrics",
		str(METRICS_CSV),
		"--out",
		str(LISTINGS_CSV),
		"--rps",
		str(int(rps)),
	]
	proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=os.environ.copy())
	proc.wait()
	st.toast(f"Listings fetch finished with exit {proc.returncode}", icon="✅" if proc.returncode == 0 else "⚠️")

# Load metrics
metrics_df = pd.read_csv(METRICS_CSV) if METRICS_CSV.exists() else pd.DataFrame(columns=["address","price_usd","ath_price_usd","ath_date","market_cap_usd","global_rank","cg_id"]) 
metrics_df = metrics_df.drop(columns=[c for c in ["symbol","name"] if c in metrics_df.columns], errors="ignore")

# Merge on address
merged = tokens_df.merge(metrics_df, on="address", how="left")

# If primary source wasn't the alpha listings, merge them now if present
if not primary_from_alpha:
	alpha_listing_path = BINANCE_LISTINGS_CSV if BINANCE_LISTINGS_CSV.exists() else (BINANCE_LISTINGS_COPY_CSV if BINANCE_LISTINGS_COPY_CSV.exists() else None)
	if alpha_listing_path:
		alpha_df = pd.read_csv(alpha_listing_path)
		alpha_df = alpha_df[[c for c in alpha_df.columns if c in {"address","listing_date_alpha","listing_price_quote","listing_quote","alpha_pair","listing_timestamp_ms"}]]
		merged = merged.merge(alpha_df, on="address", how="left", suffixes=("", "_alpha"))
		merged["listing_date"] = merged.get("listing_date") if "listing_date" in merged.columns else merged.get("listing_date_alpha")
else:
	# Already present from tokens_df
	if "listing_date" not in merged.columns and "listing_date_alpha" in merged.columns:
		merged["listing_date"] = merged["listing_date_alpha"]

# Drop columns no longer desired
merged = merged.drop(columns=[c for c in ["alpha_pair","listing_quote"] if c in merged.columns], errors="ignore")

# Derived columns
for col in ["price_usd", "ath_price_usd", "market_cap_usd", "global_rank"]:
	if col not in merged.columns:
		merged[col] = pd.NA

# Sanity guards for outliers
px = pd.to_numeric(merged["price_usd"], errors="coerce")
ath_raw = pd.to_numeric(merged["ath_price_usd"], errors="coerce")
lq_raw = pd.to_numeric(merged.get("listing_price_quote"), errors="coerce") if "listing_price_quote" in merged.columns else pd.Series([])
ath_outlier = (ath_raw <= 0) | ((px > 0) & (ath_raw / px > 10000)) | (ath_raw > 1_000_000)
lq_outlier = (lq_raw <= 0) | ((px > 0) & (lq_raw / px > 10000)) | (lq_raw > 1_000_000)
merged["ath_outlier"] = ath_outlier.fillna(False)
if not lq_raw.empty:
	merged["listing_outlier"] = lq_outlier.fillna(False)
merged["ath_price_usd"] = ath_raw.mask(ath_outlier)
if "listing_price_quote" in merged.columns:
	merged["listing_price_quote"] = lq_raw.mask(lq_outlier)

# Helpers: CoinGecko utilities
@st.cache_data(show_spinner=False)

def cg_price_near_ts(cg_id: str, ts_ms: int, offset_sec: int = 60, window_sec: int = 300):
	if not cg_id or not ts_ms:
		return None
	base = "https://api.coingecko.com/api/v3"
	headers = {"Accept": "application/json"}
	params = {
		"vs_currency": "usd",
		"from": int((ts_ms + offset_sec * 1000 - window_sec * 1000) / 1000),
		"to": int((ts_ms + offset_sec * 1000 + window_sec * 1000) / 1000),
	}
	api_key = os.getenv("COINGECKO_API_KEY") or os.getenv("CG_API_KEY")
	if api_key:
		base = "https://pro-api.coingecko.com/api/v3"
		headers["x-cg-pro-api-key"] = api_key
		params["x_cg_pro_api_key"] = api_key
	try:
		with httpx.Client(base_url=base, headers=headers, follow_redirects=True, timeout=30) as client:
			r = client.get(f"/coins/{cg_id}/market_chart/range", params=params)
			if r.status_code != 200:
				return None
			prices = (r.json() or {}).get("prices") or []
			if not prices:
				return None
			target = ts_ms + offset_sec * 1000
			after = [p for p in prices if int(p[0]) >= target]
			chosen = after[0] if after else min(prices, key=lambda p: abs(int(p[0]) - target))
			return float(chosen[1])
	except Exception:
		return None

@st.cache_data(show_spinner=False)

def cg_ath_after(cg_id: str, from_ts_ms: int):
	if not cg_id:
		return None, None
	base = "https://api.coingecko.com/api/v3"
	headers = {"Accept": "application/json"}
	params = {
		"vs_currency": "usd",
		"from": int(from_ts_ms / 1000),
		"to": int(time.time()),
	}
	api_key = os.getenv("COINGECKO_API_KEY") or os.getenv("CG_API_KEY")
	if api_key:
		base = "https://pro-api.coingecko.com/api/v3"
		headers["x-cg-pro-api-key"] = api_key
		params["x_cg_pro_api_key"] = api_key
	try:
		with httpx.Client(base_url=base, headers=headers, follow_redirects=True, timeout=40) as client:
			r = client.get(f"/coins/{cg_id}/market_chart/range", params=params)
			if r.status_code != 200:
				return None, None
			prices = (r.json() or {}).get("prices") or []
			if not prices:
				return None, None
			mx = max(prices, key=lambda p: float(p[1]))
			return float(mx[1]), int(mx[0])
	except Exception:
		return None, None

@st.cache_data(show_spinner=False)

def cg_id_by_address(address: str):
	if not address:
		return None
	base = "https://api.coingecko.com/api/v3"
	headers = {"Accept": "application/json"}
	params = {}
	api_key = os.getenv("COINGECKO_API_KEY") or os.getenv("CG_API_KEY")
	if api_key:
		base = "https://pro-api.coingecko.com/api/v3"
		headers["x-cg-pro-api-key"] = api_key
		params["x_cg_pro_api_key"] = api_key
	try:
		with httpx.Client(base_url=base, headers=headers, follow_redirects=True, timeout=20) as client:
			r = client.get(f"/coins/binance-smart-chain/contract/{address}", params=params)
			if r.status_code != 200:
				return None
			data = r.json() or {}
			return data.get("id")
	except Exception:
		return None

# 1) Fill NaN ATH using price 1 minute after ath_date
if "ath_date" in merged.columns:
	mask_na = merged["ath_price_usd"].isna() & merged["ath_date"].notna()
	if mask_na.any():
		for idx, row in merged[mask_na].iterrows():
			try:
				cgid = str(row.get("cg_id") or "")
				if not cgid:
					cgid = cg_id_by_address(str(row.get("address") or ""))
					replaced = False
				ts = int(pd.to_datetime(row["ath_date"]).value // 1_000_000)
				price = cg_price_near_ts(cgid, ts, offset_sec=60, window_sec=300) if cgid else None
				if price and price < 1_000_000:
					merged.at[idx, "ath_price_usd"] = price
					merged.at[idx, "ath_outlier"] = False
			except Exception:
				continue

# 2) If ATH date is before listing date, recompute ATH only after listing
if "listing_date" in merged.columns and "ath_date" in merged.columns:
	mask_before = merged["ath_price_usd"].notna() & merged["ath_date"].notna() & merged["listing_date"].notna()
	if mask_before.any():
		for idx, row in merged[mask_before].iterrows():
			try:
				cgid = str(row.get("cg_id") or "") or cg_id_by_address(str(row.get("address") or ""))
				ath_ts = int(pd.to_datetime(row["ath_date"]).value // 1_000_000)
				lst_ts = int(pd.to_datetime(row["listing_date"]).value // 1_000_000)
				if cgid and ath_ts < lst_ts:
					val, ts2 = cg_ath_after(cgid, lst_ts)
					if val and val < 1_000_000:
						merged.at[idx, "ath_price_usd"] = val
						if ts2:
							merged.at[idx, "ath_date"] = pd.to_datetime(ts2, unit="ms", utc=True).isoformat()
			except Exception:
				continue

# 3) For remaining outliers/missing, fall back to global max
mask = merged["ath_price_usd"].isna() | merged["ath_outlier"]
if mask.any():
	for idx, row in merged[mask].iterrows():
		cgid = str(row.get("cg_id") or "") or cg_id_by_address(str(row.get("address") or ""))
		if not cgid:
			continue
		val, ts = cg_ath_after(cgid, 0)
		if val and val < 1_000_000:
			merged.at[idx, "ath_price_usd"] = val
			if ts:
				merged.at[idx, "ath_date"] = pd.to_datetime(ts, unit="ms", utc=True).isoformat()
			merged.at[idx, "ath_outlier"] = False

# Recompute metrics
merged["% from ATH"] = (px - merged["ath_price_usd"].astype(float)) / merged["ath_price_usd"].astype(float) * 100.0
merged["% to ATH"] = (merged["ath_price_usd"].astype(float) - px) / px * 100.0

# ROI using Alpha listing price (USDT≈USD) — in percentages
if "listing_price_quote" in merged.columns:
	lq = pd.to_numeric(merged["listing_price_quote"], errors="coerce")
	merged["ROI %"] = ((px / lq) - 1.0).where((lq > 0) & px.notna()) * 100.0
	merged["ROI %"] = pd.to_numeric(merged["ROI %"], errors="coerce").round(2)
	merged["ATH ROI %"] = ((merged["ath_price_usd"].astype(float) / lq) - 1.0).where((lq > 0) & merged["ath_price_usd"].notna()) * 100.0
	merged["ATH ROI %"] = pd.to_numeric(merged["ATH ROI %"], errors="coerce").round(2)

# Focus view for BSC-only highlights (defaulting to entire dataset if chain info missing)
focus_df = merged.copy()
if "chain" in focus_df.columns:
	bsc_mask = focus_df["chain"].astype(str).str.upper() == "BSC"
	if bsc_mask.any():
		focus_df = focus_df[bsc_mask]
if focus_df.empty:
	focus_df = merged.copy()

for col in ["% from ATH", "% to ATH", "ROI %", "ATH ROI %", "price_usd", "market_cap_usd", "listing_date"]:
	if col not in focus_df.columns:
		focus_df[col] = pd.NA

roi_all = pd.to_numeric(merged.get("ROI %"), errors="coerce")
ath_roi_all = pd.to_numeric(merged.get("ATH ROI %"), errors="coerce")
listing_dates_all = pd.to_datetime(merged.get("listing_date"), errors="coerce", utc=True)

# Rank: keep only CoinGecko global rank (no local fallback)
merged["rank"] = pd.to_numeric(merged.get("global_rank"), errors="coerce").astype("Int64")

# Display columns
desired_cols = [
	"rank",
	"name",
	"symbol",
	"address",
	"price_usd",
	"market_cap_usd",
	"ath_price_usd",
	"ROI %",
	"ATH ROI %",
	"ath_date",
	"listing_date",
	"listing_price_quote",
	"% from ATH",
	"% to ATH",
	"ath_outlier",
	"listing_outlier",
]
available_cols = [c for c in desired_cols if c in merged.columns]

st.subheader("Summary")
colA, colB, colC, colD = st.columns(4)
focus_from_ath = pd.to_numeric(focus_df["% from ATH"], errors="coerce")
focus_to_ath = pd.to_numeric(focus_df["% to ATH"], errors="coerce")
focus_prices = pd.to_numeric(focus_df["price_usd"], errors="coerce")
focus_valid_from = focus_from_ath.notna()
focus_valid_to = focus_to_ath.notna() & focus_prices.gt(0) & focus_to_ath.ge(0)
from_ath_avg = (
	focus_from_ath[focus_valid_from].clip(
		lower=focus_from_ath[focus_valid_from].quantile(0.01),
		upper=focus_from_ath[focus_valid_from].quantile(0.99),
	).mean()
	if focus_valid_from.any()
	else float("nan")
)
if focus_valid_to.any():
	to_clip = focus_to_ath[focus_valid_to]
	to_ath_avg = to_clip.clip(upper=to_clip.quantile(0.99)).mean()
else:
	to_ath_avg = float("nan")
with colA:
	st.metric("Tokens", len(focus_df))
with colB:
	st.metric("With metrics", int(focus_df["price_usd"].notna().sum()))
with colC:
	st.metric("Avg % from ATH", f"{from_ath_avg:.2f}%")
with colD:
	st.metric("Avg % to ATH", f"{to_ath_avg:.2f}%")

if cross_chain_df is not None and not cross_chain_df.empty:
	st.subheader("Cross-chain Alpha listings (H2 2025)")
	try:
		cc = cross_chain_df.copy()
		cc["listing_date_alpha"] = pd.to_datetime(cc["listing_date_alpha"], errors="coerce", utc=True)
		cc["listing_price_usdt"] = pd.to_numeric(cc["listing_price_usdt"], errors="coerce")
		start_h2 = pd.Timestamp(year=2025, month=7, day=1, tz="UTC")
		cc_h2 = cc[cc["listing_date_alpha"] >= start_h2].copy()
		if cc_h2.empty:
			st.info("No cross-chain Binance Alpha listings recorded for the second half of 2025 yet.")
		else:
			summary = (
				cc_h2.groupby("chain")
				.agg(
					tokens=("address", "nunique"),
					avg_listing_price_usdt=("listing_price_usdt", "mean"),
					first_listing=("listing_date_alpha", "min"),
					last_listing=("listing_date_alpha", "max"),
				)
				.reset_index()
			)
			chain_order = ["BSC", "Ethereum", "Solana", "Base"]
			summary = summary.set_index("chain").reindex(chain_order).reset_index()
			summary["tokens"] = summary["tokens"].fillna(0).astype(int)
			summary["avg_listing_price_usdt"] = summary["avg_listing_price_usdt"].round(4)
			summary["first_listing"] = summary["first_listing"].dt.date
			summary["last_listing"] = summary["last_listing"].dt.date
			total_tokens_h2 = summary["tokens"].sum()
			if total_tokens_h2 > 0:
				summary["share_of_total_%"] = (summary["tokens"] / total_tokens_h2 * 100.0).round(1)
			else:
				summary["share_of_total_%"] = 0.0
			bsc_tokens_h2 = summary.loc[summary["chain"] == "BSC", "tokens"]
			if not bsc_tokens_h2.empty and bsc_tokens_h2.iloc[0] > 0:
				summary["vs_BSC_tokens"] = (summary["tokens"] / bsc_tokens_h2.iloc[0]).round(2)
			else:
				summary["vs_BSC_tokens"] = pd.NA

			kc1, kc2, kc3 = st.columns(3)
			with kc1:
				st.metric("Total listings (H2 2025)", int(total_tokens_h2))
			with kc2:
				st.metric("Chains with listings", int((summary["tokens"] > 0).sum()))
			with kc3:
				top_chain = summary.loc[summary["tokens"].idxmax(), "chain"] if total_tokens_h2 > 0 else "–"
				st.metric("Largest chain by listings", top_chain)

			st.bar_chart(summary.set_index("chain")["tokens"], height=240)
			st.dataframe(
				summary.rename(
					columns={
						"tokens": "Tokens",
						"avg_listing_price_usdt": "Avg listing price (USDT)",
						"first_listing": "First listing date",
						"last_listing": "Last listing date",
						"share_of_total_%": "Share of H2 total (%)",
						"vs_BSC_tokens": "× vs BSC (tokens)",
					}
				),
				hide_index=True,
				use_container_width=True,
			)
			st.caption(
				"Counts and averages consider only listings dated July 1, 2025 or later. "
				"Values marked 0 indicate no captured USDT pair listings for that chain in H2 2025 yet."
			)
	except Exception as cross_err:
		st.warning(f"Unable to build cross-chain comparison: {cross_err}")

if not focus_df.empty and "listing_date" in focus_df.columns:
	bsc_dates = pd.to_datetime(focus_df["listing_date"], errors="coerce", utc=True)
	valid_dates = bsc_dates.dropna()
	if not valid_dates.empty:
		st.subheader("BSC listings per month (Binance Alpha)")
		monthly_counts = (
			valid_dates.dt.to_period("M")
			.value_counts()
			.sort_index()
			.rename("tokens")
			.to_frame()
		)
		monthly_counts.index = monthly_counts.index.astype(str)
		st.bar_chart(monthly_counts, height=250)
		st.dataframe(
			monthly_counts.reset_index().rename(columns={"index": "month"}),
			hide_index=True,
			use_container_width=True,
		)
	else:
		st.info("No Binance Alpha listing dates available to build the BSC monthly breakdown.")

# Highlights / KPIs
st.subheader("Highlights")
now = pd.Timestamp.utcnow()
focus_list_dates = pd.to_datetime(focus_df.get("listing_date"), errors="coerce", utc=True)
days_since = (now - focus_list_dates).dt.days
roi_focus = pd.to_numeric(focus_df.get("ROI %"), errors="coerce")
ath_roi_focus = pd.to_numeric(focus_df.get("ATH ROI %"), errors="coerce")
near_ath = (focus_to_ath <= 25).sum() if "% to ATH" in focus_df else 0

# Robust averages (trim 1% tails to reduce outlier impact)

def trimmed_mean(s: pd.Series) -> float:
	v = s.dropna()
	if v.empty:
		return float("nan")
	lo, hi = v.quantile(0.01), v.quantile(0.99)
	return v.clip(lo, hi).mean()

# USD formatter

def fmt_usd(v: float | None) -> str:
	try:
		if v is None or pd.isna(v):
			return "–"
		return f"${v:,.0f}"
	except Exception:
		return "–"

# Global crypto market cap fetch
@st.cache_data(show_spinner=False)

def cg_global_market_cap_usd() -> float | None:
	base = "https://api.coingecko.com/api/v3"
	headers = {"Accept": "application/json"}
	params = {}
	api_key = os.getenv("COINGECKO_API_KEY") or os.getenv("CG_API_KEY")
	if api_key:
		base = "https://pro-api.coingecko.com/api/v3"
		headers["x-cg-pro-api-key"] = api_key
		params["x_cg_pro_api_key"] = api_key
	try:
		with httpx.Client(base_url=base, headers=headers, follow_redirects=True, timeout=20) as client:
			r = client.get("/global", params=params)
			if r.status_code != 200:
				return None
			data = r.json() or {}
			return float(((data.get("data") or {}).get("total_market_cap") or {}).get("usd"))
	except Exception:
		return None

k1, k2, k3, k4 = st.columns(4)
with k1:
	st.metric("Median ROI % (since Binance Alpha)", f"{roi_focus.median():.2f}%" if roi_focus.notna().any() else "–")
with k2:
	st.metric("Average ROI % (trimmed)", f"{trimmed_mean(roi_focus):.2f}%" if roi_focus.notna().any() else "–")
with k3:
	st.metric("Median ATH ROI % (since Binance Alpha)", f"{ath_roi_focus.median():.2f}%" if ath_roi_focus.notna().any() else "–")
with k4:
	st.metric("Average ATH ROI % (trimmed)", f"{trimmed_mean(ath_roi_focus):.2f}%" if ath_roi_focus.notna().any() else "–")

m1, m2, m3, m4 = st.columns(4)
with m1:
	pos = (roi_focus > 0).sum() if roi_focus.notna().any() else 0
	total = roi_focus.notna().sum()
	st.metric("Share with positive ROI", f"{(100*pos/max(total,1)):.1f}%")
with m2:
	st.metric("Tokens within 25% of ATH", int(near_ath))
with m3:
	st.metric("Median days since listing", int(days_since.median()) if days_since.notna().any() else "–")
with m4:
	st.metric("90th pct ROI %", f"{roi_focus.quantile(0.90):.2f}%" if roi_focus.notna().any() else "–")

# Total market cap for BSC Alpha and share of crypto market
mc = pd.to_numeric(focus_df.get("market_cap_usd"), errors="coerce")
alpha_total_mc = float(mc.dropna().sum()) if mc.notna().any() else None
global_mc = cg_global_market_cap_usd()
share = (alpha_total_mc / global_mc * 100.0) if (alpha_total_mc and global_mc and global_mc > 0) else None

s1, s2, s3, s4 = st.columns(4)
with s1:
	st.metric("Total market cap — BSC Alpha", fmt_usd(alpha_total_mc))
with s2:
	st.metric("Share of total crypto market", f"{share:.2f}%" if share is not None else "–")
# Counts above/below listing price
roi_all = pd.to_numeric(merged.get("ROI %"), errors="coerce")
pos_count = int((roi_focus > 0).sum()) if roi_focus.notna().any() else 0
neg_count = int((roi_focus <= 0).sum()) if roi_focus.notna().any() else 0
with s3:
	st.metric("Coins above listing price", f"{pos_count}")
with s4:
	st.metric("Coins at/below listing price", f"{neg_count}")

st.caption("Median resists outliers; averages shown are trimmed (1%-99%) to avoid extreme spikes.")
with st.expander("How we calculate ROI metrics"):
	st.markdown(
		"""
		- **ROI % (since Binance Alpha)**: ((current price − listing price) / listing price) × 100. Uses `listing_price_quote` (USDT ≈ USD).
		- **ATH ROI % (since Binance Alpha)**: ((ATH price − listing price) / listing price) × 100. ATH is sanitized and recomputed only using data after the Binance Alpha listing date.
		- **Median vs Average (trimmed)**:
		  - Median is the middle value — robust to extreme outliers (good for typical performance).
		  - Trimmed average clips the top/bottom 1% before averaging — reflects the mean while reducing distortion from spikes.
		- **90th pct ROI %**: the value at which 90% of tokens have ROI ≤ this number (top 10% perform above it).
		- **Above vs below listing**: counts are based on ROI sign (ROI > 0 = above listing; ROI ≤ 0 = at or below listing).
		"""
	)

# Top and Bottom movers
merged["_mc_num"] = mc

colt1, colt2, colt3 = st.columns(3)
with colt1:
	st.caption("Top market cap")
	top_mc = merged.dropna(subset=["_mc_num"]).nlargest(10, "_mc_num")[['name','symbol','_mc_num']]
	st.dataframe(top_mc.rename(columns={'_mc_num':'market cap usd'}), hide_index=True, use_container_width=True)
with colt2:
	st.caption("Top ROI since Binance Alpha")
	top_roi = merged.dropna(subset=["ROI %"]).nlargest(10, "ROI %")[['name','symbol','ROI %']]
	st.dataframe(top_roi, hide_index=True, use_container_width=True)
with colt3:
	st.caption("Top ATH ROI since Binance Alpha")
	top_ath_roi = merged.dropna(subset=["ATH ROI %"]).nlargest(10, "ATH ROI %")[['name','symbol','ATH ROI %']]
	st.dataframe(top_ath_roi, hide_index=True, use_container_width=True)

colb1, colb2 = st.columns(2)
with colb1:
	st.caption("Bottom ROI since Binance Alpha")
	bot_roi = merged.dropna(subset=["ROI %"]).nsmallest(10, "ROI %")[['name','symbol','ROI %']]
	st.dataframe(bot_roi, hide_index=True, use_container_width=True)
with colb2:
	st.caption("Bottom ATH ROI since Binance Alpha")
	bot_ath_roi = merged.dropna(subset=["ATH ROI %"]).nsmallest(10, "ATH ROI %")[['name','symbol','ATH ROI %']]
	st.dataframe(bot_ath_roi, hide_index=True, use_container_width=True)

st.subheader("Coins vs listing price")
view_cols = [
	"name",
	"symbol",
	"price_usd",
	"listing_price_quote",
	"ROI %",
	"ATH ROI %",
	"ath_price_usd",
	"ath_date",
	"listing_date",
]
view_cols = [c for c in view_cols if c in merged.columns]
rename_map_view = {c: c.replace("_", " ") for c in view_cols}
rename_map_view["ROI %"] = "ROI % (since Binance Alpha)"
rename_map_view["ATH ROI %"] = "ATH ROI % (since Binance Alpha)"
above_df = merged[roi_all > 0].copy() if roi_all.notna().any() else merged.iloc[0:0]
below_df = merged[roi_all <= 0].copy() if roi_all.notna().any() else merged.iloc[0:0]
tab_above, tab_below = st.tabs(["Above listing price", "At or below listing price"])
with tab_above:
	st.caption(f"{len(above_df)} tokens currently trade above their Binance Alpha listing price.")
	if view_cols:
		st.dataframe(
			above_df.sort_values("ROI %", ascending=False, na_position="last")[view_cols].rename(columns=rename_map_view),
			use_container_width=True,
			hide_index=True,
		)
with tab_below:
	st.caption(f"{len(below_df)} tokens are at or below their Binance Alpha listing price.")
	if view_cols:
		st.dataframe(
			below_df.sort_values("ROI %", ascending=True, na_position="last")[view_cols].rename(columns=rename_map_view),
			use_container_width=True,
			hide_index=True,
		)


st.subheader("Tokens")
if not available_cols:
	st.warning("No displayable columns found.")
else:
	default_sort = "rank" if "rank" in available_cols else ("market_cap_usd" if "market_cap_usd" in available_cols else available_cols[0])
	sort_by = st.selectbox("Sort by", options=available_cols, index=available_cols.index(default_sort))
	ascending = st.checkbox("Ascending", value=True if sort_by == "rank" else (sort_by != "rank"))
	if sort_by == "rank":
		_tmp = merged.copy()
		_tmp["_rank_isna"] = _tmp["rank"].isna()
		_sorted = _tmp.sort_values(by=["_rank_isna", "rank"], ascending=[True, ascending], na_position="last")
		filtered = _sorted[available_cols]
	else:
		filtered = merged[available_cols].sort_values(sort_by, ascending=ascending, na_position="last")
	# Rename columns for display: remove underscores and annotate ROI columns
	rename_map = {c: c.replace("_", " ") for c in filtered.columns}
	rename_map["ROI %"] = "ROI % (since Binance Alpha)"
	rename_map["ATH ROI %"] = "ATH ROI % (since Binance Alpha)"
	st.dataframe(filtered.rename(columns=rename_map), use_container_width=True, hide_index=True)

st.subheader("Export")
@st.cache_data

def to_csv_bytes(df: pd.DataFrame) -> bytes:
	buf = io.StringIO()
	df.to_csv(buf, index=False)
	return buf.getvalue().encode("utf-8")

if 'filtered' in locals():
	# Export with same user-friendly headers
	rename_map = {c: c.replace("_", " ") for c in filtered.columns}
	rename_map["ROI %"] = "ROI % (since Binance Alpha)"
	rename_map["ATH ROI %"] = "ATH ROI % (since Binance Alpha)"
	st.download_button("Download table as CSV", to_csv_bytes(filtered.rename(columns=rename_map)), file_name="bsc_alpha_dashboard.csv", mime="text/csv")
