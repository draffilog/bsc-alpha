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
	merged["ATH ROI %"] = ((merged["ath_price_usd"].astype(float) / lq) - 1.0).where((lq > 0) & merged["ath_price_usd"].notna()) * 100.0
	merged["ATH ROI %"] = pd.to_numeric(merged["ATH ROI %"], errors="coerce").round(2)

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
from_ath_series = pd.to_numeric(merged["% from ATH"], errors="coerce")
to_ath_series = pd.to_numeric(merged["% to ATH"], errors="coerce")
price_series = pd.to_numeric(merged["price_usd"], errors="coerce")
valid_from = from_ath_series.notna()
valid_to = to_ath_series.notna() & price_series.gt(0) & to_ath_series.ge(0)
from_ath_avg = (
	from_ath_series[valid_from].clip(
		lower=from_ath_series[valid_from].quantile(0.01),
		upper=from_ath_series[valid_from].quantile(0.99),
	).mean()
	if valid_from.any()
	else float("nan")
)
if valid_to.any():
	to_clip = to_ath_series[valid_to]
	to_ath_avg = to_clip.clip(upper=to_clip.quantile(0.99)).mean()
else:
	to_ath_avg = float("nan")
with colA:
	st.metric("Tokens", len(merged))
with colB:
	st.metric("With metrics", int(merged["price_usd"].notna().sum()))
with colC:
	st.metric("Avg % from ATH", f"{from_ath_avg:.2f}%")
with colD:
	st.metric("Avg % to ATH", f"{to_ath_avg:.2f}%")

# Highlights / KPIs
st.subheader("Highlights")
now = pd.Timestamp.utcnow()
list_dates = pd.to_datetime(merged.get("listing_date"), errors="coerce", utc=True)
days_since = (now - list_dates).dt.days
roi = pd.to_numeric(merged.get("ROI %"), errors="coerce")
ath_roi = pd.to_numeric(merged.get("ATH ROI %"), errors="coerce")
near_ath = (to_ath_series <= 25).sum() if "% to ATH" in merged else 0

# Robust averages (trim 1% tails to reduce outlier impact)
def trimmed_mean(s: pd.Series) -> float:
	v = s.dropna()
	if v.empty:
		return float("nan")
	lo, hi = v.quantile(0.01), v.quantile(0.99)
	return v.clip(lo, hi).mean()

k1, k2, k3, k4 = st.columns(4)
with k1:
	st.metric("Median ROI % (since Binance Alpha)", f"{roi.median():.2f}%" if roi.notna().any() else "–")
with k2:
	st.metric("Average ROI % (trimmed)", f"{trimmed_mean(roi):.2f}%" if roi.notna().any() else "–")
with k3:
	st.metric("Median ATH ROI % (since Binance Alpha)", f"{ath_roi.median():.2f}%" if ath_roi.notna().any() else "–")
with k4:
	st.metric("Average ATH ROI % (trimmed)", f"{trimmed_mean(ath_roi):.2f}%" if ath_roi.notna().any() else "–")

m1, m2, m3, m4 = st.columns(4)
with m1:
	pos = (roi > 0).sum() if roi.notna().any() else 0
	total = roi.notna().sum()
	st.metric("Share with positive ROI", f"{(100*pos/max(total,1)):.1f}%")
with m2:
	st.metric("Tokens within 25% of ATH", int(near_ath))
with m3:
	st.metric("Median days since listing", int(days_since.median()) if days_since.notna().any() else "–")
with m4:
	st.metric("90th pct ROI %", f"{roi.quantile(0.90):.2f}%" if roi.notna().any() else "–")

st.caption("Median resists outliers; averages shown are trimmed (1%-99%) to avoid extreme spikes.")

# Top and Bottom movers
mc = pd.to_numeric(merged.get("market_cap_usd"), errors="coerce")
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
