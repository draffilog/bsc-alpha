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

left, right = st.columns([3, 2])

with left:
	st.markdown("Use the fetcher to refresh metrics from CoinGecko, or let the app auto-refresh if missing.")
	st.code("python -m alpha_contracts.metrics --in ./data/alpha/alpha_tokens.csv --out ./data/alpha/metrics.csv", language="bash")
	st.caption("Note: CoinGecko Pro uses header x-cg-pro-api-key and query param x_cg_pro_api_key. 400 Bad Request usually means the contract isn't indexed/unsupported.")

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

# Controls
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

# Derived columns
for col in ["price_usd", "ath_price_usd", "market_cap_usd", "global_rank"]:
	if col not in merged.columns:
		merged[col] = pd.NA

merged["% from ATH"] = (merged["price_usd"].astype(float) - merged["ath_price_usd"].astype(float)) / merged["ath_price_usd"].astype(float) * 100.0
merged["% to ATH"] = (merged["ath_price_usd"].astype(float) - merged["price_usd"].astype(float)) / merged["price_usd"].astype(float) * 100.0

# Listing price USD via CoinGecko using listing_date
@st.cache_data(show_spinner=False)

def fetch_price_at_ts(cg_id: str, ts_ms: int, api_key: str | None) -> float | None:
	if not cg_id or not ts_ms:
		return None
	base = "https://api.coingecko.com/api/v3"
	headers = {"Accept": "application/json"}
	params = {"vs_currency": "usd", "from": int(ts_ms/1000) - 3600, "to": int(ts_ms/1000) + 3600}
	if api_key:
		base = "https://pro-api.coingecko.com/api/v3"
		headers["x-cg-pro-api-key"] = api_key
		params["x_cg_pro_api_key"] = api_key
	try:
		with httpx.Client(base_url=base, headers=headers, follow_redirects=True, timeout=30) as client:
			r = client.get(f"/coins/{cg_id}/market_chart/range", params=params)
			if r.status_code != 200:
				return None
			data = r.json()
			prices = data.get("prices") or []
			if not prices:
				return None
			closest = min(prices, key=lambda p: abs(int(p[0]) - ts_ms))
			return float(closest[1])
	except Exception:
		return None

# Compute listing USD automatically when we have cg_id and listing date
api_key = os.getenv("COINGECKO_API_KEY") or os.getenv("CG_API_KEY")
if "listing_date" in merged.columns and "cg_id" in merged.columns:
	lst_prices = []
	for _, r in merged.iterrows():
		cgid = str(r.get("cg_id") or "")
		ts = None
		if pd.notna(r.get("listing_timestamp_ms")):
			try:
				ts = int(r.get("listing_timestamp_ms"))
			except Exception:
				ts = None
		if ts is None and pd.notna(r.get("listing_date")):
			try:
				ts = int(pd.to_datetime(r.get("listing_date")).value // 1_000_000)
			except Exception:
				ts = None
		price = fetch_price_at_ts(cgid, ts, api_key) if (cgid and ts) else None
		lst_prices.append(price)
	merged["listing_price_usd_cg"] = lst_prices

# ROI using CoinGecko listing USD
lp = pd.to_numeric(merged.get("listing_price_usd_cg"), errors="coerce") if "listing_price_usd_cg" in merged.columns else pd.Series([])
px = pd.to_numeric(merged.get("price_usd"), errors="coerce")
ath = pd.to_numeric(merged.get("ath_price_usd"), errors="coerce")
if not lp.empty:
	merged["ROI_cg_x"] = (px / lp).where((lp > 0) & px.notna())
	merged["ATH_ROI_cg_x"] = (ath / lp).where((lp > 0) & ath.notna())

# Also compute ROI from Alpha-quoted price if present (USDT≈USD)
if "listing_price_quote" in merged.columns:
	lq = pd.to_numeric(merged["listing_price_quote"], errors="coerce")
	merged["ROI_alpha_x"] = (px / lq).where((lq > 0) & px.notna())
	merged["ATH_ROI_alpha_x"] = (ath / lq).where((lq > 0) & ath.notna())

# Rank: prefer global_rank
try:
	merged["market_cap_usd_num"] = pd.to_numeric(merged["market_cap_usd"], errors="coerce")
except Exception:
	merged["market_cap_usd_num"] = pd.NA
merged["local_rank"] = (-merged["market_cap_usd_num"]).rank(method="dense").astype("Int64")
merged["rank"] = pd.to_numeric(merged["global_rank"], errors="coerce")
merged.loc[merged["rank"].isna(), "rank"] = merged.loc[merged["rank"].isna(), "local_rank"]
merged["rank"] = merged["rank"].astype("Int64")

# Display columns
desired_cols = [
	"rank",
	"name",
	"symbol",
	"address",
	"price_usd",
	"market_cap_usd",
	"ath_price_usd",
	"ath_date",
	"listing_date",
	"listing_price_usd_cg",
	"ROI_cg_x",
	"ATH_ROI_cg_x",
	"listing_price_quote",
	"listing_quote",
	"alpha_pair",
	"ROI_alpha_x",
	"ATH_ROI_alpha_x",
	"% from ATH",
	"% to ATH",
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

st.subheader("Tokens")
if not available_cols:
	st.warning("No displayable columns found.")
else:
	default_sort = "rank" if "rank" in available_cols else ("market_cap_usd" if "market_cap_usd" in available_cols else available_cols[0])
	sort_by = st.selectbox("Sort by", options=available_cols, index=available_cols.index(default_sort))
	ascending = st.checkbox("Ascending", value=(sort_by != "rank"))
	filtered = merged[available_cols].sort_values(sort_by, ascending=ascending, na_position="last")
	st.dataframe(filtered, use_container_width=True)

st.subheader("Export")
@st.cache_data

def to_csv_bytes(df: pd.DataFrame) -> bytes:
	buf = io.StringIO()
	df.to_csv(buf, index=False)
	return buf.getvalue().encode("utf-8")

if 'filtered' in locals():
	st.download_button("Download table as CSV", to_csv_bytes(filtered), file_name="bsc_alpha_dashboard.csv", mime="text/csv")
