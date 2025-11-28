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

# Ensure project root is on sys.path so we can import alpha_contracts
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

TOKENS_CSV = Path("./data/alpha/alpha_tokens.csv")
METRICS_CSV = Path("./data/alpha/metrics.csv")
LISTINGS_CSV = Path("./data/alpha/listings.csv")
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

# Load tokens
if not TOKENS_CSV.exists():
	st.error(f"Missing {TOKENS_CSV}")
	st.stop()

tokens_df = pd.read_csv(TOKENS_CSV)

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
	force_listing = st.button("Refresh listings now")

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

# Optional: fetch listings
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

# Load metrics (may still be empty for some rows)
metrics_df = pd.read_csv(METRICS_CSV) if METRICS_CSV.exists() else pd.DataFrame(columns=["address","price_usd","ath_price_usd","ath_date","market_cap_usd","global_rank"]) 
metrics_df = metrics_df.drop(columns=[c for c in ["symbol","name"] if c in metrics_df.columns], errors="ignore")

# Merge on address
merged = tokens_df.merge(metrics_df, on="address", how="left")

# Merge listings if available
if LISTINGS_CSV.exists():
	listings_df = pd.read_csv(LISTINGS_CSV)
	merged = merged.merge(listings_df, on="address", how="left")

# Compute derived columns safely
for col in ["price_usd", "ath_price_usd", "market_cap_usd", "global_rank"]:
	if col not in merged.columns:
		merged[col] = pd.NA

merged["% from ATH"] = (merged["price_usd"].astype(float) - merged["ath_price_usd"].astype(float)) / merged["ath_price_usd"].astype(float) * 100.0
merged["% to ATH"] = (merged["ath_price_usd"].astype(float) - merged["price_usd"].astype(float)) / merged["price_usd"].astype(float) * 100.0

# ROI columns from listing
if "listing_price_usd" in merged.columns:
	lp = pd.to_numeric(merged["listing_price_usd"], errors="coerce")
	px = pd.to_numeric(merged["price_usd"], errors="coerce")
	ath = pd.to_numeric(merged["ath_price_usd"], errors="coerce")
	merged["ROI_x"] = (px / lp).where((lp > 0) & px.notna())
	merged["ATH_ROI_x"] = (ath / lp).where((lp > 0) & ath.notna())

# Create display rank: prefer CoinGecko global_rank; fallback to local rank by market cap (dense)
try:
	merged["market_cap_usd_num"] = pd.to_numeric(merged["market_cap_usd"], errors="coerce")
except Exception:
	merged["market_cap_usd_num"] = pd.NA
merged["local_rank"] = (-merged["market_cap_usd_num"]).rank(method="dense").astype("Int64")
merged["rank"] = pd.to_numeric(merged["global_rank"], errors="coerce")
merged.loc[merged["rank"].isna(), "rank"] = merged.loc[merged["rank"].isna(), "local_rank"]
merged["rank"] = merged["rank"].astype("Int64")

# Prepare display columns dynamically
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
	"listing_price_usd",
	"ROI_x",
	"ATH_ROI_x",
	"% from ATH",
	"% to ATH",
]
available_cols = [c for c in desired_cols if c in merged.columns]

st.subheader("Summary")
colA, colB, colC, colD = st.columns(4)
# Robust averages: ignore zero/NaN, clip extreme outliers at 99th percentile
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
