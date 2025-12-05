import sys
from pathlib import Path

import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
	sys.path.insert(0, str(CURRENT_DIR))

from alpha_dashboard import DashboardConfig, render_dashboard

# Ensure project root is on sys.path so we can import alpha_contracts modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

TOKENS_CSV = Path("./data/alpha/alpha_tokens.csv")
METRICS_CSV = Path("./data/alpha/metrics.csv")
LISTINGS_CSV = Path("./data/alpha/listings.csv")
BINANCE_LISTINGS_CSV = Path("./data/alpha/binance_listings.csv")
BINANCE_LISTINGS_COPY_CSV = Path("./data/alpha/binance_listings copy.csv")
LOG_PATH = Path("./data/alpha/metrics_log_bsc.txt")


def build_metrics_cmd(rps: int) -> list[str]:
	return [
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


def build_listing_cmd(rps: int) -> list[str]:
	return [
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


st.set_page_config(page_title="BSC Alpha Tokens Dashboard", layout="wide")

config = DashboardConfig(
	chain_label="BSC",
	page_heading="BSC Alpha Tokens — Performance Dashboard",
	project_root=PROJECT_ROOT,
	primary_listing_candidates=[BINANCE_LISTINGS_CSV, BINANCE_LISTINGS_COPY_CSV],
	fallback_tokens_csv=TOKENS_CSV,
	metrics_csv=METRICS_CSV,
	log_path=LOG_PATH,
	metrics_cmd_builder=build_metrics_cmd,
	listing_cmd_builder=build_listing_cmd,
	listing_button_label="Refresh listings now (CoinGecko)",
	cg_platform="binance-smart-chain",
	tokens_label="BSC Alpha",
	primary_missing_hint="Run python -m alpha_contracts.listing_alpha to regenerate Binance Alpha listings.",
	fallback_missing_hint="Ensure alpha_tokens.csv exists (python -m alpha_contracts.fetch_contracts ...).",
	download_basename="bsc_alpha_dashboard",
)

render_dashboard(config)
