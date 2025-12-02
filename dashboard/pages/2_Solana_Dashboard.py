import sys
from pathlib import Path

import streamlit as st

from dashboard.alpha_dashboard import DashboardConfig, render_dashboard

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

SOLANA_TOKENS_CSV = Path("./data/alpha/solana_tokens.csv")
SOLANA_LISTINGS_CSV = Path("./data/alpha/solana_listings.csv")
SOLANA_METRICS_CSV = Path("./data/alpha/solana_metrics.csv")
SOLANA_LOG_PATH = Path("./data/alpha/metrics_log_solana.txt")


def build_solana_metrics_cmd(rps: int):
	if not SOLANA_TOKENS_CSV.exists():
		return None
	return [
		sys.executable,
		"-m",
		"alpha_contracts.metrics",
		"--in",
		str(SOLANA_TOKENS_CSV),
		"--out",
		str(SOLANA_METRICS_CSV),
		"--rps",
		str(int(rps)),
		"--platform",
		"solana",
		"--onchain-network",
		"none",
	]


st.set_page_config(page_title="Solana Alpha Tokens Dashboard", layout="wide")

config = DashboardConfig(
	chain_label="Solana",
	page_heading="Solana Alpha Tokens — Performance Dashboard",
	project_root=PROJECT_ROOT,
	primary_listing_candidates=[SOLANA_LISTINGS_CSV],
	fallback_tokens_csv=SOLANA_TOKENS_CSV,
	metrics_csv=SOLANA_METRICS_CSV,
	log_path=SOLANA_LOG_PATH,
	metrics_cmd_builder=build_solana_metrics_cmd,
	listing_cmd_builder=None,
	cg_platform="solana",
	tokens_label="Solana Alpha",
	primary_missing_hint="Run python -m alpha_contracts.prepare_solana to regenerate solana_listings.csv.",
	fallback_missing_hint="Run python -m alpha_contracts.prepare_solana to generate solana_tokens.csv.",
	download_basename="solana_alpha_dashboard",
)

render_dashboard(config)

