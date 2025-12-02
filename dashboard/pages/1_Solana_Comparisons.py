import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

CROSS_CHAIN_CSV = PROJECT_ROOT / "data/alpha/cross_chain_listings.csv"

st.set_page_config(page_title="Solana Comparisons", layout="wide")
st.title("Solana Comparisons — Similar Listings on Solana vs BSC")
st.caption("We match Solana tokens to BSC listings by symbol (case-insensitive) to spot familiar names across chains.")


@st.cache_data(show_spinner=False)
def load_cross_chain(path: Path) -> pd.DataFrame:
	if not path.exists():
		return pd.DataFrame()
	df = pd.read_csv(path)
	df["listing_timestamp_ms"] = pd.to_numeric(df.get("listing_timestamp_ms"), errors="coerce")
	df["listing_price_usdt"] = pd.to_numeric(df.get("listing_price_usdt"), errors="coerce")
	df["listing_date_alpha"] = pd.to_datetime(df.get("listing_date_alpha"), errors="coerce", utc=True)
	df["symbol_norm"] = df.get("symbol").astype(str).str.strip().str.upper()
	df["chain"] = df.get("chain").fillna("").astype(str)
	df["chain_id"] = df.get("chain_id").fillna("").astype(str)
	return df


def slice_chain(df: pd.DataFrame, chain: str) -> pd.DataFrame:
	chain = chain.lower()
	if chain == "solana":
		mask = df["chain"].str.lower().eq("solana") | df["chain_id"].str.lower().eq("ct_501")
	elif chain == "bsc":
		mask = df["chain"].str.lower().eq("bsc") | df["chain_id"].astype(str).eq("56")
	else:
		mask = pd.Series(False, index=df.index)
	return df[mask].copy()


def dedupe_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
	df = df[df["symbol_norm"].notna() & df["symbol_norm"].ne("")].copy()
	df["_sort_key"] = df["listing_timestamp_ms"]
	missing_ts = df["_sort_key"].isna() & df["listing_date_alpha"].notna()
	if missing_ts.any():
		df.loc[missing_ts, "_sort_key"] = (
			df.loc[missing_ts, "listing_date_alpha"].view("int64") / 1_000_000
		)
	df["_sort_key"] = df["_sort_key"].fillna(np.inf)
	df = df.sort_values(["symbol_norm", "_sort_key"])
	df = df.drop_duplicates("symbol_norm", keep="first")
	return df.drop(columns="_sort_key")


def fmt_price(v: float | None) -> str:
	try:
		if v is None or pd.isna(v):
			return "–"
		if v >= 1:
			return f"${v:,.2f}"
		return f"${v:,.6f}"
	except Exception:
		return "–"


df = load_cross_chain(CROSS_CHAIN_CSV)
if df.empty:
	st.error("`data/alpha/cross_chain_listings.csv` is missing. Run the cross-chain export first.")
	st.stop()

solana_df = dedupe_by_symbol(slice_chain(df, "solana"))
bsc_df = dedupe_by_symbol(slice_chain(df, "bsc"))

if solana_df.empty:
	st.warning("No Solana listings found in the dataset yet.")
	st.stop()

paired = solana_df.merge(
	bsc_df,
	on="symbol_norm",
	how="left",
	suffixes=("_sol", "_bsc"),
)
paired["listing_gap_days"] = (
	paired["listing_date_alpha_sol"] - paired["listing_date_alpha_bsc"]
).dt.total_seconds() / 86_400
paired["price_ratio_bsc_vs_sol"] = (
	paired["listing_price_usdt_bsc"] / paired["listing_price_usdt_sol"]
)
paired["price_ratio_sol_vs_bsc"] = (
	paired["listing_price_usdt_sol"] / paired["listing_price_usdt_bsc"]
)

solana_with_match = paired["address_bsc"].notna().sum()
solana_without_match = len(solana_df) - solana_with_match

cover_cols = st.columns(4)
with cover_cols[0]:
	st.metric("Solana tokens tracked", len(solana_df))
with cover_cols[1]:
	st.metric("With BSC symbol match", solana_with_match)
with cover_cols[2]:
	st.metric("No BSC listing yet", solana_without_match)
with cover_cols[3]:
	median_gap = paired["listing_gap_days"].dropna()
	st.metric(
		"Median listing gap (days)",
		f"{median_gap.median():.1f}" if not median_gap.empty else "–",
	)

matched_only = paired[paired["address_bsc"].notna()].copy()
if matched_only.empty:
	st.warning(
		"No overlapping symbols between Solana and BSC have been found so far. "
		"Once a token launches on both chains with the same ticker, it will show up here."
	)
else:
	st.subheader("Solana ↔ BSC symbol overlaps")
	ratio_series = matched_only["price_ratio_bsc_vs_sol"].replace([np.inf, -np.inf], np.nan).dropna()
	gap_series = matched_only["listing_gap_days"].abs().dropna()
	ratio_slider_max = float(ratio_series.max()) if not ratio_series.empty else 5.0
	ratio_slider_max = max(0.5, ratio_slider_max)
	gap_slider_max = int(np.ceil(gap_series.max())) if not gap_series.empty else 30
	gap_slider_max = max(1, gap_slider_max)
	cols = st.columns([2, 1, 1, 1, 1, 1])
	with cols[0]:
		min_ratio = st.slider(
			"Min BSC/Solana listing price ratio",
			min_value=0.0,
			max_value=ratio_slider_max,
			value=0.0,
			step=0.1,
		)
	with cols[1]:
		max_days = st.slider(
			"Max listing gap (days)",
			min_value=0,
			max_value=gap_slider_max,
			value=gap_slider_max,
		)
	filtered = matched_only.copy()
	filtered = filtered[
		(filtered["price_ratio_bsc_vs_sol"].fillna(0) >= min_ratio)
		& (filtered["listing_gap_days"].abs().fillna(max_days) <= max_days)
	]

	def fmt_date(series: pd.Series) -> pd.Series:
		return series.dt.strftime("%Y-%m-%d %H:%M UTC")

	display_cols = [
		"symbol_norm",
		"name_sol",
		"listing_date_alpha_sol",
		"listing_price_usdt_sol",
		"name_bsc",
		"listing_date_alpha_bsc",
		"listing_price_usdt_bsc",
		"price_ratio_bsc_vs_sol",
		"listing_gap_days",
	]
	view = filtered[display_cols].rename(
		columns={
			"symbol_norm": "Symbol",
			"name_sol": "Solana name",
			"listing_date_alpha_sol": "Solana listed at",
			"listing_price_usdt_sol": "Solana listing price",
			"name_bsc": "BSC name",
			"listing_date_alpha_bsc": "BSC listed at",
			"listing_price_usdt_bsc": "BSC listing price",
			"price_ratio_bsc_vs_sol": "Price ratio (BSC/SOL)",
			"listing_gap_days": "Listing gap (days)",
		}
	)
	view["Solana listed at"] = fmt_date(view["Solana listed at"])
	view["BSC listed at"] = fmt_date(view["BSC listed at"])
	view["Solana listing price"] = view["Solana listing price"].apply(fmt_price)
	view["BSC listing price"] = view["BSC listing price"].apply(fmt_price)
	view["Price ratio (BSC/SOL)"] = view["Price ratio (BSC/SOL)"].apply(
		lambda v: "–" if pd.isna(v) or np.isinf(v) else f"{v:.2f}×"
	)
	view["Listing gap (days)"] = view["Listing gap (days)"].apply(
		lambda v: "–" if pd.isna(v) else f"{v:+.1f}"
	)
	st.dataframe(view, use_container_width=True, hide_index=True)

	st.caption(
		"A positive listing gap means the Solana listing came after the BSC listing; "
		"negative values mean Solana launched first."
	)

st.subheader("Inspect a specific Solana token")
options = solana_df.sort_values("symbol_norm")
option_labels = {
	row.symbol_norm: f"{row.symbol_norm} — {row.name or 'Unnamed'}"
	for row in options.itertuples()
}
selected_symbol = st.selectbox(
	"Pick a Solana ticker",
	options=list(option_labels.keys()),
	format_func=lambda sym: option_labels.get(sym, sym),
)

solana_row = solana_df[solana_df["symbol_norm"] == selected_symbol].iloc[0]
bsc_matches = bsc_df[bsc_df["symbol_norm"] == selected_symbol]

info_cols = st.columns(3)
with info_cols[0]:
	st.metric("Solana listing date", solana_row["listing_date_alpha"].strftime("%Y-%m-%d %H:%M UTC") if pd.notna(solana_row["listing_date_alpha"]) else "–")
with info_cols[1]:
	st.metric("Solana listing price", fmt_price(solana_row["listing_price_usdt"]))
with info_cols[2]:
	st.metric("Alpha pair", solana_row.get("alpha_pair", "–"))

if bsc_matches.empty:
	st.info("No BSC listing detected with the same symbol yet.")
else:
	st.write("BSC listings with the same symbol:")
	match_view = bsc_matches.copy()
	match_view["listing_date_alpha"] = match_view["listing_date_alpha"].dt.strftime("%Y-%m-%d %H:%M UTC")
	match_view["listing_price_usdt"] = match_view["listing_price_usdt"].apply(fmt_price)
	st.dataframe(
		match_view[["name", "listing_date_alpha", "listing_price_usdt", "alpha_pair"]],
		use_container_width=True,
		hide_index=True,
	)

st.subheader("Solana listings with no BSC counterpart")
sol_only = solana_df[~solana_df["symbol_norm"].isin(bsc_df["symbol_norm"])].copy()
if sol_only.empty:
	st.success("Every Solana symbol currently has a BSC listing.")
else:
	sol_only["listing_date_alpha"] = sol_only["listing_date_alpha"].dt.strftime("%Y-%m-%d %H:%M UTC")
	sol_only["listing_price_usdt"] = sol_only["listing_price_usdt"].apply(fmt_price)
	st.dataframe(
		sol_only[["symbol_norm", "name", "listing_date_alpha", "listing_price_usdt", "alpha_pair"]].rename(
			columns={
				"symbol_norm": "Symbol",
				"name": "Name",
				"listing_date_alpha": "Listing date",
				"listing_price_usdt": "Listing price",
				"alpha_pair": "Alpha pair",
			}
		),
		use_container_width=True,
		hide_index=True,
	)


