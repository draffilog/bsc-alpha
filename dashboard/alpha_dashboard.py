import io
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import httpx
import pandas as pd
import streamlit as st


@dataclass
class DashboardConfig:
	chain_label: str
	page_heading: str
	project_root: Path
	primary_listing_candidates: Sequence[Path]
	fallback_tokens_csv: Path
	metrics_csv: Path
	log_path: Path
	metrics_cmd_builder: Optional[Callable[[int], Optional[List[str]]]] = None
	listing_cmd_builder: Optional[Callable[[int], Optional[List[str]]]] = None
	listing_button_label: str = "Refresh listings now (CoinGecko)"
	cg_platform: str = "binance-smart-chain"
	tokens_label: str = "Alpha"
	primary_missing_hint: Optional[str] = None
	fallback_missing_hint: Optional[str] = None
	download_basename: str = "alpha_dashboard"
	auto_refresh_metrics: bool = True


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


@st.cache_data(show_spinner=False)
def cg_price_near_ts(platform: str, cg_id: str, ts_ms: int, offset_sec: int = 60, window_sec: int = 300):
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
def cg_ath_after(platform: str, cg_id: str, from_ts_ms: int):
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
def cg_id_by_address(platform: str, address: str):
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
			r = client.get(f"/coins/{platform}/contract/{address}", params=params)
			if r.status_code != 200:
				return None
			data = r.json() or {}
			return data.get("id")
	except Exception:
		return None


@st.cache_data(show_spinner=False)
def cg_global_market_cap_usd() -> Optional[float]:
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


def trimmed_mean(series: pd.Series) -> float:
	v = series.dropna()
	if v.empty:
		return float("nan")
	lo, hi = v.quantile(0.01), v.quantile(0.99)
	return v.clip(lo, hi).mean()


def fmt_usd(value: Optional[float]) -> str:
	try:
		if value is None or pd.isna(value):
			return "–"
		return f"${value:,.0f}"
	except Exception:
		return "–"


def render_dashboard(config: DashboardConfig) -> None:
	st.title(config.page_heading)

	# Propagate secrets to env (for Streamlit Cloud deploys)
	if "COINGECKO_API_KEY" in st.secrets:
		os.environ["COINGECKO_API_KEY"] = st.secrets["COINGECKO_API_KEY"]
	if "BSCSCAN_API_KEY" in st.secrets:
		os.environ["BSCSCAN_API_KEY"] = st.secrets["BSCSCAN_API_KEY"]

	primary_from_alpha = False
	primary_path: Optional[Path] = None
	for candidate in config.primary_listing_candidates:
		if candidate and candidate.exists():
			primary_from_alpha = True
			primary_path = candidate
			break
	if primary_path is None:
		primary_path = config.fallback_tokens_csv
		primary_from_alpha = False

	if not primary_path or not primary_path.exists():
		hint = config.primary_missing_hint if primary_from_alpha else config.fallback_missing_hint
		msg = f"Missing {primary_path}"
		if hint:
			msg = f"{msg}. {hint}"
		st.error(msg)
		st.stop()

	raw_df = pd.read_csv(primary_path)
	if primary_from_alpha:
		keep = [
			c
			for c in raw_df.columns
			if c
			in {
				"address",
				"symbol",
				"name",
				"listing_date_alpha",
				"listing_timestamp_ms",
				"listing_price_quote",
				"listing_quote",
				"alpha_pair",
			}
		]
		tokens_df = raw_df[keep].copy()
		if "listing_date_alpha" in tokens_df.columns:
			tokens_df["listing_date"] = tokens_df["listing_date_alpha"]
	else:
		tokens_df = raw_df.copy()

	# Controls
	right = st.container()
	with right:
		colA, colB = st.columns(2)
		with colA:
			force_refresh = st.button("Refresh metrics now", disabled=config.metrics_cmd_builder is None)
		with colB:
			rps = st.number_input("Requests/sec", min_value=1, max_value=50, value=8, step=1)
		if config.listing_cmd_builder:
			force_listing = st.button(config.listing_button_label)
		else:
			force_listing = False

	log_area = st.empty()

	metrics_should_refresh = False
	if config.metrics_cmd_builder is not None:
		if force_refresh:
			metrics_should_refresh = True
		elif config.auto_refresh_metrics and need_refresh(config.metrics_csv, len(tokens_df)):
			metrics_should_refresh = True

	if metrics_should_refresh:
		cmd = config.metrics_cmd_builder(int(rps))
		if not cmd:
			st.warning("Metrics refresh command is not configured. Ensure token CSV exists.")
		else:
			st.info("Fetching metrics from CoinGecko... logs streaming below")
			try:
				if config.log_path.exists():
					config.log_path.unlink()
			except Exception:
				pass
			env = os.environ.copy()
			proc = subprocess.Popen(cmd, cwd=str(config.project_root), env=env)
			while proc.poll() is None:
				if config.log_path.exists():
					try:
						text = config.log_path.read_text()
						lines = text.strip().splitlines()
						log_area.code("\n".join(lines[-200:]))
					except Exception:
						pass
				time.sleep(0.8)
			if config.log_path.exists():
				try:
					text = config.log_path.read_text()
					lines = text.strip().splitlines()
					log_area.code("\n".join(lines[-200:]))
				except Exception:
					pass
			st.toast(
				f"Metrics fetch finished with exit {proc.returncode}",
				icon="✅" if proc.returncode == 0 else "⚠️",
			)

	if force_listing and config.listing_cmd_builder:
		cmd = config.listing_cmd_builder(int(rps))
		if not cmd:
			st.warning("Listing refresh command is not configured.")
		else:
			st.info("Fetching listing date/price from CoinGecko market charts (first data point)")
			proc = subprocess.Popen(cmd, cwd=str(config.project_root), env=os.environ.copy())
			proc.wait()
			st.toast(
				f"Listings fetch finished with exit {proc.returncode}",
				icon="✅" if proc.returncode == 0 else "⚠️",
			)

	metrics_df = (
		pd.read_csv(config.metrics_csv)
		if config.metrics_csv.exists()
		else pd.DataFrame(columns=["address", "price_usd", "ath_price_usd", "ath_date", "market_cap_usd", "global_rank", "cg_id"])
	)
	metrics_df = metrics_df.drop(columns=[c for c in ["symbol", "name"] if c in metrics_df.columns], errors="ignore")

	merged = tokens_df.merge(metrics_df, on="address", how="left")

	if not primary_from_alpha:
		alpha_listing_path = None
		for candidate in config.primary_listing_candidates:
			if candidate and candidate.exists():
				alpha_listing_path = candidate
				break
		if alpha_listing_path:
			alpha_df = pd.read_csv(alpha_listing_path)
			alpha_df = alpha_df[
				[
					c
					for c in alpha_df.columns
					if c
					in {
						"address",
						"listing_date_alpha",
						"listing_price_quote",
						"listing_quote",
						"alpha_pair",
						"listing_timestamp_ms",
					}
				]
			]
			merged = merged.merge(alpha_df, on="address", how="left", suffixes=("", "_alpha"))
			merged["listing_date"] = merged.get("listing_date") if "listing_date" in merged.columns else merged.get("listing_date_alpha")
	else:
		if "listing_date" not in merged.columns and "listing_date_alpha" in merged.columns:
			merged["listing_date"] = merged["listing_date_alpha"]

	merged = merged.drop(columns=[c for c in ["alpha_pair", "listing_quote"] if c in merged.columns], errors="ignore")

	for col in ["price_usd", "ath_price_usd", "market_cap_usd", "global_rank"]:
		if col not in merged.columns:
			merged[col] = pd.NA

	px = pd.to_numeric(merged["price_usd"], errors="coerce")
	ath_raw = pd.to_numeric(merged["ath_price_usd"], errors="coerce")
	lq_raw = (
		pd.to_numeric(merged.get("listing_price_quote"), errors="coerce")
		if "listing_price_quote" in merged.columns
		else pd.Series([], dtype=float)
	)
	ath_outlier = (ath_raw <= 0) | ((px > 0) & (ath_raw / px > 10000)) | (ath_raw > 1_000_000)
	lq_outlier = (lq_raw <= 0) | ((px > 0) & (lq_raw / px > 10000)) | (lq_raw > 1_000_000)
	merged["ath_outlier"] = ath_outlier.fillna(False)
	if not lq_raw.empty:
		merged["listing_outlier"] = lq_outlier.fillna(False)
	merged["ath_price_usd"] = ath_raw.mask(ath_outlier)
	if "listing_price_quote" in merged.columns:
		merged["listing_price_quote"] = lq_raw.mask(lq_outlier)

	if "ath_date" in merged.columns:
		mask_na = merged["ath_price_usd"].isna() & merged["ath_date"].notna()
		if mask_na.any():
			for idx, row in merged[mask_na].iterrows():
				try:
					cgid = str(row.get("cg_id") or "")
					if not cgid:
						cgid = cg_id_by_address(config.cg_platform, str(row.get("address") or ""))
					ts = int(pd.to_datetime(row["ath_date"]).value // 1_000_000)
					price = cg_price_near_ts(config.cg_platform, cgid, ts, offset_sec=60, window_sec=300) if cgid else None
					if price and price < 1_000_000:
						merged.at[idx, "ath_price_usd"] = price
						merged.at[idx, "ath_outlier"] = False
				except Exception:
					continue

	if "listing_date" in merged.columns and "ath_date" in merged.columns:
		mask_before = merged["ath_price_usd"].notna() & merged["ath_date"].notna() & merged["listing_date"].notna()
		if mask_before.any():
			for idx, row in merged[mask_before].iterrows():
				try:
					cgid = str(row.get("cg_id") or "") or cg_id_by_address(config.cg_platform, str(row.get("address") or ""))
					ath_ts = int(pd.to_datetime(row["ath_date"]).value // 1_000_000)
					lst_ts = int(pd.to_datetime(row["listing_date"]).value // 1_000_000)
					if cgid and ath_ts < lst_ts:
						val, ts2 = cg_ath_after(config.cg_platform, cgid, lst_ts)
						if val and val < 1_000_000:
							merged.at[idx, "ath_price_usd"] = val
							if ts2:
								merged.at[idx, "ath_date"] = pd.to_datetime(ts2, unit="ms", utc=True).isoformat()
				except Exception:
					continue

	mask = merged["ath_price_usd"].isna() | merged["ath_outlier"]
	if mask.any():
		for idx, row in merged[mask].iterrows():
			cgid = str(row.get("cg_id") or "") or cg_id_by_address(config.cg_platform, str(row.get("address") or ""))
			if not cgid:
				continue
			val, ts = cg_ath_after(config.cg_platform, cgid, 0)
			if val and val < 1_000_000:
				merged.at[idx, "ath_price_usd"] = val
				if ts:
					merged.at[idx, "ath_date"] = pd.to_datetime(ts, unit="ms", utc=True).isoformat()
				merged.at[idx, "ath_outlier"] = False

	merged["% from ATH"] = (px - merged["ath_price_usd"].astype(float)) / merged["ath_price_usd"].astype(float) * 100.0
	merged["% to ATH"] = (merged["ath_price_usd"].astype(float) - px) / px * 100.0

	if "listing_price_quote" in merged.columns:
		lq = pd.to_numeric(merged["listing_price_quote"], errors="coerce")
		merged["ROI %"] = ((px / lq) - 1.0).where((lq > 0) & px.notna()) * 100.0
		merged["ROI %"] = pd.to_numeric(merged["ROI %"], errors="coerce").round(2)
		merged["ATH ROI %"] = ((merged["ath_price_usd"].astype(float) / lq) - 1.0).where((lq > 0) & merged["ath_price_usd"].notna()) * 100.0
		merged["ATH ROI %"] = pd.to_numeric(merged["ATH ROI %"], errors="coerce").round(2)

	merged["rank"] = pd.to_numeric(merged.get("global_rank"), errors="coerce").astype("Int64")

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

	st.subheader("Highlights")
	now = pd.Timestamp.utcnow()
	list_dates = pd.to_datetime(merged.get("listing_date"), errors="coerce", utc=True)
	days_since = (now - list_dates).dt.days
	roi = pd.to_numeric(merged.get("ROI %"), errors="coerce")
	ath_roi = pd.to_numeric(merged.get("ATH ROI %"), errors="coerce")
	near_ath = (to_ath_series <= 25).sum() if "% to ATH" in merged else 0

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
		st.metric("Share with positive ROI", f"{(100 * pos / max(total, 1)):.1f}%")
	with m2:
		st.metric("Tokens within 25% of ATH", int(near_ath))
	with m3:
		st.metric("Median days since listing", int(days_since.median()) if days_since.notna().any() else "–")
	with m4:
		st.metric("90th pct ROI %", f"{roi.quantile(0.90):.2f}%" if roi.notna().any() else "–")

	mc = pd.to_numeric(merged.get("market_cap_usd"), errors="coerce")
	alpha_total_mc = float(mc.dropna().sum()) if mc.notna().any() else None
	global_mc = cg_global_market_cap_usd()
	share = (alpha_total_mc / global_mc * 100.0) if (alpha_total_mc and global_mc and global_mc > 0) else None

	s1, s2, s3, s4 = st.columns(4)
	with s1:
		st.metric(f"Total market cap — {config.tokens_label}", fmt_usd(alpha_total_mc))
	with s2:
		st.metric("Share of total crypto market", f"{share:.2f}%" if share is not None else "–")
	pos_count = int((roi > 0).sum()) if roi.notna().any() else 0
	neg_count = int((roi <= 0).sum()) if roi.notna().any() else 0
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

	merged["_mc_num"] = mc

	colt1, colt2, colt3 = st.columns(3)
	with colt1:
		st.caption("Top market cap")
		top_mc = merged.dropna(subset=["_mc_num"]).nlargest(10, "_mc_num")[["name", "symbol", "_mc_num"]]
		st.dataframe(top_mc.rename(columns={"_mc_num": "market cap usd"}), hide_index=True, use_container_width=True)
	with colt2:
		st.caption("Top ROI since Binance Alpha")
		top_roi = merged.dropna(subset=["ROI %"]).nlargest(10, "ROI %")[["name", "symbol", "ROI %"]]
		st.dataframe(top_roi, hide_index=True, use_container_width=True)
	with colt3:
		st.caption("Top ATH ROI since Binance Alpha")
		top_ath_roi = merged.dropna(subset=["ATH ROI %"]).nlargest(10, "ATH ROI %")[["name", "symbol", "ATH ROI %"]]
		st.dataframe(top_ath_roi, hide_index=True, use_container_width=True)

	colb1, colb2 = st.columns(2)
	with colb1:
		st.caption("Bottom ROI since Binance Alpha")
		bot_roi = merged.dropna(subset=["ROI %"]).nsmallest(10, "ROI %")[["name", "symbol", "ROI %"]]
		st.dataframe(bot_roi, hide_index=True, use_container_width=True)
	with colb2:
		st.caption("Bottom ATH ROI since Binance Alpha")
		bot_ath_roi = merged.dropna(subset=["ATH ROI %"]).nsmallest(10, "ATH ROI %")[["name", "symbol", "ATH ROI %"]]
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
	above_df = merged[roi > 0].copy() if roi.notna().any() else merged.iloc[0:0]
	below_df = merged[roi <= 0].copy() if roi.notna().any() else merged.iloc[0:0]
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

	if "filtered" in locals():
		rename_map = {c: c.replace("_", " ") for c in filtered.columns}
		rename_map["ROI %"] = "ROI % (since Binance Alpha)"
		rename_map["ATH ROI %"] = "ATH ROI % (since Binance Alpha)"
		st.download_button(
			"Download table as CSV",
			to_csv_bytes(filtered.rename(columns=rename_map)),
			file_name=f"{config.download_basename}.csv",
			mime="text/csv",
		)

