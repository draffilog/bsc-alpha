import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console

console = Console()


def filter_solana(df: pd.DataFrame) -> pd.DataFrame:
	mask = (
		df["chain"].astype(str).str.lower().eq("solana")
		| df["chain_id"].astype(str).str.lower().eq("ct_501")
	)
	solana = df[mask].copy()
	return solana


def write_tokens(df: pd.DataFrame, out_path: Path) -> int:
	cols = ["address", "symbol", "name"]
	missing = [c for c in cols if c not in df.columns]
	if missing:
		raise ValueError(f"Missing columns for tokens export: {', '.join(missing)}")
	tokens = df[cols].drop_duplicates("address")
	out_path.parent.mkdir(parents=True, exist_ok=True)
	tokens.to_csv(out_path, index=False)
	return len(tokens)


def write_listings(df: pd.DataFrame, out_path: Path) -> int:
	required = [
		"address",
		"symbol",
		"name",
		"listing_timestamp_ms",
		"listing_date_alpha",
		"listing_price_usdt",
	]
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f"Missing columns for listings export: {', '.join(missing)}")
	listings = df[required + ["alpha_pair"]].copy()
	listings = listings.rename(columns={"listing_price_usdt": "listing_price_quote"})
	listings["listing_quote"] = "USDT"
	out_path.parent.mkdir(parents=True, exist_ok=True)
	listings.to_csv(
		out_path,
		index=False,
		columns=[
			"address",
			"symbol",
			"name",
			"listing_timestamp_ms",
			"listing_date_alpha",
			"listing_price_quote",
			"listing_quote",
			"alpha_pair",
		],
	)
	return len(listings)


def run(cross_chain_csv: Path, tokens_out: Path, listings_out: Path) -> int:
	if not cross_chain_csv.exists():
		console.print(f"[red]Missing cross-chain listings file {cross_chain_csv}[/]")
		return 2
	df = pd.read_csv(cross_chain_csv)
	if df.empty:
		console.print("[yellow]Cross-chain listings CSV is empty[/]")
		return 0
	solana = filter_solana(df)
	if solana.empty:
		console.print("[yellow]No Solana rows found in cross-chain listings CSV[/]")
		return 0
	token_count = write_tokens(solana, tokens_out)
	listing_count = write_listings(solana, listings_out)
	console.print(f"[green]Saved {token_count} Solana tokens -> {tokens_out}[/]")
	console.print(f"[green]Saved {listing_count} Solana listings -> {listings_out}[/]")
	return 0


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Prepare Solana token/listing CSVs from cross-chain listings export")
	parser.add_argument(
		"--cross",
		dest="cross_chain_csv",
		type=Path,
		default=Path("./data/alpha/cross_chain_listings.csv"),
		help="Path to the cross-chain listings CSV file.",
	)
	parser.add_argument(
		"--tokens-out",
		dest="tokens_out",
		type=Path,
		default=Path("./data/alpha/solana_tokens.csv"),
		help="Output path for the Solana tokens CSV.",
	)
	parser.add_argument(
		"--listings-out",
		dest="listings_out",
		type=Path,
		default=Path("./data/alpha/solana_listings.csv"),
		help="Output path for the Solana listings CSV.",
	)
	args = parser.parse_args()
	raise SystemExit(run(args.cross_chain_csv, args.tokens_out, args.listings_out))

