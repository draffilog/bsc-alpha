import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import httpx
from aiolimiter import AsyncLimiter
from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

PANCAKE_LIST = "https://tokens.pancakeswap.finance/pancakeswap-extended.json"
COINGECKO_LIST = "https://tokens.coingecko.com/binance-smart-chain/all.json"
BSCSCAN_API = "https://api.bscscan.com/api"


class TokenItem(BaseModel):
    address: str
    chainId: int
    symbol: Optional[str] = None
    name: Optional[str] = None


@dataclass
class AddressSource:
    name: str
    url: str


ADDRESS_SOURCES = {
    "pancakeswap": AddressSource("pancakeswap", PANCAKE_LIST),
    "coingecko": AddressSource("coingecko", COINGECKO_LIST),
}


def normalize_address(addr: str) -> str:
    a = addr.strip().lower()
    if a.startswith("0x") and len(a) == 42:
        return a
    return a


async def fetch_json(client: httpx.AsyncClient, url: str) -> dict:
    r = await client.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


async def gather_tokens(source_keys: List[str]) -> Dict[str, dict]:
    """Return mapping of address -> {address, symbol, name}."""
    tokens_map: Dict[str, dict] = {}
    async with httpx.AsyncClient(follow_redirects=True) as client:
        for key in source_keys:
            src = ADDRESS_SOURCES[key]
            try:
                data = await fetch_json(client, src.url)
            except Exception as e:
                console.print(f"[yellow]Warn:[/] Failed to fetch {src.name}: {e}")
                continue
            tokens = data.get("tokens") or data.get("data") or []
            for t in tokens:
                try:
                    item = TokenItem.model_validate(t)
                except Exception:
                    continue
                if item.chainId == 56:
                    addr = normalize_address(item.address)
                    if addr not in tokens_map:
                        tokens_map[addr] = {
                            "address": addr,
                            "symbol": item.symbol or "",
                            "name": item.name or "",
                        }
    return tokens_map


class BscScanClient:
    def __init__(self, api_key: str, rps: int = 3) -> None:
        self.api_key = api_key
        self.client = httpx.AsyncClient(base_url=BSCSCAN_API, follow_redirects=True, timeout=40)
        self.rate = AsyncLimiter(rps, 1)

    async def close(self) -> None:
        await self.client.aclose()

    async def get_verified_source(self, address: str) -> Optional[dict]:
        params = {
            "module": "contract",
            "action": "getsourcecode",
            "address": address,
            "apikey": self.api_key,
        }
        async with self.rate:
            resp = await self.client.get("", params=params)
        resp.raise_for_status()
        payload = resp.json()
        status = payload.get("status")
        if status != "1":
            return None
        result = payload.get("result", [])
        return result[0] if result else None


def save_source_record(out_dir: Path, address: str, record: dict) -> None:
    target_dir = out_dir / address
    target_dir.mkdir(parents=True, exist_ok=True)

    # Always save raw metadata
    (target_dir / "bscscan_metadata.json").write_text(json.dumps(record, indent=2))

    content = record.get("SourceCode") or ""
    contract_name = record.get("ContractName") or "Contract"

    # BscScan wraps multi-file sources as a JSON object within braces sometimes; detect that
    if content.strip().startswith("{{"):  # single string with nested json
        try:
            # Some responses are wrapped like: {{"language":"Solidity", "sources": {...}}}
            # Remove one level of braces if duplicated
            trimmed = content.strip()[1:-1]
            obj = json.loads(trimmed)
        except Exception:
            obj = None
        if obj and isinstance(obj, dict):
            sources = None
            if "sources" in obj and isinstance(obj["sources"], dict):
                sources = obj["sources"]
            elif "Sources" in obj and isinstance(obj["Sources"], dict):
                sources = obj["Sources"]
            if sources:
                for relpath, src in sources.items():
                    code = src.get("content") if isinstance(src, dict) else str(src)
                    path = target_dir / relpath
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(code)
                return

    # Fallback: write single file
    (target_dir / f"{contract_name}.sol").write_text(content)


async def download_worker(addresses: Iterable[str], client: BscScanClient, out_dir: Path, progress, task_id: int) -> Tuple[int, int]:
    ok = 0
    skipped = 0
    for address in addresses:
        try:
            record = await client.get_verified_source(address)
            if not record:
                skipped += 1
            else:
                save_source_record(out_dir, address, record)
                ok += 1
        except Exception as e:
            console.print(f"[red]Error[/] {address}: {e}")
            skipped += 1
        finally:
            try:
                progress.update(task_id, advance=1)
            except Exception:
                pass
    return ok, skipped


def chunked(seq: List[str], n: int) -> List[List[str]]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def save_tokens_index(out_dir: Path, tokens_map: Dict[str, dict]) -> None:
    # tokens.json
    tokens_list = list(tokens_map.values())
    (out_dir / "tokens.json").write_text(json.dumps(tokens_list, indent=2))
    # tokens.csv
    import csv
    with (out_dir / "tokens.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["address", "symbol", "name"])
        for t in tokens_list:
            writer.writerow([t.get("address", ""), t.get("symbol", ""), t.get("name", "")])


def save_token_record(out_dir: Path, token: dict) -> None:
    target_dir = out_dir / token["address"]
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "token.json").write_text(json.dumps(token, indent=2))


async def run(source: str, limit: int, out: str, api_key: Optional[str], concurrency: int, bscscan_rps: int, addresses_only: bool) -> int:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine address sources
    if source == "both":
        sources = ["pancakeswap", "coingecko"]
    else:
        sources = [source]

    tokens_map = await gather_tokens(sources)
    if not tokens_map:
        console.print("[red]No addresses found[/]")
        return 1

    # De-dup and limit
    addresses = list(tokens_map.keys())

    if limit and limit > 0:
        addresses = addresses[:limit]

    # Persist token indices
    limited_tokens_map = {addr: tokens_map[addr] for addr in addresses}
    save_tokens_index(out_dir, limited_tokens_map)
    for token in limited_tokens_map.values():
        save_token_record(out_dir, token)

    if addresses_only:
        console.print(f"[green]Saved token metadata for {len(addresses)} addresses. Skipping source downloads (--addresses-only).")
        return 0

    api_key = api_key or os.getenv("BSCSCAN_API_KEY")
    if not api_key:
        console.print("[red]Missing BscScan API key. Provide --api-key or set BSCSCAN_API_KEY.[/]")
        return 2

    client = BscScanClient(api_key=api_key, rps=bscscan_rps)

    total = len(addresses)
    batches = chunked(addresses, max(1, total // max(1, concurrency)))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task_id = progress.add_task(f"Downloading {total} contracts from BscScan", total=total)

        # Sequential per batch to keep code simple; still parallel per BscScan RPS
        results: List[Tuple[int, int]] = []
        for batch in batches:
            ok, skipped = await download_worker(batch, client, out_dir, progress, task_id)
            results.append((ok, skipped))

    await client.close()

    total_ok = sum(o for o, _ in results)
    total_skipped = sum(s for _, s in results)
    console.print(f"[green]Saved[/] {total_ok}  [yellow]skipped[/] {total_skipped}  [white]total[/] {total}")
    return 0


def parse_args(argv: List[str]):
    import argparse

    parser = argparse.ArgumentParser(description="Fetch BSC token contract sources via BscScan")
    parser.add_argument("--source", choices=["pancakeswap", "coingecko", "both"], default="both")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", type=str, default="./data/contracts")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("CONCURRENCY", "2")))
    parser.add_argument("--bscscan-rps", type=int, default=int(os.getenv("BSCSCAN_RPS", "3")))
    parser.add_argument("--addresses-only", action="store_true", help="Only fetch addresses and tickers; skip source downloads")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return asyncio.run(
        run(
            source=args.source,
            limit=args.limit,
            out=args.out,
            api_key=args.api_key,
            concurrency=args.concurrency,
            bscscan_rps=args.bscscan_rps,
            addresses_only=args.addresses_only,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
