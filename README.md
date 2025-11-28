# Binance Alpha: Token Contract Source Fetcher

This utility fetches smart contract source code for BSC (chainId 56) token contracts from reliable public indexers, avoiding brittle scraping of the Binance Alpha UI.

- Primary flow: gather token contract addresses from public token lists (CoinGecko, PancakeSwap), then fetch verified source code from BscScan and save them locally.
- Why not scrape the Alpha UI? The Alpha page (e.g., `https://www.binance.com/en/alpha/bsc/0xe6df05ce8c8301223373cf5b969afcb1498c5528`) uses internal, undocumented endpoints that can change and rate-limit without notice.

## Quick start

1) Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Provide a BscScan API key (required to download verified source code):

- Copy `.env.example` to `.env` and fill `BSCSCAN_API_KEY`, or
- Pass `--api-key` on the CLI, or
- Set env var `BSCSCAN_API_KEY` in your shell

3) Run the fetcher (examples):

```bash
# Fetch from PancakeSwap list and download up to 200 verified contract sources to ./data/contracts
python -m alpha_contracts.fetch_contracts --source pancakeswap --limit 200 --out ./data/contracts

# Fetch from CoinGecko (can be large, use --limit) and download sources
python -m alpha_contracts.fetch_contracts --source coingecko --limit 300 --out ./data/contracts

# Combine both sources, skip addresses already fetched, and increase concurrency
python -m alpha_contracts.fetch_contracts --source both --limit 500 --concurrency 4 --out ./data/contracts
```

Outputs are stored under `./data/contracts/<address>/`. When a contract has multi-file sources, all files are saved preserving paths; single-file sources are saved as `<ContractName>.sol`.

## Dashboard

- Streamlit app: `dashboard/app.py`
- Metrics fetcher (CoinGecko): `alpha_contracts/metrics.py`
- Run locally:

```bash
COINGECKO_API_KEY=your_key streamlit run dashboard/app.py
```

## Deploy (Streamlit Cloud)

1) Push this repo to GitHub.
2) Go to Streamlit Community Cloud → New app → select your repo/branch → `dashboard/app.py`.
3) In Secrets, add:

```
COINGECKO_API_KEY = "your_pro_key"
BSCSCAN_API_KEY = "optional"
```

4) Deploy. The app reads secrets automatically and streams fetch logs live.

## Deploy (Docker)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENV COINGECKO_API_KEY=your_key
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

```bash
docker build -t bsc-alpha .
docker run -p 8501:8501 -e COINGECKO_API_KEY=your_key bsc-alpha
```

## Legal

- Use responsibly and respect API rate limits and Terms of Service of data providers.
- Binance Alpha page referenced: https://www.binance.com/en/alpha/bsc/0xe6df05ce8c8301223373cf5b969afcb1498c5528

---

## Token classification pipeline

This script classifies tokens from `data/alpha/alpha_tokens.csv` into BNB vs other L1/L2, category, and BD attribution, using BscScan official links, Perplexity summarization, optional GitHub activity, and optional Monday.com matching.

### Env vars

- `BSCSCAN_KEY` or `BSCSCAN_API_KEY` (recommended)
- `PPLX_KEY` (Perplexity; optional but recommended)
- `GITHUB_TOKEN` (optional)
- `MONDAY_TOKEN` (optional, for BD attribution matching)

You can place these in a `.env` file at repo root.

### Run

```bash
python -m alpha_contracts.classify_tokens \
  --input ./data/alpha/alpha_tokens.csv \
  --out ./data/alpha \
  --board-id <monday_board_id_optional> \
  --overrides ./data/alpha/overrides.csv  # optional CSV: name,domain,bnb_launch_date
```

Flags: `--limit N` to process a subset, `--sleep-ms` to pace API calls.

Output: `data/alpha/classified_tokens.csv` and a printed summary of counts by `primary_chain`, `category`, and `bd_attribution`.
