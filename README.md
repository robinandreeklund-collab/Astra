# Astra — Local AI Paper Trading

Local-only paper-trading platform for S&P 500 stocks. A local LLM (via LM Studio)
makes BUY/SELL/HOLD decisions based on signals from Finnhub. The AI learns from
its own trade history. Portfolio can be reset while the strategy memory persists
across resets.

**No real money. Ever.** The codebase has no live broker integration.

## Quick start

```bash
# 1. Install
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env: add FINNHUB_API_KEY, ensure LM Studio is running on :1234

# 3. Run
python -m astra
# Open http://127.0.0.1:8765
```

## Architecture

```
FastAPI ── SQLite (portfolio.db + memory.db)
   │
   ├── Finnhub REST/WS  (signals)
   ├── LM Studio /v1    (decisions)
   └── APScheduler      (5-min tick loop)
```

See [docs/architecture.md] for details (or `astra/__init__.py` docstring).

## Modes

- **paper** — live signals + simulated fills (default)
- **backtest** — historical replay against past candles

## Reset

`POST /api/reset` wipes `portfolio.db` (cash, positions, trades, equity) but
keeps `memory.db` (pattern stats + strategy lessons). A new `generation` row is
written so the AI knows it's starting fresh with prior knowledge.

## License

MIT
