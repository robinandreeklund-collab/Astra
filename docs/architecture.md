# Astra — architecture

```
┌──────────────────────────────────────────────────────┐
│ Browser (HTMX + Chart.js, SSE for live updates)      │
└─────────────────┬────────────────────────────────────┘
                  │ HTTP / SSE
┌─────────────────▼────────────────────────────────────┐
│ FastAPI app (astra.app)                              │
│   astra.api.routes — pages + JSON + SSE              │
│   astra.engine.runner — tick loop                    │
│   astra.engine.backtest — historical replay          │
│   astra.engine.broker — paper fills                  │
│   astra.engine.risk — position sizing + caps         │
│   astra.signals.* — TA + news + insider + funds      │
│   astra.llm.* — LM Studio client + decision engine   │
│   astra.memory.* — recorder + retriever + reflector  │
│   astra.data.finnhub — REST client (rate-limited)    │
└─┬───────────────┬───────────────┬────────────────────┘
  │               │               │
┌─▼───┐  ┌────────▼────────┐  ┌───▼─────────────┐
│SQLite│ │ Finnhub REST    │  │ LM Studio /v1   │
│ 3 db │ │ (live signals)  │  │ (decisions +    │
│      │ │                 │  │  reflection)    │
└──────┘ └─────────────────┘  └─────────────────┘
```

## Data stores

Three SQLite databases under `data/` (path configurable via `ASTRA_DATA_DIR`):

| File           | Purpose                                                        | Wiped on reset |
| -------------- | -------------------------------------------------------------- | -------------- |
| `portfolio.db` | account, positions, trades, equity curve, AI thought log        | yes            |
| `memory.db`    | pattern stats, strategy lessons, generations                   | **no**         |
| `cache.db`     | Finnhub response cache (TTL) + candle bars                     | safe to delete |

## Decision flow per tick

1. `TradingEngine.tick()` runs.
2. Universe is the S&P 500 (Finnhub `/index/constituents` with a static
   fallback list of large-caps).
3. Watchlist = `held_positions ∪ first N tickers` where N = `ASTRA_WATCHLIST_SIZE`.
4. For each symbol:
   - `SignalAggregator.fetch()` collects in parallel: quote, daily candles
     (→ technical indicators), company news, news sentiment, insider
     transactions, analyst recommendations, earnings, earnings calendar,
     social sentiment.
   - A pattern fingerprint (`pattern_hash`) is computed by bucketing the
     feature vector (RSI band, MACD sign, BB position, SMA cross, insider
     direction, sentiment bucket, analyst trend).
   - `MemoryRetriever` fetches the top-N strategy lessons + relevant pattern
     stats from `memory.db`.
   - `DecisionEngine` calls LM Studio (`response_format=json_object`) with
     a structured prompt. If LM Studio is unreachable, a deterministic
     heuristic decides instead (still records reasoning).
   - `RiskManager` sizes the order (max position cap + confidence) and
     blocks trades on a daily loss limit breach.
   - `PaperBroker` simulates a fill (slippage in bps + flat fee per trade)
     and writes a `trades` row with the full signal snapshot.
5. Mark-to-market: a new `equity_curve` point is appended.
6. SSE broadcasts `thought`, `trade`, and `equity` events to any subscribed
   clients.

## Learning

* Every closed `SELL` calls `MemoryRecorder.record_close`, updating
  `pattern_stats(wins, losses, total_pnl, avg_hold_minutes, confidence)`.
  Confidence uses the Wilson lower bound so a 3W/0L pattern doesn't outrank
  a 30W/2L pattern.
* `Reflector.reflect()` (manual via `/api/engine/reflect`, or schedulable)
  sends the most recent 30 closed trades to LM Studio and asks for up to
  three concrete lessons, stored in `strategy_lessons` with a score.

## Reset semantics

`POST /api/reset` with a new starting balance:

1. Closes the current `generations` row with `ending_equity` + trade count.
2. Stops the engine, wipes `portfolio.db`.
3. Creates a new account + new generation row.
4. Starts the engine.

`memory.db` is **untouched**. The next decision cycle therefore starts with
all pattern stats and lessons learned from prior generations intact, but a
clean cash balance and empty position book.

## Backtest

`run_backtest(symbols, starting_balance, days, use_llm=False)`:

* Fetches daily candles for the symbols + benchmark (SPY) once.
* Walks bar-by-bar. At each bar, only bars ≤ current ts are visible
  (no lookahead).
* Uses isolated temp databases so production portfolio + memory are
  unaffected.
* Returns equity curve, trade list, win-rate, alpha vs SPY.

## No real money

The codebase has no live-broker integration. The account `mode` column is
restricted to `paper | backtest`; broker module imports no third-party
trading SDK. UI shows a permanent `PAPER` badge.
