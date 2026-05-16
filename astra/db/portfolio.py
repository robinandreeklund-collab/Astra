"""Portfolio database — wiped on reset."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from astra.db._base import SQLiteDB

SCHEMA = """
CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    starting_balance REAL NOT NULL,
    cash REAL NOT NULL,
    created_at TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('paper','backtest'))
);

CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    qty REAL NOT NULL,
    avg_price REAL NOT NULL,
    opened_at TEXT NOT NULL,
    high_water_mark REAL
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
    qty REAL NOT NULL,
    price REAL NOT NULL,
    fees REAL NOT NULL,
    executed_at TEXT NOT NULL,
    signal_snapshot TEXT NOT NULL,
    llm_reasoning TEXT,
    pnl REAL,
    pattern_hash TEXT,
    closed_trade_id INTEGER REFERENCES trades(id)
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, executed_at);
CREATE INDEX IF NOT EXISTS idx_trades_pattern ON trades(pattern_hash);

CREATE TABLE IF NOT EXISTS equity_curve (
    ts TEXT PRIMARY KEY,
    equity REAL NOT NULL,
    cash REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_thoughts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT,
    action TEXT,
    confidence REAL,
    reasoning TEXT NOT NULL,
    accepted INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_thoughts_ts ON ai_thoughts(ts);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PortfolioDB(SQLiteDB):
    def __init__(self, path: Path) -> None:
        super().__init__(path)

    async def init(self) -> None:
        await self.init_schema(SCHEMA)
        # Migration: add high_water_mark to pre-existing positions tables.
        async with self.session() as conn:
            cols = await (await conn.execute("PRAGMA table_info(positions)")).fetchall()
            names = {c["name"] for c in cols}
            if "high_water_mark" not in names:
                await conn.execute(
                    "ALTER TABLE positions ADD COLUMN high_water_mark REAL"
                )

    # ---- account ----

    async def get_account(self) -> dict[str, Any] | None:
        async with self.session() as conn:
            row = await (await conn.execute("SELECT * FROM account WHERE id=1")).fetchone()
            return dict(row) if row else None

    async def create_account(self, starting_balance: float, mode: str = "paper") -> dict[str, Any]:
        async with self.session() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO account(id, starting_balance, cash, created_at, mode) "
                "VALUES (1, ?, ?, ?, ?)",
                (starting_balance, starting_balance, _now(), mode),
            )
        acc = await self.get_account()
        assert acc is not None
        return acc

    async def update_cash(self, cash: float) -> None:
        async with self.session() as conn:
            await conn.execute("UPDATE account SET cash=? WHERE id=1", (cash,))

    # ---- positions ----

    async def get_positions(self) -> list[dict[str, Any]]:
        async with self.session() as conn:
            rows = await (await conn.execute("SELECT * FROM positions ORDER BY symbol")).fetchall()
            return [dict(r) for r in rows]

    async def get_position(self, symbol: str) -> dict[str, Any] | None:
        async with self.session() as conn:
            row = await (
                await conn.execute("SELECT * FROM positions WHERE symbol=?", (symbol,))
            ).fetchone()
            return dict(row) if row else None

    async def upsert_position(
        self, symbol: str, qty: float, avg_price: float,
        high_water_mark: float | None = None,
    ) -> None:
        async with self.session() as conn:
            hwm = high_water_mark if high_water_mark is not None else avg_price
            await conn.execute(
                "INSERT INTO positions(symbol, qty, avg_price, opened_at, high_water_mark) "
                "VALUES(?,?,?,?,?) "
                "ON CONFLICT(symbol) DO UPDATE SET qty=excluded.qty, "
                "avg_price=excluded.avg_price",
                (symbol, qty, avg_price, _now(), hwm),
            )

    async def update_high_water_mark(self, symbol: str, hwm: float) -> None:
        async with self.session() as conn:
            await conn.execute(
                "UPDATE positions SET high_water_mark=? WHERE symbol=?",
                (hwm, symbol),
            )

    async def delete_position(self, symbol: str) -> None:
        async with self.session() as conn:
            await conn.execute("DELETE FROM positions WHERE symbol=?", (symbol,))

    # ---- trades ----

    async def insert_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        fees: float,
        signal_snapshot: dict[str, Any],
        llm_reasoning: str | None,
        pnl: float | None = None,
        pattern_hash: str | None = None,
        closed_trade_id: int | None = None,
        executed_at: str | None = None,
    ) -> int:
        async with self.session() as conn:
            cur = await conn.execute(
                """INSERT INTO trades(symbol, side, qty, price, fees, executed_at,
                                      signal_snapshot, llm_reasoning, pnl,
                                      pattern_hash, closed_trade_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    symbol,
                    side,
                    qty,
                    price,
                    fees,
                    executed_at or _now(),
                    json.dumps(signal_snapshot),
                    llm_reasoning,
                    pnl,
                    pattern_hash,
                    closed_trade_id,
                ),
            )
            return cur.lastrowid or 0

    async def list_trades(self, limit: int = 200) -> list[dict[str, Any]]:
        async with self.session() as conn:
            rows = await (
                await conn.execute(
                    "SELECT * FROM trades ORDER BY executed_at DESC LIMIT ?", (limit,)
                )
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                try:
                    d["signal_snapshot"] = json.loads(d["signal_snapshot"])
                except Exception:
                    pass
                out.append(d)
            return out

    async def closed_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self.session() as conn:
            rows = await (
                await conn.execute(
                    "SELECT * FROM trades WHERE side='SELL' AND pnl IS NOT NULL "
                    "ORDER BY executed_at DESC LIMIT ?",
                    (limit,),
                )
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                try:
                    d["signal_snapshot"] = json.loads(d["signal_snapshot"])
                except Exception:
                    pass
                out.append(d)
            return out

    async def get_trade(self, trade_id: int) -> dict[str, Any] | None:
        async with self.session() as conn:
            row = await (
                await conn.execute("SELECT * FROM trades WHERE id=?", (trade_id,))
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            try:
                d["signal_snapshot"] = json.loads(d["signal_snapshot"])
            except Exception:
                pass
            return d

    async def find_sell_for_buy(self, buy_id: int) -> dict[str, Any] | None:
        """The SELL trade that closed a given BUY, if any."""
        async with self.session() as conn:
            row = await (
                await conn.execute(
                    "SELECT * FROM trades WHERE side='SELL' AND closed_trade_id=? "
                    "ORDER BY executed_at ASC LIMIT 1",
                    (buy_id,),
                )
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            try:
                d["signal_snapshot"] = json.loads(d["signal_snapshot"])
            except Exception:
                pass
            return d

    async def open_buy_trade_for(self, symbol: str) -> dict[str, Any] | None:
        """Last unmatched BUY trade for a symbol (FIFO)."""
        async with self.session() as conn:
            row = await (
                await conn.execute(
                    """SELECT * FROM trades
                       WHERE symbol=? AND side='BUY'
                         AND id NOT IN (SELECT closed_trade_id FROM trades
                                        WHERE side='SELL' AND closed_trade_id IS NOT NULL)
                       ORDER BY executed_at ASC LIMIT 1""",
                    (symbol,),
                )
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            try:
                d["signal_snapshot"] = json.loads(d["signal_snapshot"])
            except Exception:
                pass
            return d

    # ---- equity curve ----

    async def append_equity(self, equity: float, cash: float, ts: str | None = None) -> None:
        async with self.session() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO equity_curve(ts, equity, cash) VALUES(?,?,?)",
                (ts or _now(), equity, cash),
            )

    async def equity_history(self, limit: int = 1000) -> list[dict[str, Any]]:
        async with self.session() as conn:
            rows = await (
                await conn.execute(
                    "SELECT ts, equity, cash FROM equity_curve "
                    "ORDER BY ts DESC LIMIT ?",
                    (limit,),
                )
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    # ---- thoughts ----

    async def log_thought(
        self,
        symbol: str | None,
        action: str | None,
        confidence: float | None,
        reasoning: str,
        accepted: bool = True,
    ) -> None:
        async with self.session() as conn:
            await conn.execute(
                "INSERT INTO ai_thoughts(ts, symbol, action, confidence, reasoning, accepted) "
                "VALUES(?,?,?,?,?,?)",
                (_now(), symbol, action, confidence, reasoning, 1 if accepted else 0),
            )

    async def recent_thoughts(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self.session() as conn:
            rows = await (
                await conn.execute(
                    "SELECT * FROM ai_thoughts ORDER BY ts DESC LIMIT ?", (limit,)
                )
            ).fetchall()
            return [dict(r) for r in rows]

    # ---- reset ----

    async def wipe(self) -> None:
        async with self.session() as conn:
            await conn.executescript(
                "DELETE FROM account; DELETE FROM positions; DELETE FROM trades; "
                "DELETE FROM equity_curve; DELETE FROM ai_thoughts;"
            )
