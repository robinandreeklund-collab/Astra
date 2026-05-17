"""Memory database — survives portfolio reset."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astra.db._base import SQLiteDB

SCHEMA = """
CREATE TABLE IF NOT EXISTS pattern_stats (
    pattern_hash TEXT PRIMARY KEY,
    pattern_desc TEXT NOT NULL,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    total_pnl REAL NOT NULL DEFAULT 0,
    avg_hold_minutes REAL,
    last_seen TEXT,
    confidence REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS strategy_lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    generation INTEGER NOT NULL,
    lesson TEXT NOT NULL,
    evidence_trade_ids TEXT,
    score REAL NOT NULL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_lessons_score ON strategy_lessons(score DESC);

CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    reset_at TEXT,
    starting_balance REAL,
    ending_equity REAL,
    trade_count INTEGER,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS stock_profiles (
    symbol TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    key TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def wilson_lower_bound(wins: int, losses: int, z: float = 1.96) -> float:
    n = wins + losses
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (centre - margin) / denom


class MemoryDB(SQLiteDB):
    def __init__(self, path: Path) -> None:
        super().__init__(path)

    async def init(self) -> None:
        await self.init_schema(SCHEMA)

    # ---- patterns ----

    async def record_pattern_outcome(
        self,
        pattern_hash: str,
        pattern_desc: str,
        win: bool,
        pnl: float,
        hold_minutes: float,
    ) -> None:
        async with self.session() as conn:
            row = await (
                await conn.execute(
                    "SELECT wins, losses, total_pnl, avg_hold_minutes "
                    "FROM pattern_stats WHERE pattern_hash=?",
                    (pattern_hash,),
                )
            ).fetchone()

            if row is None:
                wins = 1 if win else 0
                losses = 0 if win else 1
                avg_hold = hold_minutes
                total_pnl = pnl
            else:
                wins = row["wins"] + (1 if win else 0)
                losses = row["losses"] + (0 if win else 1)
                total_pnl = row["total_pnl"] + pnl
                n = wins + losses
                prev_avg = row["avg_hold_minutes"] or 0.0
                avg_hold = (prev_avg * (n - 1) + hold_minutes) / n

            confidence = wilson_lower_bound(wins, losses)
            await conn.execute(
                """INSERT INTO pattern_stats(pattern_hash, pattern_desc, wins, losses,
                                              total_pnl, avg_hold_minutes, last_seen, confidence)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(pattern_hash) DO UPDATE SET
                       pattern_desc=excluded.pattern_desc,
                       wins=excluded.wins, losses=excluded.losses,
                       total_pnl=excluded.total_pnl,
                       avg_hold_minutes=excluded.avg_hold_minutes,
                       last_seen=excluded.last_seen,
                       confidence=excluded.confidence""",
                (pattern_hash, pattern_desc, wins, losses, total_pnl,
                 avg_hold, _now(), confidence),
            )

    async def relevant_patterns(
        self, current_hash: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        async with self.session() as conn:
            row = await (
                await conn.execute(
                    "SELECT * FROM pattern_stats WHERE pattern_hash=?", (current_hash,)
                )
            ).fetchone()
            results: list[dict[str, Any]] = []
            if row is not None:
                results.append(dict(row))
            rows = await (
                await conn.execute(
                    "SELECT * FROM pattern_stats WHERE pattern_hash != ? "
                    "ORDER BY (wins + losses) DESC, confidence DESC LIMIT ?",
                    (current_hash, limit - len(results)),
                )
            ).fetchall()
            results.extend(dict(r) for r in rows)
            return results

    async def all_patterns(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self.session() as conn:
            rows = await (
                await conn.execute(
                    "SELECT * FROM pattern_stats ORDER BY (wins+losses) DESC LIMIT ?",
                    (limit,),
                )
            ).fetchall()
            return [dict(r) for r in rows]

    # ---- lessons ----

    async def add_lesson(
        self,
        lesson: str,
        generation: int,
        evidence_trade_ids: list[int] | None = None,
        score: float = 1.0,
    ) -> int:
        async with self.session() as conn:
            cur = await conn.execute(
                "INSERT INTO strategy_lessons(created_at, generation, lesson, "
                "evidence_trade_ids, score) VALUES(?,?,?,?,?)",
                (
                    _now(),
                    generation,
                    lesson,
                    json.dumps(evidence_trade_ids or []),
                    score,
                ),
            )
            return cur.lastrowid or 0

    async def top_lessons(self, limit: int = 8) -> list[dict[str, Any]]:
        async with self.session() as conn:
            rows = await (
                await conn.execute(
                    "SELECT * FROM strategy_lessons ORDER BY score DESC, created_at DESC LIMIT ?",
                    (limit,),
                )
            ).fetchall()
            return [dict(r) for r in rows]

    async def all_lessons(self) -> list[dict[str, Any]]:
        async with self.session() as conn:
            rows = await (
                await conn.execute(
                    "SELECT * FROM strategy_lessons ORDER BY created_at DESC"
                )
            ).fetchall()
            return [dict(r) for r in rows]

    async def adjust_lesson_score(self, lesson_id: int, delta: float) -> None:
        async with self.session() as conn:
            await conn.execute(
                "UPDATE strategy_lessons SET score = MAX(0, score + ?) WHERE id=?",
                (delta, lesson_id),
            )

    # ---- generations ----

    async def start_generation(self, starting_balance: float) -> int:
        async with self.session() as conn:
            cur = await conn.execute(
                "INSERT INTO generations(started_at, starting_balance) VALUES(?,?)",
                (_now(), starting_balance),
            )
            return cur.lastrowid or 0

    async def close_generation(
        self, gen_id: int, ending_equity: float, trade_count: int, notes: str = ""
    ) -> None:
        async with self.session() as conn:
            await conn.execute(
                "UPDATE generations SET reset_at=?, ending_equity=?, trade_count=?, notes=? "
                "WHERE id=?",
                (_now(), ending_equity, trade_count, notes, gen_id),
            )

    async def current_generation(self) -> dict[str, Any] | None:
        async with self.session() as conn:
            row = await (
                await conn.execute(
                    "SELECT * FROM generations WHERE reset_at IS NULL "
                    "ORDER BY id DESC LIMIT 1"
                )
            ).fetchone()
            return dict(row) if row else None

    async def all_generations(self) -> list[dict[str, Any]]:
        async with self.session() as conn:
            rows = await (
                await conn.execute("SELECT * FROM generations ORDER BY id DESC")
            ).fetchall()
            return [dict(r) for r in rows]

    # ---- stock profiles (per-stock adaptive memory) ----

    async def get_stock_profile(self, symbol: str) -> dict[str, Any] | None:
        async with self.session() as conn:
            row = await (
                await conn.execute(
                    "SELECT data FROM stock_profiles WHERE symbol=?", (symbol,)
                )
            ).fetchone()
            if not row:
                return None
            try:
                return json.loads(row["data"])
            except Exception:
                return None

    async def save_stock_profile(self, symbol: str, data: dict[str, Any]) -> None:
        async with self.session() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO stock_profiles(symbol, data, updated_at) "
                "VALUES(?,?,?)",
                (symbol, json.dumps(data), _now()),
            )

    async def all_stock_profiles(self) -> list[dict[str, Any]]:
        async with self.session() as conn:
            rows = await (
                await conn.execute("SELECT data FROM stock_profiles")
            ).fetchall()
            out: list[dict[str, Any]] = []
            for r in rows:
                try:
                    out.append(json.loads(r["data"]))
                except Exception:
                    continue
            return out

    # ---- generic model store (contextual bandit, etc.) ----

    async def get_model(self, key: str) -> dict[str, Any] | None:
        async with self.session() as conn:
            row = await (
                await conn.execute("SELECT data FROM models WHERE key=?", (key,))
            ).fetchone()
            if not row:
                return None
            try:
                return json.loads(row["data"])
            except Exception:
                return None

    async def save_model(self, key: str, data: dict[str, Any]) -> None:
        async with self.session() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO models(key, data, updated_at) VALUES(?,?,?)",
                (key, json.dumps(data), _now()),
            )
