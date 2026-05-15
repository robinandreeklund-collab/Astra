"""Cache database — for Finnhub responses (candles, news, fundamentals)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from astra.db._base import SQLiteDB

SCHEMA = """
CREATE TABLE IF NOT EXISTS kv_cache (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    expires_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_kv_expires ON kv_cache(expires_at);

CREATE TABLE IF NOT EXISTS candle_cache (
    symbol TEXT NOT NULL,
    resolution TEXT NOT NULL,
    ts INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (symbol, resolution, ts)
);
CREATE INDEX IF NOT EXISTS idx_candle_sym_res ON candle_cache(symbol, resolution, ts);
"""


class CacheDB(SQLiteDB):
    def __init__(self, path: Path) -> None:
        super().__init__(path)

    async def init(self) -> None:
        await self.init_schema(SCHEMA)

    async def get(self, key: str) -> Any | None:
        async with self.session() as conn:
            row = await (
                await conn.execute(
                    "SELECT value, expires_at FROM kv_cache WHERE key=?", (key,)
                )
            ).fetchone()
            if not row:
                return None
            if row["expires_at"] < int(time.time()):
                return None
            try:
                return json.loads(row["value"])
            except Exception:
                return None

    async def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        async with self.session() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO kv_cache(key, value, expires_at) VALUES(?,?,?)",
                (key, json.dumps(value), int(time.time()) + ttl_seconds),
            )

    async def purge_expired(self) -> None:
        async with self.session() as conn:
            await conn.execute(
                "DELETE FROM kv_cache WHERE expires_at < ?", (int(time.time()),)
            )

    async def store_candles(
        self,
        symbol: str,
        resolution: str,
        rows: list[dict[str, Any]],
    ) -> None:
        if not rows:
            return
        async with self.session() as conn:
            await conn.executemany(
                "INSERT OR REPLACE INTO candle_cache(symbol, resolution, ts, "
                "open, high, low, close, volume) VALUES(?,?,?,?,?,?,?,?)",
                [
                    (
                        symbol,
                        resolution,
                        int(r["t"]),
                        float(r["o"]),
                        float(r["h"]),
                        float(r["l"]),
                        float(r["c"]),
                        float(r.get("v", 0)),
                    )
                    for r in rows
                ],
            )

    async def load_candles(
        self,
        symbol: str,
        resolution: str,
        from_ts: int | None = None,
        to_ts: int | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM candle_cache WHERE symbol=? AND resolution=?"
        params: list[Any] = [symbol, resolution]
        if from_ts is not None:
            q += " AND ts >= ?"
            params.append(from_ts)
        if to_ts is not None:
            q += " AND ts <= ?"
            params.append(to_ts)
        q += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        async with self.session() as conn:
            rows = await (await conn.execute(q, params)).fetchall()
            return [dict(r) for r in reversed(rows)]
