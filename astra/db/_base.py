"""Shared SQLite helpers."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite


class SQLiteDB:
    """Tiny async SQLite wrapper with WAL + sane defaults."""

    def __init__(self, path: Path) -> None:
        self.path = path

    async def connect(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(str(self.path))
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @asynccontextmanager
    async def session(self) -> AsyncIterator[aiosqlite.Connection]:
        conn = await self.connect()
        try:
            yield conn
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
        finally:
            await conn.close()

    async def init_schema(self, schema_sql: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with self.session() as conn:
            await conn.executescript(schema_sql)
