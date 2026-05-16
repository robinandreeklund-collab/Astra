"""Shared pytest fixtures."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio


@pytest.fixture(scope="session", autouse=True)
def isolate_settings():
    """Use a temp data dir + dummy keys for all tests."""
    tmp = tempfile.mkdtemp(prefix="astra-test-")
    os.environ.setdefault("ASTRA_DATA_DIR", tmp)
    os.environ.setdefault("FINNHUB_API_KEY", "test")
    os.environ.setdefault("LM_STUDIO_URL", "http://127.0.0.1:0")  # unreachable on purpose
    os.environ.setdefault("ASTRA_TICK_SECONDS", "3600")
    yield


@pytest.fixture(autouse=True)
def reset_settings():
    """Snapshot the global Settings before each test and restore it after,
    so tests that tweak settings.* don't leak config into each other."""
    from astra.config import settings
    snapshot = settings.model_dump()
    yield
    for key, value in snapshot.items():
        setattr(settings, key, value)


@pytest_asyncio.fixture
async def portfolio_db(tmp_path):
    from astra.db.portfolio import PortfolioDB
    db = PortfolioDB(tmp_path / "portfolio.db")
    await db.init()
    return db


@pytest_asyncio.fixture
async def memory_db(tmp_path):
    from astra.db.memory import MemoryDB
    db = MemoryDB(tmp_path / "memory.db")
    await db.init()
    return db


@pytest_asyncio.fixture
async def cache_db(tmp_path):
    from astra.db.cache import CacheDB
    db = CacheDB(tmp_path / "cache.db")
    await db.init()
    return db
