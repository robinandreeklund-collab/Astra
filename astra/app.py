"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from astra.api.routes import build_router
from astra.config import settings
from astra.db.cache import CacheDB
from astra.db.memory import MemoryDB
from astra.db.portfolio import PortfolioDB
from astra.engine.runner import EngineState, TradingEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

PKG_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PKG_DIR / "api" / "templates"
STATIC_DIR = PKG_DIR / "static"


def create_app() -> FastAPI:
    settings.ensure_dirs()

    portfolio = PortfolioDB(settings.portfolio_db_path)
    memory = MemoryDB(settings.memory_db_path)
    cache = CacheDB(settings.cache_db_path)
    state = EngineState()
    engine = TradingEngine(portfolio, memory, cache, state)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await portfolio.init()
        await memory.init()
        await cache.init()
        # The engine starts PAUSED. It only auto-resumes if the user had
        # explicitly pressed Play before (the preference is persisted), so a
        # fresh install or a restart never starts trading on its own.
        pref = await memory.get_model("engine_pref")
        if pref and pref.get("running") and await portfolio.get_account():
            await engine.start()
        yield
        await engine.stop()

    app = FastAPI(title="Astra", lifespan=lifespan)
    app.state.portfolio = portfolio
    app.state.memory = memory
    app.state.cache = cache
    app.state.engine = engine
    app.state.engine_state = state
    app.state.training_status = {"running": False, "message": "", "result": None}
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    app.include_router(build_router(app))
    return app


app = create_app()
