"""Finnhub REST client with rate limiting + caching."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from astra.config import settings
from astra.db.cache import CacheDB

log = logging.getLogger(__name__)

BASE_URL = "https://finnhub.io/api/v1"


class TokenBucket:
    def __init__(self, rate_per_minute: int = 55) -> None:
        self.rate = rate_per_minute
        self.tokens = float(rate_per_minute)
        self.updated = time.monotonic()
        self.lock = asyncio.Lock()

    async def take(self) -> None:
        async with self.lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.updated
                self.tokens = min(self.rate, self.tokens + elapsed * (self.rate / 60.0))
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) / (self.rate / 60.0)
                await asyncio.sleep(wait)


class FinnhubClient:
    def __init__(self, api_key: str | None = None, cache: CacheDB | None = None) -> None:
        self.api_key = api_key or settings.finnhub_api_key
        self.cache = cache
        self.client = httpx.AsyncClient(timeout=15.0)
        self.bucket = TokenBucket(55)

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "FinnhubClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        cache_ttl: int | None = None,
    ) -> Any:
        params = dict(params or {})
        params["token"] = self.api_key
        cache_key = None
        if cache_ttl and self.cache is not None:
            cache_key = f"GET:{path}:{sorted([(k,v) for k,v in params.items() if k != 'token'])}"
            hit = await self.cache.get(cache_key)
            if hit is not None:
                return hit

        await self.bucket.take()
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = await self.client.get(f"{BASE_URL}{path}", params=params)
                if resp.status_code == 429:
                    await asyncio.sleep(2 + attempt * 2)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if cache_key and cache_ttl and self.cache is not None:
                    await self.cache.set(cache_key, data, cache_ttl)
                return data
            except (httpx.HTTPError, httpx.RequestError) as e:
                last_exc = e
                await asyncio.sleep(1 + attempt)
        raise RuntimeError(f"Finnhub GET {path} failed: {last_exc}")

    # ---- endpoints we use ----

    async def quote(self, symbol: str) -> dict[str, Any]:
        return await self._get("/quote", {"symbol": symbol}, cache_ttl=30)

    async def candles(
        self,
        symbol: str,
        resolution: str,
        from_ts: int,
        to_ts: int,
    ) -> dict[str, Any]:
        return await self._get(
            "/stock/candle",
            {"symbol": symbol, "resolution": resolution, "from": from_ts, "to": to_ts},
            cache_ttl=300,
        )

    async def recommendation(self, symbol: str) -> list[dict[str, Any]]:
        return await self._get(
            "/stock/recommendation", {"symbol": symbol}, cache_ttl=3600 * 6
        ) or []

    async def insider_transactions(self, symbol: str) -> dict[str, Any]:
        return await self._get(
            "/stock/insider-transactions", {"symbol": symbol}, cache_ttl=3600 * 12
        )

    async def company_news(
        self, symbol: str, from_date: str, to_date: str
    ) -> list[dict[str, Any]]:
        return await self._get(
            "/company-news",
            {"symbol": symbol, "from": from_date, "to": to_date},
            cache_ttl=900,
        ) or []

    async def news_sentiment(self, symbol: str) -> dict[str, Any]:
        return await self._get(
            "/news-sentiment", {"symbol": symbol}, cache_ttl=3600
        )

    async def earnings_calendar(
        self, from_date: str, to_date: str, symbol: str | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"from": from_date, "to": to_date}
        if symbol:
            params["symbol"] = symbol
        return await self._get("/calendar/earnings", params, cache_ttl=3600 * 6)

    async def earnings(self, symbol: str) -> list[dict[str, Any]]:
        return await self._get(
            "/stock/earnings", {"symbol": symbol}, cache_ttl=3600 * 6
        ) or []

    async def social_sentiment(self, symbol: str) -> dict[str, Any]:
        return await self._get(
            "/stock/social-sentiment", {"symbol": symbol}, cache_ttl=3600
        )

    async def index_constituents(self, symbol: str = "^GSPC") -> dict[str, Any]:
        return await self._get(
            "/index/constituents", {"symbol": symbol}, cache_ttl=3600 * 24
        )
