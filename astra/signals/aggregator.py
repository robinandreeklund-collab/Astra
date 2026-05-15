"""Combine all signal sources into a single bundle per symbol."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any

from astra.data.finnhub import FinnhubClient
from astra.data.yahoo import fetch_daily_candles
from astra.signals import fundamentals, insider, news, technical

log = logging.getLogger(__name__)


@dataclass
class SignalBundle:
    symbol: str
    fetched_at: float
    quote: dict[str, Any] = field(default_factory=dict)
    technical: dict[str, Any] = field(default_factory=dict)
    news: dict[str, Any] = field(default_factory=dict)
    sentiment: dict[str, Any] = field(default_factory=dict)
    insider: dict[str, Any] = field(default_factory=dict)
    recommendations: dict[str, Any] = field(default_factory=dict)
    earnings: dict[str, Any] = field(default_factory=dict)
    earnings_calendar: dict[str, Any] = field(default_factory=dict)
    social: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def pattern_hash(self) -> tuple[str, str]:
        """Bucketed feature fingerprint + human description."""
        t = self.technical
        rsi_v = t.get("rsi14")
        if rsi_v is None:
            rsi_b = "na"
        elif rsi_v < 30:
            rsi_b = "lt30"
        elif rsi_v < 50:
            rsi_b = "30-50"
        elif rsi_v < 70:
            rsi_b = "50-70"
        else:
            rsi_b = "gt70"
        macd_b = t.get("macd_state") or "na"
        bb_p = t.get("bb_position")
        if bb_p is None:
            bb_b = "na"
        elif bb_p < 0.2:
            bb_b = "lower"
        elif bb_p < 0.8:
            bb_b = "mid"
        else:
            bb_b = "upper"
        sma_b = t.get("sma_cross") or "na"

        ins = self.insider.get("net_direction") or "na"

        s = self.sentiment
        sent_b = "na"
        cns = s.get("company_news_score")
        if isinstance(cns, (int, float)):
            sent_b = "pos" if cns > 0.55 else "neg" if cns < 0.45 else "neu"

        recs = self.recommendations.get("trend") or "na"

        key = f"RSI:{rsi_b}|MACD:{macd_b}|BB:{bb_b}|SMA:{sma_b}|INS:{ins}|SENT:{sent_b}|REC:{recs}"
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        return h, key


class SignalAggregator:
    def __init__(self, client: FinnhubClient, cache=None) -> None:
        self.client = client
        self.cache = cache if cache is not None else getattr(client, "cache", None)

    async def fetch(self, symbol: str) -> SignalBundle:
        b = SignalBundle(symbol=symbol, fetched_at=time.time())

        async def _quote() -> None:
            try:
                b.quote = await self.client.quote(symbol)
            except PermissionError:
                pass  # known disabled endpoint
            except Exception as e:
                b.errors.append(f"quote:{e}")

        async def _tech() -> None:
            # Primary source: yfinance with disk cache. Finnhub free tier
            # no longer includes /stock/candle.
            try:
                rows = await fetch_daily_candles(symbol, days=180, cache=self.cache)
                if rows:
                    b.technical = technical.compute_indicators(rows)
                else:
                    b.technical = {"available": False, "reason": "no_candles"}
            except Exception as e:
                b.errors.append(f"candles:{e}")
                b.technical = {"available": False, "reason": str(e)}

        async def _news() -> None:
            try:
                fr, to = news.date_range(5)
                items = await self.client.company_news(symbol, fr, to)
                b.news = news.summarize_headlines(items)
            except PermissionError:
                pass
            except Exception as e:
                b.errors.append(f"news:{e}")

        async def _sent() -> None:
            try:
                s = await self.client.news_sentiment(symbol)
                b.sentiment = news.summarize_sentiment(s)
            except PermissionError:
                pass
            except Exception as e:
                b.errors.append(f"sentiment:{e}")

        async def _ins() -> None:
            try:
                d = await self.client.insider_transactions(symbol)
                b.insider = insider.summarize_insider(d)
            except PermissionError:
                pass
            except Exception as e:
                b.errors.append(f"insider:{e}")

        async def _recs() -> None:
            try:
                r = await self.client.recommendation(symbol)
                b.recommendations = fundamentals.summarize_recommendations(r)
            except PermissionError:
                pass
            except Exception as e:
                b.errors.append(f"recs:{e}")

        async def _earn() -> None:
            try:
                e = await self.client.earnings(symbol)
                b.earnings = fundamentals.summarize_earnings(e)
            except PermissionError:
                pass
            except Exception as e:
                b.errors.append(f"earnings:{e}")

        async def _cal() -> None:
            try:
                today = date.today()
                fr = today.isoformat()
                to = (today + timedelta(days=21)).isoformat()
                cal = await self.client.earnings_calendar(fr, to, symbol)
                b.earnings_calendar = fundamentals.summarize_calendar(cal, symbol)
            except PermissionError:
                pass
            except Exception as e:
                b.errors.append(f"calendar:{e}")

        async def _social() -> None:
            try:
                s = await self.client.social_sentiment(symbol)
                if s:
                    reddit = s.get("reddit") or []
                    twitter = s.get("twitter") or []
                    def avg(items: list[dict[str, Any]], key: str) -> float | None:
                        vals = [float(i[key]) for i in items if i.get(key) is not None]
                        return sum(vals) / len(vals) if vals else None
                    b.social = {
                        "available": True,
                        "reddit_score_avg": avg(reddit, "score"),
                        "twitter_score_avg": avg(twitter, "score"),
                        "reddit_mentions": sum(int(i.get("mention", 0) or 0) for i in reddit),
                        "twitter_mentions": sum(int(i.get("mention", 0) or 0) for i in twitter),
                    }
                else:
                    b.social = {"available": False}
            except PermissionError:
                pass
            except Exception as e:
                b.errors.append(f"social:{e}")

        await asyncio.gather(
            _quote(), _tech(), _news(), _sent(), _ins(), _recs(), _earn(), _cal(), _social(),
            return_exceptions=False,
        )
        return b
