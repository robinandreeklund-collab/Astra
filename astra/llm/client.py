"""Thin OpenAI-compatible client pointed at LM Studio."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from astra.config import settings

log = logging.getLogger(__name__)


class LMStudioClient:
    """Direct httpx-based client. Avoids openai-sdk's heavy retry/streaming setup
    so we can run fully offline against LM Studio."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = (base_url or settings.lm_studio_url).rstrip("/")
        self.api_key = api_key or settings.lm_studio_api_key
        self.model = model or settings.lm_studio_model
        self.client = httpx.AsyncClient(timeout=120.0)

    async def close(self) -> None:
        await self.client.aclose()

    async def health(self) -> bool:
        try:
            resp = await self.client.get(f"{self.base_url}/models", headers=self._headers())
            return resp.status_code == 200
        except Exception:
            return False

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 600,
    ) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        resp = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return self._parse_json(content)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        # Strip leading prose if any
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        return json.loads(text)
