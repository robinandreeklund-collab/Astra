"""Thin OpenAI-compatible client pointed at LM Studio.

We try `response_format: json_object` first (works on GPT/Qwen2.5/Llama3.1).
Many local models (Gemma, older Llamas, Mistral 7B) return HTTP 400 for that
parameter — so we transparently fall back to a plain chat completion and let
the parser extract the JSON object from the response. The fallback decision
is sticky per process so we don't keep paying the 400 round-trip cost.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from astra.config import settings

log = logging.getLogger(__name__)


class LMStudioClient:
    # Process-wide flag: once we've seen a 400 on json mode, stop using it.
    _supports_json_mode: bool = True

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

    def _build_body(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> dict[str, Any]:
        # When json mode is unavailable we add a strong nudge in the user msg.
        if not json_mode:
            user = user + "\n\nReturn ONLY a valid JSON object. No prose, no markdown."
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    async def _post_chat(self, body: dict[str, Any]) -> httpx.Response:
        return await self.client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=body,
        )

    async def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 600,
    ) -> dict[str, Any]:
        # First attempt: with json mode if we believe it works.
        use_json = LMStudioClient._supports_json_mode
        body = self._build_body(system, user, temperature, max_tokens, json_mode=use_json)
        resp = await self._post_chat(body)

        # If the server rejected json mode, retry once without it and remember.
        if resp.status_code == 400 and use_json:
            try:
                detail = resp.json()
            except Exception:
                detail = {"text": resp.text[:200]}
            log.warning(
                "LM Studio rejected response_format=json_object (%s) — "
                "falling back to plain mode for the rest of this run",
                detail,
            )
            LMStudioClient._supports_json_mode = False
            body = self._build_body(system, user, temperature, max_tokens, json_mode=False)
            resp = await self._post_chat(body)

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
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        return json.loads(text)
