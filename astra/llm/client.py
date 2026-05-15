"""LM Studio OpenAI-compatible client.

We negotiate the strongest structured-output mode the server accepts:

  1. response_format = {"type": "json_schema", json_schema: {...}}   (strict)
  2. response_format = {"type": "json_object"}                        (loose)
  3. no response_format — rely on the model's natural JSON output    (Nemotron,
     Qwen2.5, Llama3.1 all handle this fine when asked clearly)

LM Studio responds 400 when a particular form isn't supported by the loaded
model's chat template. Each fallback is sticky for the rest of the process
so we don't pay the rejection round-trip more than once.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from astra.config import settings

log = logging.getLogger(__name__)


DECISION_SCHEMA: dict[str, Any] = {
    "name": "trading_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
            "size_pct": {"type": "number", "minimum": 0, "maximum": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reasoning": {"type": "string"},
        },
        "required": ["action", "size_pct", "confidence", "reasoning"],
        "additionalProperties": False,
    },
}

REFLECTION_SCHEMA: dict[str, Any] = {
    "name": "trading_lessons",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "lessons": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5,
            },
        },
        "required": ["lessons"],
        "additionalProperties": False,
    },
}


# Process-wide negotiation state, shared by all client instances.
class _State:
    schema_ok: bool = True
    object_ok: bool = True


class LMStudioClient:
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
        mode: str,
        schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        # In plain mode, nudge the model with an explicit "JSON only" directive.
        if mode == "plain":
            user = user + "\n\nReturn ONLY a valid JSON object. No prose, no markdown fences."
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if mode == "schema" and schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": schema}
        elif mode == "object":
            body["response_format"] = {"type": "json_object"}
        return body

    async def _post_chat(self, body: dict[str, Any]) -> httpx.Response:
        return await self.client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=body,
        )

    @staticmethod
    def _explain_400(resp: httpx.Response) -> str:
        try:
            j = resp.json()
            return json.dumps(j)[:500]
        except Exception:
            return resp.text[:500]

    async def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 600,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Build the list of modes to try, in order of strictness.
        order: list[str] = []
        if _State.schema_ok and schema is not None:
            order.append("schema")
        if _State.object_ok:
            order.append("object")
        order.append("plain")

        last_400: tuple[str, str] | None = None
        for mode in order:
            body = self._build_body(system, user, temperature, max_tokens, mode, schema)
            resp = await self._post_chat(body)
            if resp.status_code == 400:
                detail = self._explain_400(resp)
                last_400 = (mode, detail)
                # Disable this mode for the rest of the run.
                if mode == "schema":
                    _State.schema_ok = False
                    log.warning(
                        "LM Studio rejected json_schema mode: %s — trying json_object", detail
                    )
                elif mode == "object":
                    _State.object_ok = False
                    log.warning(
                        "LM Studio rejected json_object mode: %s — using plain mode", detail
                    )
                else:
                    log.error("LM Studio rejected even plain mode: %s", detail)
                    resp.raise_for_status()
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return self._parse_json(content)

        # Should be unreachable — plain mode either succeeded or raised above.
        raise RuntimeError(f"LM Studio request failed; last 400: {last_400}")

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
