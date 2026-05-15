"""Retrieve relevant lessons + pattern stats for LLM prompt."""

from __future__ import annotations

from typing import Any

from astra.db.memory import MemoryDB


class MemoryRetriever:
    def __init__(self, memory: MemoryDB) -> None:
        self.memory = memory

    async def for_decision(
        self,
        pattern_hash: str,
        max_lessons: int = 6,
        max_patterns: int = 5,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        lessons_rows = await self.memory.top_lessons(limit=max_lessons)
        lessons = [r["lesson"] for r in lessons_rows]
        patterns = await self.memory.relevant_patterns(pattern_hash, limit=max_patterns)
        return lessons, patterns
