"""CLI entrypoint: `python -m astra`."""

from __future__ import annotations

import sys

import uvicorn

from astra.config import settings


def main() -> None:
    uvicorn.run(
        "astra.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
