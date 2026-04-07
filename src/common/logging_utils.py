from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure a consistent root logger for all scripts."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
