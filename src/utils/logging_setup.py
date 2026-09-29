"""Shared logging setup for the CLI and HTTP API — console plus a rotating
file, so logs survive after the terminal/process closes."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from src.config import LOG_PATH

_configured = False


def configure_logging(level: int = logging.INFO, log_path: Optional[Path] = None, force: bool = False) -> None:
    """Idempotent by default (safe to call from multiple entrypoints/imports).
    ``force=True`` reconfigures even if already set up — used by tests."""
    global _configured
    if _configured and not force:
        return
    _configured = True

    path = Path(log_path) if log_path else Path(LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(threadName)-14s | %(name)-28s | %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=10_000_000, backupCount=5, encoding="utf-8",
    )
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    if force:
        for h in list(root.handlers):
            root.removeHandler(h)
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
