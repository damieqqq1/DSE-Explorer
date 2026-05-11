"""Global logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import TextIO
from logging import Logger


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str = "INFO", stream: TextIO | None = None) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    stream = stream or sys.stdout

    if root.handlers:
        root.setLevel(numeric_level)
        for handler in root.handlers:
            handler.setLevel(numeric_level)
        return

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    handler.setLevel(numeric_level)

    root.addHandler(handler)
    root.setLevel(numeric_level)


def get_logger(name: str) -> Logger:
    return logging.getLogger(name)
