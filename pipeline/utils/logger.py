import logging
import sys
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_fmt = logging.Formatter(
    "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Shared error-only file handler (all loggers write ERRORs here)
_error_handler = logging.FileHandler(_LOG_DIR / "error.log", encoding="utf-8")
_error_handler.setLevel(logging.ERROR)
_error_handler.setFormatter(_fmt)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        # stdout — all levels
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(_fmt)
        logger.addHandler(stream)
        # error.log — errors only
        logger.addHandler(_error_handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger
