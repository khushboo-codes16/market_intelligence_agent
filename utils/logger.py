# utils/logger.py
# ============================================================
# Centralised logging using loguru
# ============================================================

import sys
from pathlib import Path
from loguru import logger
from config.settings import LOG_LEVEL, LOG_FILE

# Remove default handler
logger.remove()

# Console handler — pretty for development
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
    colorize=True,
)

# File handler — full details for debugging
logger.add(
    LOG_FILE,
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
)

__all__ = ["logger"]
