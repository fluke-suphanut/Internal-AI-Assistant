from __future__ import annotations

import logging
import sys
from typing import Optional

from app.core.config import get_settings


def setup_logging(log_level: Optional[str] = None) -> logging.Logger:
    """
    Configure app-wide logging.
    - Uses stdout handler (docker-friendly)
    - Avoids duplicate handlers on reload
    """
    settings = get_settings()
    level_name = (log_level or settings.LOG_LEVEL).upper()

    logger = logging.getLogger()  # root logger
    logger.setLevel(getattr(logging, level_name, logging.INFO))

    # Prevent duplicated handlers
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level_name, logging.INFO))

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Make noisy libs quieter if needed
    logging.getLogger("uvicorn").setLevel(getattr(logging, level_name, logging.INFO))
    logging.getLogger("uvicorn.error").setLevel(getattr(logging, level_name, logging.INFO))
    logging.getLogger("uvicorn.access").setLevel(getattr(logging, level_name, logging.INFO))

    return logging.getLogger(__name__)
