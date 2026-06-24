import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.logging import RichHandler


def setup_logging() -> None:
    from app.utils.config import get_settings
    cfg = get_settings().logging

    app_level = getattr(logging, cfg.level.upper(), logging.INFO)
    third_party_level = getattr(logging, cfg.third_party_level.upper(), logging.WARNING)
    handler_level = min(app_level, third_party_level)  # handlers must pass the most verbose

    handlers: List[logging.Handler] = []

    if cfg.rich:
        console_handler: logging.Handler = RichHandler(
            console=Console(stderr=True),
            rich_tracebacks=True,
            show_path=True,
            markup=True,
        )
    else:
        console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(handler_level)
    handlers.append(console_handler)

    log_path = Path(cfg.log_dir) / cfg.file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=cfg.rotation_mb * 1024 * 1024,
        backupCount=cfg.backups,
        encoding="utf-8",
    )
    file_handler.setLevel(handler_level)
    handlers.append(file_handler)

    # Root level governs unconfigured (mostly third-party) loggers; our code logs at app_level.
    logging.basicConfig(level=third_party_level, handlers=handlers, force=True)
    logging.getLogger("app").setLevel(app_level)

    # Pin known-noisy libraries so they never flood, regardless of root level.
    for name in cfg.noisy_loggers:
        logging.getLogger(name).setLevel(third_party_level)


def log_llm_request(model: str, messages: List[Dict[str, str]], **params: Any) -> None:
    """One place every LLM call logs through: an INFO summary plus DEBUG detail."""
    logger = logging.getLogger("app.llm")
    logger.info("Sending to LLM (model=%s, messages=%d)", model, len(messages))
    if not logger.isEnabledFor(logging.DEBUG):
        return
    prompt = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    logger.debug("Prompt to LLM: %s", prompt)
    logger.debug("Messages to LLM: %s", messages)
    logger.debug("Params to LLM: %s", params)


def log_llm_response(raw: str) -> None:
    """Raw text the model returned, before any parsing."""
    logging.getLogger("app.llm").debug("Raw LLM response: %s", raw)


def log_llm_parsed(schema: str, parsed: Any) -> None:
    """The Pydantic-parsed result of a structured-output call."""
    logger = logging.getLogger("app.llm")
    logger.info("Parsing LLM response via Pydantic (%s)", schema)
    if not logger.isEnabledFor(logging.DEBUG):
        return
    detail = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
    logger.debug("Parsed output via Pydantic: %s", detail)
