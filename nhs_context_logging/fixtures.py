import logging
from collections.abc import Iterable
from typing import Any

import pytest

from nhs_context_logging import app_logger
from nhs_context_logging.handlers import capturing_log_handlers

__all__ = ["log_capture_fixture", "log_capture_global_fixture"]


@pytest.fixture(scope="session", name="log_capture_global")
def log_capture_global_fixture() -> Iterable[tuple[list[dict], list[dict]]]:
    std_out: list[dict[str, Any]] = []
    std_err: list[dict[str, Any]] = []

    capturing_handlers = capturing_log_handlers(std_out, std_err)

    app_logger.setup("pytest")

    for handler in capturing_handlers:
        logging.root.addHandler(handler)

    yield std_out, std_err

    for handler in capturing_handlers:
        logging.root.removeHandler(handler)


@pytest.fixture(name="log_capture")
def log_capture_fixture(log_capture_global) -> Iterable[tuple[list[dict], list[dict]]]:
    std_out, std_err = log_capture_global

    std_out.clear()
    std_err.clear()

    log_at_level = app_logger.log_at_level

    app_logger.log_at_level = app_logger.DEBUG

    yield std_out, std_err

    app_logger.log_at_level = log_at_level
