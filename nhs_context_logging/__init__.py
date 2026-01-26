import logging
from collections.abc import Callable

from nhs_context_logging.constants import Constants
from nhs_context_logging.logger import (
    LogActionContextManager,
    TemporaryGlobalFieldsContextManager,
    logging_context,
)
from nhs_context_logging.logger import app_logger as _app_logger

__version__ = "0.0.0"

DEFAULT_LOG_LEVEL = Constants.DEFAULT_LOG_LEVEL
LOG_AT_LEVEL = Constants.LOG_AT_LEVEL
LOG_LEVEL = Constants.LOG_LEVEL

CRITICAL = _app_logger.CRITICAL
FATAL = _app_logger.FATAL
ERROR = _app_logger.ERROR
AUDIT = _app_logger.AUDIT
WARNING = _app_logger.WARNING
WARN = _app_logger.WARN
NOTICE = _app_logger.NOTICE
INFO = _app_logger.INFO
DEBUG = _app_logger.DEBUG
TRACE = _app_logger.TRACE


app_logger = _app_logger


def add_fields(**kwargs):
    """Add success fields to the current action

    Args:
        **kwargs: fields to be added
    """
    action = logging_context.current()
    if action:
        action.add_fields(**kwargs)
    else:
        raise ValueError("Add fields called with no current log_action")


def debug_fields(fun_fields: Callable[[], dict]):
    """Add success fields to the current action

    Args:
        fun_fields (Callable[dict]): factory to create fields on demand
    """
    action = logging_context.current()
    if action:
        if logging_context.log_at_level() <= logging.DEBUG:
            fields = fun_fields()
            action.add_fields(**fields)
    else:
        raise ValueError("Add fields called with no current log_action")


log_action = LogActionContextManager

temporary_global_fields = TemporaryGlobalFieldsContextManager
