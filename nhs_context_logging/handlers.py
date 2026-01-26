import logging
import sys

from nhs_context_logging.formatters import StructuredFormatter

_filter_are_errors = staticmethod(lambda r: bool(r.levelno >= logging.ERROR))
_filter_not_errors = staticmethod(lambda r: bool(r.levelno < logging.ERROR))


class StructuredCapturingHandler(logging.Handler):
    """log emitter"""

    def __init__(self, messages: list[dict], level=logging.NOTSET):
        super().__init__(level)
        self.messages = messages
        self._formatter = StructuredFormatter()
        self.formatter = self._formatter

    def emit(self, record: logging.LogRecord):
        log = self._formatter.format(record)
        self.messages.append(log)


def capturing_log_handlers(stdout_cap: list[dict], stderr_cap: list[dict]):
    stdout_handler = StructuredCapturingHandler(stdout_cap)
    stdout_handler.addFilter(type("", (logging.Filter,), {"filter": _filter_not_errors}))

    stderr_handler = StructuredCapturingHandler(stderr_cap)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.addFilter(type("", (logging.Filter,), {"filter": _filter_are_errors}))

    return [stdout_handler, stderr_handler]


def sys_std_handlers(formatter: StructuredFormatter):
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(type("", (logging.Filter,), {"filter": _filter_not_errors}))

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)
    stderr_handler.addFilter(type("", (logging.Filter,), {"filter": _filter_are_errors}))

    return [stdout_handler, stderr_handler]
