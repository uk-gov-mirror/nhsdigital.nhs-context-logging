import json
import logging
import os
import re
from collections.abc import Generator, Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import date, datetime
from decimal import Decimal
from typing import (
    Any,
    cast,
)

from nhs_context_logging.constants import Constants
from nhs_context_logging.utils import get_error_info, is_dataclass_instance


def json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if is_dataclass_instance(obj):
        return asdict(obj)

    if isinstance(obj, type) or callable(obj):
        return repr(obj)

    if isinstance(obj, Decimal):
        return str(obj)

    if isinstance(obj, Mapping):
        return {json_serializer(k): json_serializer(v) for k, v in obj.items()}

    if isinstance(obj, str):
        return obj

    return str(repr(obj))


_PATH_RE = re.compile("^/.*/site-packages/")


class StructuredFormatter(logging.Formatter):
    """provides structured format for logging"""

    def format(self, record: logging.LogRecord) -> dict:  # type: ignore[override]  # noqa: C901
        log: dict[str, Any] = {}

        log.update(
            {
                Constants.TIMESTAMP_FIELD: datetime.fromtimestamp(record.created).timestamp(),
            }
        )

        log_args: dict[str, Any] = {}
        if record.args:
            args = record.args

            if isinstance(args, list) and callable(args[0]):
                args = args[0]()
            if not isinstance(args, dict):
                args = {"args": record.args}

            if args:
                rec_ts = cast(float, args.get(Constants.TIMESTAMP_FIELD))
                if rec_ts:
                    args[Constants.TIMESTAMP_FIELD] = datetime.fromtimestamp(rec_ts).timestamp()

                log_args.update(args)

        if Constants.EXPECTED_ERRORS in log_args:
            del log_args[Constants.EXPECTED_ERRORS]

        include_traceback = log_args.get(Constants.INCLUDE_TRACEBACK, True)
        if Constants.INCLUDE_TRACEBACK in log_args:
            del log_args[Constants.INCLUDE_TRACEBACK]

        if Constants.REDACT_FIELDS in log_args:
            del log_args[Constants.REDACT_FIELDS]

        for priority_field in (Constants.LOG_REFERENCE_FIELD, Constants.LOG_CORRELATION_ID_FIELD):
            field_value = log_args.pop(priority_field, None)
            if field_value:
                log[priority_field] = field_value

        log.update(log_args)

        # /opt/api/.venv/lib/python3.10/site-packages
        log_info = {
            "level": record.levelname,
            "path": _PATH_RE.sub("", record.pathname),
            "line_no": record.lineno,
            "func": record.funcName,
            "pid": record.process,
        }

        hostname = os.getenv("HOSTNAME")

        if hostname:
            log_info["hostname"] = hostname

        if record.thread:
            log_info["thread"] = record.thread

        if record.msg:
            log["message"] = record.getMessage()

        log[Constants.LOG_INFO] = log_info

        exc_info = record.__dict__.get(Constants.EXC_INFO, None)

        if exc_info:
            _, exc_val, _ = exc_info

            error_info = get_error_info(exc_val, include_tb=include_traceback)
            log[Constants.ERROR_INFO_FIELD] = error_info

            if record.stack_info:
                log_info[Constants.STACK_INFO] = record.stack_info

        return log


class JSONFormatter(StructuredFormatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        record_dict = super().format(record)

        return json.dumps(record_dict, default=json_serializer) + "\n"


class KeyValueFormatter(StructuredFormatter):
    def __init__(self, drop_fields: Sequence[str] | None = None, list_delimiter: str = ",", datefmt: str | None = None):
        super().__init__(datefmt=datefmt)
        self.drop_fields = drop_fields or []
        self.list_delimiter = list_delimiter

    @staticmethod
    def _format_value(value) -> str:
        if isinstance(value, (datetime, date)):
            return value.isoformat()

        if isinstance(value, type) or callable(value):
            return repr(value)

        if isinstance(value, str):
            return value

        if isinstance(value, Decimal):
            return str(value)

        return str(repr(value))

    def _flatten_fields(
        self, record_dict: Mapping[str, Any], parent_key: str | None = None
    ) -> Generator[tuple[str, str], None, None]:
        for key, value in record_dict.items():
            key = f"{parent_key}_{key}" if parent_key else key

            if key in self.drop_fields:
                continue

            if value is None:
                yield key, "null"
                continue

            if isinstance(value, Mapping):
                yield from self._flatten_fields(value, parent_key=key)
                continue

            if hasattr(value, "as_dict"):
                yield from self._flatten_fields(value.as_dict())
                continue

            if is_dataclass_instance(value):
                yield from self._flatten_fields(asdict(value), parent_key=key)
                continue

            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                yield key, self.list_delimiter.join(self._format_value(item) for item in value)
                continue

            yield key, self._format_value(value)

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        record_dict = super().format(record)
        timestamp = record_dict.pop("timestamp", record.created)
        _ = record_dict.pop("level", None)
        message = record_dict.pop("message", record.getMessage())
        # ?timestamp = record_dict.pop("timestamp")
        if self.datefmt:
            timestamp = datetime.fromtimestamp(timestamp).strftime(self.datefmt)

        parts = [str(timestamp), record.levelname]
        if message and message != "None":
            parts.append(message)

        parts.extend(f"{key}={value}" for key, value in self._flatten_fields(record_dict))

        return " ".join(parts) + "\n"
