import logging
import os
import re
import traceback
from collections.abc import Callable
from dataclasses import is_dataclass
from types import FrameType
from typing import Any

from nhs_context_logging.constants import Constants


def is_dataclass_instance(obj) -> bool:
    return is_dataclass(obj) and not isinstance(obj, type)


def find_caller_info(caller_of: Callable | None = None) -> tuple[str, int, str, FrameType | None]:
    """cloned from python logger to allow caller info from app logger context"""

    try:
        return _find_caller_info(caller_of)
    except ValueError:  # pragma: no cover
        return "(unknown file)", 0, "(unknown function)", None


def _find_caller_info(caller_of: Callable | None = None) -> tuple[str, int, str, FrameType | None]:
    """cloned from python logger to allow caller info from app logger context"""
    frame = logging.currentframe()

    callee_func = caller_of.__code__.co_name if caller_of else None
    callee_path = os.path.normcase(caller_of.__code__.co_filename) if caller_of else None

    if frame is not None:
        frame = frame.f_back  # type: ignore[assignment]

    while hasattr(frame, "f_code"):
        fco = frame.f_code
        filename = os.path.normcase(fco.co_filename)
        func_name = fco.co_name

        if callee_path and filename == callee_path:
            frame = frame.f_back  # type: ignore[assignment]
            if callee_func and callee_func == func_name:
                callee_path = None
            continue

        if func_name == "<listcomp>":
            frame = frame.f_back  # type: ignore[assignment]
            continue

        if not callee_path and is_logger_module(filename):
            frame = frame.f_back  # type: ignore[assignment]
            continue

        return fco.co_filename, frame.f_lineno, fco.co_name, frame

    return "(unknown file)", 0, "(unknown function)", None


_THIS_MODULE = os.path.dirname(os.path.normcase(find_caller_info.__code__.co_filename))


def is_logger_module(filename: str) -> bool:
    return filename.startswith(_THIS_MODULE) and not filename.endswith("logger_tests.py")


def find_tb_source_frame(ex_tb):
    if not ex_tb:
        return None
    while ex_tb.tb_next:
        ex_tb = ex_tb.tb_next

    frame = ex_tb.tb_frame

    while hasattr(frame, "f_code"):
        fco = frame.f_code
        filename = os.path.normcase(fco.co_filename)
        func_name = fco.co_name

        if func_name == "<listcomp>":
            frame = frame.f_back  # type: ignore[assignment]
            continue

        if is_logger_module(filename):
            frame = frame.f_back  # type: ignore[assignment]
            continue
        break

    return frame


_PATH_RE = re.compile("^/.*/site-packages/")


def get_error_info(exc_val: BaseException, include_tb: bool = True) -> dict[str, Any]:
    exc_type = type(exc_val)
    error_info: dict[str, Any] = {
        Constants.ERROR_FIELD: repr(exc_val),
        Constants.ERROR_TYPE: exc_type.__name__,
        Constants.ERROR_FULLY_QUALIFIED_TYPE: f"{exc_type.__module__}.{exc_type.__name__}",
        Constants.ERROR_PROPERTIES_FIELD: exc_val.__dict__,
    }

    if exc_val.args:
        error_info[Constants.ERROR_ARGS] = exc_val.args

    exc_tb = exc_val.__traceback__
    frame = find_tb_source_frame(exc_val.__traceback__)

    if frame:
        if hasattr(frame, "f_code"):
            f_code = frame.f_code
            error_info["func"] = f_code.co_name
            error_info["path"] = _PATH_RE.sub("", f_code.co_filename)
        error_info["line_no"] = frame.f_lineno

    if include_tb:
        error_info["traceback"] = "\n".join(line.strip() for line in traceback.format_tb(exc_tb)).strip()

    if exc_val.__cause__:
        error_info["cause"] = get_error_info(exc_val.__cause__)

    return error_info
