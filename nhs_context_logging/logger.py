import asyncio
import inspect
import io
import logging
import os
import sys
import threading
import traceback
import typing
from asyncio import Task
from concurrent.futures import ThreadPoolExecutor, _base, thread
from concurrent.futures.thread import BrokenThreadPool
from dataclasses import asdict, dataclass
from functools import partial, wraps
from logging.handlers import RotatingFileHandler
from time import time
from types import FrameType, TracebackType
from typing import (
    Any,
    AnyStr,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from uuid import uuid4
from weakref import WeakKeyDictionary

from nhs_context_logging.constants import Constants
from nhs_context_logging.formatters import JSONFormatter, KeyValueFormatter
from nhs_context_logging.handlers import sys_std_handlers
from nhs_context_logging.utils import find_caller_info, is_dataclass_instance

_DEFAULT_REDACTIONS = {"password", ":password", "nhsnumber", "nhs_number", "authorization", "authorisation"}


def uuid4_hex_string() -> str:
    return uuid4().hex


class LoggerError(Exception):
    pass


class ActionNotInStack(LoggerError):
    pass


class MismatchedActionError(LoggerError):
    pass


@dataclass
class LogConfig:
    prepend_module_name: bool = False


class _Logger:
    CRITICAL = logging.CRITICAL
    FATAL = logging.CRITICAL
    ERROR = logging.ERROR
    AUDIT = 35
    WARNING = logging.WARN
    WARN = WARNING
    NOTICE = 25
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    TRACE = 5

    def __init__(self):
        logging.addLevelName(self.TRACE, "TRACE")
        logging.addLevelName(self.NOTICE, "NOTICE")
        logging.addLevelName(self.AUDIT, "AUDIT")
        logging.addLevelName(self.FATAL, "FATAL")

        formatter = (
            KeyValueFormatter(
                drop_fields=os.environ.get("LOG_KV_DROP_FIELDS", "").split(","),
                datefmt=os.environ.get("LOG_KV_DATE_FMT"),
            )
            if os.environ.get("LOG_FORMATTER", "json").lower() == "keyvalue"
            else JSONFormatter()
        )

        for handler in logging.root.handlers:
            handler.formatter = formatter

        self.formatter = formatter
        self.service_name = "NOT SET"
        self._logger: Optional[logging.Logger] = None
        self.log_at_level = logging.getLevelName(os.environ.get("LOG_AT_LEVEL", Constants.DEFAULT_LOG_AT_LEVEL))
        self.config = LogConfig()
        self._is_setup = False

    def setup_file_log(
        self,
        service_name: str,
        log_dir: str = "/var/log",
        **kwargs,
    ):
        log_file = os.path.join(log_dir, f"{service_name}.log")
        self.setup(service_name, handlers=[RotatingFileHandler(filename=log_file)], **kwargs)

    def setup(  # noqa: C901
        self,
        service_name: str,
        handlers: Optional[List[logging.Handler]] = None,
        append: bool = False,
        overwrite: bool = False,
        is_async: bool = False,
        redact_fields: Optional[Set[str]] = None,
        internal_id_factory: Callable[[], str] = uuid4_hex_string,
        config_kwargs: Optional[dict] = None,
        on_init: Optional[Callable[[], None]] = None,
        force_reinit: Optional[bool] = None,
        **kwargs,
    ):
        """
            Set up the logger with a bunch of handlers. Is a no-op if the setup has already been performed.
        Args:
            service_name: name of the logger
            handlers: override list of handlers
            append: append to existing log handlers ( if handlers are supplied )
            overwrite: overwrite log handlers ( if handlers are supplied )
            is_async: configure async log context storage
            redact_fields: override set of field names to override
            internal_id_factory: optional callable to configure the internal_id factory
            config_kwargs: optional configuration parameters, as described in LogConfig
            on_init: init function called as context is initialised, defaults to setup_logging_tpe for async
            force_reinit: force re-setup of the logger ( ignore _is_setup )
            **kwargs: other args added as global log items
        Returns:

        """
        if self._is_setup and not force_reinit:
            return

        logging_context._needs_init = True
        logging_context._on_init = on_init

        if is_async:
            logging_context.setup_async()
            if not on_init:
                logging_context._on_init = setup_logging_tpe

        if redact_fields is None:
            redact_fields = _DEFAULT_REDACTIONS

        if kwargs:
            redact = kwargs.pop(Constants.REDACT_FIELDS, [])
            assert isinstance(redact, (tuple, set, list))
            redact_fields = redact_fields | set(redact)

        app_globals = {Constants.REDACT_FIELDS: redact_fields}
        if kwargs:
            app_globals.update(kwargs)

        logging_context.add_app_globals(**app_globals)
        LogActionContextManager.internal_id_factory = internal_id_factory

        if not logging.root.handlers:
            handlers = handlers or sys_std_handlers(self.formatter)

        if handlers:
            if append:
                logging.root.handlers += handlers
            elif overwrite or not logging.root.handlers:
                logging.root.handlers = handlers

        for handler in logging.root.handlers:
            if handler.formatter is not None:
                continue
            handler.formatter = self.formatter

        for logger_name in logging.root.manager.loggerDict:
            override_logger = logging.getLogger(logger_name)
            for handler in override_logger.handlers:
                handler.setFormatter(self.formatter)

        if config_kwargs is None:
            config_kwargs = {}
        self.config = LogConfig(**config_kwargs)

        self._logger = None
        self.service_name = service_name
        self._is_setup = True

    @staticmethod
    def add_app_globals(**kwargs):
        logging_context.add_app_globals(**kwargs)

    def logger(self) -> logging.Logger:
        if not self._logger:
            self._logger = logging.getLogger(self.service_name)
            self._logger.setLevel(self.log_at_level)

        return self._logger

    def log(  # noqa: C901
        self,
        log_level: int = logging.INFO,
        exc_info=None,
        args: Optional[Union[str, Mapping[str, Any], Any]] = None,
        caller_info=None,
        add_context_fields=True,
        **kwargs,
    ):
        if log_level < self.log_at_level:
            return
        include_stack_info = log_level >= logging.ERROR
        pathname, line_no, func, frame = caller_info if caller_info else find_caller_info()
        stack_info: Optional[str] = None
        if include_stack_info and frame:
            with io.StringIO() as sio:
                sio.write("Stack (most recent call last):\n")
                traceback.print_stack(frame, file=sio)
                stack_info = sio.getvalue()
                if stack_info[-1] == "\n":
                    stack_info = stack_info[:-1]

        msg = None
        if isinstance(args, str):
            msg = args
            args = {"message": args}

        if kwargs:
            args = args or {}
            if isinstance(args, dict):
                kwargs.update(args)
                args = kwargs

        current_args = args
        global_fields = None
        if add_context_fields:
            global_fields = logging_context.get_context_fields()

        def resolve_args():
            new_args = current_args
            if callable(current_args):
                new_args = current_args()

            if isinstance(new_args, dict):
                kwargs.update(new_args)
                new_args = kwargs
                if global_fields:
                    global_fields.update(new_args)
                    new_args = global_fields

            return self.safe_args(new_args)

        args = resolve_args

        log_record = logging.LogRecord(
            msg=msg,
            name=self.service_name,
            level=log_level,
            pathname=pathname,
            lineno=line_no,
            args=[args],  # type: ignore[arg-type]
            exc_info=exc_info,
            func=func,
            sinfo=stack_info,
        )

        self.logger().handle(log_record)

    def trace(self, args: Optional[Union[str, dict, Callable[[], dict]]] = None, **kwargs):
        self.log(log_level=5, args=args, **kwargs)

    def debug(self, args: Optional[Union[str, dict, Callable[[], dict]]] = None, **kwargs):
        self.log(log_level=logging.DEBUG, args=args, **kwargs)

    def info(self, args: Optional[Union[str, dict, Callable[[], dict]]] = None, **kwargs):
        self.log(log_level=logging.INFO, args=args, **kwargs)

    def notice(self, args: Optional[Union[str, dict, Callable[[], dict]]] = None, **kwargs):
        self.log(log_level=25, args=args, **kwargs)

    def audit(self, args: Optional[Union[str, dict, Callable[[], dict]]] = None, **kwargs):
        self.log(log_level=logging.WARN, args=args, **kwargs)

    def warning(self, args: Optional[Union[str, dict, Callable[[], dict]]] = None, **kwargs):
        self.warn(args=args, **kwargs)

    def warn(self, args: Optional[Union[str, dict, Callable[[], dict]]] = None, **kwargs):
        self.log(log_level=logging.WARN, args=args, **kwargs)

    def error(self, args: Optional[Union[str, dict, Callable[[], dict]]] = None, **kwargs):
        self.log(log_level=logging.ERROR, args=args, **kwargs)

    def fatal(self, args: Optional[Union[str, dict, Callable[[], dict]]] = None, **kwargs):
        self.log(log_level=logging.FATAL, args=args, **kwargs)

    def critical(self, args: Optional[Union[str, dict, Callable[[], dict]]] = None, **kwargs):
        self.log(log_level=logging.CRITICAL, args=args, **kwargs)

    def exception(
        self,
        args: Optional[Union[str, dict, Callable[[], dict]]] = None,
        exc_info: Optional[
            Union[
                Tuple[Type[Any], Exception, object],
                Tuple[Type[BaseException], BaseException, TracebackType],
                Tuple[None, None, None],
            ]
        ] = None,
        log_level: Optional[int] = None,
        **kwargs,
    ):
        exc_info = exc_info or sys.exc_info()

        log_level = log_level or logging.ERROR

        self.log(log_level=log_level, exc_info=exc_info, args=args, **kwargs)

    @staticmethod
    def safe_arg(value: Any, field: Optional[str], redact_fields: Set[str]):
        if not value or isinstance(value, type):
            return value

        if callable(field):
            field = repr(field)

        if field and field.lower() in redact_fields:
            return "--REDACTED--"

        if hasattr(value, "as_dict"):
            return value.as_dict()

        if is_dataclass_instance(value):
            return asdict(value)

        if isinstance(value, (tuple, list, set)):
            return [_Logger.safe_arg(val, None, redact_fields) for val in value]

        if isinstance(value, Mapping):
            return {k: _Logger.safe_arg(v, k, redact_fields) for k, v in value.items()}

        return value

    @staticmethod
    def safe_args(args: dict):
        if not args:
            return args

        if isinstance(args, (list, tuple, set)):
            return [_Logger.safe_arg(v, None, set()) for v in args]

        if not hasattr(args, "items"):
            return args

        dont_redact = args.pop(Constants.DONT_REDACT, set())
        redact_fields = args.pop(Constants.REDACT_FIELDS, set())
        redact_fields = redact_fields - dont_redact

        _ = args.pop(Constants.EXPECTED_ERRORS, [])
        return {k: _Logger.safe_arg(v, k, redact_fields) for k, v in args.items()}


app_logger = _Logger()

CallerInfo = Tuple[str, int, str, Optional[FrameType]]


def get_args_map(func, *args, **kwargs) -> Dict[str, Any]:
    """Get a map of arguments to their argument name as defined by the function."""
    args_map: Dict[str, Any] = {}
    if not args and not kwargs:
        return args_map

    try:
        sig = inspect.Signature.from_callable(func)
        bound_args = typing.cast(inspect.BoundArguments, sig.bind(*args, **kwargs))
        args_map.update(bound_args.arguments)
    except TypeError:
        pass  # if the caller hasn't supplied the right arguments allow this to continue,
        # so the exception will come from the target function

    return args_map


def get_method_name(func, *args, **kwargs):
    args_map = get_args_map(func, *args, **kwargs)
    if "self" in args_map:
        cls = args_map["self"].__class__
        method_name = f"{cls.__name__}.{func.__name__}"
    elif "cls" in args_map:
        cls = args_map["cls"]
        method_name = f"{cls.__name__}.{func.__name__}"
    else:
        method_name = func.__name__

    if app_logger.config.prepend_module_name:
        method_name = f"{func.__module__}.{method_name}"
    return method_name


def get_args(arg_list, func, *args, **kwargs):
    """Returns a dictionary of the specified arguments keyed against their argument name."""
    args_map = get_args_map(func, *args, **kwargs)
    if not arg_list or not args_map:
        return {}
    specific_args = {k: v for k, v in args_map.items() if k in arg_list}
    return specific_args


class TemporaryGlobalFieldsContextManager(threading.local):
    def __init__(self, add_to_context: bool = False, **fields):
        self._add_to_context = add_to_context
        self._globals = _Globals(fields)

    def __enter__(self):
        if self._add_to_context:
            context = logging_context.current()
            if context:
                context.add_fields(**self._globals.globals)
        logging_context.add_temporary_globals(self._globals)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging_context.pop_temporary_globals(self._globals)

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__exit__(exc_type, exc_val, exc_tb)


FuncT = TypeVar("FuncT", bound=Callable[..., Any])


class LogActionContextManager(threading.local):
    internal_id_factory = uuid4_hex_string

    def __init__(
        self,
        action: Optional[str] = None,
        log_reference: Optional[str] = None,
        log_level: Optional[Union[str, int]] = None,
        log_args: Optional[List[str]] = None,
        log_result: Optional[bool] = False,
        forced_log_level: bool = False,
        caller_info: Optional[CallerInfo] = None,
        **fields,
    ):
        self.log_args = log_args
        self.log_result = log_result
        self.action_fields: Dict[str, Any] = {"log_reference": log_reference}
        if log_level:
            self.action_fields[Constants.LOG_LEVEL] = log_level
        if action:
            self.action_fields[Constants.ACTION_FIELD] = action
        if fields:
            self.action_fields.update(fields)

        self.fields: Dict[str, Any] = {}  # this for the fields used within the action (initialised in start_action)
        self.forced_log_level = forced_log_level
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._caller_info = caller_info

    def _recreate_cm(self, func, wrapper, *args, **inner_kwargs):
        caller_inf, args_to_log = self._get_log_args(func, wrapper, self._caller_info, *args, **inner_kwargs)
        new_context = LogActionContextManager(
            forced_log_level=self.forced_log_level,
            log_result=self.log_result,
            caller_info=caller_inf,
        )
        if args_to_log:
            new_context.action_fields.update(args_to_log)

        return new_context

    def _get_log_args(
        self, func, wrapper, caller_inf: Optional[CallerInfo], *args, **inner_kwargs
    ) -> Tuple[CallerInfo, Dict[str, Any]]:
        caller_inf = caller_inf or find_caller_info(wrapper)

        args_to_log = get_args(self.log_args, func, *args, **inner_kwargs) if self.log_args else {}

        self.action_fields[Constants.ACTION_FIELD] = self.action_fields.get(
            Constants.ACTION_FIELD, get_method_name(func, *args, **inner_kwargs)
        )

        if self.action_fields:
            args_to_log.update(self.action_fields)

        return caller_inf, args_to_log

    def __call__(self, func: FuncT) -> FuncT:
        if inspect.isasyncgenfunction(func):

            @wraps(func)
            async def _async_gen_inner(*args, **inner_kwargs):
                async with self._recreate_cm(func, _async_gen_inner, *args, **inner_kwargs):
                    async for res in func(*args, **inner_kwargs):
                        yield res

            return typing.cast(FuncT, _async_gen_inner)

        if inspect.isgeneratorfunction(func):

            @wraps(func)
            def _sync_gen_inner(*args, **inner_kwargs):
                with self._recreate_cm(func, _sync_gen_inner, *args, **inner_kwargs):
                    yield from func(*args, **inner_kwargs)

            return typing.cast(FuncT, _sync_gen_inner)

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def _async_inner(*args, **inner_kwargs):
                async with self._recreate_cm(func, _async_inner, *args, **inner_kwargs) as cm:
                    return await cm.async_handle_result(func, *args, **inner_kwargs)

            return typing.cast(FuncT, _async_inner)

        @wraps(func)
        def _sync_inner(*args, **inner_kwargs):
            with self._recreate_cm(func, _sync_inner, *args, **inner_kwargs) as cm:
                return cm.handle_result(func, *args, **inner_kwargs)

        return typing.cast(FuncT, _sync_inner)

    def __enter__(self):
        self._start_action(self.__enter__)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._end_action(exc_type, exc_val, exc_tb)

    async def __aenter__(self):
        self._start_action(self.__aenter__)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self._end_action(exc_type, exc_val, exc_tb)

    @property
    def action_type(self):
        return self.fields.get(Constants.ACTION_FIELD, "not_set")

    @property
    def log_level(self):
        return self.fields.get(Constants.LOG_LEVEL, Constants.DEFAULT_LOG_LEVEL)

    @property
    def internal_id(self) -> str:
        return str(self.fields[Constants.LOG_CORRELATION_ID_FIELD])

    @property
    def log_reference(self):
        return self.fields.get(Constants.LOG_REFERENCE_FIELD)

    async def async_handle_result(self, func, *args, **inner_kwargs):
        res = await func(*args, **inner_kwargs)
        if self.log_result:
            self.fields[Constants.ACTION_RESULT] = res
        return res

    def handle_result(self, func, *args, **inner_kwargs):
        res = func(*args, **inner_kwargs)
        if self.log_result:
            self.fields[Constants.ACTION_RESULT] = res
        return res

    def add_fields(self, **fields):
        if not fields:
            return

        self.fields.update(fields)

    def _start_action(self, caller: Callable):
        self._caller_info = self._caller_info or find_caller_info(caller)
        self.fields = logging_context.current_global_fields  # this is a copy
        global_expected_errs = self.fields.get(Constants.EXPECTED_ERRORS, ())
        self.fields.update(self.action_fields)
        action_expected_errs = self.action_fields.get(Constants.EXPECTED_ERRORS, ())
        self.fields[Constants.EXPECTED_ERRORS] = global_expected_errs + action_expected_errs
        requested_log_level = self.fields.get(Constants.LOG_AT_LEVEL)
        if requested_log_level is not None:
            del self.fields[Constants.LOG_AT_LEVEL]
            logging_context.force_log_at_level(requested_log_level)

        self.fields[Constants.LOG_LEVEL] = self.fields.get(Constants.LOG_LEVEL, Constants.DEFAULT_LOG_LEVEL)

        if not self.fields.get(Constants.LOG_CORRELATION_ID_FIELD):
            internal_id = logging_context.current_internal_id() or LogActionContextManager.internal_id_factory()
            self.fields[Constants.LOG_CORRELATION_ID_FIELD] = internal_id

        self.start_time = time()

        logging_context.push(self)

    def _end_action(self, exc_type, exc_val, exc_tb):
        message = {}
        message.update(self.fields)

        try:
            popped_action = logging_context.pop(self)
            if popped_action != self:
                raise MismatchedActionError("Mismatch action popped from stack!")
        except ActionNotInStack as err:
            message[Constants.LOG_LEVEL] = logging.WARNING
            message[Constants.LOGGER_ERROR] = err

        self.end_time = time()

        if self.end_time is not None and self.start_time is not None:
            message[Constants.ACTION_DURATION] = message.get(
                Constants.ACTION_DURATION, float(f"{(self.end_time - self.start_time):0.7f}")
            )

        if exc_val and not isinstance(exc_val, GeneratorExit):
            self._add_error_fields(message, exc_type, exc_val, exc_tb)
        else:
            message[Constants.ACTION_STATUS] = Constants.STATUS_SUCCEEDED
        message.pop(Constants.LOG_REFERENCE_ON_ERR_FIELD, None)
        logging_context.emit_message(message)

        return False

    @staticmethod
    def _get_expected_error_level(error_levels: typing.Sequence[Tuple[type, int]], exc_type) -> int:
        if not error_levels:
            return logging.INFO

        for exp_type, level in error_levels:
            if exp_type == exc_type:
                return LogActionContextManager._ensure_int_log_level(level)

        return logging.INFO

    @staticmethod
    def _ensure_int_log_level(log_level: Union[str, int]) -> int:
        if isinstance(log_level, int):
            return log_level

        level_int = logging.getLevelName(log_level)
        if isinstance(level_int, int):
            return level_int

        raise ValueError(f"log level: {log_level}")

    @staticmethod
    def _add_error_fields(message: Dict[str, Any], exc_type, exc_val, exc_tb):
        # if exceptions are being used for 'flow control', these can be logged as normal using "expected_errors" in the
        # in the action context or in the global context .. e.g.  @log_action(expected_errors=(ValueError,))

        # Add exc_info so the formatter can explode it into the component parts
        message[Constants.EXC_INFO] = (exc_type, exc_val, exc_tb)
        if Constants.LOG_REFERENCE_ON_ERR_FIELD in message:
            new_log_reference = message.pop(Constants.LOG_REFERENCE_ON_ERR_FIELD)
            if new_log_reference is None:
                message.pop(Constants.LOG_REFERENCE_FIELD, None)
            else:
                message[Constants.LOG_REFERENCE_FIELD] = new_log_reference

        # check if this is an 'expected error'
        expected_error_types = message.pop(Constants.EXPECTED_ERRORS, ())
        expected_error_levels = message.pop(Constants.ERROR_LEVELS, None)

        if not expected_error_types or not issubclass(exc_type, expected_error_types):
            message[Constants.INCLUDE_TRACEBACK] = True
            message[Constants.ACTION_STATUS] = Constants.STATUS_FAILED
            message[Constants.LOG_LEVEL] = logging.ERROR
            message[Constants.MESSAGE_TYPE_FIELD] = message.get(Constants.MESSAGE_TYPE_FIELD, Constants.TRACEBACK_FIELD)
            return

        message[Constants.INCLUDE_TRACEBACK] = False
        error_level = LogActionContextManager._get_expected_error_level(expected_error_levels, exc_type)
        current_level = LogActionContextManager._ensure_int_log_level(message.get(Constants.LOG_LEVEL, error_level))

        message[Constants.LOG_LEVEL] = max(current_level, error_level)
        message[Constants.ACTION_STATUS] = Constants.STATUS_ERROR


class _Globals:
    __slots__ = ["globals"]

    globals: Dict[str, Any]

    def __init__(self, fields: Dict[str, Any]):
        self.globals = fields


class _ThreadLocalContextStorage(threading.local):
    def __init__(self):
        self._stack: List[LogActionContextManager] = []
        self._globals_stack: List[_Globals] = []
        self._force_log_at_levels: List[int] = []

    @property
    def stack(self) -> List[LogActionContextManager]:
        return self._stack

    @property
    def current_context(self) -> Optional[LogActionContextManager]:
        if not self._stack:
            return None
        return self._stack[-1]

    @property
    def globals_stack(self) -> List[_Globals]:
        return self._globals_stack

    @property
    def current_global_fields(self) -> Dict[str, Any]:
        global_fields = {}
        for layer in self.globals_stack:
            if layer.globals:
                global_fields.update(layer.globals)

        return global_fields

    @property
    def forced_log_levels(self) -> List[int]:
        return self._force_log_at_levels

    @property
    def current_forced_log_level(self) -> Optional[int]:
        if not self._force_log_at_levels:
            return None

        return self._force_log_at_levels[-1]


class _TaskLoggingTaskStore:
    __slots__ = ("forced_levels", "globals_stack", "stack")

    stack: List[LogActionContextManager]
    globals_stack: List[_Globals]
    forced_levels: List[int]

    def __init__(self):
        self.stack = []
        self.globals_stack = []
        self.forced_levels = []


class _TaskIsolatedContextStorage:
    __slots__ = ("_task_store",)
    _task_store: Dict[Task, _TaskLoggingTaskStore]

    def __init__(self):
        self._task_store = WeakKeyDictionary()  # type: ignore

    def _get_task_store(self, task: Optional[Task] = None) -> Optional[_TaskLoggingTaskStore]:
        try:
            task = task or asyncio.current_task()
        except RuntimeError:
            # No loop for this thread:
            return None
        if not task:
            return None

        if task not in self._task_store:
            self._task_store[task] = _TaskLoggingTaskStore()

        return self._task_store[task]

    @property
    def stack(self) -> Optional[List[LogActionContextManager]]:
        store = self._get_task_store()
        if not store:
            return None

        return store.stack

    @property
    def current_context(self) -> Optional[LogActionContextManager]:
        for store in self._get_stack_stores():
            if store and store.stack:
                return store.stack[-1]

        return None

    def _get_stack_stores(self) -> typing.Generator[_TaskLoggingTaskStore, None, None]:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            return

        if not task:
            return

        while task:
            store = self._get_task_store(task)
            if not store:
                break
            yield store
            if not hasattr(task, "parent_task"):
                break
            task = task.parent_task  # type: ignore[attr-defined]

    @property
    def globals_stack(self) -> Optional[List[_Globals]]:
        store = self._get_task_store()
        if not store:
            return None

        return store.globals_stack

    @property
    def current_global_fields(self) -> Dict[str, Any]:
        global_fields: Dict[str, Any] = {}

        for store in reversed(list(self._get_stack_stores())):
            if not store:
                return global_fields

            for layer in store.globals_stack:
                if layer.globals:
                    global_fields.update(layer.globals)

        return global_fields

    @property
    def forced_log_levels(self) -> Optional[List[int]]:
        store = self._get_task_store()
        if not store:
            return None

        return store.forced_levels

    @property
    def current_forced_log_level(self) -> Optional[int]:
        for store in self._get_stack_stores():
            if store and store.forced_levels:
                return store.forced_levels[-1]

        return None


_app_globals: Dict[str, Any] = {}


def set_async_tpe(tpe_type: Type[ThreadPoolExecutor]):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    if isinstance(loop._default_executor, tpe_type):  # type: ignore[attr-defined]
        return
    tpe = tpe_type()
    loop.set_default_executor(tpe)


class _LoggingContext(threading.local):
    def __init__(self):
        self._on_init: Optional[Callable[[], None]] = None
        self._needs_init = True
        self._storage: Optional[Union[_TaskIsolatedContextStorage, _ThreadLocalContextStorage]] = None
        self._storage_factory: Callable[[], Union[_TaskIsolatedContextStorage, _ThreadLocalContextStorage]] = (
            _ThreadLocalContextStorage
        )

    configured = False

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["_storage"] = None
        return attributes

    def _maybe_init(self):
        if not self._on_init or not self._needs_init:
            return
        self._on_init()
        self._needs_init = False

    @property
    def app_globals(self) -> Dict[str, Any]:
        return _app_globals

    @property
    def storage(self):
        self._maybe_init()
        if self._storage is None:
            self._storage = self._storage_factory()

        return self._storage

    def force_log_at_level(self, log_at_level: Union[int, str]):
        log_level = logging.getLevelName(log_at_level) if isinstance(log_at_level, str) else log_at_level

        self.storage.forced_log_levels.append(log_level)

    def undo_force_log_at_level(self):
        self.storage.forced_log_levels.pop(-1)

    def log_at_level(self) -> int:
        log_level: Optional[int] = self.storage.current_forced_log_level
        if log_level is None:
            log_level = typing.cast(int, app_logger.log_at_level)
        return log_level

    def push(self, action: LogActionContextManager):
        self.storage.stack.append(action)

    def pop(self, item: LogActionContextManager) -> LogActionContextManager:
        stack = self.storage.stack

        for i in range(len(stack), 0, -1):
            if stack[i - 1] != item:
                continue
            return typing.cast(LogActionContextManager, stack.pop(i - 1))
        raise ActionNotInStack

    def current(self) -> Optional[LogActionContextManager]:
        return typing.cast(Optional[LogActionContextManager], self.storage.current_context)

    def current_internal_id(self) -> Optional[str]:
        current_action = self.current()
        if not current_action:
            return None
        return current_action.fields.get(Constants.LOG_CORRELATION_ID_FIELD)

    def get_context_fields(self):
        fields = {}
        fields.update(self.current_global_fields)
        internal_id = self.current_internal_id()
        if internal_id:
            fields[Constants.LOG_CORRELATION_ID_FIELD] = internal_id
        return fields

    @staticmethod
    def add_app_globals(**kwargs):
        if kwargs:
            _app_globals.update(kwargs)

    def add_temporary_globals(self, globals_layer: _Globals):
        self.storage.globals_stack.append(globals_layer)

    def pop_temporary_globals(self, item: _Globals):
        stack = self.storage.globals_stack

        for i in range(len(stack), 0, -1):
            if stack[i - 1] != item:
                continue
            stack.pop(i - 1)
            return
        raise ValueError("item not found")

    @property
    def current_global_fields(self) -> Dict[str, Any]:
        global_fields = _app_globals.copy()
        storage_fields = self.storage.current_global_fields
        if storage_fields:
            global_fields.update(storage_fields)
        return global_fields

    def emit_message(self, message):
        log_level = message.get(Constants.LOG_LEVEL, Constants.DEFAULT_LOG_LEVEL)

        log_level = logging.getLevelName(log_level) if isinstance(log_level, str) else log_level

        if log_level < self.log_at_level():
            # do nothing
            return

        final_message = {}
        final_message.update(message)
        final_message[Constants.TIMESTAMP_FIELD] = final_message.get(Constants.TIMESTAMP_FIELD, time())
        final_message.pop(Constants.LOG_LEVEL, None)

        exc_info = final_message.pop(Constants.EXC_INFO, None)
        caller_info = final_message.pop(Constants.CALLER_INFO_FIELD, None)

        app_logger.log(
            log_level, exc_info, args=lambda: final_message, caller_info=caller_info, add_context_fields=False
        )

    def reset_storage(self):
        self._storage = None

    def setup_async(self):
        self.async_context_storage()

    def async_context_storage(self):
        self._storage = None
        self._storage_factory = _async_storage_factory

    def thread_local_context_storage(self):
        self._storage = None
        self._storage_factory = _ThreadLocalContextStorage


def _context_storage_task_factory(loop, coro):
    # This is the default way to create a child task.
    child_task = asyncio.tasks.Task(coro, loop=loop)

    # Retrieve the request from the parent task...
    parent_task = asyncio.current_task(loop=loop)

    # ...and store it in the child task too.
    child_task.parent_task = parent_task  # type: ignore[attr-defined]

    return child_task


def _async_storage_factory():
    loop = asyncio.get_running_loop()
    loop.set_task_factory(_context_storage_task_factory)
    return _TaskIsolatedContextStorage()


logging_context = _LoggingContext()


def safe_str(obj: AnyStr) -> bytes:
    """Convert any arbitrary object into a UTF-8 encoded str"""
    if isinstance(obj, bytes):
        return obj

    return obj.encode("utf-8", errors="backslashreplace")


class LoggingContextWorkItem(thread._WorkItem):  # type: ignore[attr-defined]
    """
    Capture the global fields and internal id in the constructor (from the thread that is going to spawn the worker
    threads) and use that data within the new threads from where the run() method is called.
    """

    def __init__(self, future, fn, args, kwargs):
        super().__init__(future, fn, args, kwargs)
        self.global_fields = logging_context.current_global_fields

        internal_id = self.global_fields.get(Constants.LOG_CORRELATION_ID_FIELD, logging_context.current_internal_id())
        if internal_id:
            self.global_fields[Constants.LOG_CORRELATION_ID_FIELD] = internal_id

    def run(self):
        temp_globals = _Globals(self.global_fields)
        logging_context.add_temporary_globals(temp_globals)
        try:
            super().run()
        finally:
            logging_context.pop_temporary_globals(temp_globals)


class WorkItemThreadPoolExecutor(ThreadPoolExecutor):

    def __init__(
        self,
        max_workers=None,
        thread_name_prefix: str = "",
        initializer=None,
        initargs=(),
        work_item_type: Type[thread._WorkItem] = LoggingContextWorkItem,
    ):
        self._work_item_type = work_item_type
        super().__init__(
            max_workers=max_workers, thread_name_prefix=thread_name_prefix, initializer=initializer, initargs=initargs
        )

    def submit(self, fn, /, *args, **kwargs) -> _base.Future:  # type: ignore[override]
        def _inner() -> _base.Future:
            if self._broken:  # type: ignore[attr-defined]
                raise BrokenThreadPool(self._broken)  # type: ignore[attr-defined]

            if self._shutdown:  # type: ignore[attr-defined]
                raise RuntimeError("cannot schedule new futures after shutdown")

            if thread._shutdown:  # type: ignore[attr-defined]
                raise RuntimeError("cannot schedule new futures after interpreter shutdown")

            future: _base.Future = _base.Future()
            work_item = self._work_item_type(future, fn, args, kwargs)

            self._work_queue.put(work_item)
            ThreadPoolExecutor._adjust_thread_count(self)  # type: ignore[misc]
            return future

        if hasattr(thread, "_global_shutdown_lock"):
            with self._shutdown_lock, thread._global_shutdown_lock:  # type: ignore[attr-defined]
                return _inner()

        with self._shutdown_lock:
            return _inner()


LoggingThreadPoolExecutor = WorkItemThreadPoolExecutor

setup_logging_tpe = partial(set_async_tpe, tpe_type=LoggingThreadPoolExecutor)
