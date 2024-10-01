# type: ignore
import asyncio
import contextlib
import inspect
import json
import logging
import time
from concurrent.futures import thread
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    Generator,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    cast,
)
from uuid import uuid4

import pytest

from nhs_context_logging import (
    INFO,
    add_fields,
    app_logger,
    log_action,
    logging_context,
    temporary_global_fields,
)
from nhs_context_logging.formatters import (
    JSONFormatter,
    KeyValueFormatter,
    StructuredFormatter,
)
from nhs_context_logging.logger import (
    ActionNotInStack,
    LoggingThreadPoolExecutor,
    set_async_tpe,
)
from tests.utils import concurrent_tasks, create_task, run_in_executor


def assert_single_internal_id(log_capture: Tuple[List[dict], List[dict]]):
    internal_ids = {line["internal_id"] for line in (log_capture[0] + log_capture[1])}
    assert len(internal_ids) == 1, internal_ids


@dataclass
class MyModel:
    name: str


FuncT = TypeVar("FuncT", bound=Callable[..., Any])


class _LoggingDecoratorClass:
    def __init__(self, some_arg: int = 6):
        self._some_arg = some_arg

    def __call__(self, func: FuncT) -> FuncT:
        @wraps(func)
        def _wrapper(*args, **kwargs):
            with temporary_global_fields(some_arg=self._some_arg):
                result = func(*args, **kwargs)

            return result

        @wraps(func)
        async def _async_wrapper(*args, **kwargs):
            async with temporary_global_fields(some_arg=self._some_arg):
                result = await func(*args, **kwargs)

            return result

        if inspect.iscoroutinefunction(func):
            return cast(FuncT, _async_wrapper)

        return cast(FuncT, _wrapper)


some_logging_decorator = _LoggingDecoratorClass


def test_setup_file_log():
    app_logger.setup_file_log("testing", log_dir="/tmp", is_async=True, redact_fields={"*"})


def test_setup_override_internal_id_factory(log_capture):
    def return_of_the_bob() -> str:
        return "bob"

    service = uuid4().hex
    app_logger.setup(service, force_reinit=True)
    assert app_logger.service_name == service

    service = uuid4().hex
    app_logger.setup(service, internal_id_factory=return_of_the_bob, force_reinit=True)
    assert app_logger.service_name == service

    @log_action()
    def do_a_thing():
        pass

    do_a_thing()

    std_out, _ = log_capture
    assert len(std_out) == 1
    assert std_out[0]["action"] == do_a_thing.__name__
    assert std_out[0]["internal_id"] == "bob"


def test_setup_default_internal_id_factory(log_capture):
    app_logger._is_setup = False
    app_logger.setup("testing")

    @log_action()
    def do_a_thing():
        pass

    do_a_thing()

    std_out, _ = log_capture
    assert len(std_out) == 1
    assert std_out[0]["action"] == do_a_thing.__name__
    assert std_out[0]["internal_id"] != "bob"


def test_logging_simple(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    app_logger.info(lambda: {"test": 123})

    assert len(std_out) == 1
    assert len(std_err) == 0
    assert std_out[0]["test"] == 123


def test_logging_simple_is_lazy(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    prev_level = app_logger.log_at_level

    try:
        app_logger.log_at_level = logging.WARN

        calls = []

        def create_logging_args():
            calls.append(1)
            return {"things": 123}

        app_logger.info(create_logging_args)

        assert len(std_out) == 0
        assert len(std_err) == 0
        assert len(calls) == 0
    finally:
        app_logger.log_at_level = prev_level


def test_logging_default_logger(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture
    level = logging.root.level
    logging.root.setLevel(INFO)
    logging.info("test")
    logging.root.setLevel(level)
    assert len(std_err) == 0
    assert std_out[0]["message"] == "test"


def test_log_exception(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    try:
        raise ValueError("testing")
    except ValueError:
        app_logger.exception(lambda: {"things": 123})

    assert len(std_out) == 0
    assert len(std_err) == 1

    err = std_err[0]
    assert err["things"] == 123
    assert err["error_info"]["error"] == "ValueError('testing')"
    assert err["error_info"]["type"] == "ValueError"
    assert err["error_info"]["fq_type"] == "builtins.ValueError"
    assert isinstance(err["error_info"]["line_no"], int)
    assert err["error_info"]["line_no"] > 0
    assert "traceback" in err["error_info"]
    assert 'raise ValueError("testing")' in err["error_info"]["traceback"]


def test_with_action_logging(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    with log_action(field=123):
        num = 1 + 1

        add_fields(num=num)

    assert len(std_err) == 0
    assert len(std_out) == 1

    log = std_out[0]

    assert log["num"] == 2
    assert log["field"] == 123

    assert "internal_id" in log


def test_with_action_logging_exploded_model(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    with log_action(field=MyModel(name="vic")):
        num = 1 + 1

        add_fields(num=num)

    assert len(std_err) == 0
    assert len(std_out) == 1

    log = std_out[0]

    assert log["num"] == 2

    assert isinstance(log["field"], dict), "Model was not exploded to primitive form!"
    assert log["field"]["name"] == "vic"

    assert "internal_id" in log


def test_with_action_logging_exploded_model_added_after(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    with log_action(field=123):
        add_fields(obj=MyModel(name="vic"))

    assert len(std_err) == 0
    assert len(std_out) == 1

    log = std_out[0]

    assert log["field"] == 123

    assert isinstance(log["obj"], dict), "Model was not exploded to primitive form!"
    assert log["obj"]["name"] == "vic"

    assert "internal_id" in log


def test_with_action_logging_exception(log_capture: Tuple[List[dict], List[dict]]):
    _, std_err = log_capture

    try:
        with log_action(field=123):
            num = 1 + 1

            add_fields(num=num)

            raise ValueError("Test")

    except ValueError:
        pass

    assert len(std_err) == 1

    log = std_err[0]

    assert log["num"] == 2
    assert log["field"] == 123

    assert "internal_id" in log


def test_log_action(log_capture: Tuple[List[dict], List[dict]]):
    std_out, _ = log_capture

    @log_action()
    @some_logging_decorator()
    def test_function():
        add_fields(field=123)

    test_function()

    assert len(std_out) == 1

    log = std_out[0]

    assert log["field"] == 123

    assert log["action"] == "test_function"
    assert log["action_status"] == "succeeded"


def test_log_action_with_args(log_capture: Tuple[List[dict], List[dict]]):
    std_out, _ = log_capture

    @log_action(log_args=["_bob"])
    @some_logging_decorator(some_arg=2)
    def test_function(_bob):  # noqa: PT019
        add_fields(field=123)

    test_function("vic")

    assert len(std_out) == 1

    log = std_out[0]

    assert log["field"] == 123

    assert log["action"] == "test_function"
    assert log["action_status"] == "succeeded"

    assert log["_bob"] == "vic"


def test_log_action_with_args_and_prepend_module_name(unlock_global_setup, log_capture: Tuple[List[dict], List[dict]]):
    app_logger.setup(service_name="myApp", config_kwargs={"prepend_module_name": True})

    std_out, _ = log_capture

    @log_action(log_args=["_bob"], prepend_module_name=True)
    @some_logging_decorator(some_arg=2)
    def test_function(_bob):  # noqa: PT019
        add_fields(field=123)

    test_function("vic")

    assert len(std_out) == 1

    log = std_out[0]

    assert log["field"] == 123

    assert log["action"] == "tests.logger_tests.test_function"
    assert log["action_status"] == "succeeded"

    assert log["_bob"] == "vic"


def test_log_action_with_model_exploded(log_capture: Tuple[List[dict], List[dict]]):
    std_out, _ = log_capture

    @log_action(log_args=["_bob"])
    def test_function(_bob):  # noqa: PT019
        add_fields(field=123)

    test_function(MyModel(name="vic"))

    assert len(std_out) == 1

    log = std_out[0]

    assert log["field"] == 123

    assert log["action"] == "test_function"
    assert log["action_status"] == "succeeded"

    assert isinstance(log["_bob"], dict), "Model was not exploded to primitive form!"
    assert log["_bob"]["name"] == "vic"


def test_log_action_with_named_action(log_capture: Tuple[List[dict], List[dict]]):
    std_out, _ = log_capture

    @log_action(action="test", log_args=["_bob"])
    def test_function(_bob):  # noqa: PT019
        add_fields(field=123)

    test_function("vic")

    assert len(std_out) == 1

    log = std_out[0]

    assert log["field"] == 123

    assert log["action"] == "test"
    assert log["action_status"] == "succeeded"

    assert log["_bob"] == "vic"


def test_log_action_exception(log_capture: Tuple[List[dict], List[dict]]):
    _, std_err = log_capture

    @log_action()
    def test_function():
        add_fields(field=123)
        raise ValueError("eek")

    with contextlib.suppress(ValueError):
        test_function()

    assert len(std_err) == 1

    log = std_err[0]

    assert log["field"] == 123

    assert log["action"] == "test_function"
    assert log["action_status"] == "failed"


async def test_async_logging_context_concurrent(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    @log_action()
    async def my_coro2(task_id: str):
        res = await asyncio.gather(
            *[
                my_coro(f"{task_id}-task1", 1),
                my_coro(f"{task_id}-task2", 0.5),
                my_coro(f"{task_id}-task3", 0.25),
                my_coro(f"{task_id}-task4", 0.33),
            ]
        )
        action = logging_context.current()
        return task_id, action.internal_id, res

    @log_action(log_args=["task_id"])
    async def my_coro(task_id: str, wait: float):
        action = logging_context.current()
        await asyncio.sleep(wait)
        return action.internal_id

    # Wrapper sets the context, otherwise all the child coroutines are not under the same context
    with log_action("wrapper"):
        await asyncio.gather(*[my_coro2("2task1"), my_coro2("2task2"), my_coro2("2task3"), my_coro2("2task4")])

    assert len(std_err) == 0
    assert len(std_out) > 1

    assert_single_internal_id(log_capture)


async def test_async_logging_context_linear(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    @log_action()
    async def my_coro2(task_id: str):
        print(f"starting task {task_id}")
        res = await my_coro("task3", 0.25)
        print(f"ending task {task_id}")
        action = logging_context.current()
        return task_id, action.internal_id, res

    @log_action()
    async def my_coro(task_id: str, wait: float):
        action = logging_context.current()
        print(f"starting task {task_id}")
        await asyncio.sleep(wait)
        print(f"ending task {task_id}")
        return action.internal_id

    await my_coro2("2task1")

    assert len(std_err) == 0
    assert len(std_out) == 2

    assert_single_internal_id(log_capture)


async def test_async_logging_context_concurrent_tasks(log_capture: Tuple[List[dict], List[dict]]):
    @log_action()
    async def my_coro2(task_id: str):
        print(f"starting task {task_id}")
        res = await asyncio.gather(
            *[
                await create_task(my_coro, "task1", 1),
                await create_task(my_coro, "task2", 0.5),
                await create_task(my_coro, "task3", 0.25),
                await create_task(my_coro, "task4", 0.33),
            ]
        )
        print(f"ending task {task_id}")
        action = logging_context.current()
        return task_id, action.internal_id, res

    @log_action()
    async def my_coro(task_id: str, wait: float):
        action = logging_context.current()
        print(f"starting task {task_id}")
        await asyncio.sleep(wait)
        print(f"ending task {task_id}")
        return action.internal_id

    await asyncio.gather(
        *[
            await create_task(my_coro2, "2task1"),
            await create_task(my_coro2, "2task2"),
            await create_task(my_coro2, "2task3"),
            await create_task(my_coro2, "2task4"),
        ]
    )

    assert_single_internal_id(log_capture)


def test_concurrent_with_global_logging_context(log_capture: Tuple[List[dict], List[dict]]):
    global_id = uuid4().hex

    @log_action()
    def my_task(task_id: str, wait: float):
        print(f"starting task {task_id}")
        time.sleep(wait)
        print(f"ending task {task_id}")
        return task_id

    internal_id = uuid4().hex
    with temporary_global_fields(internal_id=internal_id, global_id=global_id):
        concurrent_tasks(
            [
                ("task1", my_task, ["task1", 1]),
                ("task2", my_task, ["task2", 0.5]),
                ("task3", my_task, ["task3", 0.25]),
                ("task4", my_task, ["task4", 0.33]),
            ]
        )

    std_out, std_err = log_capture

    assert len(std_err) == 0
    assert len(std_out) == 4

    assert_single_internal_id(log_capture)

    assert std_out[0]["internal_id"] == internal_id

    my_io_global_id = {line["global_id"] for line in std_out if line["action"] == "my_task"}
    assert len(my_io_global_id) == 1
    assert my_io_global_id == {global_id}


def test_concurrent_with_logging_context(log_capture: Tuple[List[dict], List[dict]]):
    @log_action()
    def my_task(task_id: str, wait: float):
        print(f"starting task {task_id}")
        time.sleep(wait)
        print(f"ending task {task_id}")
        return task_id

    internal_id = uuid4().hex

    with log_action(internal_id=internal_id):
        concurrent_tasks(
            [
                ("task1", my_task, ["task1", 1]),
                ("task2", my_task, ["task2", 0.5]),
                ("task3", my_task, ["task3", 0.25]),
                ("task4", my_task, ["task4", 0.33]),
            ]
        )

    std_out, std_err = log_capture

    assert len(std_err) == 0
    assert len(std_out) == 5

    assert_single_internal_id(log_capture)

    assert std_out[0]["internal_id"] == internal_id


def test_concurrent_logging_with_no_context(log_capture: Tuple[List[dict], List[dict]]):
    global_id = uuid4().hex

    @log_action()
    def my_task(task_id: str, wait: float):
        print(f"starting task {task_id}")
        time.sleep(wait)
        print(f"ending task {task_id}")
        return task_id

    with temporary_global_fields(global_id=global_id):
        concurrent_tasks(
            [
                ("task1", my_task, ["task1", 1]),
                ("task2", my_task, ["task2", 0.5]),
                ("task3", my_task, ["task3", 0.25]),
                ("task4", my_task, ["task4", 0.33]),
            ]
        )

    std_out, std_err = log_capture

    assert len(std_err) == 0
    assert len(std_out) == 4

    internal_ids = {line["internal_id"] for line in (log_capture[0] + log_capture[1])}
    assert len(internal_ids) == 4, internal_ids

    my_io_global_id = {line["global_id"] for line in std_out if line["action"] == "my_task"}
    assert len(my_io_global_id) == 1
    assert my_io_global_id == {global_id}


def test_generator(log_capture: Tuple[List[dict], List[dict]]):
    @log_action()
    def inner_action() -> str:
        action = logging_context.current()
        return action.internal_id

    @log_action()
    def generator_action(count: int) -> Generator[Optional[str], None, None]:
        for _ in range(count):
            yield logging_context.current_internal_id(), inner_action()

    @log_action()
    def outer_action():
        res = list(generator_action(3))
        action = logging_context.current()
        return action.internal_id, res

    outer_action()

    assert_single_internal_id(log_capture)


async def test_generator_exit(log_capture: Tuple[List[dict], List[dict]], tmp_path):
    @log_action()
    async def outer_action():
        raise GeneratorExit

    with pytest.raises(GeneratorExit):
        await outer_action()

    assert_single_internal_id(log_capture)
    std_out, std_err = log_capture
    assert len(std_err) == 0
    assert len(std_out) == 1
    assert std_out[0]["log_info"]["level"] == "INFO"


async def test_end_action_when_action_already_popped(log_capture: Tuple[List[dict], List[dict]], tmp_path):
    async with log_action(internal_id="bob") as action:
        logging_context.pop(action)

    std_out, std_err = log_capture

    assert not std_err
    assert len(std_out) == 1
    assert type(std_out[0]["logger_error"]) is ActionNotInStack
    assert std_out[0]["log_info"]["level"] == "WARNING"


async def test_end_action_when_action_already_popped_with_exception(
    log_capture: Tuple[List[dict], List[dict]], tmp_path
):
    with pytest.raises(ValueError, match="eek"):  # noqa: PT012
        async with log_action(internal_id="bob") as action:
            logging_context.pop(action)
            raise ValueError("eek")

    std_out, std_err = log_capture

    assert not std_out
    assert len(std_err) == 1
    assert type(std_err[0]["logger_error"]) is ActionNotInStack
    assert std_err[0]["log_info"]["level"] == "ERROR"


async def test_async_logging_context_run_in_executor(log_capture: Tuple[List[dict], List[dict]]):

    app_logger.setup("pytest", is_async=True, force_reinit=True)
    global_id = uuid4().hex

    std_out, std_err = log_capture

    @log_action()
    async def my_coro2(task_id: str):
        with temporary_global_fields(global_id=global_id):
            print(f"{datetime.now()} -starting task {task_id}")
            res = await asyncio.gather(
                *[
                    run_in_executor(my_io, f"{task_id}-task1"),
                    run_in_executor(my_io, f"{task_id}-task2", wait=0.5),
                    run_in_executor(my_io, f"{task_id}-task3", wait=0.25),
                    run_in_executor(my_io, f"{task_id}-task4", wait=0.33),
                ]
            )
            print(f"{datetime.now()} -ending task {task_id}")
            action = logging_context.current()
        return task_id, action.internal_id, res

    @log_action()
    def my_io(task_id: str, wait: float = 1):
        print(f"{datetime.now()} - starting task {task_id}")
        time.sleep(wait)
        print(f"{datetime.now()} - ending task {task_id}")
        return task_id

    async with log_action("wrapper"):
        await asyncio.gather(*[my_coro2("2task1"), my_coro2("2task2"), my_coro2("2task3"), my_coro2("2task4")])

    assert len(std_err) == 0
    assert len(std_out) == 21

    assert_single_internal_id(log_capture)

    my_io_global_id = {line["global_id"] for line in std_out if line["action"] == "my_io"}
    assert len(my_io_global_id) == 1
    assert my_io_global_id == {global_id}


class NoLoggingContextWorkItem(thread._WorkItem):  # type: ignore[attr-defined]
    pass


class CustomExecutor(LoggingThreadPoolExecutor):
    def __init__(self, max_workers=None, thread_name_prefix: str = "", initializer=None, initargs=()):
        super().__init__(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
            initializer=initializer,
            initargs=initargs,
            work_item_type=NoLoggingContextWorkItem,
        )


setup_no_context_tpe = partial(set_async_tpe, tpe_type=CustomExecutor)


async def test_async_logging_context_run_in_executor_override_init(log_capture: Tuple[List[dict], List[dict]]):

    app_logger.setup("pytest", is_async=True, force_reinit=True, on_init=setup_no_context_tpe)
    global_id = uuid4().hex

    std_out, std_err = log_capture

    @log_action()
    async def my_coro2(task_id: str):
        with temporary_global_fields(global_id=global_id):
            print(f"{datetime.now()} -starting task {task_id}")
            res = await asyncio.gather(
                *[
                    run_in_executor(my_io, f"{task_id}-task1"),
                    run_in_executor(my_io, f"{task_id}-task2", wait=0.5),
                    run_in_executor(my_io, f"{task_id}-task3", wait=0.25),
                    run_in_executor(my_io, f"{task_id}-task4", wait=0.33),
                ]
            )
            print(f"{datetime.now()} -ending task {task_id}")
            action = logging_context.current()
        return task_id, action.internal_id, res

    @log_action()
    def my_io(task_id: str, wait: float = 1):
        print(f"{datetime.now()} - starting task {task_id}")
        time.sleep(wait)
        print(f"{datetime.now()} - ending task {task_id}")
        return task_id

    async with log_action("wrapper"):
        await asyncio.gather(*[my_coro2("2task1"), my_coro2("2task2"), my_coro2("2task3"), my_coro2("2task4")])

    assert len(std_err) == 0
    assert len(std_out) == 21

    internal_ids = {line["internal_id"] for line in (log_capture[0] + log_capture[1])}
    assert len(internal_ids) > 10, internal_ids


def test_key_value_formatter_drop_log_info():
    formatter = KeyValueFormatter(datefmt="%Y-%m-%d %H:%M:%S.%f", drop_fields="log_info")
    record = logging.LogRecord(
        name="aname",
        level=logging.ERROR,
        pathname="/test.py",
        lineno=1234,
        msg=None,
        args=[{"internal_id": "testid"}],
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert " ERROR " in formatted
    assert "log_info" not in formatted
    assert " internal_id=testid" in formatted


def test_key_value_formatter_drop_part_log_info():
    class MyMapping(Mapping):
        def __init__(self, *args, **kwargs):
            self._d = dict(*args, **kwargs)

        def __iter__(self):
            return self._d.__iter__()

        def __len__(self):
            return self._d.__len__()

        def __getitem__(self, key):
            return self._d.__getitem__(key)

    formatter = KeyValueFormatter(
        drop_fields=[
            "log_info_logger",
            "log_info_thread",
            "log_info_thread_name",
            "log_info_process_name",
            "log_info_filename",
            "log_info_module",
            "log_info_level",
        ]
    )

    record = logging.LogRecord(
        name="aname",
        level=logging.ERROR,
        pathname="/test.py",
        lineno=1234,
        msg=" HI! ",
        args=[
            {
                "internal_id": "testid",
                "list_type": [1, 2, 3],
                "nested": {
                    "list": [4, 5, 6],
                    "key": "secret",
                    "my_mapping": MyMapping(another_key="another_secret"),
                },
            },
        ],
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert " ERROR " in formatted
    for field in formatter.drop_fields:
        assert field not in formatted
    assert "log_info_pid" in formatted
    assert "log_info_line_no" in formatted
    assert "log_info_func" in formatted
    assert "log_info_path" in formatted
    assert " internal_id=testid " in formatted
    assert " list_type=1,2,3 " in formatted
    assert " nested_list=4,5,6 " in formatted
    assert " nested_key=secret" in formatted
    assert " nested_my_mapping_another_key=another_secret" in formatted


@pytest.mark.parametrize("log_result", [True, False])
def test_sync_context(log_result: bool, log_capture: Tuple[List[dict], List[dict]]):
    std_out, _ = log_capture

    @log_action(log_reference="bob", log_result=log_result)
    def s_function(aaa):
        app_logger.info("smiddle")
        return aaa

    res = s_function(1)
    assert res == 1

    assert std_out[0]["message"] == "smiddle"
    assert std_out[-1]["log_reference"] == "bob"

    logged_results = [log["action_result"] for log in std_out if "action_result" in log]
    if log_result:
        assert logged_results == [res]
    else:
        assert logged_results == []


def test_default_redaction(log_capture: Tuple[List[dict], List[dict]]):
    std_out, _ = log_capture

    @log_action(log_reference="bob", nhs_number="yes", password="yes")
    def s_function(aaa):
        app_logger.info("smiddle")
        return aaa

    res = s_function(1)
    assert res == 1

    assert std_out[0]["message"] == "smiddle"
    assert std_out[-1]["log_reference"] == "bob"
    assert std_out[-1]["nhs_number"] == "--REDACTED--"
    assert std_out[-1]["password"] == "--REDACTED--"


class FrozenDict(Mapping):
    """An implementation of a frozen dict, lifted from https://stackoverflow.com/a/2704866/1571593"""

    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)
        self._hash = None

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __str__(self):
        return str(self._d)

    def __repr__(self):
        return repr(self._d)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenDict):
            return other._d == self._d
        return False

    def __hash__(self):
        # It would have been simpler and maybe more obvious to
        # use hash(tuple(sorted(self._d.iteritems()))) from this discussion
        # so far, but this solution is O(n). I don't know what kind of
        # n we are going to run into, but sometimes it's hard to resist the
        # urge to optimize when it will gain improved algorithmic performance.
        if self._hash is None:
            hash_ = 0
            for pair in self.items():
                hash_ ^= hash(pair)
            self._hash = hash_
        return self._hash


def test_default_redaction_with_general_mapping(log_capture: Tuple[List[dict], List[dict]]):
    std_out, _ = log_capture

    thing_to_redact = FrozenDict(root={"branch": {"nhs_number": {"a complex object"}}})

    @log_action(log_reference="bob", password="yes", log_result=True)
    def s_function():
        return dict(thing_to_redact)

    assert s_function() == dict(thing_to_redact)

    (log,) = std_out
    assert log["log_reference"] == "bob"
    assert log["password"] == "--REDACTED--"
    assert log["action_result"] == {"root": {"branch": {"nhs_number": "--REDACTED--"}}}


def test_default_redaction_with_general_mapping_with_function_key(log_capture: Tuple[List[dict], List[dict]]):
    std_out, _ = log_capture

    def function_to_use_as_a_key():
        return "123"

    thing_to_redact = FrozenDict(root={function_to_use_as_a_key: {"nhs_number": {"a complex object"}}})

    @log_action(log_reference="bob", password="yes", log_result=True)
    def s_function():
        return dict(thing_to_redact)

    assert s_function() == dict(thing_to_redact)

    (log,) = std_out
    assert log["log_reference"] == "bob"
    assert log["password"] == "--REDACTED--"
    assert log["action_result"] == {"root": {function_to_use_as_a_key: {"nhs_number": "--REDACTED--"}}}


def test_default_redaction_exclusion(log_capture: Tuple[List[dict], List[dict]]):
    std_out, _ = log_capture

    @log_action(log_reference="bob", nhs_number="yes", password="yes", dont_redact={"nhs_number"})
    def s_function(aaa):
        app_logger.info("smiddle")
        return aaa

    res = s_function(1)
    assert res == 1

    assert std_out[0]["message"] == "smiddle"
    assert std_out[-1]["log_reference"] == "bob"
    assert std_out[-1]["nhs_number"] == "yes"
    assert std_out[-1]["password"] == "--REDACTED--"


def test_sync_generator(log_capture: Tuple[List[dict], List[dict]]):
    std_out, _ = log_capture

    @log_action(log_reference="test")
    def sgen_function(count):
        for i in range(count):
            app_logger.info(f"{i}middle")
            yield i

    res = list(sgen_function(3))
    assert len(res) == 3

    messages = [log["message"] for log in std_out if "message" in log]
    assert messages == ["0middle", "1middle", "2middle"]
    assert std_out[-1]["log_info"]["func"] == "test_sync_generator"


@pytest.mark.parametrize("log_result", [True, False])
async def test_async_context(log_result: bool, log_capture: Tuple[List[dict], List[dict]]):
    std_out, _ = log_capture

    @log_action(log_result=log_result)
    async def a_function(aaa):
        app_logger.info("amiddle")
        return aaa

    res = await a_function(1)
    assert res == 1

    messages = [log["message"] for log in std_out if "message" in log]
    assert messages == ["amiddle"]
    assert std_out[-1]["log_info"]["func"] == "test_async_context"

    logged_results = [log["action_result"] for log in std_out if "action_result" in log]
    if log_result:
        assert logged_results == [res]
    else:
        assert logged_results == []


async def test_async_generator(log_capture: Tuple[List[dict], List[dict]]):
    std_out, _ = log_capture

    @log_action()
    async def asg_function(count):
        total = 0
        for i in range(count):
            app_logger.info(f"{i}middle")
            total += 1
            yield i
        add_fields(total=total)

    res = [i async for i in asg_function(3)]
    assert len(res) == 3

    messages = [log["message"] for log in std_out if "message" in log]
    assert messages == ["0middle", "1middle", "2middle"]
    assert std_out[-1]["total"] == 3
    assert std_out[-1]["log_info"]["func"] == "test_async_generator"


def test_sync_generator_context_manager(log_capture: Tuple[List[dict], List[dict]]):
    std_out, _ = log_capture

    def sg_function(count):
        total = 0
        for i in range(count):
            app_logger.info(f"{i}middle")
            total += 1
            yield i
        add_fields(total=total)

    with temporary_global_fields(test_global=123), log_action():
        res = list(sg_function(3))
        assert len(res) == 3

    app_logger.info("fin")

    assert std_out[-2]["total"] == 3
    assert std_out[-2]["log_info"]["func"] == "test_sync_generator_context_manager"
    assert std_out[-2]["test_global"] == 123
    assert "test_global" not in std_out[-1]

    assert "internal_id" not in std_out[-1]
    refs = [log["internal_id"] for log in std_out[:-1]]
    assert set(refs) == {std_out[-2]["internal_id"]}

    messages = [log["message"] for log in std_out if "message" in log]
    assert messages == ["0middle", "1middle", "2middle", "fin"]


async def test_add_fields_can_change_log_level(log_capture: Tuple[List[dict], List[dict]]):
    std_out, _ = log_capture

    @log_action(log_level=logging.NOTSET)
    async def a_function():
        add_fields(log_level=logging.INFO)

    await a_function()

    assert len(std_out) == 1
    assert std_out[0]["log_info"]["level"] == "INFO"


@pytest.mark.skip
async def test_async_generator_resolved_later(log_capture: Tuple[List[dict], List[dict]]):
    _, _ = log_capture

    async def asg_function(count):
        total = 0
        for i in range(count):
            app_logger.info(f"{i}middle")
            total += 1
            yield i
        add_fields(total=total)

    @log_action()
    async def outer_func():
        async with temporary_global_fields(test_global=123):
            return asg_function(3)

    gen = await outer_func()
    _ = [num async for num in gen]


def test_logger_kwargs(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    app_logger.info(log_reference="MESH1234", another="test")

    assert len(std_err) == 0
    assert len(std_out) == 1

    log = std_out[0]

    assert log["log_reference"] == "MESH1234"
    assert log["another"] == "test"


def test_logger_args_and_kwargs(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    app_logger.info({"test": 1234}, log_reference="MESH1234", test=0)

    assert len(std_err) == 0
    assert len(std_out) == 1

    log = std_out[0]

    assert log["log_reference"] == "MESH1234"
    assert log["test"] == 1234


def test_logger_message_and_kwargs(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    app_logger.info("test", log_reference="MESH1234", test=0, message="test2")

    assert len(std_err) == 0
    assert len(std_out) == 1

    log = std_out[0]

    assert log["log_reference"] == "MESH1234"
    assert log["test"] == 0
    assert log["message"] == "test"


def test_logger_list_args_and_kwargs(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    app_logger.info(123, log_reference="MESH1234", message="test2")

    assert len(std_err) == 0
    assert len(std_out) == 1

    log = std_out[0]

    assert "log_reference" not in log
    assert "message" not in log


def test_logger_callable_args_and_kwargs(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    def get_args():
        return {"message": 1234, "thing": "bob"}

    app_logger.info(get_args, log_reference="MESH1234")

    assert len(std_err) == 0
    assert len(std_out) == 1

    log = std_out[0]

    assert log["thing"] == "bob"
    assert log["message"] == 1234
    assert "log_reference" in log


def test_expected_errors(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    @log_action(action="ex_test", expected_errors=(ValueError,))
    def error_function(raise_value_error):
        if raise_value_error:
            raise ValueError("test")

        raise NotImplementedError

    with pytest.raises(NotImplementedError) as ex:
        error_function(raise_value_error=False)

    assert ex.type is NotImplementedError

    assert len(std_err) == 1

    std_out.clear()
    std_err.clear()

    with pytest.raises(ValueError, match="test") as ex:
        error_function(raise_value_error=True)

    assert ex.type is ValueError
    assert ex.value

    assert len(std_err) == 0
    assert len(std_out) == 1
    log = std_out[0]
    assert log["action"] == "ex_test"
    assert log["log_info"]["level"] == "INFO"
    assert log["action_status"] == "error"


def test_expected_errors_subclass(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    class SubValueError(ValueError):
        pass

    @log_action(action="ex_test", expected_errors=(ValueError,))
    def error_function(raise_value_error):
        if raise_value_error:
            raise SubValueError("test")

        raise NotImplementedError

    with pytest.raises(NotImplementedError) as ex:
        error_function(raise_value_error=False)

    assert ex.type is NotImplementedError

    assert len(std_err) == 1

    std_out.clear()
    std_err.clear()

    with pytest.raises(SubValueError) as ex:
        error_function(raise_value_error=True)

    assert ex.type == SubValueError
    assert ex.value

    assert len(std_err) == 0
    assert len(std_out) == 1
    log = std_out[0]
    assert log["action"] == "ex_test"
    assert log["log_info"]["level"] == "INFO"
    assert log["action_status"] == "error"


def test_expected_errors_raise_log_level_to_info(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    @log_action(action="ex_test", log_level=logging.DEBUG, expected_errors=(ValueError,))
    def error_function():
        raise ValueError("test")

    with pytest.raises(ValueError, match="test"):
        error_function()

    assert len(std_err) == 0
    assert len(std_out) == 1
    log = std_out[0]
    assert log["action"] == "ex_test"
    assert log["log_info"]["level"] == "INFO"
    assert log["action_status"] == "error"


def test_expected_errors_specific_levels(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    @log_action(
        action="ex_test",
        log_level=logging.DEBUG,
        expected_errors=(ValueError,),
        error_levels=((ValueError, logging.DEBUG),),
    )
    def error_function():
        raise ValueError("test")

    with pytest.raises(ValueError, match="test"):
        error_function()

    assert len(std_err) == 0
    assert len(std_out) == 1
    log = std_out[0]
    assert log["action"] == "ex_test"
    assert log["log_info"]["level"] == "DEBUG"
    assert log["action_status"] == "error"


def test_expected_errors_raise_doesnt_lower_log_level(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    @log_action(action="ex_test", log_level=logging.WARN, expected_errors=(ValueError,))
    def error_function():
        raise ValueError("test")

    with pytest.raises(ValueError, match="test"):
        error_function()

    assert len(std_err) == 0
    assert len(std_out) == 1
    log = std_out[0]
    assert log["action"] == "ex_test"
    assert log["log_info"]["level"] == "WARNING"
    assert log["action_status"] == "error"


def test_expected_errors_global_fields(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    @log_action(action="ex_test")
    def error_function(raise_value_error):
        if raise_value_error:
            raise ValueError("test")

        raise NotImplementedError

    with temporary_global_fields(expected_errors=(ValueError,)), pytest.raises(ValueError, match="test") as ex:
        error_function(raise_value_error=True)

    assert ex.type is ValueError
    assert ex.value

    assert len(std_err) == 0
    assert len(std_out) == 1
    log = std_out[0]
    assert log["action"] == "ex_test"
    assert log["log_info"]["level"] == "INFO"
    assert log["action_status"] == "error"
    assert log["error_info"]["type"] == "ValueError"
    assert log["error_info"]["fq_type"] == "builtins.ValueError"
    assert log["error_info"]["args"] == ("test",)


def test_expected_errors_in_both(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    @log_action(action="ex_test", expected_errors=(NotImplementedError,))
    def error_function(raise_value_error):
        if raise_value_error:
            raise ValueError("test")

        raise NotImplementedError

    with temporary_global_fields(expected_errors=(ValueError,)), pytest.raises(ValueError, match="test") as ex:
        error_function(raise_value_error=True)

    assert ex.type is ValueError
    assert ex.value

    assert len(std_err) == 0
    assert len(std_out) == 1
    log = std_out[0]
    assert log["action"] == "ex_test"
    assert log["log_info"]["level"] == "INFO"
    assert log["action_status"] == "error"
    assert log["error_info"]["type"] == "ValueError"
    assert log["error_info"]["args"] == ("test",)

    std_out.clear()
    std_err.clear()

    with temporary_global_fields(expected_errors=(NotImplementedError,)), pytest.raises(NotImplementedError) as ex:
        error_function(raise_value_error=False)

    assert ex.type is NotImplementedError
    assert ex.value

    assert len(std_err) == 0
    assert len(std_out) == 1
    log = std_out[0]
    assert log["action"] == "ex_test"
    assert log["log_info"]["level"] == "INFO"
    assert log["action_status"] == "error"
    assert log["error_info"]["type"] == "NotImplementedError"
    assert log["error_info"]["fq_type"] == "builtins.NotImplementedError"


async def test_expected_errors_run_in_executor(log_capture: Tuple[List[dict], List[dict]]):
    app_logger.setup("pytest", is_async=True, force_reinit=True)

    std_out, std_err = log_capture

    @log_action(
        action="ex_test",
    )
    def error_function(exception_class=None):
        if exception_class:
            raise exception_class("test")

        raise NotImplementedError

    with pytest.raises(NotImplementedError) as ex:
        async with log_action("wrapper", expected_errors=(ValueError,)):
            await run_in_executor(error_function)

    assert ex.type is NotImplementedError

    assert len(std_out) == 0
    assert len(std_err) == 2

    assert_single_internal_id(log_capture)

    std_out.clear()
    std_err.clear()

    async with temporary_global_fields(expected_errors=(ValueError,)):
        with pytest.raises(ValueError, match="test") as ex:
            async with log_action(
                "wrapper",
            ):
                await run_in_executor(error_function, exception_class=ValueError)

    assert len(std_out) == 2
    assert len(std_err) == 0
    assert all(line["action_status"] == "error" for line in std_out)

    assert_single_internal_id(log_capture)

    std_out.clear()
    std_err.clear()

    class SubValueError(ValueError):
        pass

    async with temporary_global_fields(expected_errors=(ValueError,)):
        with pytest.raises(SubValueError) as ex:
            async with log_action(
                "wrapper",
            ):
                await run_in_executor(error_function, exception_class=SubValueError)

    assert len(std_out) == 2
    assert len(std_err) == 0
    assert all(line["action_status"] == "error" for line in std_out)

    assert_single_internal_id(log_capture)


class _HTTPException(Exception):
    def __init__(self, status_code: int, detail: Optional[str] = None) -> None:
        super().__init__()
        self.status_code = status_code
        self.detail = detail


def test_expected_errors_complex_exception(log_capture: Tuple[List[dict], List[dict]]):
    std_out, std_err = log_capture

    @log_action(expected_errors=(_HTTPException,))
    def error_function():
        raise _HTTPException(status_code=123, detail={"test": 1234})

    with pytest.raises(_HTTPException) as ex:
        error_function()

    assert ex.type == _HTTPException
    assert ex.value

    assert len(std_err) == 0
    assert len(std_out) == 1
    log = std_out[0]
    assert log["action"] == "error_function"
    assert log["log_info"]["level"] == "INFO"
    assert log["action_status"] == "error"

    assert log["error_info"]["type"] == _HTTPException.__name__
    assert log["error_info"]["fq_type"] == f"{_HTTPException.__module__}.{_HTTPException.__name__}"
    assert log["error_info"]["func"] == "error_function"
    assert isinstance(log["error_info"]["line_no"], int)


async def test_async_sync_concurrent_tasks_transfer(log_capture: Tuple[List[dict], List[dict]]):
    app_logger.setup("pytest", is_async=True, force_reinit=True)

    @log_action()
    def my_task():
        return "from task"

    @log_action()
    def my_sync():
        concurrent_tasks([("test", my_task, [])])

    @log_action()
    async def async_fn():
        return await asyncio.gather(
            *[
                run_in_executor(my_sync),
                run_in_executor(my_sync),
                run_in_executor(my_sync),
            ]
        )

    std_out, std_err = log_capture
    async with log_action("wrapper"):
        await async_fn()

    assert len(std_err) == 0
    assert len(std_out) == 8

    for line in std_out:
        print(line)

    assert_single_internal_id(log_capture)


async def test_async_to_run_in_executor_sync(log_capture: Tuple[List[dict], List[dict]]):
    app_logger.setup("pytest", is_async=True, force_reinit=True)

    global_id = uuid4().hex

    std_out, std_err = log_capture

    @log_action()
    async def async_fn():
        async with temporary_global_fields(global_id=global_id):
            return await asyncio.gather(
                *[
                    run_in_executor(sync_fn1),
                    run_in_executor(sync_fn1),
                    run_in_executor(sync_fn1),
                ]
            )

    @log_action()
    def sync_fn1():
        with temporary_global_fields(sync_global_id=uuid4().hex):
            sync_fn2()

    @log_action()
    def sync_fn2():
        add_fields(fn2_id=uuid4().hex)

    async with log_action("wrapper"):
        await async_fn()

    assert len(std_err) == 0
    assert len(std_out) == 8, std_out
    for line in std_out:
        print(line)

    sync_fn_global_ids = {line["global_id"] for line in std_out if line["action"].startswith("sync_fn")}
    assert len(sync_fn_global_ids) == 1
    assert sync_fn_global_ids == {global_id}

    sync_fn_ids = {line["sync_global_id"] for line in std_out if line["action"].startswith("sync_fn2")}
    assert len(sync_fn_ids) == 3

    fn2_ids = {line["fn2_id"] for line in std_out if line["action"].startswith("sync_fn2")}
    assert len(fn2_ids) == 3

    assert_single_internal_id(log_capture)


def test_log_action_exception_with_log_ref_override(log_capture: Tuple[List[dict], List[dict]]):
    _, std_err = log_capture

    @log_action(log_reference="BANANA", log_reference_on_error="PLUM")
    def test_function():
        add_fields(field=123)
        raise ValueError("eek")

    with contextlib.suppress(ValueError):
        test_function()

    assert len(std_err) == 1

    log = std_err[0]

    assert log["field"] == 123

    assert log["action"] == "test_function"
    assert log["action_status"] == "failed"
    assert log["log_reference"] == "PLUM"
    assert "log_reference_on_error" not in log


def test_log_action_exception_with_log_ref_unset(log_capture: Tuple[List[dict], List[dict]]):
    _, std_err = log_capture

    @log_action(log_reference="BANANA", log_reference_on_error=None)
    def test_function():
        add_fields(field=123)
        raise ValueError("eek")

    with contextlib.suppress(ValueError):
        test_function()

    assert len(std_err) == 1

    log = std_err[0]

    assert log["field"] == 123

    assert log["action"] == "test_function"
    assert log["action_status"] == "failed"
    assert "log_reference" not in log
    assert "log_reference_on_error" not in log


def test_log_action_exception_with_log_ref_unset_on_log_ref(log_capture: Tuple[List[dict], List[dict]]):
    _, std_err = log_capture

    @log_action(log_reference_on_error="BANANA")
    def test_function():
        add_fields(field=123)
        raise ValueError("eek")

    with contextlib.suppress(ValueError):
        test_function()

    assert len(std_err) == 1

    log = std_err[0]

    assert log["field"] == 123

    assert log["action"] == "test_function"
    assert log["action_status"] == "failed"
    assert log["log_reference"] == "BANANA"


def test_decimal_json_formatter():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="aname",
        level=logging.ERROR,
        pathname="/test.py",
        lineno=1234,
        msg=None,
        args=[{"dec1": Decimal("24.534"), "sub": {"dec2": Decimal("523.109")}}],
        exc_info=None,
    )
    formatted = formatter.format(record)

    data = json.loads(formatted)
    assert data["dec1"] == "24.534"
    assert data["sub"]["dec2"] == "523.109"


def test_decimal_key_value_formatter():
    formatter = KeyValueFormatter()
    record = logging.LogRecord(
        name="aname",
        level=logging.ERROR,
        pathname="/test.py",
        lineno=1234,
        msg=None,
        args=[{"dec1": Decimal("24.534"), "sub": {"dec2": Decimal("523.109")}}],
        exc_info=None,
    )
    formatted = formatter.format(record)

    assert "dec1=24.534" in formatted
    assert "sub_dec2=523.109" in formatted


def test_decimal_structured_formatter():
    formatter = StructuredFormatter()
    record = logging.LogRecord(
        name="aname",
        level=logging.ERROR,
        pathname="/test.py",
        lineno=1234,
        msg=None,
        args=[{"dec1": Decimal("24.534"), "sub": {"dec2": Decimal("523.109")}}],
        exc_info=None,
    )
    formatted = formatter.format(record)

    assert formatted["dec1"] == Decimal("24.534")
    assert formatted["sub"]["dec2"] == Decimal("523.109")


def test_set_internal_id_from_temporary_global_fields(log_capture: Tuple[List[dict], List[dict]]):
    _, std_err = log_capture

    @log_action(log_reference="BANANA", log_reference_on_error=None)
    def test_function():
        add_fields(field=123)
        raise ValueError("eek")

    with pytest.raises(ValueError, match="eek"), temporary_global_fields(internal_id="RASPBERRY"):
        test_function()

    assert len(std_err) == 1

    log = std_err[0]

    assert log["field"] == 123

    assert log["action"] == "test_function"
    assert log["action_status"] == "failed"
    assert log["internal_id"] == "RASPBERRY"


async def test_async_set_internal_id_from_temporary_global_fields(log_capture: Tuple[List[dict], List[dict]]):
    std_out, _ = log_capture

    @log_action(log_reference="BANANA", log_reference_on_error=None)
    async def test_function():
        add_fields(field=123)
        return "test"

    async with log_action(action="outer"), temporary_global_fields(internal_id="RASPBERRY"):
        result = await test_function()
        add_fields(result=result)

    assert len(std_out) == 2

    log = std_out[0]

    assert log["field"] == 123

    assert log["action"] == "test_function"
    assert log["action_status"] == "succeeded"
    assert log["internal_id"] == "RASPBERRY"

    log = std_out[1]
    assert log["action"] == "outer"
    assert log["result"] == "test"
