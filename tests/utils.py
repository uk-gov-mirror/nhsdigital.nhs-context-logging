import asyncio
import concurrent.futures
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import (
    Any,
    TypeVar,
    cast,
)

from nhs_context_logging.logger import LoggingThreadPoolExecutor

TReturn = TypeVar("TReturn")


async def run_in_executor(func: Callable[..., TReturn], *args, **kwargs) -> TReturn:
    """
        async wrapper for sync code
    Args:
        func: the function to call
        *args: positional args to pass to fund
        **kwargs: kwargs to pass to func

    Returns:

    """
    loop = asyncio.get_running_loop()

    to_execute = partial(func, *args, **kwargs)
    result = cast(TReturn, await loop.run_in_executor(None, to_execute))

    return result


async def coro(func: Callable, *args, **kwargs):
    """
        coroutine wrapper to wrap non async calls
    Args:
        func: the function to call
        *args: positional args to pass to fund
        **kwargs: kwargs to pass to func

    Returns:

    """

    return func(*args, **kwargs)


def create_task(func: Callable, *args, **kwargs):
    """
        coroutine wrapper to wrap non async calls
    Args:
        func: the function to call
        *args: positional args to pass to fund
        **kwargs: kwargs to pass to func

    Returns:

    """

    return asyncio.ensure_future(coro(func, *args, **kwargs))


def concurrent_tasks(
    parallels: list[tuple[str, Callable, Sequence[Any]]],
    raise_if_ex: bool = True,
    with_results=True,
    max_workers: int | None = None,
    tpe_type: type[ThreadPoolExecutor] = LoggingThreadPoolExecutor,
) -> Mapping[str, Any]:
    """
    Execute a collection of tasks in parallel, wait for all tasks to complete and return the
    outcomes
    Args:
        parallels: The tasks to be queued to execute, each specified as a three-tuple:
            the first element is a string to use as an identifier for the task;
            the second is the callable to be invoked;
            the third a sequence of arguments to be passed to the callable
        raise_if_ex: Whether to raise an instance of ConcurrentExceptions should any of the
            tasks raise an exception
        with_results: Whether to return the task results:
            if set to false, the returned dictionary will just contain either a True or a False
            value depending on the success of the task
        max_workers: max concurrency
    Returns:
        A dictionary keyed by the task IDs and whose values are either the value produced by the
         task or the exception raised by the task; or, if with_results is set to False, True
         for tasks that ran successfully and False for tasks which raised exceptions
    """
    with tpe_type(max_workers=max_workers) as executor:
        task_ids = {}
        futures = []
        for task_id, func, args in parallels:
            future = executor.submit(func, *args)
            task_ids[id(future)] = str(task_id)
            futures.append(future)

        results = {}
        exceptions = []
        for future in concurrent.futures.as_completed(futures):
            task_id = task_ids[id(future)]
            ex = future.exception()
            if not ex:
                results[task_id] = future.result() if with_results else True
                continue
            results[task_id] = ex if with_results else False
            exceptions.append((future, ex))

        if not raise_if_ex or not exceptions:
            return results

        if len(exceptions) == 1:
            exceptions[0][0].result()

        result_exc = [(task_ids[id(f)], e) for f, e in exceptions]

        raise ConcurrentExceptions(*result_exc)  # type: ignore[arg-type]


class ConcurrentExceptions(Exception):
    """custom concurrent exception implementation, with extended output"""

    def __init__(self, *exceptions: tuple[str, Exception]):
        super().__init__(*exceptions)

        self.exceptions = dict(exceptions)

    def __repr__(self):
        return "\n".join(t + ": " + repr(e) for t, e in self.exceptions.items())

    def __str__(self):
        child_errors = "\n".join(t + ": " + str(e) for t, e in self.exceptions.items())
        return "Inner Exceptions:\n" + child_errors
