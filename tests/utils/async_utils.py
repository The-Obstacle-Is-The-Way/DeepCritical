import asyncio
import inspect
import time
from typing import Awaitable, Callable, TypeVar, Union

T = TypeVar("T")


async def wait_for_condition(
    condition_func: Callable[[], Union[bool, T, Awaitable[Union[bool, T]]]],
    timeout: float = 5.0,
    interval: float = 0.1,
    error_msg: str = "Condition not met within timeout",
) -> T:
    """Wait for a condition to become true. Handles both sync and async functions."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        result = condition_func()

        # If result is a coroutine, await it
        if inspect.isawaitable(result):
            result = await result

        if result:
            return result  # type: ignore
        await asyncio.sleep(interval)

    raise TimeoutError(error_msg)
