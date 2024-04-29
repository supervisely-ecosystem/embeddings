import asyncio
from collections import namedtuple
from functools import wraps
from time import perf_counter

import supervisely as sly

ImageInfoLite = namedtuple("ImageInfoLite", ["id", "url", "updated_at"])


def timer(func):
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = await func(*args, **kwargs)
            end_time = perf_counter()
            sly.logger.debug(
                f"Function {func.__name__} executed in {end_time - start_time} seconds"
            )
            return result

        return async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = func(*args, **kwargs)
            end_time = perf_counter()
            sly.logger.debug(
                f"Function {func.__name__} executed in {end_time - start_time} seconds"
            )
            return result

        return sync_wrapper
