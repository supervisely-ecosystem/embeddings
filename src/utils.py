import asyncio
from collections import namedtuple
from functools import wraps
from time import perf_counter

import supervisely as sly

ImageInfoLite = namedtuple("ImageInfoLite", ["id", "url", "updated_at"])
LOG_THRESHOLD = 0.02


def timer(func):
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = await func(*args, **kwargs)
            end_time = perf_counter()
            execution_time = end_time - start_time
            if execution_time > LOG_THRESHOLD:
                sly.logger.debug(
                    f"BAD SPEED | {func.__name__}: {execution_time:.4f} sec"
                )
            else:
                sly.logger.debug(
                    f"OK  SPEED | {func.__name__}: {execution_time:.4f} sec"
                )
            return result

        return async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = func(*args, **kwargs)
            end_time = perf_counter()
            execution_time = end_time - start_time
            if execution_time > LOG_THRESHOLD:
                sly.logger.debug(
                    f"BAD SPEED | {func.__name__}: {execution_time:.4f} sec"
                )
            else:
                sly.logger.debug(
                    f"OK  SPEED | {func.__name__}: {execution_time:.4f} sec"
                )
            return result

        return sync_wrapper
