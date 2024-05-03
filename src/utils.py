import asyncio
from collections import namedtuple
from functools import wraps
from time import perf_counter

import supervisely as sly

ImageInfoLite = namedtuple(
    "ImageInfoLite",
    ["id", "dataset_id", "full_url", "cas_url", "hdf5_url", "updated_at"],
)
TileInfo = namedtuple(
    "TileInfo",
    ["id", "unitSize", "url", "thumbnail"],
)


LOG_THRESHOLD = 0.02
SHOULD_NOT_BE_ASYNC_THRESHOLD = 0.1


def timer(func):
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = await func(*args, **kwargs)
            end_time = perf_counter()
            execution_time = end_time - start_time
            if execution_time < SHOULD_NOT_BE_ASYNC_THRESHOLD:
                sly.logger.warning(
                    f"WARNING!  | {func.__name__} is too fast to be async."
                )
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
