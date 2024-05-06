import asyncio
from collections import namedtuple
from functools import wraps
from time import perf_counter
from typing import Callable, List

import cv2
import numpy as np
import supervisely as sly

ImageInfoLite = namedtuple(
    "ImageInfoLite",
    ["id", "dataset_id", "full_url", "cas_url", "hdf5_url", "updated_at"],
)
TileInfo = namedtuple(
    "TileInfo",
    ["id", "unitSize", "url", "thumbnail"],
)
PointCloudTileInfo = namedtuple(
    "PointCloudTileInfo",
    ["atlasId", "atlasIndex", "imageId", "vector"],
)

LOG_THRESHOLD = 0.02
SHOULD_NOT_BE_ASYNC_THRESHOLD = 0.1


def with_retries(retries: int = 3, sleep_time: int = 1) -> Callable:
    """Decorator to retry the function in case of an exception.

    :param retries: Number of retries.
    :type retries: int
    :param sleep_time: Time to sleep between retries.
    :type sleep_time: int
    :return: Decorator.
    :rtype: Callable
    """

    def retry_decorator(func):
        @wraps(func)
        async def async_function_with_retries(*args, **kwargs):
            for _ in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    sly.logger.debug(
                        f"Failed to execute {func.__name__}, retrying. Error: {str(e)}"
                    )
                    await asyncio.sleep(sleep_time)
            raise Exception(
                f"Failed to execute {func.__name__} after {retries} retries."
            )

        return async_function_with_retries

    return retry_decorator


def timer(func: Callable) -> Callable:
    """Decorator to measure the execution time of the function.
    Works with both async and sync functions.

    :param func: Function to measure the execution time of.
    :type func: Callable
    :return: Decorated function.
    :rtype: Callable
    """

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


@timer
def rgb_to_rgba(nps: List[np.ndarray], size: int) -> List[np.ndarray]:
    """Resize list of numpy arrays to the square images with alpha channel.

    :param nps: List of numpy arrays to resize.
    :type nps: List[np.ndarray]
    :param size: Size of the square image.
    :type size: int
    :return: List of resized images with alpha channel.
    :rtype: List[np.ndarray]
    """
    return [_rgb_to_rgba(np_image, size) for np_image in nps]


def _rgb_to_rgba(rgb: np.ndarray, size: int) -> np.ndarray:
    """Resizing RGB image while keeping the aspect ratio and adding alpha channel.
    Returns square image with alpha channel.

    :param rgb: RGB image as numpy array.
    :type rgb: np.ndarray
    :param size: Size of the square image.
    :type size: int
    :return: Resized image with alpha channel.
    :rtype: np.ndarray
    """

    # Calculate the aspect ratio
    h, w = rgb.shape[:2]
    aspect_ratio = w / h

    # Calculate new width and height while keeping the aspect ratio
    if w > h:
        new_w = size
        new_h = int(size / aspect_ratio)
    else:
        new_h = size
        new_w = int(size * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(rgb, (new_w, new_h))

    # Create a new image with alpha channel
    alpha_image = np.zeros((size, size, 4), dtype=np.uint8)

    # Calculate padding
    y_pad = (size - new_h) // 2
    x_pad = (size - new_w) // 2

    # Add the resized image to the new image
    alpha_image[y_pad : y_pad + new_h, x_pad : x_pad + new_w, :3] = resized_image

    # Set alpha channel to 255 where the image is
    alpha_image[y_pad : y_pad + new_h, x_pad : x_pad + new_w, 3] = 255

    return alpha_image
