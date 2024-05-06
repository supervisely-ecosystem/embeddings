import asyncio
from collections import namedtuple
from functools import wraps
from time import perf_counter
from typing import Callable, List

import aiohttp
import cv2
import numpy as np
import supervisely as sly
from fastapi import Request
from supervisely._utils import resize_image_url

import src.globals as g

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


def to_thread(func: Callable) -> Callable:
    """Decorator to run the function in a separate thread.

    :param func: Function to run in a separate thread.
    :type func: Callable
    :return: Decorated function.
    :rtype: Callable
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.to_thread(func, *args, **kwargs)

    return wrapper


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


@timer
async def download_items(urls: List[str]) -> List[np.ndarray]:
    """Asynchronously download images from the list of URLs.

    :param urls: List of URLs to download images from.
    :type urls: List[str]
    :return: List of downloaded images as numpy arrays.
    :rtype: List[np.ndarray]
    """
    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*[download_item(session, url) for url in urls])


async def download_item(session: aiohttp.ClientSession, url: str) -> np.ndarray:
    """Asynchronously download image from the URL and return it as numpy array.

    :param session: Instance of aiohttp ClientSession.
    :type session: aiohttp.ClientSession
    :param url: URL to download image from.
    :type url: str
    :return: Downloaded image as numpy array.
    :rtype: np.ndarray
    """
    async with session.get(url) as response:
        image_bytes = await response.read()

    return sly.image.read_bytes(image_bytes)


@to_thread
@timer
def get_datasets(api: sly.Api, project_id: int) -> List[sly.DatasetInfo]:
    """Returns list of datasets from the project.

    :param api: Instance of supervisely API.
    :type api: sly.Api
    :param project_id: ID of the project to get datasets from.
    :type project_id: int
    :return: List of datasets.
    :rtype: List[sly.DatasetInfo]
    """
    return api.dataset.get_list(project_id)


@to_thread
@timer
def get_image_infos(
    api: sly.Api,
    cas_size: int,
    hdf5_size: int,
    dataset_id: int = None,
    image_ids: List[int] = None,
) -> List[ImageInfoLite]:
    """Returns lite version of image infos to cut off unnecessary data.
    Uses either dataset_id or image_ids to get image infos.
    If dataset_id is provided, it will be used to get all images from the dataset.
    If image_ids are provided, they will be used to get image infos.

    :param api: Instance of supervisely API.
    :type api: sly.Api
    :param cas_size: Size of the image for CAS, it will be added to URL.
    :type cas_size: int
    :param hdf5_size: Size of the image for HDF5, it will be added to URL.
    :type hdf5_size: int
    :param dataset_id: ID of the dataset to get images from.
    :type dataset_id: int, optional
    :param image_ids: List of image IDs to get image infos.
    :type image_ids: List[int], optional
    :return: List of lite version of image infos.
    :rtype: List[ImageInfoLite]
    """

    if dataset_id:
        image_infos = api.image.get_list(dataset_id)
    elif image_ids:
        image_infos = api.image.get_info_by_id_batch(image_ids)

    return [
        ImageInfoLite(
            id=image_info.id,
            dataset_id=image_info.dataset_id,
            full_url=image_info.full_storage_url,
            cas_url=resize_image_url(
                image_info.full_storage_url,
                method="fit",
                width=cas_size,
                height=cas_size,
            ),
            hdf5_url=resize_image_url(
                image_info.full_storage_url,
                method="fit",
                width=hdf5_size,
                height=hdf5_size,
            ),
            updated_at=image_info.updated_at,
        )
        for image_info in image_infos
    ]


def _get_api_from_request(request: Request) -> sly.Api:
    """Returns API instance from the request. In development mode, API instance
    is stored in the global variable, in production mode it's stored in the state.

    :param request: FastAPI request instance.
    :type request: Request
    :return: Instance of supervisely API.
    :rtype: sly.Api
    """
    if sly.is_development():
        return g.api
    else:
        return request.state.api
