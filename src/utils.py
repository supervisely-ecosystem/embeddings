import asyncio
from collections import namedtuple
from functools import partial, wraps
from time import perf_counter
from typing import Callable, List

import aiohttp
import cv2
import numpy as np
import supervisely as sly
from fastapi import Request
from supervisely._utils import resize_image_url

import src.globals as g


class TupleFields:
    """Fields of the named tuples used in the project."""

    ID = "id"
    DATASET_ID = "dataset_id"
    FULL_URL = "full_url"
    CAS_URL = "cas_url"
    HDF5_URL = "hdf5_url"
    UPDATED_AT = "updated_at"
    UNIT_SIZE = "unitSize"
    URL = "url"
    THUMBNAIL = "thumbnail"
    ATLAS_ID = "atlasId"
    ATLAS_INDEX = "atlasIndex"
    VECTOR = "vector"
    FILE_NAME = "fileName"
    IMAGES = "images"


class StateFields:
    """Fields of the context in request objects."""

    PROJECT_ID = "project_id"
    IMAGE_IDS = "image_ids"
    FORCE = "force"
    QUERY = "query"
    LIMIT = "limit"
    METHOD = "method"


class QdrantFields:
    """Fields for the queries to the Qdrant API."""

    KMEANS = "kmeans"
    FPS = "fps"
    NUM_CLUSTERS = "num_clusters"
    OPTION = "option"
    INITIAL_VECTOR = "initial_vector"
    RANDOM = "random"
    CENTROIDS = "centroids"


ImageInfoLite = namedtuple(
    "ImageInfoLite",
    [
        TupleFields.ID,
        TupleFields.DATASET_ID,
        TupleFields.FULL_URL,
        TupleFields.CAS_URL,
        TupleFields.HDF5_URL,
        TupleFields.UPDATED_AT,
    ],
)
TileInfo = namedtuple(
    "TileInfo",
    [TupleFields.ID, TupleFields.UNIT_SIZE, TupleFields.URL, TupleFields.THUMBNAIL],
)
PointCloudTileInfo = namedtuple(
    "PointCloudTileInfo",
    ["atlasId", "atlasIndex", "imageId", "vector"],
)


def to_thread(func: Callable) -> Callable:
    """Decorator to run the function in a separate thread.
    Can be used for slow synchronous functions inside of the asynchronous code
    to avoid blocking the event loop.

    :param func: Function to run in a separate thread.
    :type func: Callable
    :return: Decorated function.
    :rtype: Callable
    """

    # For Python 3.9+.
    # @wraps(func)
    # def wrapper(*args, **kwargs):
    #     return asyncio.to_thread(func, *args, **kwargs)

    # For Python 3.7+.
    @wraps(func)
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        func_with_args = partial(func, *args, **kwargs)
        return loop.run_in_executor(None, func_with_args)

    return wrapper


def with_retries(
    retries: int = 3, sleep_time: int = 1, on_failure: Callable = None
) -> Callable:
    """Decorator to retry the function in case of an exception.
    Works only with async functions. Custom function can be executed on failure.
    NOTE: The on_failure function should be idempotent and synchronous.

    :param retries: Number of retries.
    :type retries: int
    :param sleep_time: Time to sleep between retries.
    :type sleep_time: int
    :param on_failure: Function to execute on failure, if None, raise an exception.
    :type on_failure: Callable, optional
    :raises Exception: If the function fails after all retries.
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
            if on_failure is not None:
                return on_failure()
            else:
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
            sly.logger.debug(f"{execution_time:.4f} sec | {func.__name__}")
            return result

        return async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = func(*args, **kwargs)
            end_time = perf_counter()
            execution_time = end_time - start_time
            _log_execution_time(func.__name__, execution_time)
            return result

        return sync_wrapper


def _log_execution_time(function_name: str, execution_time: float) -> None:
    """Log the execution time of the function.

    :param function_name: Name of the function.
    :type function_name: str
    :param execution_time: Execution time of the function.
    :type execution_time: float
    """
    sly.logger.debug(f"{execution_time:.4f} sec | {function_name}")


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


@with_retries(
    on_failure=lambda: np.zeros(
        (g.IMAGE_SIZE_FOR_ATLAS, g.IMAGE_SIZE_FOR_ATLAS, 3), dtype=np.uint8
    )
)
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
        sly.logger.debug("Not in development mode, getting API from the state.")
        return request.state.api
