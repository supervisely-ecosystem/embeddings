import asyncio
import os
from typing import List

import aiohttp
import cv2
import h5py
import numpy as np
import supervisely as sly
from fastapi import Request

import src.cas as cas
import src.globals as g
import src.qdrant as qdrant
from src.utils import ImageInfoLite, timer

app = sly.Application()
server = app.get_server()


@server.post("/test")
@timer
async def test(request: Request):
    # Step 0: Unpack request, get context and instance of API.
    # * for debug using api from globals module
    api = g.api
    context = request.state.context
    project_id = context["project_id"]

    # Step 1: Ensure collection exists in Qdrant.
    await qdrant.get_or_create_collection(project_id)

    # Step 2: Get datasets from project.
    datasets = await asyncio.to_thread(get_datasets, api, project_id)

    # Step 3: Iterate over datasets.
    for dataset in datasets:
        # Step 4: Get lite version of image infos to cut off unnecessary data.
        # URLs will lead to resized images.
        all_image_infos = await asyncio.to_thread(
            get_image_infos, api, g.IMAGE_SIZE_FOR_CAS, dataset.id
        )

        # Step 4.1: Get diff of image infos, check if they are already
        # in the Qdrant collection and have the same updated_at field.
        image_infos = await qdrant.get_diff(project_id, all_image_infos)

        # Step 5: Batch image urls.
        for image_batch in sly.batched(image_infos):
            url_batch = [image_info.url for image_info in image_batch]
            ids_batch = [image_info.id for image_info in image_batch]

            # Step 6: Download images as numpy arrays.
            nps_batch = await download_items(url_batch)

            # Step 6.1: Resize images.
            nps_batch = resize_nps(nps_batch, g.IMAGE_SIZE_FOR_ATLAS)

            # Step 7: Save images to hdf5.
            save_to_hdf5(dataset.id, nps_batch, ids_batch, project_id)

            # Step 8: Get vectors from images.
            vectors_batch = await get_vectors(url_batch)
            updated_at_batch = [image_info.updated_at for image_info in image_batch]

            # Step 9: Upsert vectors to Qdrant.
            await qdrant.upsert(
                project_id, vectors_batch, ids_batch, updated_at=updated_at_batch
            )


@timer
def save_to_hdf5(
    dataset_id: int, vectors: List[np.ndarray], ids: List[int], project_id: int
):
    """Save vector data to the HDF5 file.

    :param dataset_id: ID of the dataset. It will be used as a group name.
    :type dataset_id: int
    :param vectors: List of vectors to save.
    :type vectors: List[np.ndarray]
    :param ids: List of IDs of the images. It will be used as a dataset name.
        Order of IDs should match the order of vectors.
    :type ids: List[int]
    :param project_id: ID of the project. It will be used as a file name.
    :type project_id: int
    """
    file_path = os.path.join(g.HDF5_DIR, f"{project_id}.hdf5")

    with h5py.File(file_path, "a") as file:
        # Using Dataset ID as a group name.
        # Require method works as get or create, to speed up the process.
        # If group already exists, it will be returned, otherwise created.
        group = file.require_group(str(dataset_id))
        for vector, id in zip(vectors, ids):
            # For each vector, we create a dataset with ID which comes from image ID.
            # Same, require method works as get or create.
            # So if vector for image with specific ID already exists, it will be overwritten.
            group.require_dataset(
                str(id), data=vector, shape=vector.shape, dtype=vector.dtype
            )


@timer
async def get_vectors(image_urls: List[str]) -> List[np.ndarray]:
    """Use CAS to get vectors from the list of images.

    :param image_urls: List of URLs to get vectors from.
    :type image_urls: List[str]
    :return: List of vectors.
    :rtype: List[np.ndarray]
    """
    vectors = await cas.client.aencode(image_urls)
    return [vector.tolist() for vector in vectors]


@timer
def resize_nps(nps: List[np.ndarray], size: int) -> List[np.ndarray]:
    """Resize list of numpy arrays to the square images with alpha channel.

    :param nps: List of numpy arrays to resize.
    :type nps: List[np.ndarray]
    :param size: Size of the square image.
    :type size: int
    :return: List of resized images with alpha channel.
    :rtype: List[np.ndarray]
    """
    return [_resize_to_rgba(np_image, size) for np_image in nps]


def _resize_to_rgba(rgb: np.ndarray, size: int) -> np.ndarray:
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


@timer
def get_image_infos(
    api: sly.Api, size: int, dataset_id: int = None, image_ids: List[int] = None
) -> List[ImageInfoLite]:
    """Returns lite version of image infos to cut off unnecessary data.
    Uses either dataset_id or image_ids to get image infos.
    If dataset_id is provided, it will be used to get all images from the dataset.
    If image_ids are provided, they will be used to get image infos.

    :param api: Instance of supervisely API.
    :type api: sly.Api
    :param size: Size of the images to resize.
    :type size: int
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
            url=api.image.resize_image_url(
                image_info.full_storage_url,
                method="fit",
                width=size,
                height=size,
            ),
            updated_at=image_info.updated_at,
        )
        for image_info in image_infos
    ]
