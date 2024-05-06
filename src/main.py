import asyncio
import json
import os
from typing import Dict, List, Tuple

import aiohttp
import cv2
import numpy as np
import supervisely as sly
from fastapi import Request
from pympler import asizeof
from supervisely._utils import resize_image_url

import src.atlas as atlas
import src.cas as cas
import src.globals as g
import src.pointclouds as pointclouds
import src.qdrant as qdrant
import src.thumbnails as thumbnails
from src.utils import ImageInfoLite, rgb_to_rgba, timer

app = sly.Application()
server = app.get_server()


@server.post("/create_atlas")
@timer
async def create_atlas(request: Request):
    # ! DEBUG, will receive instance of API in request later.
    # api = g.api
    # end of debug

    # Step 1: Unpack data from the request.
    context = request.state.context
    project_id = context.get("project_id")
    sly.logger.info(f"Creating atlas for project {project_id}...")

    # Step 2: Prepare the project directory.
    project_atlas_dir = os.path.join(g.ATLAS_DIR, str(project_id))
    sly.fs.mkdir(project_atlas_dir, remove_content_if_exists=True)

    # Step 3: Save atlas pages and prepare vector and atlas maps.
    atlas_map, vector_map = await process_atlas(project_id, project_atlas_dir)

    # Step 4: Save atlas map to the JSON file.
    json.dump(
        atlas_map,
        open(os.path.join(project_atlas_dir, f"{project_id}.json"), "w"),
        indent=4,
    )

    # Step 5: Create and save the vector map to the PCD file.
    await pointclouds.create_pointcloud(project_id, vector_map)

    sly.logger.info(f"Atlas for project {project_id} has been created.")


@server.post("/embeddings")
@timer
async def create_embeddings(request: Request):
    # ! DEBUG, will receive instance of API in request later.
    api = g.api
    # end of debug

    # Step 1: Unpack data from the request.
    context = request.state.context
    project_id = context.get("project_id")
    # If context contains list of image_ids it means that we're
    # updating embeddings for specific images, otherwise we're updating
    # the whole project.
    image_ids = context.get("image_ids")
    force = context.get("force")

    if force:
        # Step 1.1: If force is True, delete the collection and recreate it.
        sly.logger.debug(f"Force enabled, deleting collection {project_id}.")
        await qdrant.delete_collection(project_id)
        sly.logger.debug(f"Deleting HDF5 file for project {project_id}.")
        thumbnails.remove_hdf5(project_id)

    # Step 2: Ensure collection exists in Qdrant.
    await qdrant.get_or_create_collection(project_id)

    # Step 3: Process images.
    if not image_ids:
        # Step 3A: If image_ids are not provided, get all datasets from the project.
        # Then iterate over datasets and process images from each dataset.
        datasets = await asyncio.to_thread(get_datasets, api, project_id)
        for dataset in datasets:
            await process_images(api, project_id, dataset_id=dataset.id)
    else:
        # Step 3B: If image_ids are provided, process images with specific IDs.
        await process_images(api, project_id, image_ids=image_ids)


@timer
async def process_atlas(
    project_id: int, project_atlas_dir: str
) -> Tuple[List[Dict], List[Dict]]:
    atlas_map = []
    vector_map = []

    for idx, tiles in enumerate(thumbnails.get_tiles_from_hdf5(project_id=project_id)):
        sly.logger.debug(
            f"Processing {idx + 1} batch of tiles. Received {len(tiles)} tiles."
        )

        # Get atlas image and map. Image is a numpy array, map is a list of dictionaries.
        page_image, page_map = await atlas.save_atlas(
            g.ATLAS_SIZE, g.IMAGE_SIZE_FOR_ATLAS, tiles
        )

        # Add page elements to the vector and atlas maps, which will be saved to the files
        # without splitting into pages.
        vector_map.extend(await pointclouds.get_vector_map(idx, project_id, page_map))
        atlas_map.extend(page_map)

        if sly.is_development():
            # Checking how much memory the vector map takes.
            vector_map_size = asizeof.asizeof(vector_map) / 1024 / 1024
            sly.logger.debug(f"Vector map size: {vector_map_size:.2f} MB")
        # Convert RGBA to BGRA for compatibility with OpenCV.
        page_image = cv2.cvtColor(page_image, cv2.COLOR_RGBA2BGRA)
        # Save atlas image to the file.
        cv2.imwrite(
            os.path.join(project_atlas_dir, f"{project_id}_{idx}.png"), page_image
        )

    return atlas_map, vector_map


@timer
async def process_images(
    api: sly.Api, project_id: int, dataset_id: int = None, image_ids: List[int] = None
):
    image_infos = await asyncio.to_thread(
        get_image_infos,
        api,
        g.IMAGE_SIZE_FOR_CAS,
        g.IMAGE_SIZE_FOR_ATLAS,
        dataset_id=dataset_id,
        image_ids=image_ids,
    )

    # Get diff of image infos, check if they are already
    # in the Qdrant collection and have the same updated_at field.
    diff_image_infos = await qdrant.get_diff(project_id, image_infos)

    # Batch image urls.
    for image_batch in sly.batched(diff_image_infos):
        # url_batch = [image_info.url for image_info in image_batch]
        # ids_batch = [image_info.id for image_info in image_batch]

        # Download images as numpy arrays using URLs resized
        # for for specified HDF5 sizes (for thumbnails in the atlas).
        nps_batch = await download_items(
            [image_info.hdf5_url for image_info in image_batch]
        )

        # Convert tiles into square images with alpha channel.
        nps_batch = rgb_to_rgba(nps_batch, g.IMAGE_SIZE_FOR_ATLAS)

        # Save images to hdf5.
        thumbnails.save_to_hdf5(nps_batch, image_batch, project_id)

        # Get vectors from images.
        vectors_batch = await cas.get_vectors(
            [image_info.cas_url for image_info in image_batch]
        )

        # Upsert vectors to Qdrant.
        await qdrant.upsert(
            project_id,
            vectors_batch,
            [image_info.id for image_info in image_batch],
            updated_at=[image_info.updated_at for image_info in image_batch],
        )


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
