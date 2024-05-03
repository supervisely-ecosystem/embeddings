import asyncio
import json
import os
from typing import Dict, List, Union

import aiohttp
import cv2
import h5py
import numpy as np
import supervisely as sly
import umap
from fastapi import Request
from pympler import asizeof
from pypcd4 import Encoding, PointCloud

import src.atlas as atlas
import src.cas as cas
import src.globals as g
import src.qdrant as qdrant
from src.utils import ImageInfoLite, PointCloudTileInfo, timer

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

    # Step 2: Get number of tiles in the atlas.
    num_tiles = atlas.tiles_in_atlas(g.ATLAS_SIZE, 512)
    sly.logger.debug(f"Full atlas should be containing {num_tiles} tiles.")

    # Step 2.1: Prepare empty vector and atlas maps.
    atlas_map = []
    vector_map = []

    # Step 3: Get tiles from the HDF5 file and save them into separate atlas images.
    for idx, tiles in enumerate(
        atlas.get_tiles_from_hdf5(get_hdf5_path(project_id), num_tiles)
    ):
        sly.logger.debug(
            f"Processing {idx + 1} batch of tiles. Received {len(tiles)} tiles."
        )

        page_image, page_map = await atlas.save_atlas(g.ATLAS_SIZE, 512, tiles)
        vector_map.extend(await get_vector_map(idx, project_id, page_map))
        atlas_map.extend(page_map)

        if sly.is_development():
            vector_map_size = asizeof.asizeof(vector_map) / 1024 / 1024
            sly.logger.debug(f"Vector map size: {vector_map_size:.2f} MB")

        # Convert RGBA to BGRA for compatibility with OpenCV.
        page_image = cv2.cvtColor(page_image, cv2.COLOR_RGBA2BGRA)

        # Save atlas image to the file.
        cv2.imwrite(os.path.join(g.ATLAS_DIR, f"{project_id}_{idx}.png"), page_image)

    json.dump(
        atlas_map,
        open(os.path.join(g.ATLAS_DIR, f"{project_id}.json"), "w"),
        indent=4,
    )

    umap_vectors = await asyncio.to_thread(
        create_umap, [entry.vector for entry in vector_map]
    )
    cloud = await asyncio.to_thread(create_point_cloud, umap_vectors, vector_map)
    cloud.save(
        os.path.join(g.ATLAS_DIR, f"{project_id}.pcd"),
        encoding=Encoding.BINARY_COMPRESSED,
    )

    if sly.is_development():
        cloud = PointCloud.from_path(os.path.join(g.ATLAS_DIR, f"{project_id}.pcd"))
        sly.logger.debug(f"Number of points in the cloud: {cloud.points}")
        sly.logger.debug(f"Fields in the PCD file: {cloud.fields}")

    sly.logger.info(f"Atlas for project {project_id} has been created.")


@timer
def create_point_cloud(
    umap_vectors: np.ndarray, vector_map: List[PointCloudTileInfo]
) -> PointCloud:
    custom_fields = [field for field in PointCloudTileInfo._fields if field != "vector"]
    return PointCloud.from_points(
        points=_add_metadata_to_umap(umap_vectors, vector_map),
        fields=g.DEFAULT_PCD_FIELDS + custom_fields,
        types=[np.float32, np.float32, np.float32, np.int32, np.int32, np.int32],
    )


@timer
def _add_metadata_to_umap(
    umap_vectors: np.ndarray, vector_map: List[PointCloudTileInfo]
) -> np.ndarray:
    with_metadata = np.hstack(
        (
            umap_vectors,
            np.array(
                [
                    [entry.atlasId, entry.atlasIndex, entry.imageId]
                    for entry in vector_map
                ]
            ),
        )
    )
    sly.logger.debug(f"UMAP vectors with metadata shape: {with_metadata.shape}")
    return with_metadata


@timer
def create_umap(vectors: List[np.ndarray], n_components: int = 3) -> np.ndarray:
    reducer = umap.UMAP(n_components=n_components)
    umap_vectors = reducer.fit_transform(vectors)
    sly.logger.debug(f"UMAP vectors shape: {umap_vectors.shape}")
    return umap_vectors


@timer
async def get_vector_map(
    atlas_id: int, project_id: int, atlas_map: List[Dict[str, Union[str, int]]]
) -> List[PointCloudTileInfo]:
    vector_map = []
    image_ids = []
    for entry in atlas_map:
        image_id = entry.pop("image_id")
        vector_map.append(
            {
                "image_id": image_id,
                "id": entry.get("id"),
            }
        )
        image_ids.append(image_id)

    vectors = await qdrant.get_vectors(project_id, image_ids)

    return [
        PointCloudTileInfo(
            atlasId=atlas_id,
            atlasIndex=entry.get("id"),
            imageId=entry.get("image_id"),
            vector=vector,
        )
        for entry, vector in zip(vector_map, vectors)
    ]


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
        sly.logger.debug(f"Force is True, deleting collection {project_id}.")
        await qdrant.delete_collection(project_id)

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

    # Step 4.1: Get diff of image infos, check if they are already
    # in the Qdrant collection and have the same updated_at field.
    diff_image_infos = await qdrant.get_diff(project_id, image_infos)

    # Step 5: Batch image urls.
    for image_batch in sly.batched(diff_image_infos):
        # url_batch = [image_info.url for image_info in image_batch]
        # ids_batch = [image_info.id for image_info in image_batch]

        # Step 6: Download images as numpy arrays using URLs resized
        # for for specified HDF5 sizes (for thumbnails in the atlas).
        nps_batch = await download_items(
            [image_info.hdf5_url for image_info in image_batch]
        )

        # Step 6.1: Convert tiles into square images with alpha channel.
        nps_batch = rgb_to_rgba(nps_batch, g.IMAGE_SIZE_FOR_ATLAS)

        # Step 7: Save images to hdf5.
        save_to_hdf5(nps_batch, image_batch, project_id)

        # Step 8: Get vectors from images.
        vectors_batch = await get_vectors(
            [image_info.cas_url for image_info in image_batch]
        )

        # Step 9: Upsert vectors to Qdrant.
        await qdrant.upsert(
            project_id,
            vectors_batch,
            [image_info.id for image_info in image_batch],
            updated_at=[image_info.updated_at for image_info in image_batch],
        )


@timer
def save_to_hdf5(
    image_nps: List[np.ndarray],
    image_infos: List[ImageInfoLite],
    project_id: int,
):
    """Save thumbnail numpy array data to the HDF5 file.

    :param image_np: List of image_np to save.
    :type image_np: List[np.ndarray]
    :param image_infos: List of image infos to get image IDs.
        Image ID will be used as a dataset name. Length of this list should be the same as image_np.
    :type image_infos: List[ImageInfoLite]
    :param project_id: ID of the project. It will be used as a file name.
    :type project_id: int
    """

    with h5py.File(get_hdf5_path(project_id), "a") as file:
        for image_np, image_info in zip(image_nps, image_infos):
            # Using Dataset ID as a group name.
            # Require method works as get or create, to speed up the process.
            # If group already exists, it will be returned, otherwise created.
            group = file.require_group(str(image_info.dataset_id))
            # For each numpy array, we create a dataset with ID which comes from image ID.
            # Same, require method works as get or create.
            # So if numpy_array for image with specific ID already exists, it will be overwritten.
            dataset = group.require_dataset(
                str(image_info.id),
                data=image_np,
                shape=image_np.shape,
                dtype=image_np.dtype,
            )

            dataset.attrs[g.HDF5_URL_KEY] = image_info.full_url


def get_hdf5_path(project_id: int):
    return os.path.join(g.HDF5_DIR, f"{project_id}.hdf5")


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
            cas_url=api.image.resize_image_url(
                image_info.full_storage_url,
                method="fit",
                width=cas_size,
                height=cas_size,
            ),
            hdf5_url=api.image.resize_image_url(
                image_info.full_storage_url,
                method="fit",
                width=hdf5_size,
                height=hdf5_size,
            ),
            updated_at=image_info.updated_at,
        )
        for image_info in image_infos
    ]
