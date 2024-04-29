import asyncio
import os
from typing import List

# import aiohttp
import h5py
import numpy as np
import supervisely as sly
from fastapi import Request
from supervisely._utils import compress_image_url

import src.cas as cas
import src.globals as g
import src.qdrant as qdrant
from src.utils import ImageInfoLite, timer

app = sly.Application()
server = app.get_server()


WIDTH = 32
HEIGHT = 32


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
        all_image_infos = await asyncio.to_thread(get_image_infos, api, dataset.id)

        # Step 4.1: Get diff of image infos, check if they are already
        # in the Qdrant collection and have the same updated_at field.
        image_infos = await qdrant.get_diff(project_id, all_image_infos)

        # Step 5: Batch image urls.
        for image_batch in sly.batched(image_infos):
            url_batch = [image_info.url for image_info in image_batch]
            ids_batch = [image_info.id for image_info in image_batch]

            # Step 6: Download images as numpy arrays.
            # nps_batch = await download_items(url_batch)
            # ! DEBUG FUNCTION, REPLACE WITH COMMENTED ABOVE.
            nps_batch = await asyncio.to_thread(
                download_items,
                dataset.id,
                ids_batch,
            )

            # ! Resize won't be needed with correct download function.
            # Step 6.1: Resize images.
            nps_batch = resize_nps(nps_batch)

            # Step 7: Save images to hdf5.
            save_to_hdf5(dataset.id, nps_batch, ids_batch, project_id)

            # Step 8: Get vectors from images.
            vectors_batch = await get_vectors(url_batch)
            updated_at_batch = [image_info.updated_at for image_info in image_batch]

            # Step 9: Upsert vectors to Qdrant.
            await qdrant.upsert(
                project_id, vectors_batch, ids_batch, updated_at=updated_at_batch
            )


# async def download_item(session: aiohttp.ClientSession, url: str) -> np.ndarray:
#     async with session.get(url) as response:
#         image_bytes = await response.read()

#     return sly.image.read_bytes(image_bytes)


# * ADD GROUPING BY DATASET_ID INTO HDF5 TO SPEED UP THE RANDOM ACCESS LATENCY!
@timer
def save_to_hdf5(
    dataset_id, vectors: List[np.ndarray], ids: List[int], project_id: int
):
    file_path = os.path.join(g.HDF5_DIR, f"{project_id}.hdf5")

    # Dataset ID will be used as a group name in hdf5 file.

    with h5py.File(file_path, "a") as file:
        group = file.require_group(str(dataset_id))
        for vector, id in zip(vectors, ids):
            group.require_dataset(
                str(id), data=vector, shape=vector.shape, dtype=vector.dtype
            )


@timer
async def get_vectors(image_urls: List[str]) -> List[np.ndarray]:
    vectors = await cas.client.aencode(image_urls)
    return [vector.tolist() for vector in vectors]


@timer
def resize_nps(nps: List[np.ndarray]) -> List[np.ndarray]:
    return [sly.image.resize(nps, out_size=(WIDTH, HEIGHT)) for nps in nps]


# ! DEBUG FUNCTION, REPLACE WITH COMMENTED ONE.
@timer
def download_items(dataset_id: int, ids: List[int]) -> List[np.ndarray]:
    api = g.api
    image_nps = api.image.download_nps(dataset_id, ids)
    return image_nps


# @timer
# async def download_items(urls: List[str]) -> List[np.ndarray]:
#     async with aiohttp.ClientSession() as session:
#         return await asyncio.gather(*[download_item(session, url) for url in urls])


@timer
def get_datasets(api: sly.Api, project_id: int) -> List[sly.DatasetInfo]:
    return api.dataset.get_list(project_id)


@timer
def get_image_infos(
    api: sly.Api, dataset_id: int = None, image_ids: List[int] = None
) -> List[ImageInfoLite]:
    if dataset_id:
        image_infos = api.image.get_list(dataset_id)
    elif image_ids:
        image_infos = api.image.get_info_by_id_batch(image_ids)
    elif dataset_id and image_ids:
        raise ValueError("Either dataset_id or image_ids should be provided.")
    elif not dataset_id and not image_ids:
        raise ValueError("Either dataset_id or image_ids should be provided.")
    return [
        ImageInfoLite(
            id=image_info.id,
            url=compress_image_url(
                image_info.full_storage_url, width=WIDTH, height=HEIGHT
            ),
            updated_at=image_info.updated_at,
        )
        for image_info in image_infos
    ]
