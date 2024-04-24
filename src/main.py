import asyncio
from collections import namedtuple
from functools import wraps
from time import perf_counter
from typing import List

import aiohttp
import numpy as np
import supervisely as sly
from fastapi import Request
from supervisely._utils import compress_image_url

import src.cas as cas
import src.globals as g
import src.qdrant as qdrant

app = sly.Application()
server = app.get_server()

ImageInfoLite = namedtuple("ImageInfoLite", ["id", "url", "updated_at"])
WIDTH = 32
HEIGHT = 32


def timer(func):
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = await func(*args, **kwargs)
            end_time = perf_counter()
            print(
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
            print(
                f"Function {func.__name__} executed in {end_time - start_time} seconds"
            )
            return result

        return sync_wrapper


@server.post("/test")
@timer
async def test(request: Request):
    # Step 1: Unpack request, get context and instance of API.
    # * for debug using api from globals module
    api = g.api
    context = request.state.context
    project_id = context["project_id"]
    collection_name = str(project_id)
    await qdrant.get_or_create_collection(collection_name)

    # Step 2: Get datasets from project.
    datasets = await asyncio.to_thread(get_datasets, api, project_id)

    # Step 3: Iterate over datasets.
    for dataset in datasets:
        # Step 4: Get comressed image urls from dataset.
        image_infos = await asyncio.to_thread(get_image_infos, api, dataset.id)

        # Step 5: Batch image urls.
        for image_batch in sly.batched(image_infos):
            url_batch = [image_info.url for image_info in image_batch]

            # Step 6: Download images as numpy arrays.
            # nps_batch = await download_items(url_batch)
            # Use later for other tasks.

            # Step 7: Get vectors from images.
            vectors_batch = await get_vectors(url_batch)
            ids_batch = [image_info.id for image_info in image_batch]

            # Step 8: Upsert vectors to Qdrant.
            await qdrant.upsert(collection_name, vectors_batch, ids_batch)


async def download_item(session: aiohttp.ClientSession, url: str) -> np.ndarray:
    async with session.get(url) as response:
        image_bytes = await response.read()

    return sly.image.read_bytes(image_bytes)


@timer
async def get_vectors(image_urls: List[str]) -> List[np.ndarray]:
    vectors = await cas.client.aencode(image_urls)
    return [vector.tolist() for vector in vectors]


@timer
async def download_items(urls: List[str]) -> List[np.ndarray]:
    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*[download_item(session, url) for url in urls])


@timer
def get_datasets(api: sly.Api, project_id: int) -> List[sly.DatasetInfo]:
    return api.dataset.get_list(project_id)


@timer
def get_image_infos(api: sly.Api, dataset_id: int) -> List[ImageInfoLite]:
    image_infos = api.image.get_list(dataset_id)
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
