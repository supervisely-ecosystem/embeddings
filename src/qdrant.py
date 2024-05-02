from typing import Any, Dict, List

import numpy as np
import supervisely as sly
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Batch, CollectionInfo, Distance, VectorParams

import src.globals as g
from src.utils import ImageInfoLite, timer

client = AsyncQdrantClient(g.qdrant_host)

try:
    QdrantClient(g.qdrant_host).get_collections()
    sly.logger.debug(f"Connected to Qdrant at {g.qdrant_host}")
except Exception as e:
    sly.logger.error(f"Failed to connect to Qdrant at {g.qdrant_host} with error: {e}")


async def delete_collection(collection_name: str):
    sly.logger.debug(f"Deleting collection {collection_name}...")
    await client.delete_collection(collection_name)


async def get_or_create_collection(
    collection_name: str, size: int = 512, distance: Distance = Distance.COSINE
) -> CollectionInfo:
    try:
        collection = await client.get_collection(collection_name)
        sly.logger.debug(f"Collection {collection_name} already exists.")
    except UnexpectedResponse:
        await client.create_collection(
            collection_name,
            vectors_config=VectorParams(size=size, distance=distance),
        )
        sly.logger.debug(f"Collection {collection_name} created.")
        collection = await client.get_collection(collection_name)
    return collection


@timer
async def upsert(
    collection_name: str, vectors: List[np.ndarray], ids: List[int], **kwargs
):
    await client.upsert(
        collection_name,
        Batch(
            vectors=vectors,
            ids=ids,
            payloads=await get_payloads(kwargs),
        ),
    )
    if sly.is_development():
        # By default qdrant should overwrite vectors with the same ids
        # so this line is only needed to check if vectors were upserted correctly.
        # Do not use this in production since it will slow down the process.
        collecton_info = await client.get_collection(collection_name)
        sly.logger.debug(
            f"Collection {collection_name} has {collecton_info.points_count} vectors."
        )


@timer
async def get_diff(
    collection_name: str, image_infos: List[ImageInfoLite]
) -> List[ImageInfoLite]:
    # Get specified ids from collection, compare updated_at and return ids that need to be updated.
    points = await client.retrieve(
        collection_name,
        [image_info.id for image_info in image_infos],
        with_payload=True,
    )
    sly.logger.debug(
        f"Retrieved {len(points)} points from collection {collection_name}"
    )

    diff = await _diff(image_infos, points)

    sly.logger.debug(f"Found {len(diff)} points that need to be updated.")
    if sly.is_development():
        # To avoid unnecessary computations in production,
        # only log the percentage of points that need to be updated in development.
        percent = round(len(diff) / len(image_infos) * 100, 2)
        sly.logger.debug(
            f"From the total of {len(image_infos)} points, {len(diff)} points need to be updated. ({percent}%)"
        )

    return diff


@timer
async def _diff(
    image_infos: List[ImageInfoLite], points: List[Dict[str, Any]]
) -> List[ImageInfoLite]:
    # If the point with the same id doesn't exist in the collection, it will be added to the diff.
    # If the point with the same id exsts - check if updated_at is exactly the same, or add to the diff.
    # Image infos and points have different length, so we need to iterate over image infos.
    diff = []
    points_dict = {point.id: point for point in points}

    for image_info in image_infos:
        point = points_dict.get(image_info.id)
        if point is None or point.payload.get("updated_at") != image_info.updated_at:
            diff.append(image_info)

    return diff


@timer
async def get_payloads(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    POSSIBLE_KEYS = ["updated_at"]
    return [{key: entry} for key in POSSIBLE_KEYS if key in data for entry in data[key]]
