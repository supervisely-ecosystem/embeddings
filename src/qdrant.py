from typing import List

import numpy as np
import supervisely as sly
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Batch, CollectionInfo, Distance, VectorParams

import src.globals as g

client = AsyncQdrantClient(g.qdrant_host)

try:
    QdrantClient(g.qdrant_host).get_collections()
    sly.logger.debug(f"Connected to Qdrant at {g.qdrant_host}")
except Exception as e:
    sly.logger.error(f"Failed to connect to Qdrant at {g.qdrant_host} with error: {e}")


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


async def upsert(collection_name: str, vectors: List[np.ndarray], ids: List[int]):
    await client.upsert(
        collection_name,
        Batch(
            vectors=vectors,
            ids=ids,
        ),
    )
    sly.logger.debug(
        f"Upserted {len(vectors)} vectors to collection {collection_name}."
    )
