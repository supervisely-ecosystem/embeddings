from math import sqrt
from random import choice
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import supervisely as sly
from pympler import asizeof
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Batch, CollectionInfo, Distance, VectorParams
from sklearn.cluster import KMeans

import src.globals as g
from src.utils import ImageInfoLite, timer, with_retries

# TODO: Return client on demand.
client = AsyncQdrantClient(g.qdrant_host)

try:
    QdrantClient(g.qdrant_host).get_collections()
    sly.logger.debug(f"Connected to Qdrant at {g.qdrant_host}")
except Exception as e:
    sly.logger.error(f"Failed to connect to Qdrant at {g.qdrant_host} with error: {e}")


@with_retries()
async def delete_collection(collection_name: str):
    sly.logger.debug(f"Deleting collection {collection_name}...")
    try:
        await client.delete_collection(collection_name)
        sly.logger.debug(f"Collection {collection_name} deleted.")
    except UnexpectedResponse:
        sly.logger.debug(f"Collection {collection_name} wasn't found while deleting.")


@with_retries()
@timer
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


@with_retries()
@timer
async def get_vectors(collection_name: str, image_ids: List[int]) -> List[np.ndarray]:
    points = await client.retrieve(
        collection_name, image_ids, with_payload=False, with_vectors=True
    )
    return [point.vector for point in points]


@with_retries(retries=5, sleep_time=2)
@timer
async def upsert(
    collection_name: str,
    vectors: List[np.ndarray],
    image_infos: List[ImageInfoLite],
):
    await client.upsert(
        collection_name,
        Batch(
            vectors=vectors,
            ids=[image_info.id for image_info in image_infos],
            payloads=get_payloads(image_infos),
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


@with_retries()
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

    diff = _diff(image_infos, points)

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
def _diff(
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
def get_payloads(image_infos: List[ImageInfoLite]) -> List[Dict[str, Any]]:
    payloads = [
        {k: v for k, v in image_info._asdict().items() if k != "id"}
        for image_info in image_infos
    ]
    return payloads


@with_retries()
@timer
async def search(
    collection_name: str, query_vector: np.ndarray, limit: int
) -> List[ImageInfoLite]:
    points = await client.search(
        collection_name,
        query_vector,
        limit=limit,
        with_payload=True,
    )
    return [ImageInfoLite(point.id, **point.payload) for point in points]


@with_retries()
@timer
async def get_items(
    collection_name: str, limit: int = None
) -> Tuple[List[ImageInfoLite], List[np.ndarray]]:
    if not limit:
        collection = await client.get_collection(collection_name)
        limit = collection.points_count
    points, _ = await client.scroll(
        collection_name, limit=limit, with_payload=True, with_vectors=True
    )
    sly.logger.debug(
        f"Retrieved {len(points)} points from collection {collection_name}."
    )
    return [ImageInfoLite(point.id, **point.payload) for point in points], [
        point.vector for point in points
    ]


@timer
async def diverse(
    collection_name: str, num_images: int, method: str, **kwargs
) -> List[ImageInfoLite]:
    if method == "kmeans":
        num_clusters = kwargs.get("num_clusters")
        option = kwargs.get("option")
        return await diverse_kmeans(collection_name, num_images, num_clusters, option)
    elif method == "fps":
        initial_vector = kwargs.get("initial_vector")
        return await diverse_fps(collection_name, num_images, initial_vector)


@timer
async def diverse_kmeans(
    collection_name: str,
    num_images: int,
    num_clusters: int = None,
    option: Literal["random", "centroids"] = None,
) -> List[ImageInfoLite]:
    """Generate a diverse population of images using KMeans clustering.
    Two options are available: "random" and "centroids".
    The "random" option chooses a random image from each cluster.
    The "centroids" option chooses the image closest to the centroid of each cluster.

    :param collection_name: The name of the collection to get items from.
    :type collection_name: str
    :param num_images: The number of diverse images to generate.
    :type num_images: int
    :param num_clusters: The number of clusters to use in KMeans clustering.
    :type num_clusters: int
    :param option: The option to use for choosing images from clusters, defaults to None.
    :type option: Literal["random", "centroids"], optional
    :return: A list of diverse images as ImageInfoLite objects.
    :rtype: List[ImageInfoLite]
    """
    image_infos, vectors = await get_items(collection_name)
    if sly.is_development():
        vectors_size = asizeof.asizeof(vectors) / 1024 / 1024
        sly.logger.debug(f"Vectors size: {vectors_size:.2f} MB.")
    if not num_clusters:
        num_clusters = int(sqrt(len(image_infos) / 2))
        sly.logger.debug(f"Number of clusters is set to {num_clusters}.")
    if not option:
        option = "random"
        sly.logger.debug(f"Option is set to {option}.")
    kmeans = KMeans(n_clusters=num_clusters).fit(vectors)
    labels = kmeans.labels_
    sly.logger.debug(
        f"KMeans clustering with {num_clusters} and {option} option is done."
    )

    diverse_images = []
    while len(diverse_images) < num_images:
        for cluster_id in set(labels):
            cluster_image_infos = [
                image_info
                for image_info, label in zip(image_infos, labels)
                if label == cluster_id
            ]
            if not cluster_image_infos:
                continue

            if option == "random":
                # Randomly choose an image from the cluster.
                image_info = choice(cluster_image_infos)
            elif option == "centroids":
                # Choose the image closest to the centroid of the cluster.
                cluster_vectors = [
                    vector
                    for vector, label in zip(vectors, labels)
                    if label == cluster_id
                ]
                centroid = np.mean(cluster_vectors, axis=0)
                distances = [
                    np.linalg.norm(vector - centroid) for vector in cluster_vectors
                ]
                image_info = cluster_image_infos[distances.index(min(distances))]
            diverse_images.append(image_info)
            image_infos.remove(image_info)
            if len(diverse_images) == num_images:
                break

    return diverse_images


@timer
async def get_single_point(
    collection_name: str,
    vector: np.ndarray,
    limit: int,
    option: Literal["farthest", "closest", "random"] = "farthest",
) -> Tuple[ImageInfoLite, np.ndarray]:
    raise NotImplementedError("This function is not working properly.")

    # TODO: Refactor and move search to a separate function with retries.
    points = await client.search(
        collection_name, vector, limit=limit, with_payload=True, with_vectors=True
    )

    if option == "farthest":
        point = points[-1]
    elif option == "closest":
        point = points[0]
    elif option == "random":
        point = choice(points)

    return ImageInfoLite(point.id, **point.payload), point.vector


@timer
async def diverse_fps(
    collection_name: str,
    num_images: int,
    initial_vector: np.ndarray = None,
) -> List[ImageInfoLite]:
    """Generate a diverse population of images using Farthest Point Sampling (FPS).

    :param collection_name: The name of the collection to get items from.
    :type collection_name: str
    :param num_images: The number of diverse images to generate.
    :type num_images: int
    :param initial_vector: The initial vector to start the sampling from, defaults to None.
    :type initial_vector: np.ndarray, optional
    :return: A list of diverse images as ImageInfoLite objects.
    :rtype: List[ImageInfoLite]
    """
    collection = await client.get_collection(collection_name)

    _, vectors = await get_items(collection_name, 1)
    vector_len = len(vectors[0])
    if not initial_vector:
        query_vector = np.random.rand(1, vector_len).flatten()
    else:
        if not len(initial_vector) == vector_len:
            raise ValueError(
                f"The length of the initial vector should be {vector_len}."
            )
        query_vector = initial_vector

    diverse_images = []
    while len(diverse_images) < num_images:
        image_info, vector = await get_single_point(
            collection_name,
            query_vector,
            limit=collection.points_count,
        )
        diverse_images.append(image_info)
        query_vector = vector

    return diverse_images
