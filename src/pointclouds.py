import asyncio
import os
from typing import Dict, List, Union

import numpy as np
import supervisely as sly
import umap
from pypcd4 import Encoding, PointCloud

import src.globals as g
import src.qdrant as qdrant
from src.utils import PointCloudTileInfo, timer


@timer
def get_pointcloud(
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


@timer
async def create_pointcloud(
    project_id: int, vector_map: List[PointCloudTileInfo]
) -> None:
    umap_vectors = await asyncio.to_thread(
        create_umap, [entry.vector for entry in vector_map]
    )

    # Step 9: Create point cloud and save it to the PCD file.
    cloud = await asyncio.to_thread(get_pointcloud, umap_vectors, vector_map)
    project_atlas_dir = os.path.join(g.ATLAS_DIR, str(project_id))
    sly.fs.mkdir(project_atlas_dir)
    cloud.save(
        os.path.join(project_atlas_dir, f"{project_id}.pcd"),
        encoding=Encoding.BINARY_COMPRESSED,
    )

    if sly.is_development():
        # Debugging information about the point cloud. Will be visible in the development mode.
        cloud = PointCloud.from_path(os.path.join(g.ATLAS_DIR, f"{project_id}.pcd"))
        sly.logger.debug(f"Number of points in the cloud: {cloud.points}")
        sly.logger.debug(f"Fields in the PCD file: {cloud.fields}")
