import asyncio
import os
from typing import List

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
async def get_tile_infos(
    atlas_id: int, project_id: int, image_ids: List[int]
) -> List[PointCloudTileInfo]:
    image_infos, vectors = await qdrant.get_items_by_ids(
        project_id, image_ids, with_vectors=True
    )

    return [
        PointCloudTileInfo(
            atlasId=atlas_id,
            atlasIndex=idx,
            imageId=image_info.id,
            vector=vector,
        )
        for idx, (image_info, vector) in enumerate(zip(image_infos, vectors))
    ]


@timer
async def create_pointcloud(
    project_id: int, vector_map: List[PointCloudTileInfo]
) -> None:
    umap_vectors = await asyncio.to_thread(
        create_umap, [entry.vector for entry in vector_map]
    )

    cloud = await asyncio.to_thread(get_pointcloud, umap_vectors, vector_map)
    project_atlas_dir = os.path.join(g.ATLAS_DIR, str(project_id))
    sly.fs.mkdir(project_atlas_dir)
    cloud.save(
        os.path.join(project_atlas_dir, f"{project_id}.pcd"),
        encoding=Encoding.BINARY_COMPRESSED,
    )

    if sly.is_development():
        # Debugging information about the point cloud. Will be visible in the development mode.
        cloud = PointCloud.from_path(
            os.path.join(project_atlas_dir, f"{project_id}.pcd")
        )
        sly.logger.debug(f"Number of points in the cloud: {cloud.points}")
        sly.logger.debug(f"Fields in the PCD file: {cloud.fields}")
        print(cloud.numpy())
