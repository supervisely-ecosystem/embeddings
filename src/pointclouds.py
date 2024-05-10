import os
from typing import List

import numpy as np
import supervisely as sly
import umap
from pypcd4 import Encoding, PointCloud

import src.globals as g
import src.qdrant as qdrant
from src.utils import PointCloudTileInfo, timer, to_thread


@to_thread
@timer
def get_pointcloud(
    umap_vectors: np.ndarray, tile_infos: List[PointCloudTileInfo]
) -> PointCloud:
    custom_fields = [field for field in PointCloudTileInfo._fields if field != "vector"]
    return PointCloud.from_points(
        points=_add_metadata_to_umap(umap_vectors, tile_infos),
        fields=g.DEFAULT_PCD_FIELDS + custom_fields,
        types=[np.float32, np.float32, np.float32, np.int32, np.int32, np.int32],
    )


@timer
def _add_metadata_to_umap(
    umap_vectors: np.ndarray, tile_infos: List[PointCloudTileInfo]
) -> np.ndarray:
    with_metadata = np.hstack(
        (
            umap_vectors,
            np.array(
                [
                    [tile_info.atlasId, tile_info.atlasIndex, tile_info.imageId]
                    for tile_info in tile_infos
                ]
            ),
        )
    )
    sly.logger.debug(f"UMAP vectors with metadata shape: {with_metadata.shape}")
    return with_metadata


@to_thread
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
async def save_pointcloud(
    project_id: int, tile_infos: List[PointCloudTileInfo]
) -> None:
    umap_vectors = await create_umap(tile_info.vector for tile_info in tile_infos)

    cloud = await get_pointcloud(umap_vectors, tile_infos)
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
