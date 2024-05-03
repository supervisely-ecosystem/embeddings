import asyncio
import math
import os
from typing import Dict, Generator, List, Tuple, Union

import cv2
import h5py
import numpy as np
import supervisely as sly
import umap
from pypcd4 import Encoding, PointCloud

import src.globals as g
import src.qdrant as qdrant
from src.utils import PointCloudTileInfo, TileInfo, timer


def tiles_in_atlas(atlas_size: int, tile_size: int) -> int:
    """Calculates the number of tiles that can fit in the atlas of given size.

    :param atlas_size: size of the atlas
    :type atlas_size: int
    :param tile_size: size of the tile
    :type tile_size: int
    :return: number of tiles that can fit in the atlas
    :rtype: int
    """
    return (atlas_size // tile_size) ** 2


@timer
async def save_atlas(
    atlas_size: int, tile_size: int, tile_infos: List[TileInfo]
) -> Tuple[np.ndarray, List[Dict[str, Union[str, int]]]]:
    # Receives a list of 4-channel RGBA numpy arrays and returns a single 4-channel RGBA numpy array.
    # Tiles of the atlas are filled row by row from the top left corner.
    # Result height of the atlas depends on the number of given image_nps (will be less than atlas_size if needed).
    num_tiles = atlas_size // tile_size
    sly.logger.debug(f"Full atlas should be {num_tiles}x{num_tiles} tiles.")
    rows = math.ceil(len(tile_infos) / num_tiles)
    sly.logger.debug(
        f"Given amount ({len(tile_infos)}) of images will fill {rows} rows."
    )

    # Create an empty atlas with rows number of rows
    atlas = np.zeros((rows * tile_size, atlas_size, 4), dtype=np.uint8)
    sly.logger.debug(f"Created an empty atlas with shape {atlas.shape}.")
    atlas_map = []

    # Iterate over the thumbnails and fill the atlas, adding metadata to the atlas_map.
    for idx, tile_info in enumerate(tile_infos):
        # Resize the thumbnail if it's not the same size as the tile_size.
        if tile_info.thumbnail.shape[0] != tile_size:
            thumbnail = resize_np(tile_info.thumbnail, tile_size)

        row = idx // num_tiles
        col = idx % num_tiles
        atlas[
            row * tile_size : (row + 1) * tile_size,
            col * tile_size : (col + 1) * tile_size,
        ] = thumbnail

        atlas_map.append(
            {
                "id": idx,
                "unitSize": tile_info.unitSize,
                "url": tile_info.url,
                "image_id": tile_info.id,
            }
        )

    return atlas, atlas_map


def resize_np(image_np: np.ndarray, size: int) -> np.ndarray:
    """Resize the given numpy array to the given size.

    :param image_np: numpy array to resize
    :type image_np: np.ndarray
    :param size: size to resize the numpy array to
    :type size: int
    :return: resized numpy array
    :rtype: np.ndarray
    """
    return cv2.resize(image_np, (size, size), interpolation=cv2.INTER_AREA)


@timer
def get_tiles_from_hdf5(
    hdf5_path: str, batch_size: int
) -> Generator[List[TileInfo], None, None]:
    """Reads HDF5 file from given path and returns a generator of numpy arrays
    split into batches of given size. At the end yields the remaining items in the batch

    :param hdf5_path: path to the HDF5 file
    :type hdf5_path: str
    :param batch_size: size of the batch
    :type batch_size: int
    :return: generator of TileInfo objects
    :rtype: Generator[List[TileInfo], None, None]
    """

    # Open the HDF5 file
    with h5py.File(hdf5_path, "r") as f:
        # Initialize a list to hold the batch of items
        batch = []
        # Iterate over groups in the HDF5 file
        for group in f.values():
            # Iterate over datasets in the group
            for dataset in group.values():
                # Add the dataset to the batch
                thumbnail = np.array(dataset)
                batch.append(
                    TileInfo(
                        id=_image_id_from_dataset(dataset.name),
                        unitSize=thumbnail.shape[0],
                        url=dataset.attrs.get(g.HDF5_URL_KEY),
                        thumbnail=thumbnail,
                    )
                )
                # If the batch has reached the desired size, yield it
                if len(batch) == batch_size:
                    yield batch
                    # Start a new batch
                    batch = []
        # If there are any remaining items in the batch, yield them
        if batch:
            yield batch


def _image_id_from_dataset(dataset_name: str) -> int:
    """Extracts the image_id from the dataset name.
    Example: "/0022233/123456", where "123456" is the image_id.

    :param dataset_name: name of the dataset
    :type dataset_name: str
    :return: image_id
    :rtype: int
    """
    return int(dataset_name.split("/")[-1])


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
    cloud.save(
        os.path.join(g.ATLAS_DIR, f"{project_id}.pcd"),
        encoding=Encoding.BINARY_COMPRESSED,
    )

    if sly.is_development():
        # Debugging information about the point cloud. Will be visible in the development mode.
        cloud = PointCloud.from_path(os.path.join(g.ATLAS_DIR, f"{project_id}.pcd"))
        sly.logger.debug(f"Number of points in the cloud: {cloud.points}")
        sly.logger.debug(f"Fields in the PCD file: {cloud.fields}")
