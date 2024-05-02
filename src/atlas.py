import math
from typing import Generator, List

import cv2
import h5py
import numpy as np
import supervisely as sly

from src.utils import timer


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
    atlas_size: int, tile_size: int, image_nps: List[np.ndarray]
) -> np.ndarray:
    # Receives a list of 4-channel RGBA numpy arrays and returns a single 4-channel RGBA numpy array.
    # Tiles of the atlas are filled row by row from the top left corner.
    # Result height of the atlas depends on the number of given image_nps (will be less than atlas_size if needed).

    num_tiles = atlas_size // tile_size
    sly.logger.debug(f"Full atlas should be {num_tiles}x{num_tiles} tiles.")
    rows = math.ceil(len(image_nps) / num_tiles)
    sly.logger.debug(
        f"Given amount ({len(image_nps)}) of images will fill {rows} rows."
    )

    # Create an empty atlas with rows number of rows
    atlas = np.zeros((rows * tile_size, atlas_size, 4), dtype=np.uint8)
    sly.logger.debug(f"Created an empty atlas with shape {atlas.shape}.")

    # Fill the atlas with tiles
    for i, image_np in enumerate(image_nps):
        # Resize the image_np if it's not the same size as the tile_size.
        if image_np.shape[0] != tile_size:
            image_np = resize_np(image_np, tile_size)

        row = i // num_tiles
        col = i % num_tiles
        atlas[
            row * tile_size : (row + 1) * tile_size,
            col * tile_size : (col + 1) * tile_size,
        ] = image_np

    return atlas


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
) -> Generator[List[np.ndarray], None, None]:
    """Reads HDF5 file from given path and returns a generator of numpy arrays
    split into batches of given size. At the end yields the remaining items in the batch

    :param hdf5_path: path to the HDF5 file
    :type hdf5_path: str
    :param batch_size: size of the batch
    :type batch_size: int
    :return: generator of numpy arrays
    :rtype: Generator[List[np.ndarray], None, None]
    """

    # Open the HDF5 file
    with h5py.File(hdf5_path, "r") as f:
        # Initialize a list to hold the batch of numpy arrays
        batch = []
        # Iterate over groups in the HDF5 file
        for group in f.values():
            # Iterate over datasets in the group
            for dataset in group.values():
                # Add the dataset to the batch
                batch.append(np.array(dataset))
                # If the batch has reached the desired size, yield it
                if len(batch) == batch_size:
                    yield batch
                    # Start a new batch
                    batch = []
        # If there are any remaining items in the batch, yield them
        if batch:
            yield batch
