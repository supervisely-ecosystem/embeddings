import math
from typing import List

import cv2
import numpy as np
import supervisely as sly

from src.utils import TileInfo, timer


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
def get_atlas(
    atlas_size: int, tile_size: int, tile_infos: List[TileInfo]
) -> np.ndarray:
    """Create an atlas from the given list of tile_infos.
    The atlas will be filled row by row from the top left corner and sliced by the height
    if it's not enough tiles to fill the whole atlas. The width of the atlas is always the same as the atlas_size.
    The atlas will be an RGBA numpy array and it expects the tile_infos to have the thumbnail as a 4-channel RGBA numpy array.

    :param atlas_size: size of the atlas, should be a power of 2, e.g. 2048, 4096, 8192.
        the width of the atlas will be the same as the atlas_size (it will be a square).
    :type atlas_size: int
    :param tile_size: size of the tile, should be a power of 2, e.g. 32, 64, 128.
    :type tile_size: int
    :param tile_infos: list of tile_infos to create the atlas from
    :type tile_infos: List[TileInfo]
    :return: 4-channel RGBA numpy array of the atlas
    :rtype: np.ndarray
    """
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

    # Iterate over the thumbnails and fill the atlas.
    for idx, tile_info in enumerate(tile_infos):
        # Resize the thumbnail if it's not the same size as the tile_size.
        if tile_info.thumbnail.shape[0] != tile_size:
            thumbnail = resize_np(tile_info.thumbnail, tile_size)
        else:
            thumbnail = tile_info.thumbnail

        row = idx // num_tiles
        col = idx % num_tiles
        atlas[
            row * tile_size : (row + 1) * tile_size,
            col * tile_size : (col + 1) * tile_size,
        ] = thumbnail

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
