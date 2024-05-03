import os
from typing import List

import h5py
import numpy as np

import src.globals as g
from src.utils import ImageInfoLite, timer


@timer
def save_to_hdf5(
    image_nps: List[np.ndarray],
    image_infos: List[ImageInfoLite],
    project_id: int,
) -> None:
    """Save thumbnail numpy array data to the HDF5 file.

    :param image_np: List of image_np to save.
    :type image_np: List[np.ndarray]
    :param image_infos: List of image infos to get image IDs.
        Image ID will be used as a dataset name. Length of this list should be the same as image_np.
    :type image_infos: List[ImageInfoLite]
    :param project_id: ID of the project. It will be used as a file name.
    :type project_id: int
    """

    with h5py.File(get_hdf5_path(project_id), "a") as file:
        for image_np, image_info in zip(image_nps, image_infos):
            # Using Dataset ID as a group name.
            # Require method works as get or create, to speed up the process.
            # If group already exists, it will be returned, otherwise created.
            group = file.require_group(str(image_info.dataset_id))
            # For each numpy array, we create a dataset with ID which comes from image ID.
            # Same, require method works as get or create.
            # So if numpy_array for image with specific ID already exists, it will be overwritten.
            dataset = group.require_dataset(
                str(image_info.id),
                data=image_np,
                shape=image_np.shape,
                dtype=image_np.dtype,
            )

            dataset.attrs[g.HDF5_URL_KEY] = image_info.full_url


def get_hdf5_path(project_id: int) -> str:
    """Get the path to the HDF5 file for the given project ID.

    :param project_id: ID of the project
    :type project_id: int
    :return: path to the HDF5 file
    :rtype: str
    """
    return os.path.join(g.HDF5_DIR, f"{project_id}.hdf5")
