import os
from typing import Generator, List

import h5py
import numpy as np
import supervisely as sly

import src.globals as g
from src.utils import ImageInfoLite, TileInfo, timer


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


def remove_hdf5(project_id: int) -> None:
    """Remove HDF5 file for the given project ID.

    :param project_id: ID of the project
    :type project_id: int
    """
    path = get_hdf5_path(project_id)
    if os.path.isfile(path):
        os.remove(path)
        sly.logger.debug(f"HDF5 file for project {project_id} removed.")


def get_hdf5_path(project_id: int) -> str:
    """Get the path to the HDF5 file for the given project ID.

    :param project_id: ID of the project
    :type project_id: int
    :return: path to the HDF5 file
    :rtype: str
    """
    return os.path.join(g.HDF5_DIR, f"{project_id}.hdf5")


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
