import json
import os
from typing import Dict, List, Tuple, Union

import supervisely as sly
from pympler import asizeof

import src.cas as cas
import src.globals as g
import src.qdrant as qdrant
import src.thumbnails as thumbnails
from src.atlas import get_atlas
from src.events import Event
from src.pointclouds import get_tile_infos, save_pointcloud
from src.utils import (
    ImageInfoLite,
    TupleFields,
    download_items,
    get_datasets,
    get_image_infos,
    rgb_to_rgba,
    timer,
)

app = sly.Application()
server = app.get_server()


@app.event(Event.Atlas, use_state=True)
@timer
async def create_atlas(api: sly.Api, event: Event.Atlas) -> None:
    sly.logger.info(f"Creating atlas for project {event.project_id}.")

    # Step 1: Prepare the project directory.
    project_atlas_dir = os.path.join(g.ATLAS_DIR, str(event.project_id))
    sly.fs.mkdir(project_atlas_dir, remove_content_if_exists=True)

    # Step 2: Save atlas pages and prepare atlas map.
    atlas_map, tile_infos = await process_atlas(
        api,
        event.project_id,
        project_atlas_dir,
        atlas_size=g.ATLAS_SIZE,
        tile_size=g.IMAGE_SIZE_FOR_ATLAS,
    )

    # Step 3: Save atlas map to the JSON file.
    json.dump(
        atlas_map,
        open(os.path.join(project_atlas_dir, f"{event.project_id}.json"), "w"),
        indent=4,
    )

    # Step 4: Create and save pointcloud into PCD file.
    await save_pointcloud(event.project_id, project_atlas_dir, tile_infos)

    sly.logger.info(f"Atlas for project {event.project_id} has been created.")


@app.event(Event.Embeddings, use_state=True)
@timer
async def create_embeddings(api: sly.Api, event: Event.Embeddings) -> None:
    sly.logger.info(
        f"Started creating embeddings for project {event.project_id}. "
        f"Force: {event.force}, Image IDs: {event.image_ids}."
    )

    if event.force:
        # Step 1: If force is True, delete the collection and recreate it.
        sly.logger.debug(f"Force enabled, deleting collection {event.project_id}.")
        await qdrant.delete_collection(event.project_id)
        sly.logger.debug(f"Deleting HDF5 file for project {event.project_id}.")
        thumbnails.remove_hdf5(event.project_id)

    # Step 2: Ensure collection exists in Qdrant.
    await qdrant.get_or_create_collection(event.project_id)

    # Step 3: Process images.
    if not event.image_ids:
        # Step 3A: If image_ids are not provided, get all datasets from the project.
        # Then iterate over datasets and process images from each dataset.
        datasets = await get_datasets(api, event.project_id)
        for dataset in datasets:
            await process_images(api, event.project_id, dataset_id=dataset.id)
    else:
        # Step 3B: If image_ids are provided, process images with specific IDs.
        await process_images(api, event.project_id, image_ids=event.image_ids)

    sly.logger.debug(f"Embeddings for project {event.project_id} have been created.")


@app.event(Event.Search, use_state=True)
@timer
async def search(api: sly.Api, event: Event.Search) -> List[ImageInfoLite]:
    sly.logger.info(
        f"Searching for similar images in project {event.project_id}. "
        f"Query: {event.query}, Limit: {event.limit}."
    )

    # ? Add support for image IDs in the query.
    # * In this case, we'll need to get the resized image URLs from the API
    # * and then get vectors from these URLs.

    # Vectorize the query data (can be a text prompt or an image URL).
    query_vectors = await cas.get_vectors([event.query])

    image_infos = await qdrant.search(event.project_id, query_vectors[0], event.limit)
    sly.logger.debug(f"Found {len(image_infos)} similar images.")

    return image_infos


@app.event(Event.Diverse, use_state=True)
@timer
async def diverse(api: sly.Api, event: Event.Diverse) -> List[ImageInfoLite]:
    sly.logger.info(
        f"Generating diverse population for project {event.project_id}. "
        f"Method: {event.query}, Limit: {event.limit}, Option: {event.option}."
    )

    image_infos = await qdrant.diverse(
        event.project_id,
        event.limit,
        event.method,
        event.option,
    )
    sly.logger.debug(f"Generated {len(image_infos)} diverse images.")

    return image_infos


@timer
async def process_atlas(
    api: sly.Api,
    project_id: int,
    project_atlas_dir: str,
    atlas_size: int,
    tile_size: int,
) -> Tuple[List[Dict], Dict[str, List[Dict[str, Union[int, str]]]]]:
    """Process the atlas for the given project.
    Generate the atlas pages and save them to the project directory (as PNG files).
    Create an atlas manifest and return it as a dictionary.
    Prepare the list of tile_infos for the pointcloud creation. The order of tile_infos
    will be the same as the order of the tiles in the atlas.

    :param api: Supervisely API object.
    :type api: sly.Api
    :param project_id: Project ID to create the atlas for.
    :type project_id: int
    :param project_atlas_dir: Directory to save the atlas pages and later the atlas manifest
        and the pointcloud.
    :type project_atlas_dir: str
    :param atlas_size: Size of the atlas in pixels. Should be a power of 2.
        The atlas will be a square with the size of atlas_size x atlas_size.
    :type atlas_size: int
    :param tile_size: Size of the tile in pixels. Should be a power of 2.
        The tiles will be square with the size of tile_size x tile_size.
    :type tile_size: int
    :return: Tuple with the atlas manifest and the list of tile_infos.
    :rtype: Tuple[List[Dict], Dict[str, List[Dict[str, Union[int, str]]]]
    """
    images = []
    tile_infos = []

    for idx, tiles in enumerate(thumbnails.get_tiles_from_hdf5(project_id=project_id)):
        file_name = f"{project_id}_{idx}.png"
        image_ids = [tile.id for tile in tiles]
        sly.logger.debug(f"Processing tile batch #{idx}, received {len(tiles)} tiles.")

        # Get 4-channel RGBA numpy array of the atlas.
        page_image = get_atlas(atlas_size, tile_size, tiles)

        # Add corresponding tile_infos to the list. It will be used to create the pointcloud.
        tile_infos.extend(await get_tile_infos(idx, project_id, image_ids))

        # This dictionary will be used to create the JSON file with the atlas manifest
        # and will be used in Embeddings widget to display the atlas.
        images.append(
            {
                TupleFields.ID: idx,
                TupleFields.UNIT_SIZE: tile_size,
                TupleFields.FILE_NAME: file_name,
                TupleFields.URL: "",  # URL will be added when serving the file, fileName will be used.
            }
        )

        if sly.is_development():
            # Checking how much memory the tile_infos take.
            tile_infos_size = asizeof.asizeof(tile_infos) / 1024 / 1024
            sly.logger.debug(f"TileInfos size: {tile_infos_size:.2f} MB")

        # Save atlas image to the file.
        sly.image.write(
            os.path.join(project_atlas_dir, file_name),
            page_image,
            remove_alpha_channel=False,
        )
        sly.logger.debug(f"Atlas page {file_name} has been saved.")

    atlas_map = {
        TupleFields.IMAGES: images,
    }

    return atlas_map, tile_infos


@timer
async def process_images(
    api: sly.Api, project_id: int, dataset_id: int = None, image_ids: List[int] = None
) -> None:
    """Process images from the specified project. Download images, save them to HDF5,
    get vectors from the images and upsert them to Qdrant.
    Either dataset_id or image_ids should be provided. If the dataset_id is provided,
    all images from the dataset will be processed. If the image_ids are provided,
    only images with the specified IDs will be processed.

    :param api: Supervisely API object.
    :type api: sly.Api
    :param project_id: Project ID to process images from.
    :type project_id: int
    :param dataset_id: Dataset ID to process images from.
    :type dataset_id: int, optional
    :param image_ids: List of image IDs to process.
    :type image_ids: List[int], optional
    """
    # Get image infos from the project.
    image_infos = await get_image_infos(
        api,
        g.IMAGE_SIZE_FOR_CAS,
        g.IMAGE_SIZE_FOR_ATLAS,
        dataset_id=dataset_id,
        image_ids=image_ids,
    )

    # Get diff of image infos, check if they are already
    # in the Qdrant collection and have the same updated_at field.
    diff_image_infos = await qdrant.get_diff(project_id, image_infos)

    # Batch image urls.
    for image_batch in sly.batched(diff_image_infos):
        # Download images as numpy arrays using URLs resized
        # for for specified HDF5 sizes (for thumbnails in the atlas).
        nps_batch = await download_items(
            [image_info.hdf5_url for image_info in image_batch]
        )

        # Convert tiles into square images with alpha channel.
        nps_batch = rgb_to_rgba(nps_batch, g.IMAGE_SIZE_FOR_ATLAS)

        # Save images to hdf5.
        thumbnails.save_to_hdf5(nps_batch, image_batch, project_id)

        # Get vectors from images.
        vectors_batch = await cas.get_vectors(
            [image_info.cas_url for image_info in image_batch]
        )

        # Upsert vectors to Qdrant.
        await qdrant.upsert(
            project_id,
            vectors_batch,
            image_batch,
        )
