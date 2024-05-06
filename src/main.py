import json
import os
from typing import Dict, List, Tuple

import cv2
import supervisely as sly
from fastapi import Request
from pympler import asizeof

import src.cas as cas
import src.globals as g
import src.pointclouds as pointclouds
import src.qdrant as qdrant
import src.thumbnails as thumbnails
from src.atlas import save_atlas
from src.layout import layout
from src.utils import (
    _get_api_from_request,
    download_items,
    get_datasets,
    get_image_infos,
    rgb_to_rgba,
    timer,
)

layout = layout if sly.is_development() else None
app = sly.Application(layout=layout)
server = app.get_server()


@server.post("/create_atlas")
@timer
async def create_atlas(request: Request):
    # Step 1: Unpack data from the request.
    context = request.state.context
    project_id = context.get("project_id")
    sly.logger.info(f"Creating atlas for project {project_id}...")

    # Step 2: Prepare the project directory.
    project_atlas_dir = os.path.join(g.ATLAS_DIR, str(project_id))
    sly.fs.mkdir(project_atlas_dir, remove_content_if_exists=True)

    # Step 3: Save atlas pages and prepare vector and atlas maps.
    atlas_map, vector_map = await process_atlas(project_id, project_atlas_dir)

    # Step 4: Save atlas map to the JSON file.
    json.dump(
        atlas_map,
        open(os.path.join(project_atlas_dir, f"{project_id}.json"), "w"),
        indent=4,
    )

    # Step 5: Create and save the vector map to the PCD file.
    await pointclouds.create_pointcloud(project_id, vector_map)

    sly.logger.info(f"Atlas for project {project_id} has been created.")


@server.post("/embeddings")
@timer
async def create_embeddings(request: Request):
    # Step 1: Unpack data from the request.
    api = _get_api_from_request(request)
    context = request.state.context
    project_id = context.get("project_id")
    # If context contains list of image_ids it means that we're
    # updating embeddings for specific images, otherwise we're updating
    # the whole project.
    image_ids = context.get("image_ids")
    force = context.get("force")

    if force:
        # Step 1.1: If force is True, delete the collection and recreate it.
        sly.logger.debug(f"Force enabled, deleting collection {project_id}.")
        await qdrant.delete_collection(project_id)
        sly.logger.debug(f"Deleting HDF5 file for project {project_id}.")
        thumbnails.remove_hdf5(project_id)

    # Step 2: Ensure collection exists in Qdrant.
    await qdrant.get_or_create_collection(project_id)

    # Step 3: Process images.
    if not image_ids:
        # Step 3A: If image_ids are not provided, get all datasets from the project.
        # Then iterate over datasets and process images from each dataset.
        datasets = await get_datasets(api, project_id)
        for dataset in datasets:
            await process_images(api, project_id, dataset_id=dataset.id)
    else:
        # Step 3B: If image_ids are provided, process images with specific IDs.
        await process_images(api, project_id, image_ids=image_ids)


@server.post("/search")
@timer
async def search(request: Request):
    try:
        context = request.state.context
    except Exception:
        # For development purposes.
        context = request.get("context")
    project_id = context.get("project_id")
    query = context.get("query")
    limit = context.get("limit", 10)
    sly.logger.debug(f"Searching for {query} in project {project_id}...")

    query_vectors = await cas.get_vectors([query])

    image_infos = await qdrant.search(project_id, query_vectors[0], limit)
    sly.logger.debug(f"Found {len(image_infos)} similar images.")

    return image_infos


@server.post("/diverse")
@timer
async def diverse(request: Request):
    try:
        context = request.state.context
    except Exception:
        # For development purposes.
        context = request.get("context")
    project_id = context.get("project_id")
    method = context.get("method")
    limit = context.get("limit", 10)
    sly.logger.debug(
        f"Generating diverse population for project {project_id} with method {method}"
    )

    image_infos = await qdrant.diverse_kmeans(project_id, limit)
    sly.logger.debug(f"Generated {len(image_infos)} diverse images.")

    return image_infos


@timer
async def process_atlas(
    project_id: int, project_atlas_dir: str
) -> Tuple[List[Dict], List[Dict]]:
    atlas_map = []
    vector_map = []

    for idx, tiles in enumerate(thumbnails.get_tiles_from_hdf5(project_id=project_id)):
        sly.logger.debug(
            f"Processing {idx + 1} batch of tiles. Received {len(tiles)} tiles."
        )

        # Get atlas image and map. Image is a numpy array, map is a list of dictionaries.
        page_image, page_map = save_atlas(
            g.ATLAS_SIZE, g.IMAGE_SIZE_FOR_ATLAS, tiles, atlas_id=idx
        )

        # Add page elements to the vector and atlas maps, which will be saved to the files
        # without splitting into pages.
        vector_map.extend(await pointclouds.get_vector_map(idx, project_id, page_map))
        atlas_map.extend(page_map)

        if sly.is_development():
            # Checking how much memory the vector map takes.
            vector_map_size = asizeof.asizeof(vector_map) / 1024 / 1024
            sly.logger.debug(f"Vector map size: {vector_map_size:.2f} MB")
        # Convert RGBA to BGRA for compatibility with OpenCV.
        page_image = cv2.cvtColor(page_image, cv2.COLOR_RGBA2BGRA)
        # Save atlas image to the file.
        cv2.imwrite(
            os.path.join(project_atlas_dir, f"{project_id}_{idx}.png"), page_image
        )

    return atlas_map, vector_map


@timer
async def process_images(
    api: sly.Api, project_id: int, dataset_id: int = None, image_ids: List[int] = None
):
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
        # url_batch = [image_info.url for image_info in image_batch]
        # ids_batch = [image_info.id for image_info in image_batch]

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
