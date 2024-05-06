from typing import List

import numpy as np
import supervisely as sly
from clip_client import Client

import src.globals as g
from src.utils import timer, with_retries

# TODO: Return client on demand.
client = Client(f"grpc://{g.cas_host}")

try:
    client.profile()
    sly.logger.info(f"Connected to CAS at {g.cas_host}")
except Exception as e:
    sly.logger.error(f"Failed to connect to CAS at {g.cas_host}: {e}")


@with_retries(retries=5, sleep_time=2)
@timer
async def get_vectors(image_urls: List[str]) -> List[np.ndarray]:
    """Use CAS to get vectors from the list of images.

    :param image_urls: List of URLs to get vectors from.
    :type image_urls: List[str]
    :return: List of vectors.
    :rtype: List[np.ndarray]
    """
    vectors = await client.aencode(image_urls)
    return [vector.tolist() for vector in vectors]
