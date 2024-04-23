import supervisely as sly
from clip_client import Client

import src.globals as g

client = Client(f"grpc://{g.cas_host}")

try:
    client.profile()
    sly.logger.info(f"Connected to CAS at {g.cas_host}")
except Exception as e:
    sly.logger.error(f"Failed to connect to CAS at {g.cas_host}: {e}")
