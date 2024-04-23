import supervisely as sly
from qdrant_client import QdrantClient

import src.globals as g

client = QdrantClient(g.qdrant_host)

try:
    client.get_collections()
    sly.logger.debug(f"Connected to Qdrant at {g.qdrant_host}")
except Exception as e:
    sly.logger.error(f"Failed to connect to Qdrant at {g.qdrant_host} with error: {e}")
