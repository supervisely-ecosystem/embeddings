import os

import supervisely as sly
from dotenv import load_dotenv

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely-prod.env"))
    load_dotenv("local.env")

# region envvars
team_id = sly.io.env.team_id()
workspace_id = sly.io.env.workspace_id()

qdrant_host = os.getenv("QDRANT_HOST")
cas_host = os.getenv("CAS_HOST")
# endregion
if not qdrant_host:
    raise ValueError("QDRANT_HOST is not set in the environment variables")
if not cas_host:
    raise ValueError("CAS_HOST is not set in the environment variables")

api: sly.Api = sly.Api.from_env()
sly.logger.debug(
    f"Connected to {api.server_address}, team_id={team_id}, workspace_id={workspace_id}"
)

# region constants
STORAGE_DIR = sly.env.agent_storage()
HDF5_DIR = os.path.join(STORAGE_DIR, "hdf5")
# endregion
sly.fs.mkdir(HDF5_DIR)
sly.logger.debug(f"Storage directory: {STORAGE_DIR}, HDF5 directory: {HDF5_DIR}")
