import os

import supervisely as sly
from aiocron import crontab
from dotenv import load_dotenv

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely-prod.env"))
    load_dotenv("local.env")

# ! DEBUG! Remove team_id and workspace_id from the environment variables.
# Also, remove creating sly.Api instance.
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
ATLAS_DIR = os.path.join(STORAGE_DIR, "atlas")
IMAGE_SIZE_FOR_CAS = 224
IMAGE_SIZE_FOR_ATLAS = 32
ATLAS_SIZE = pow(2, 13)
HDF5_URL_KEY = "url"
DEFAULT_PCD_FIELDS = ["x", "y", "z"]
QDRANT_MAX_RETRIES = 3
# endregion
sly.fs.mkdir(HDF5_DIR)
sly.fs.mkdir(ATLAS_DIR, remove_content_if_exists=True)
sly.logger.debug(
    f"Image size for CAS: {IMAGE_SIZE_FOR_CAS}, Image size for Atlas: {IMAGE_SIZE_FOR_ATLAS}"
)
sly.logger.debug(f"Storage directory: {STORAGE_DIR}, HDF5 directory: {HDF5_DIR}")
sly.logger.debug(f"Atlas directory: {ATLAS_DIR}, Atlas size: {ATLAS_SIZE}")

hdf5_storage_check_cron = "*/1 * * * *" if sly.is_development() else "0 * * * *"


@crontab(hdf5_storage_check_cron)
def log_size_of_hdf_storage() -> None:
    """Log the size of the HDF5 storage in megabytes."""
    try:
        size = sly.fs.get_directory_size(HDF5_DIR) / 1024 / 1024
        sly.logger.info(f"HDF5 storage size: {size:.2f} MB")
    except Exception:
        pass
