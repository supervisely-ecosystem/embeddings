from collections import defaultdict
from typing import Dict, List

from supervisely.app.fastapi.utils import run_sync
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    Field,
    GridGallery,
    Input,
    InputNumber,
)

from src.globals import api
from src.utils import ImageInfoLite

PROJECT_ID = 298119

text_input = Input("aeroplane", minlength=1, placeholder="Enter text query")
text_input_field = Field(text_input, title="Query", description="Enter a text query.")
limit_input = InputNumber(10, min=1, max=100, step=1)
limit_input_field = Field(limit_input, title="Limit", description="Enter a limit.")
search_button = Button("Search")
grid_gallery = GridGallery(columns_number=5)

layout = Card(
    "QDrant search",
    "Enter a text query to search for similar images.",
    content=Container(
        [text_input_field, limit_input_field, search_button, grid_gallery]
    ),
)


@search_button.click
def on_search_button_click():
    grid_gallery.clean_up()

    query = text_input.get_value()
    limit = limit_input.get_value()

    request_json = {
        "context": {
            "project_id": PROJECT_ID,
            "query": query,
            "limit": limit,
        }
    }

    from src.main import search

    image_infos = run_sync(search(request_json))
    grouped_dict = image_infos_to_dict(image_infos)

    for dataset_id, image_ids in grouped_dict.items():
        image_infos = api.image.get_info_by_id_batch(image_ids)
        for image_info in image_infos:
            grid_gallery.append(image_info.full_storage_url)


def image_infos_to_dict(image_infos: List[ImageInfoLite]) -> Dict[int, List[int]]:
    grouped_dict = defaultdict(list)
    for image_info in image_infos:
        grouped_dict[image_info.dataset_id].append(image_info.id)
    return grouped_dict
