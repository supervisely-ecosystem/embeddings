from typing import List

from supervisely.app.fastapi.utils import run_sync
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    Field,
    Flexbox,
    GridGallery,
    Input,
    InputNumber,
    Select,
)

from src.utils import ImageInfoLite

PROJECT_ID = 298119
options = ["centroids", "random"]

text_input = Input("aeroplane", minlength=1, placeholder="Enter text query")
text_input_field = Field(text_input, title="Query", description="Enter a text query.")
limit_input = InputNumber(10, min=1, max=100, step=1)
limit_input_field = Field(limit_input, title="Limit", description="Enter a limit.")
kmeans_option_select = Select([Select.Item(option) for option in options])
kmeans_option_field = Field(
    kmeans_option_select,
    title="KMeans Option",
    description="Select an option for KMeans.",
)
search_button = Button("Search")
diverse_kmeans_button = Button("Diverse KMeans")
diverse_fps_button = Button("Diverse FPS")
diverse_fps_button.disable()
buttons_flexbox = Flexbox([search_button, diverse_kmeans_button, diverse_fps_button])
grid_gallery = GridGallery(columns_number=5)

layout = Card(
    "QDrant search",
    "Enter a text query to search for similar images.",
    content=Container(
        [
            text_input_field,
            limit_input_field,
            kmeans_option_field,
            buttons_flexbox,
            grid_gallery,
        ]
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

    image_infos: List[ImageInfoLite] = run_sync(search(request_json))
    assert len(image_infos) == len(set(image_infos)), "Duplicate images found."

    for image_info in image_infos:
        grid_gallery.append(image_info.full_url)


@diverse_kmeans_button.click
def on_diverse_kmeans_button_click():
    grid_gallery.clean_up()

    limit = limit_input.get_value()
    option = kmeans_option_select.get_value()

    request_json = {
        "context": {
            "project_id": PROJECT_ID,
            "method": "kmeans",
            "limit": limit,
            "option": option,
        }
    }

    from src.main import diverse

    image_infos: List[ImageInfoLite] = run_sync(diverse(request_json))
    assert len(image_infos) == len(set(image_infos)), "Duplicate images found."

    for image_info in image_infos:
        grid_gallery.append(image_info.full_url)


@diverse_fps_button.click
def on_diverse_fps_button_click():
    grid_gallery.clean_up()

    limit = limit_input.get_value()

    request_json = {
        "context": {
            "project_id": PROJECT_ID,
            "method": "fps",
            "limit": limit,
        }
    }

    from src.main import diverse

    image_infos: List[ImageInfoLite] = run_sync(diverse(request_json))
    assert len(image_infos) == len(set(image_infos)), "Duplicate images found."

    for image_info in image_infos:
        grid_gallery.append(image_info.full_url)
