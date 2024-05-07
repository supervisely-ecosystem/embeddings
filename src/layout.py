from typing import List

from supervisely.app.fastapi.utils import run_sync
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    Field,
    Flexbox,
    GridGallery,
    Image,
    Input,
    InputNumber,
    Select,
    Table,
)

from src.globals import api
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

table = Table()

columns = [
    "dataset_id",
    "image_id",
    "image_name",
    "select",
]

rows = []
for dataset in api.dataset.get_list(PROJECT_ID):
    for image in api.image.get_list(dataset.id):
        rows.append(
            [
                dataset.id,
                image.id,
                image.name,
                Table.create_button("select"),
            ]
        )

table_data = {"columns": columns, "data": rows}
table.read_json(table_data)

image = Image()


card1 = Card(
    "QDrant search by prompt",
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

card2 = Card(
    "QDrant search by image",
    "Select an image to search for similar images.",
    content=Container([table, image], direction="horizontal"),
)

layout = Container([card1, card2])


@table.click
def handle_table_button(datapoint: Table.ClickedDataPoint):
    """Handles the click on the button in the table. Gets the image id from the
    table and calls the API to get the image info. Downloads the image to the static directory,
    reads the annotation info and shows the image in the preview widget.

    Args:
        datapoint (sly.app.widgets.Table.ClickedDataPoint): clicked datapoint in the table.
    """
    if datapoint.button_name != "select":
        return

    image_id = datapoint.row["image_id"]
    image_info = api.image.get_info_by_id(image_id)

    query = image_info.full_storage_url

    image.set(query)

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

    grid_gallery.clean_up()

    for image_info in image_infos:
        grid_gallery.append(image_info.full_url)


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
