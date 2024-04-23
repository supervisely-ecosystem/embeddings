import supervisely as sly
from supervisely.app.widgets import Card

import src.cas as cas
import src.globals as g
import src.qdrant as qdrant

test_card = Card()

app = sly.Application(layout=test_card)
