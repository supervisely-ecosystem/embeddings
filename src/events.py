from typing import Any, Dict, List, Optional

from src.utils import StateFields


class Event:
    class Embeddings:
        endpoint = "/embeddings"

        def __init__(
            self, project_id: int, force: Optional[bool], image_ids: Optional[List[int]]
        ):
            self.project_id = project_id
            self.force = force
            self.image_ids = image_ids

        @classmethod
        def from_json(cls, data: Dict[str, Any]):
            return cls(
                data.get(StateFields.PROJECT_ID),
                data.get(StateFields.FORCE),
                data.get(StateFields.IMAGE_IDS),
            )
