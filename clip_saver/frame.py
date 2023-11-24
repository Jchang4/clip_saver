from datetime import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict
from supervision import BoundingBoxAnnotator, Detections, LabelAnnotator

from .connection import Connection


class Frame(BaseModel):
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    raw_image: np.ndarray
    detections: Detections
    timestamp: datetime
    connection: Connection

    def get_annotated_image(self, class_map: dict[int, str] | None = None):
        bounding_box_annotator = BoundingBoxAnnotator()
        label_annotator = LabelAnnotator()

        assert (
            self.detections.class_id is not None
            and self.detections.confidence is not None
        )

        labels = [
            f"{class_map[class_id] if class_map else class_id} confidence={(confidence*100):2f}%"
            for class_id, confidence in zip(
                self.detections.class_id, self.detections.confidence
            )
        ]

        annotated_image = bounding_box_annotator.annotate(
            scene=self.raw_image,
            detections=self.detections,
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image,
            detections=self.detections,
            labels=labels,
        )

        return annotated_image


Frame.model_rebuild()


class StartAndEndFrames(BaseModel):
    start: Frame
    end: Frame


class MostAccurateFrame(BaseModel):
    frame: Frame
    start_time: datetime
    end_time: datetime
