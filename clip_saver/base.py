from datetime import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict
from supervision import BoxAnnotator, Detections


class Frame(BaseModel):
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    raw_image: np.ndarray
    detections: Detections
    timestamp: datetime

    def get_annotated_image(self, class_map: dict[int, str] | None = None):
        box_annotator = BoxAnnotator(
            text_thickness=1,
            text_padding=5,
            text_scale=0.3,
        )

        labels = [
            f"id={tracker_id} class={class_map[class_id] if class_map is not None and class_id is not None else class_id} {confidence:.2f}"
            for xyxy, mask, confidence, class_id, tracker_id in self.detections
        ]

        frame = box_annotator.annotate(
            scene=self.raw_image,
            detections=self.detections,
            labels=labels,
        )

        return frame


Frame.model_rebuild()


class StartAndEndFrames(BaseModel):
    start: Frame
    end: Frame


class MostAccurateFrame(BaseModel):
    frame: Frame
    start_time: datetime
    end_time: datetime
