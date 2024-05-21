from dataclasses import dataclass
from datetime import datetime

import numpy as np
from supervision import BoundingBoxAnnotator, Detections, LabelAnnotator


class Frame:
    raw_image: np.ndarray
    detections: Detections
    timestamp: datetime
    rtsp_url: str

    def __init__(
        self,
        raw_image: np.ndarray,
        detections: Detections,
        timestamp: datetime,
        rtsp_url: str,
    ):
        self.raw_image = raw_image
        self.detections = detections
        self.timestamp = timestamp
        self.rtsp_url = rtsp_url

    def get_annotated_image(self, class_map: dict[int, str] | None = None):
        bounding_box_annotator = BoundingBoxAnnotator()
        label_annotator = LabelAnnotator()

        assert (
            self.detections.class_id is not None
            and self.detections.confidence is not None
        )

        labels = [
            f"{class_map[class_id] if class_map else class_id} confidence={(confidence*100):.2f}%"
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


@dataclass(kw_only=True)
class StartAndEndFrames:
    start: Frame
    end: Frame


@dataclass(kw_only=True)
class MostAccurateFrame:
    frame: Frame
    start_time: datetime
    end_time: datetime
