from datetime import datetime
from typing import Iterable

import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from .core import Callback, Frame, VideoSource

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"


class ClipSaver:
    model_path: str
    video_source: VideoSource
    callbacks: list[Callback]
    confidence_thresholds: dict[str, float]
    model_kwargs: dict[str, str]

    def __init__(
        self,
        model_path: str,
        video_source: VideoSource,
        callbacks: list[Callback] = [],
        confidence_thresholds: dict[str, float] = {},
        model_kwargs: dict[str, str] = {},
    ):
        self.model_path = model_path
        self.video_source = video_source
        self.callbacks = callbacks
        self.confidence_thresholds = confidence_thresholds
        self.model_kwargs = model_kwargs

    def start(self):
        self.init_callbacks()
        results = self.get_iterator()
        for result in results:
            self.run(result)

    def run(self, result: Results):
        detections = sv.Detections.from_ultralytics(result)
        detections = self.filter_detections(detections, result.names)
        frame = Frame(
            image=result.orig_img,
            detections=detections,
            timestamp=datetime.now().isoformat(),
            video_path=result.path,
        )
        self.run_callbacks(frame)

    def stop(self):
        for callback in self.callbacks:
            callback.stop()

    def get_iterator(self) -> Iterable[Results]:
        # Initialize with default args
        combined_kwargs = {
            "imgsz": 640,
            "conf": 0.25,
            "line_width": 2,
            "stream": True,
            "persist": True,
            "verbose": False,
            "show": False,
            "save": False,  # save video
        }

        combined_kwargs.update(self.model_kwargs)

        return (
            YOLO(model=self.model_path, task="detect")
            .to(DEVICE)
            .track(source=self.video_source.get_video_url(), **combined_kwargs)
        )

    def filter_detections(
        self, detections: sv.Detections, classnames: list[str]
    ) -> sv.Detections:
        filtered_class_ids = []
        filtered_bboxes = []
        filtered_confidences = []
        for class_id, xyxy, conf in zip(
            detections.class_id,
            detections.xyxy,
            detections.confidence,
        ):
            if conf < self.confidence_thresholds.get(classnames[class_id], 0.0):
                continue
            filtered_class_ids.append(class_id)
            filtered_bboxes.append(xyxy)
            filtered_confidences.append(conf)

        if not filtered_class_ids:
            return sv.Detections.empty()

        detections.class_id = np.array(filtered_class_ids)
        detections.xyxy = np.array(filtered_bboxes).reshape(-1, 4)
        detections.confidence = np.array(filtered_confidences)

        return detections

    def init_callbacks(self):
        for callback in self.callbacks:
            callback.start()

    def run_callbacks(self, frame: Frame):
        for callback in self.callbacks:
            callback.run(frame)

    def stop_callbacks(self):
        for callback in self.callbacks:
            callback.stop()
