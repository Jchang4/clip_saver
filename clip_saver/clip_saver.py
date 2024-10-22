from datetime import datetime
from typing import Callable, Iterable

import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from .callbacks.base import BaseCallback
from .datatypes.frame import Frame
from .video_source.base import BaseVideoSource


class ClipSaver:
    model_path: str
    video_source: BaseVideoSource
    detections_filter: list[Callable[[sv.Detections, list[str]], sv.Detections]]
    callbacks: list[BaseCallback]
    model_kwargs: dict[str, str]

    def __init__(
        self,
        model_path: str,
        video_source: BaseVideoSource,
        detections_filter: list[
            Callable[[sv.Detections, list[str]], sv.Detections]
        ] = [],
        callbacks: list[BaseCallback] = [],
        model_kwargs: dict[str, str] = {},
    ):
        self.model_path = model_path
        self.video_source = video_source
        self.detections_filter = detections_filter
        self.callbacks = callbacks
        self.model_kwargs = model_kwargs

    def run(self, result: Results):
        frame = self.create_frame(result)
        self.run_callbacks(frame)

    def start(self):
        self.init_callbacks()
        results = self.get_iterator()
        for result in results:
            self.run(result)

    def stop(self):
        for callback in self.callbacks:
            callback.stop()

    def get_iterator(self) -> Iterable[Results]:
        # Initialize with default args
        combined_kwargs = {
            "conf": 0.25,
            "line_width": 2,
            "stream": True,
            "persist": True,
            "verbose": False,
            "show": False,
            "save": False,  # save video
        }

        combined_kwargs.update(self.model_kwargs)

        return YOLO(model=self.model_path).track(
            source=self.video_source.get_video_url(), **combined_kwargs
        )

    def filter_detections(
        self, detections: sv.Detections, classnames: list[str]
    ) -> sv.Detections:
        for filter_fn in self.detections_filter:
            detections = filter_fn(detections, classnames)
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

    def create_frame(self, result: Results) -> Frame:
        detections = sv.Detections.from_ultralytics(result)
        detections = self.filter_detections(detections, result.names)
        return Frame(
            image=result.orig_img,
            detections=detections,
            timestamp=datetime.now().isoformat(),
            video_path=result.path,
        )
