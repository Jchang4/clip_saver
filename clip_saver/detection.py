import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from supervision import Detections
from ultralytics import YOLO

from .base import Frame
from .buffer import Buffer
from .callbacks import DetectionCallback, run_in_background
from .connection import Connection


class DetectionSaver(BaseModel):
    """
    Class for saving detections to a video file.

    Args:
        model_path (str): Path to the model
        output_dir (str): Path to the output directory
        video_source (str): Path to the source video
        video_width (int): Video width
        video_height (int): Video height
        video_fps (int): Video FPS

        confidence_threshold (float): Confidence threshold
        verbose (bool): Verbose mode
    """

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    ##################
    # Input Variables
    ##################
    video_sources: list[str]
    yolo_model_path: str
    yolo_verbose: bool = False
    output_dir: str
    callbacks: list[DetectionCallback] = Field(default_factory=list)

    # Detection settings
    confidence_threshold: float = 0.25

    show: bool = False
    verbose: bool = False

    ##################
    # Class managed variables - Don't touch!
    ##################
    yolo: YOLO = Field(default=None)
    connections: list[Connection] = Field(default=None)
    buffer: Buffer = Field(default_factory=Buffer)

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.yolo = YOLO(self.yolo_model_path, task="detect")
        self.connections = [
            Connection(video_source=source) for source in self.video_sources
        ]

    def start(self):
        time_first_detection: datetime | None = None
        time_last_detection: datetime | None = None

        while True:
            images = [connection.get_frame() for connection in self.connections]
            images = [img for img in images if img is not None]
            results = self.yolo.track(
                images,
                stream=True,
                conf=self.confidence_threshold,
                verbose=self.yolo_verbose,
                show=self.show,
                imgsz=320,
            )

            # Start detection
            for result in results:
                if result.boxes.cls is None or len(result.boxes.cls) == 0:
                    continue

                time_last_detection = datetime.utcnow()
                frame = Frame(
                    raw_image=result.orig_img,
                    detections=Detections.from_ultralytics(result),
                    timestamp=datetime.utcnow(),
                )

                if not time_first_detection:
                    time_first_detection = datetime.utcnow()
                    run_in_background(
                        lambda: [
                            callback.on_detection_start(frame)
                            for callback in self.callbacks
                        ]
                    )

                self.buffer.add_frame(frame)

                run_in_background(
                    lambda: [
                        callback.on_detection(frame) for callback in self.callbacks
                    ]
                )

            # End detection
            if (
                time_first_detection is not None
                and time_last_detection is not None
                and (
                    # Greater than 30 seconds since last detection
                    (datetime.utcnow() - time_last_detection).seconds >= 30
                    or
                    # Detecting activity for more than 5 minutes
                    ((datetime.utcnow() - time_first_detection).seconds >= (60 * 5))
                )
            ):
                if self.verbose:
                    logging.info("Saving video...")

                time_first_detection = None
                time_last_detection = None
                self.buffer.save(class_map=self.yolo.names)

                run_in_background(
                    lambda: [
                        callback.on_detection_end(self.buffer.get_frames())
                        for callback in self.callbacks
                    ]
                )

                self.buffer.reset()

    def get_class_name(self, class_id: int | None):
        assert self.yolo.names
        if not class_id:
            return "None"
        return self.yolo.names[class_id]
