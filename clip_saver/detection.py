import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from supervision import Detections
from ultralytics import YOLO

from .buffer import Buffer
from .callbacks import DetectionCallback, run_in_background
from .connection import Connection
from .frame import Frame


class DetectionSaver(BaseModel):
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    yolo_model_path: Path
    confidence_threshold: float = 0.25
    wait_between_detections_secs: float = 30
    max_secs_between_detections: float = 5 * 60
    yolo_predict_kwargs: dict[str, Any] = Field(default_factory=dict)
    yolo_verbose: bool = False
    show: bool = False
    verbose: bool = False
    connections: list[Connection]
    buffer: Buffer = Field(default_factory=Buffer)
    callbacks: list[DetectionCallback] = Field(default_factory=list)

    # Class managed variables - Don't touch!
    yolo_model: YOLO
    time_first_detection: datetime | None = None
    time_last_detection: datetime | None = None

    @classmethod
    def from_model_path(
        cls,
        yolo_model_path: Path,
        callbacks: list[type[DetectionCallback]] = [],
        **kwargs: Any
    ):
        yolo_model = YOLO(yolo_model_path, task="detect")
        return cls(
            yolo_model_path=yolo_model_path,
            yolo_model=yolo_model,
            callbacks=[
                Callback(
                    yolo_model=yolo_model,
                    yolo_model_path=yolo_model_path,
                )
                for Callback in callbacks
            ],
            **kwargs,
        )

    def predict_next_frame(self):
        images = {connection: connection.get_image() for connection in self.connections}
        images = {k: v for k, v in images.items() if v is not None}

        if len(images) == 0:
            return

        results = self.yolo_model.track(
            list(images.values()),
            stream=True,
            conf=self.confidence_threshold,
            verbose=self.yolo_verbose,
            show=self.show,
            **self.yolo_predict_kwargs,
        )

        # Start detection
        for result, connection in zip(results, images.keys()):
            if result.boxes.cls is None or len(result.boxes.cls) == 0:
                continue

            self.time_last_detection = datetime.utcnow()
            frame = Frame(
                raw_image=result.orig_img,
                detections=Detections.from_ultralytics(result),
                timestamp=datetime.utcnow(),
                connection=connection,
            )

            if not self.time_first_detection:
                self.time_first_detection = datetime.utcnow()
                self.on_detection_start(frame)

            self.buffer.add_frame(frame)
            self.on_detection(frame)

        # End detection
        if (
            self.time_first_detection is not None
            and self.time_last_detection is not None
            and (
                # Greater than 30 seconds since last detection
                (datetime.utcnow() - self.time_last_detection).seconds
                >= self.wait_between_detections_secs
                or
                # Detecting activity for more than 5 minutes
                (
                    (datetime.utcnow() - self.time_first_detection).seconds
                    >= self.max_secs_between_detections
                )
            )
        ):
            if self.verbose:
                logging.info("Saving video...")

            self.time_first_detection = None
            self.time_last_detection = None
            self.on_detection_end()

            if self.verbose:
                logging.info("Resetting buffer...")

            self.buffer.reset()

    def on_detection_start(self, frame: Frame):
        for callback in self.callbacks:
            callback.on_detection_start(frame)

    def on_detection(self, frame: Frame):
        for callback in self.callbacks:
            callback.on_detection(frame)

    def on_detection_end(self):
        for callback in self.callbacks:
            callback.on_detection_end(self.buffer.get_frames())
