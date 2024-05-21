import logging
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any, Iterable

import torch
from supervision import Detections
from ultralytics import YOLO
from ultralytics.engine.results import Results

from clip_saver_old.buffer import Buffer
from clip_saver_old.callbacks import DetectionCallback
from clip_saver_old.frame import Frame
from clip_saver_old.rtsp import RtspUrl

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"


class DetectionSaver:
    rtsp_urls: list[str]
    yolo_model_path: Path
    wait_between_detections_secs: float = 30
    max_secs_between_detections: float = 30
    sleep_secs_between_detections: float = 0.1
    yolo_predict_kwargs: dict[str, Any]
    yolo_verbose: bool = False
    verbose: bool = False
    device: str = DEVICE
    buffer: Buffer
    callbacks: list[DetectionCallback]

    # Class managed variables - Don't touch!
    yolo_model: YOLO
    time_first_detection: datetime | None = None
    time_last_detection: datetime | None = None

    def __init__(
        self,
        rtsp_urls: list[str],
        yolo_model_path: Path,
        buffer: Buffer | None = None,
        callbacks: list[type[DetectionCallback]] = [],
        verbose: bool = False,
        yolo_verbose: bool = False,
        wait_between_detections_secs: float = 30,
        max_secs_between_detections: float = 30,
        sleep_secs_between_detections: float = 0.1,
        device: str = DEVICE,
        **kwargs,
    ):
        super().__init__()
        # Create a list.streams file containing the RTSP urls
        with open("list.streams", "w") as f:
            f.write("\n".join(rtsp_urls))

        self.rtsp_urls = rtsp_urls
        self.device = device
        self.yolo_model_path = yolo_model_path
        self.yolo_model = YOLO(yolo_model_path, task="detect").to(device)
        self.callbacks = [
            Callback(
                yolo_model=self.yolo_model,
                yolo_model_path=yolo_model_path,
            )
            for Callback in callbacks
        ]
        self.buffer = Buffer() if buffer is None else buffer
        self.yolo_predict_kwargs = kwargs
        self.verbose = verbose
        self.yolo_verbose = yolo_verbose
        self.wait_between_detections_secs = wait_between_detections_secs
        self.max_secs_between_detections = max_secs_between_detections
        self.sleep_secs_between_detections = sleep_secs_between_detections

    def run(self):
        results: Iterable[Results] = self.yolo_model.track(
            "list.streams",
            stream=True,
            verbose=self.yolo_verbose,
            **self.yolo_predict_kwargs,
        )

        # Start detection
        # Run 1 batch
        while results:
            self.process_batch(results=results)

    def get_iterator(self) -> Iterable[Results]:
        return self.yolo_model.track(
            "list.streams",
            stream=True,
            verbose=self.yolo_verbose,
            **self.yolo_predict_kwargs,
        )

    def process_batch(self, results: Iterable[Results]):
        for i in range(len(self.rtsp_urls)):
            result = next(iter(results))
            if (
                not isinstance(result, Results)
                or not result.boxes
                or result.boxes.cls is None
                or len(result.boxes.cls) == 0
            ):
                continue

            self.time_last_detection = datetime.utcnow()
            frame = Frame(
                raw_image=result.orig_img,
                detections=Detections.from_ultralytics(result),
                timestamp=datetime.utcnow(),
                rtsp_url=str(RtspUrl.from_yolo_path(result.path)),
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

        # Sleep between detections
        if self.sleep_secs_between_detections > 0:
            sleep(self.sleep_secs_between_detections)

    def on_detection_start(self, frame: Frame):
        for callback in self.callbacks:
            callback.on_detection_start(frame)

    def on_detection(self, frame: Frame):
        for callback in self.callbacks:
            callback.on_detection(frame)

    def on_detection_end(self):
        for callback in self.callbacks:
            callback.on_detection_end(self.buffer.get_frames())
