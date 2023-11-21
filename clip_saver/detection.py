import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from supervision import Detections
from ultralytics import YOLO

from .base import Frame
from .buffer import Buffer
from .callbacks import DetectionCallback, run_in_background
from .connection import Connection


class DetectionSaver(BaseModel):
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    yolo_model_path: str
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
    yolo_model: YOLO = Field(default=None)
    time_first_detection: datetime | None = None
    time_last_detection: datetime | None = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.yolo_model = YOLO(self.yolo_model_path, task="detect")

    def predict_next_frame(self):
        images = [connection.get_frame() for connection in self.connections]
        images = [img for img in images if img is not None]

        if len(images) == 0:
            return

        results = self.yolo_model.track(
            images,
            stream=True,
            conf=self.confidence_threshold,
            verbose=self.yolo_verbose,
            show=self.show,
            **self.yolo_predict_kwargs,
        )

        # Start detection
        for result in results:
            if result.boxes.cls is None or len(result.boxes.cls) == 0:
                continue

            self.time_last_detection = datetime.utcnow()
            frame = Frame(
                raw_image=result.orig_img,
                detections=Detections.from_ultralytics(result),
                timestamp=datetime.utcnow(),
            )

            if not self.time_first_detection:
                self.time_first_detection = datetime.utcnow()
                run_in_background(
                    lambda: [
                        callback.on_detection_start(frame)
                        for callback in self.callbacks
                    ]
                )

            self.buffer.add_frame(frame)

            run_in_background(
                lambda: [callback.on_detection(frame) for callback in self.callbacks]
            )

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
            self.buffer.save(class_map=self.yolo_model.names)

            run_in_background(
                lambda: [
                    callback.on_detection_end(self.buffer.get_frames())
                    for callback in self.callbacks
                ]
            )

            if self.verbose:
                logging.info("Resetting buffer...")
            self.buffer.reset()


# class DetectionSaver(BaseModel):
#     """
#     Class for saving detections to a video file.

#     Args:
#         model_path (str): Path to the model
#         output_dir (str): Path to the output directory
#         video_source (str): Path to the source video
#         video_width (int): Video width
#         video_height (int): Video height
#         video_fps (int): Video FPS

#         confidence_threshold (float): Confidence threshold
#         verbose (bool): Verbose mode
#     """

#     model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

#     ##################
#     # Input Variables
#     ##################
#     video_sources: list[str]
#     yolo_model_path: str
#     yolo_verbose: bool = False
#     output_dir: str
#     callbacks: list[DetectionCallback] = Field(default_factory=list)
#     wait_between_detections: float = 30
#     max_video_time_in_mins: float = 60 * 5

#     # Detection settings
#     confidence_threshold: float = 0.25

#     sleep_time_secs: float = 5
#     predict_kwargs: dict[str, Any] = Field(default_factory=dict)
#     show: bool = False
#     verbose: bool = False

#     ##################
#     # Class managed variables - Don't touch!
#     ##################
#     yolo: YOLO = Field(default=None)
#     connections: list[Connection] = Field(default=None)
#     buffer: Buffer = Field(default_factory=Buffer)
#     loop_lock: Event = Field(default_factory=Event)
#     thread: Thread | None = None

#     def __init__(self, **data: Any):
#         super().__init__(**data)
#         self.yolo = YOLO(self.yolo_model_path, task="detect")
#         self.connections = [
#             Connection(video_source=source) for source in self.video_sources
#         ]

#     @property
#     def is_running(self):
#         return self.loop_lock.is_set()

#     def start(self):
#         self.loop_lock.set()

#         if self.show:
#             return self.run()

#         if self.thread is None:
#             self.thread = Thread(target=self.run, daemon=True)
#             self.thread.start()

#     def stop(self):
#         self.loop_lock.clear()

#         if self.thread:
#             self.thread.join()
#             self.thread = None

#     def run(self):
#         time_first_detection: datetime | None = None
#         time_last_detection: datetime | None = None

#         while self.is_running:
#             images = [connection.get_frame() for connection in self.connections]
#             images = [img for img in images if img is not None]

#             if len(images) == 0:
#                 sleep(self.sleep_time_secs)
#                 continue

#             results = self.yolo.track(
#                 images,
#                 stream=True,
#                 conf=self.confidence_threshold,
#                 verbose=self.yolo_verbose,
#                 show=self.show,
#                 **self.predict_kwargs,
#             )

#             # Start detection
#             for result in results:
#                 if result.boxes.cls is None or len(result.boxes.cls) == 0:
#                     continue

#                 time_last_detection = datetime.utcnow()
#                 frame = Frame(
#                     raw_image=result.orig_img,
#                     detections=Detections.from_ultralytics(result),
#                     timestamp=datetime.utcnow(),
#                 )

#                 if not time_first_detection:
#                     time_first_detection = datetime.utcnow()
#                     run_in_background(
#                         lambda: [
#                             callback.on_detection_start(frame)
#                             for callback in self.callbacks
#                         ]
#                     )

#                 self.buffer.add_frame(frame)

#                 run_in_background(
#                     lambda: [
#                         callback.on_detection(frame) for callback in self.callbacks
#                     ]
#                 )

#             # End detection
#             if (
#                 time_first_detection is not None
#                 and time_last_detection is not None
#                 and (
#                     # Greater than 30 seconds since last detection
#                     (datetime.utcnow() - time_last_detection).seconds
#                     >= self.wait_between_detections
#                     or
#                     # Detecting activity for more than 5 minutes
#                     (
#                         (datetime.utcnow() - time_first_detection).seconds
#                         >= self.max_video_time_in_mins
#                     )
#                 )
#             ):
#                 if self.verbose:
#                     logging.info("Saving video...")

#                 time_first_detection = None
#                 time_last_detection = None
#                 self.buffer.save(class_map=self.yolo.names)

#                 run_in_background(
#                     lambda: [
#                         callback.on_detection_end(self.buffer.get_frames())
#                         for callback in self.callbacks
#                     ]
#                 )

#                 self.buffer.reset()

#             sleep(self.sleep_time_secs)

#         if (
#             len(self.buffer.get_frames()) > 0
#             and time_first_detection is not None
#             and time_last_detection is not None
#         ):
#             self.buffer.save(class_map=self.yolo.names)
#             run_in_background(
#                 lambda: [
#                     callback.on_detection_end(self.buffer.get_frames())
#                     for callback in self.callbacks
#                 ]
#             )
#             self.buffer.reset()

#     def get_class_name(self, class_id: int | None):
#         assert self.yolo.names
#         if not class_id:
#             return "None"
#         return self.yolo.names[class_id]
