import logging
import os
from datetime import datetime

from numpy import ndarray
from pydantic import BaseModel, Field
from supervision import BoxAnnotator, Detections, VideoInfo, VideoSink
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class DetectionSaver(BaseModel):
    model_path: str = Field(..., description="Path to the model")
    output_dir: str = Field(..., description="Path to the output directory")
    video_source: str = Field(..., description="Path to the source video")
    video_width: int = Field(640, description="Video width")
    video_height: int = Field(480, description="Video height")
    video_fps: int = Field(25, description="Video FPS")

    # Detection settings
    confidence_threshold: float = Field(0.25, description="Confidence threshold")
    verbose: bool = Field(False, description="Verbose mode")

    def __init__(self, **data):
        super().__init__(**data)
        self.yolo = YOLO(self.model_path)

    def start(self):
        current_time_utc = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

        # Setup YOLO
        results = self.yolo.track(
            self.video_source,
            stream=True,
            conf=self.confidence_threshold,
            verbose=self.verbose,
        )

        # Setup Video Writer
        video_file_path = os.path.join(self.output_dir, current_time_utc)
        video_info = VideoInfo(
            width=self.video_width,
            height=self.video_height,
            fps=self.video_fps,
        )

        with VideoSink(video_file_path, video_info) as sink:
            # Start detection
            for result in results:
                detections = Detections.from_ultralytics(results)
                if detections.class_id is None:
                    continue
                sink.write_frame(
                    self.annotate_image(
                        raw_image=result.orig_img,
                        detections=detections,
                    )
                )

                if self.verbose:
                    logger.info("Frame written.")

    def annotate_image(
        self,
        raw_image: ndarray,
        detections: Detections,
    ):
        box_annotator = BoxAnnotator(
            text_thickness=1,
            text_padding=5,
            text_scale=0.3,
        )

        labels = [
            f"id={tracker_id} class={self.yolo.names[class_id] if self.yolo.names and class_id else class_id} {confidence:.2f}"
            for xyxy, mask, confidence, class_id, tracker_id in detections
        ]

        frame = box_annotator.annotate(
            scene=raw_image,
            detections=detections,
            labels=labels,
        )

        return frame
