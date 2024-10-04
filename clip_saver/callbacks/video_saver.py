import numpy as np
import supervision as sv

from ..datatypes.frame import Frame
from .base import BaseCallback


class VideoSaverCallback(BaseCallback):
    output_path: str
    video_sink: sv.VideoSink | None = None
    skip_no_detections: bool = False

    def __init__(self, output_path: str, skip_no_detections: bool = False):
        self.output_path = output_path
        self.skip_no_detections = skip_no_detections

    def get_video_sink(self, width: int, height: int, fps: int):
        if self.video_sink is not None:
            return self.video_sink
        return sv.VideoSink(
            target_path=self.output_path,
            video_info=sv.VideoInfo(width=width, height=height, fps=fps),
        ).__enter__()

    def start(self):
        pass

    def run(self, frame: Frame):
        self.video_sink = self.get_video_sink(
            width=frame.image.shape[1], height=frame.image.shape[0], fps=15
        )

        if (
            not self.skip_no_detections
            and frame.detections.class_id is None
            or len(frame.detections.class_id) == 0
        ):
            self.video_sink.write_frame(frame.image)
            return

        bounding_box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_image = bounding_box_annotator.annotate(
            scene=frame.image, detections=frame.detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image,
            detections=frame.detections,
            labels=[
                f"{class_name} {confidence:.2f}"
                for class_name, confidence in zip(
                    frame.detections["class_name"], frame.detections.confidence
                )
            ],
        )

        self.video_sink.write_frame(annotated_image)

    def stop(self):
        if self.video_sink:
            self.video_sink.__exit__(None, None, None)
