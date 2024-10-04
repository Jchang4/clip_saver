from collections import defaultdict

from ..datatypes.frame import Frame
from .base import Callback


class TrackerIdCallback(Callback):
    trackid_to_label_to_frames: dict[int, dict[int, list[Frame]]]

    def __init__(self):
        self.trackid_to_label_to_frames = defaultdict(lambda: defaultdict(list))

    def start(self):
        self.trackid_to_label_to_frames = defaultdict(lambda: defaultdict(list))

    def run(self, frame: Frame):
        if frame.detections.tracker_id is None or frame.detections.class_id is None:
            return

        for track_id, class_id in zip(
            frame.detections.tracker_id, frame.detections.class_id
        ):
            self.trackid_to_label_to_frames[track_id][class_id].append(frame)

    def stop(self):
        pass

    def get_frames(self) -> list[Frame]:
        return [
            frame
            for track_id, label_to_frames in self.trackid_to_label_to_frames.items()
            for label, frames in label_to_frames.items()
            for frame in frames
        ]

    def get_latest_frame(self) -> Frame | None:
        frames = sorted(self.get_frames(), key=lambda f: f.timestamp, reverse=True)
        if not frames:
            return None
        return frames[0]
