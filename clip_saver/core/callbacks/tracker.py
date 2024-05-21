from collections import defaultdict

from ..datatypes.frame import Frame
from .base import Callback


class TrackerIdCallback(Callback):
    trackid_to_frame: dict[int, list[Frame]]

    def __init__(self):
        self.trackid_to_frame = defaultdict(list)

    def start(self):
        pass

    def run(self, frame: Frame):
        if frame.detections.tracker_id is None:
            return

        for track_id in frame.detections.tracker_id:
            self.trackid_to_frame[track_id].append(frame)

    def stop(self):
        pass
