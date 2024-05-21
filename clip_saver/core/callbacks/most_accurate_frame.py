from collections import defaultdict
from dataclasses import dataclass

from ..datatypes.frame import Frame
from .tracker import TrackerIdCallback


@dataclass(kw_only=True)
class MostAccurateFrame:
    frame: Frame
    start_time: str
    end_time: str


class MostAccurateFrameCallback(TrackerIdCallback):
    trackid_to_label_to_frames: dict[int, dict[int, list[MostAccurateFrame]]]

    def run(self, frame: Frame):
        if frame.detections.tracker_id is None or frame.detections.class_id is None:
            return

        for track_id, class_id, conf in zip(
            frame.detections.tracker_id,
            frame.detections.class_id,
            frame.detections.confidence,
        ):
            prev_most_accurate = self.trackid_to_label_to_frames.get(track_id, {}).get(
                class_id
            )
            if prev_most_accurate is None:
                self.trackid_to_label_to_frames[track_id][class_id] = [
                    MostAccurateFrame(
                        frame=frame,
                        start_time=frame.timestamp,
                        end_time=frame.timestamp,
                    )
                ]
                return

            # Find most accurate frame
            prev_most_accurate = prev_most_accurate[-1]
            prev_confidence = self.get_confidence(
                prev_most_accurate.frame, track_id, class_id
            )
            if conf > prev_confidence:
                self.trackid_to_label_to_frames[track_id][class_id] = [
                    MostAccurateFrame(
                        frame=frame,
                        start_time=prev_most_accurate.start_time,
                        end_time=frame.timestamp,
                    )
                ]
            else:
                prev_most_accurate.end_time = frame.timestamp

    def get_confidence(self, frame: Frame, track_id: int, class_id: int) -> float:
        for frame_track_id, frame_class_id, conf in zip(
            frame.detections.tracker_id,
            frame.detections.class_id,
            frame.detections.confidence,
        ):
            if frame_track_id == track_id and frame_class_id == class_id:
                return conf
        return 0
