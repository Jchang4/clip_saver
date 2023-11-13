from collections import defaultdict
from datetime import datetime

from pydantic import BaseModel, Field
from supervision import VideoInfo, VideoSink

from .base import Frame, MostAccurateFrame, StartAndEndFrames


class Buffer(BaseModel):
    frames: list[Frame] = Field(default_factory=list)

    def add_frame(self, frame: Frame):
        """Adds a frame to the buffer."""
        self.frames.append(frame)

    def get_frames(self) -> list[Frame]:
        return self.frames

    def reset(self):
        self.frames.clear()

    def save(self, class_map: dict[int, str] | None = None):
        """Saves the buffer to disk."""
        if not self.frames:
            return

        output_path = f"outputs/{datetime.utcnow().isoformat()}.mp4"

        video_info = VideoInfo(
            width=self.frames[0].raw_image.shape[1],
            height=self.frames[0].raw_image.shape[0],
            fps=25,
        )

        with VideoSink(output_path, video_info) as sink:
            for frame in self.frames:
                sink.write_frame(frame=frame.get_annotated_image(class_map=class_map))


class SamplingBuffer(Buffer):
    """Buffer that samples frames."""

    # Sample every N seconds
    sample_interval: int
    frames_per_second: int = 15
    # Initialize sample_count to sample_every so we start sampling right away.
    sample_count: int = 0

    def add_frame(self, frame: Frame):
        sample_frame_interval = self.sample_interval * self.frames_per_second
        sample_count = self.sample_count

        self.sample_count %= sample_frame_interval

        if sample_count % (sample_frame_interval) == 0:
            super().add_frame(frame)


class MostAccurateFrameBuffer(Buffer):
    # TrackID to Class ID to AccurateFrame
    frames: dict[int, dict[int, MostAccurateFrame]] = Field(
        default_factory=lambda: defaultdict(dict)
    )

    def add_frame(self, frame: Frame):
        if (
            not frame.detections.class_id
            or not frame.detections.confidence
            or not frame.detections.tracker_id
        ):
            return

        for i, (class_id, confidence, tracker_id) in enumerate(
            zip(
                frame.detections.class_id,
                frame.detections.confidence,
                frame.detections.tracker_id,
            )
        ):
            accurate_frame = self.frames.get(tracker_id, {}).get(class_id)
            if not accurate_frame:
                self.frames[tracker_id][class_id] = MostAccurateFrame(
                    frame=frame,
                    start_time=frame.timestamp,
                    end_time=frame.timestamp,
                )
                return

            prev_frame = accurate_frame.frame

            assert (
                prev_frame.detections.class_id
                and prev_frame.detections.confidence
                and prev_frame.detections.tracker_id
            )

            prev_confidence = 0
            for prev_class_id, prev_conf, prev_tracker_id in zip(
                prev_frame.detections.class_id,
                prev_frame.detections.confidence,
                prev_frame.detections.tracker_id,
            ):
                if prev_class_id == class_id and prev_tracker_id == tracker_id:
                    prev_confidence = max(prev_confidence, prev_conf)

            if confidence > prev_confidence:
                accurate_frame.frame = frame

            accurate_frame.start_time = min(
                accurate_frame.start_time,
                frame.timestamp,
            )
            accurate_frame.end_time = max(
                accurate_frame.end_time,
                frame.timestamp,
            )

    def get_frames(self) -> list[Frame]:
        return [
            accurate_frame.frame
            for tracker_id, class_id_to_accurate_frame in self.frames.items()
            for accurate_frame in class_id_to_accurate_frame.values()
        ]


class StartAndEndFramesBuffer(Buffer):
    # Tracker ID to Class ID to StartAndEndFrames
    frames: dict[int, dict[int, StartAndEndFrames]] = Field(
        default_factory=lambda: defaultdict(dict)
    )

    def add_frame(self, frame: Frame):
        if not frame.detections.class_id or not frame.detections.tracker_id:
            return

        for class_id, tracker_id in zip(
            frame.detections.class_id, frame.detections.tracker_id
        ):
            start_and_end_frames = self.frames.get(tracker_id, {}).get(class_id)
            if not start_and_end_frames:
                self.frames[tracker_id][class_id] = StartAndEndFrames(
                    start=frame,
                    end=frame,
                )
                return

            start_and_end_frames.start = min(
                start_and_end_frames.start,
                frame,
                key=lambda frame: frame.timestamp,
            )
            start_and_end_frames.end = max(
                start_and_end_frames.end,
                frame,
                key=lambda frame: frame.timestamp,
            )

    def get_frames(self) -> list[Frame]:
        return [
            frame
            for tracker_id, class_id_to_start_and_end_frames in self.frames.items()
            for start_and_end_frames in class_id_to_start_and_end_frames.values()
            for frame in [start_and_end_frames.start, start_and_end_frames.end]
            if frame
        ]


# class SampleWithStartAndEndFrames(SamplingBuffer, StartAndEndFrames):
#     pass
