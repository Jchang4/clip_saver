from datetime import datetime

from .base import BaseCallback


class PrintInfoCallback(BaseCallback):
    def start(self):
        print("Starting")

    def run(self, frame):
        if not frame or not frame.detections:
            return

        detections = frame.detections
        print(f"Detections: {len(detections)} - {datetime.now().isoformat()}")

        for i, (class_id, conf) in enumerate(
            zip(detections.class_id, detections.confidence)
        ):
            track_id = (
                detections.tracker_id[i] if detections.tracker_id is not None else -1
            )
            print(f"Track ID: {track_id}, Class ID: {class_id}, Confidence: {conf}")

    def stop(self):
        print("Stopping")
