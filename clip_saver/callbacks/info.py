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

        for track_id, class_id, conf in zip(
            detections.tracker_id, detections.class_id, detections.confidence
        ):
            print(f"Track ID: {track_id}, Class ID: {class_id}, Confidence: {conf}")

    def stop(self):
        print("Stopping")
