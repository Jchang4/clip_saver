from pathlib import Path
from threading import Thread
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict
from ultralytics import YOLO

from .frame import Frame


class Callback(BaseModel):
    """
    Base class for all callbacks.
    """

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)
    yolo_model: YOLO
    yolo_model_path: Path


class DetectionCallback(Callback):
    def on_detection_start(self, frame: Frame):
        pass

    def on_detection(self, frame: Frame):
        pass

    def on_detection_end(self, frames: list[Frame]):
        pass


def run_in_background(fn: Callable, *args: Any):
    # Wait for thread to finish before shutting down
    thread = Thread(target=fn, args=args, daemon=False)
    thread.start()
    return thread
