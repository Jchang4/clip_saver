from abc import ABC, abstractmethod
from threading import Thread
from typing import Any, Callable

from pydantic import BaseModel

from .base import Frame


class Callback(BaseModel):
    """
    Base class for all callbacks.
    """

    pass


class DetectionCallback(Callback, ABC):
    @abstractmethod
    def on_detection_start(self, frame: Frame):
        pass

    @abstractmethod
    def on_detection(self, frame: Frame):
        pass

    @abstractmethod
    def on_detection_end(self, frames: list[Frame]):
        pass


def run_in_background(fn: Callable):
    thread = Thread(target=fn)
    thread.start()
    return thread
