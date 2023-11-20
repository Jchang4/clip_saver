from threading import Thread
from typing import Any, Iterable

import cv2
from cv2.typing import MatLike
from pydantic import BaseModel, ConfigDict, Field


class Connection(BaseModel):
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    video_source: str
    max_retries: int = 10
    retry_count: int = 0

    thread: Thread | None = None
    video_buffer: cv2.VideoCapture = Field(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_buffer = self._get_bufer()

    def get_frame(self) -> MatLike | None:
        ret, raw_frame = self.video_buffer.read()

        if not ret:
            self.reconnect()
            return None

        return raw_frame

    def disconnect(self):
        self.video_buffer.release()

    @property
    def is_connected(self):
        return self.video_buffer.isOpened()

    def _get_bufer(self):
        return cv2.VideoCapture(self.video_source)

    def reconnect(self):
        # Use a thread so we reconnect in the background
        if self.thread:
            self.retry_count += 1
            return

        def _reconnect():
            self.disconnect()

            if self.retry_count >= self.max_retries:
                raise ConnectionError(f"Failed to connect to {self.video_source}")

            self.video_buffer = cv2.VideoCapture(self.video_source)

            if self.is_connected:
                self.retry_count = 0
                self.thread = None

        self.thread = Thread(target=_reconnect, daemon=True)
        self.thread.start()


def get_rtsp_url(
    username: str,
    password: str,
    ip: str,
    port: int = 554,
    channel: int = 1,
    subtype: int = 1,
):
    return f"rtsp://{username}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"
