from threading import Thread
from typing import Any, Iterable

import cv2
from cv2.typing import MatLike
from pydantic import BaseModel, ConfigDict


class Connection(BaseModel):
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    video_source: str

    max_retries: int = 10
    retry_count: int = 0
    thread: Thread | None = None

    def get_frame(self) -> MatLike | None:
        video_buffer = cv2.VideoCapture(self.video_source)
        ret, raw_frame = video_buffer.read()
        return raw_frame if ret else None

    # def disconnect(self):
    #     pass
    #     # self.video_buffer.release()

    @property
    def is_connected(self):
        return True
        # return self.video_buffer.isOpened()

    # def reconnect(self):
    #     # Use a thread so we reconnect in the background
    #     if self.thread:
    #         self.retry_count += 1
    #         return

    #     def _reconnect():
    #         self.disconnect()

    #         if self.retry_count >= self.max_retries:
    #             raise ConnectionError(f"Failed to connect to {self.video_source}")

    #         self.video_buffer = cv2.VideoCapture(self.video_source)

    #         if self.is_connected:
    #             self.retry_count = 0
    #             self.thread = None

    #     self.thread = Thread(target=_reconnect)
    #     self.thread.start()


def get_rtsp_url(
    username: str,
    password: str,
    ip: str,
    port: int = 554,
    channel: int = 1,
    subtype: int = 1,
):
    return f"rtsp://{username}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"
