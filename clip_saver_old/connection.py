from threading import Lock, Thread

import cv2
from cv2.typing import MatLike

from .rtsp import RtspUrl

cv2.CAP_PROP_BUFFERSIZE = 1


class Connection:
    rtsp_url: RtspUrl
    buffer: cv2.VideoCapture
    last_frame: MatLike | None = None
    last_ready: bool | None = None
    lock: Lock

    def __init__(self, rtsp_url: str):
        self.rtsp_url = RtspUrl.from_rtsp_url(rtsp_url)
        self.buffer = self.get_bufer()
        self.lock = Lock()
        thread = Thread(
            target=self.rtsp_cam_buffer,
            name="rtsp_read_thread",
            daemon=True,
        )
        thread.start()

    def rtsp_cam_buffer(self):
        while True:
            with self.lock:
                self.last_ready, self.last_frame = self.buffer.read()
                if not self.last_ready:
                    self.buffer.release()
                    self.get_bufer()

    def get_image(self) -> MatLike | None:
        if (self.last_ready is not None) and (self.last_frame is not None):
            return self.last_frame.copy()
        return None

    def get_bufer(self):
        self.buffer = cv2.VideoCapture(str(self.rtsp_url))
        return self.buffer
