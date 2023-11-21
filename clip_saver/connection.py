from threading import Thread
from typing import Any, Iterable

import cv2
from cv2.typing import MatLike
from pydantic import BaseModel, ConfigDict, Field


class Connection(BaseModel):
    """Connect to a video source, such as an RTSP stream.

    Args:
        video_source (str): The video source to connect to
    """

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    video_source: str

    def get_frame(self) -> MatLike | None:
        """Get the latest frame from the video source.

        We always reconnect when getting frames to ensure
        we always get the latest frame.

        Returns:
            MatLike | None: The latest frame from the video source
        """
        buffer = self._get_bufer()
        ret, raw_frame = buffer.read()
        buffer.release()
        return raw_frame if ret else None

    def _get_bufer(self):
        return cv2.VideoCapture(self.video_source)


def get_rtsp_url(
    username: str,
    password: str,
    ip: str,
    port: int = 554,
    channel: int = 1,
    subtype: int = 1,
):
    """Get the RTSP URL for a camera.

    Args:
        username (str): username used to login to the camera
        password (str): password used to login to the camera
        ip (str): IP address of the camera
        port (int, optional): port of the camera. Defaults to 554.
        channel (int, optional): channel to use. Defaults to 1.
        subtype (int, optional): subtype to use. For Lorex cameras this sets the video quality. Defaults to 1.

    Returns:
        str: The RTSP URL
    """
    return f"rtsp://{username}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"
