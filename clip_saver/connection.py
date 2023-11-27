from threading import Lock, Thread

import cv2
from cv2.typing import MatLike

cv2.CAP_PROP_BUFFERSIZE = 1


class Connection:
    video_source: str
    buffer: cv2.VideoCapture
    last_frame: MatLike | None = None
    last_ready: bool | None = None
    lock: Lock

    def __init__(self, video_source: str):
        self.video_source = video_source
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
        self.buffer = cv2.VideoCapture(self.video_source)
        return self.buffer


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


def get_camera_from_rtsp_url(rtsp_url: str):
    """Get the camera from an RTSP URL.

    Args:
        rtsp_url (str): The RTSP URL in the format
                `rtsp://username:password@ip:port/cam/realmonitor?channel=channel&subtype=subtype`

    Returns:
        dict[str, Any]: The camera details
            - username (str): username used to login to the camera
            - password (str): password used to login to the camera
            - ip_address (str): IP address of the camera
            - port (int): port of the camera
            - channel (int): channel to use. Defaults to 1.
            - subtype (int): subtype to use. For Lorex cameras this sets the video quality. Defaults to 1.
    """
    username, password = rtsp_url.split("@")[0].split("://")[1].split(":")
    ip, port = rtsp_url.split("@")[1].split("/")[0].split(":")
    channel, subtype = rtsp_url.split("?")[1].split("&")
    channel = int(channel.split("=")[1])
    subtype = int(subtype.split("=")[1])

    return {
        "username": username,
        "password": password,
        "ip_address": ip,
        "port": port,
        "channel": channel,
        "subtype": subtype,
    }
