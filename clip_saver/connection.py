import cv2
from cv2.typing import MatLike
from pydantic import BaseModel, ConfigDict


class Connection(BaseModel):
    """Connect to a video source, such as an RTSP stream.

    Args:
        video_source (str): The video source to connect to
    """

    model_config = ConfigDict(frozen=True)

    video_source: str

    def get_image(self) -> MatLike | None:
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
