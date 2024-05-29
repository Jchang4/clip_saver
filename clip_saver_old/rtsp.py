from dataclasses import dataclass


@dataclass
class RtspUrl:
    username: str
    password: str
    ip_address: str
    port: int
    channel: int
    subtype: int

    def __str__(self):
        return f"rtsp://{self.username}:{self.password}@{self.ip_address}:{self.port}/cam/realmonitor?channel={self.channel}&subtype={self.subtype}"

    @classmethod
    def from_rtsp_url(cls, rtsp_url: str) -> "RtspUrl":
        """Converts RTSP URL to a RtspUrl.

        Example:
            rtsp://admin:password@22.222.222.222:80/cam/realmonitor?channel=1&subtype=1

        Args:
            rtsp_url (str): _description_

        Returns:
            str: _description_
        """
        username, password = rtsp_url.split("@")[0].split("://")[1].split(":")
        ip, port = rtsp_url.split("@")[1].split("/")[0].split(":")
        channel, subtype = rtsp_url.split("?")[1].split("&")
        channel = int(channel.split("=")[1])
        subtype = int(subtype.split("=")[1])

        return cls(
            username=username,
            password=password,
            ip_address=ip,
            port=int(port),
            channel=channel,
            subtype=subtype,
        )

    @classmethod
    def from_yolo_path(cls, result_path: str) -> "RtspUrl":
        """Converts YOLO prediction `path` to a valid RTSP URL.

        Example:
            rtsp_//admin_password_22.222.222.222_80/cam/realmonitor_channel_1_subtype_1
            rtsp://admin_password@22.222.222.222:80/cam/realmonitor?channel=1&subtype=1

        Args:
            rtsp_url (str): _description_

        Returns:
            str: _description_
        """
        rtsp_url = result_path.split("//")[1].split("/")[0]
        username, password, ip_address, port = rtsp_url.split("_")

        options = (
            result_path.split("//")[1]
            .split("/", maxsplit=1)[1]
            .replace("cam/realmonitor_", "")
            .split("_")
        )
        _, channel, _, subtype = options
        return cls(
            username=username,
            password=password,
            ip_address=ip_address,
            port=int(port),
            channel=int(channel),
            subtype=int(subtype),
        )
