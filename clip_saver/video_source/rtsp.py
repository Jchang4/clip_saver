from .base import BaseVideoSource


class RTSPVideoSource(BaseVideoSource):
    rtsp_url: str

    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url

    def get_video_url(self) -> str:
        return self.rtsp_url


class MultiRTSPVideoSource(BaseVideoSource):
    rtsp_urls: list[str]

    def __init__(self, rtsp_urls: list[str]):
        self.rtsp_urls = rtsp_urls

        with open("list.streams", "w") as f:
            f.write("\n".join(rtsp_urls))

    def get_video_url(self) -> str:
        return "list.streams"
