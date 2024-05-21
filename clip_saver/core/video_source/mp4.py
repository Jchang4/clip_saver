from .base import VideoSource


class MP4VideoSource(VideoSource):
    source: str

    def __init__(self, source: str):
        self.source = source

    def get_video_url(self) -> str:
        return self.source
