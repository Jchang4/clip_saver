from .base import BaseVideoSource


class FileVideoSource(BaseVideoSource):
    file_path: str

    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_video_url(self) -> str:
        return self.file_path
