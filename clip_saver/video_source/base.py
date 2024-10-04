from abc import ABC, abstractmethod


class BaseVideoSource(ABC):
    @abstractmethod
    def get_video_url(self) -> str: ...
