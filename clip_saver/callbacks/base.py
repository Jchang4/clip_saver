from abc import ABC, abstractmethod

from ..datatypes.frame import Frame


class BaseCallback(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def run(self, frame: Frame):
        pass

    @abstractmethod
    def stop(self):
        pass
