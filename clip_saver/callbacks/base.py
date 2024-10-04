from ..datatypes.frame import Frame


class Callback:
    def start(self):
        raise NotImplementedError()

    def run(self, frame: Frame):
        raise NotImplementedError()

    def stop(self):
        raise NotImplementedError()
