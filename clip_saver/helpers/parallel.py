from threading import Thread
from typing import Any, Callable


def run_in_background(fn: Callable, *args: Any):
    # Wait for thread to finish before shutting down
    thread = Thread(target=fn, args=args, daemon=False)
    thread.start()
    return thread
