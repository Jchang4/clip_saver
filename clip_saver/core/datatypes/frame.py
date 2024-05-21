from dataclasses import dataclass

import numpy as np
import supervision as sv


@dataclass
class Frame:
    image: np.ndarray
    detections: sv.Detections
    timestamp: str
    video_path: str
