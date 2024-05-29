from dataclasses import dataclass

import numpy as np
import supervision as sv
from PIL import Image

from ..helpers.image import bgr_to_rgb


@dataclass
class Frame:
    image: np.ndarray
    detections: sv.Detections
    timestamp: str
    video_path: str

    def get_image(self) -> Image.Image:
        return Image.fromarray(bgr_to_rgb(self.image))
