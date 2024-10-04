import cv2
import numpy as np


def bgr_to_rgb(image: np.ndarray):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
