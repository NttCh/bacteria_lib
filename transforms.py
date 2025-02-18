"""
Transforms module for bacteria_lib.

This module provides custom Albumentations transforms, such as ToGray3.
"""

from .utils import load_obj  # If needed
from .callbacks import PlotMetricsCallback  # Example import if needed

class ToGray3:
    """
    Custom transform to convert an image to grayscale and replicate it to create three channels.
    """
    def __init__(self, always_apply: bool = False, p: float = 1.0):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, image):
        import cv2
        gray = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_RGB2GRAY)
        gray_3 = np.stack([gray, gray, gray], axis=-1).astype("float32")
        return gray_3
