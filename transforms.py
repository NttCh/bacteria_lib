"""
Transforms module for bacteria_lib.

Provides custom Albumentations transforms, such as ToGray3.
"""

import cv2
import numpy as np
import albumentations as A

class ToGray3(A.ImageOnlyTransform):
    """
    Custom Albumentations transform to convert an image to grayscale and 
    replicate it to create a three-channel image.
    """
    def __init__(self, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Convert the image to grayscale and replicate the grayscale image
        into three channels.
        
        Args:
            image (np.ndarray): The input image.
            
        Returns:
            np.ndarray: The transformed image.
        """
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_3 = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
        return gray_3
