import cv2
import numpy as np
import albumentations as A

class ToGray3(A.ImageOnlyTransform):
    """
    Convert an RGB image to grayscale and replicate it to create a three-channel image.

    This transform converts the given RGB image to a grayscale image using OpenCV and 
    then replicates the grayscale channel into three channels. This is useful when you 
    need to provide a three-channel input to a model or further processing steps but want 
    to work with grayscale data.

    Args:
        always_apply (bool): If True, always apply the transform. Defaults to False.
        p (float): The probability of applying the transform. Defaults to 1.0.
    """
    def __init__(self, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply the transform to an image.

        Args:
            image (np.ndarray): The input RGB image as a NumPy array.
            **params: Additional keyword arguments.

        Returns:
            np.ndarray: The transformed image in grayscale with three channels.
        """
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_3 = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
        return gray_3
