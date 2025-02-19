"""
Data module for bacteria_lib.

Provides the PatchClassificationDataset for loading image data and labels.
"""

import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Any, Optional, Tuple


class PatchClassificationDataset(Dataset):
    """
    A dataset class for patch-based image classification.
    """
    def __init__(self, data: Any, image_dir: str, transforms: Optional[Any] = None) -> None:
        """
        Initialize the dataset.

        Args:
            data (Any): A CSV file path or a DataFrame containing image paths and labels.
            image_dir (str): Directory where images are stored.
            transforms (Optional[Any]): Albumentations transforms to apply.
        """
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data

        valid_rows = []
        for idx, row in df.iterrows():
            primary_path = os.path.join(image_dir, row["image_path"])
            alternative_folder = image_dir.replace("Fa", "test")
            alternative_path = os.path.join(alternative_folder, row["image_path"])
            if os.path.exists(primary_path) or os.path.exists(alternative_path):
                valid_rows.append(row)
            else:
                print(f"Warning: Image not found for row {idx}: {primary_path} or {alternative_path}. Skipping sample.")
        self.df = pd.DataFrame(valid_rows)
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """
        Get an image and its label.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Any, int]: The transformed image and its label.
        """
        row = self.df.iloc[idx]
        primary_path = os.path.join(self.image_dir, row["image_path"])
        if os.path.exists(primary_path):
            image_path = primary_path
        else:
            alternative_folder = self.image_dir.replace("Fa", "test")
            alternative_path = os.path.join(alternative_folder, row["image_path"])
            if os.path.exists(alternative_path):
                image_path = alternative_path
                print(f"Using alternative image path: {image_path}")
            else:
                raise FileNotFoundError(
                    f"Image not found in either location: {primary_path} or {alternative_path}"
                )
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        label = int(row["label"])
        return image, label
