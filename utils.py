"""
Utility module for bacteria_lib.

This module includes helper functions like load_obj and set_seed.
"""

import numpy as np
import torch

def load_obj(obj_path: str) -> any:
    """
    Dynamically load an object from a given module path.

    Args:
        obj_path (str): Dotted module path.

    Returns:
        any: The loaded object.
    """
    parts = obj_path.split(".")
    module_path = ".".join(parts[:-1])
    obj_name = parts[-1]
    module = __import__(module_path, fromlist=[obj_name])
    return getattr(module, obj_name)


def set_seed(seed: int = 666) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The random seed.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
