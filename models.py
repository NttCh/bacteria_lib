"""
Models module for bacteria_lib.

Contains functions to build the classifier model, including backbone selection.
"""

import torch.nn as nn
from omegaconf import DictConfig
from .utils import load_obj


def build_classifier(cfg: DictConfig, num_classes: int) -> nn.Module:
    """
    Build the classifier model using a specified backbone.

    Args:
        cfg (DictConfig): The configuration.
        num_classes (int): The number of output classes.

    Returns:
        nn.Module: The classifier model.
    """
    from torchvision.models import ResNet50_Weights
    backbone_cls = load_obj(cfg.model.backbone.class_name)
    weights_str = cfg.model.backbone.params.get("weights", None)
    if weights_str is not None and isinstance(weights_str, str):
        if weights_str == "ResNet50_Weights.IMAGENET1K_V1":
            cfg.model.backbone.params["weights"] = ResNet50_Weights.IMAGENET1K_V1
        elif weights_str == "ResNet50_Weights.DEFAULT":
            cfg.model.backbone.params["weights"] = ResNet50_Weights.DEFAULT
    model = backbone_cls(**cfg.model.backbone.params)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
