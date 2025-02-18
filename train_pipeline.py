#!/usr/bin/env python
"""
Train pipeline module for bacteria_lib.

This module contains functions for training a model using either a single 
train/validation split or k-fold cross-validation, for continuing training, 
and for evaluating the model.
"""

import os
import datetime
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from typing import Any, Tuple, Optional

from omegaconf import DictConfig
import torch

from .utils import load_obj
from .data import PatchClassificationDataset
from .callbacks import PlotMetricsCallback, OptunaReportingCallback


def train_stage(cfg: DictConfig, csv_path: str, num_classes: int, stage_name: str,
                trial: Optional[pl.Trainer] = None, suppress_metrics: bool = False) -> Tuple[pl.LightningModule, Any]:
    """
    Train a model using a single train/validation split.

    Args:
        cfg (DictConfig): Configuration object.
        csv_path (str): Path to the CSV file with training data.
        num_classes (int): Number of output classes.
        stage_name (str): Stage name (e.g., "detection" or "classification").
        trial (Optional): An Optuna trial object for hyperparameter tuning.
        suppress_metrics (bool): If True, metric plotting is suppressed.

    Returns:
        Tuple[pl.LightningModule, Any]: The trained model and the validation accuracy.
    """
    full_df = pd.read_csv(csv_path)
    train_df, valid_df = train_test_split(
        full_df,
        test_size=cfg.data.valid_split,
        random_state=cfg.training.seed,
        stratify=full_df[cfg.data.label_col]
    )
    train_transforms = cfg.augmentation.train.augs
    valid_transforms = cfg.augmentation.valid.augs
    # Create Albumentations Compose objects from the list of dicts.
    import albumentations as A
    train_transforms = A.Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.train.augs])
    valid_transforms = A.Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs])
    
    train_dataset = PatchClassificationDataset(train_df, cfg.data.folder_path, transforms=train_transforms)
    valid_dataset = PatchClassificationDataset(valid_df, cfg.data.folder_path, transforms=valid_transforms)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    
    from .models import build_classifier
    from .utils import set_seed
    model = build_classifier(cfg, num_classes=num_classes)
    
    from .train_pipeline import LitClassifier  # If you decide to place your Lightning module here.
    # Alternatively, if your Lightning module is defined elsewhere (e.g., in main.py),
    # you can import it from that module.
    lit_model = LitClassifier(cfg, model, num_classes)
    
    stage_id = f"{stage_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    save_dir = os.path.join(cfg.general.save_dir, stage_id)
    os.makedirs(save_dir, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(save_dir=save_dir, name=f"{cfg.general.project_name}_{stage_name}")
    
    if stage_name == "detection":
        max_epochs = cfg.training.tuning_epochs_detection
    else:
        max_epochs = cfg.training.tuning_epochs_classification

    callbacks = []
    if not suppress_metrics:
        callbacks.append(PlotMetricsCallback())
    callbacks.append(pl.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min"))
    callbacks.append(pl.callbacks.ModelCheckpoint(
        dirpath=save_dir, monitor="val_acc", mode="max",
        filename=f"{stage_name}" + "-{{epoch:02d}}-{{val_acc:.4f}}"))
    if trial is not None:
        callbacks.append(OptunaReportingCallback(trial, metric_name="val_acc"))
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        logger=logger,
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 10),
        callbacks=callbacks
    )
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    return lit_model, trainer.callback_metrics.get("val_acc")


def train_with_cross_validation(cfg: DictConfig, csv_path: str, num_classes: int, stage_name: str,
                                trial: Optional[pl.Trainer] = None, suppress_metrics: bool = False) -> Tuple[pl.LightningModule, float]:
    """
    Train a model using k-fold stratified cross-validation.

    Args:
        cfg (DictConfig): Configuration object.
        csv_path (str): Path to the CSV file with data.
        num_classes (int): Number of output classes.
        stage_name (str): Stage name.
        trial (Optional): An Optuna trial object for tuning.
        suppress_metrics (bool): If True, suppress metric plotting.

    Returns:
        Tuple[pl.LightningModule, float]: The best model and average validation accuracy.
    """
    full_df = pd.read_csv(csv_path)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=cfg.training.num_folds, shuffle=True, random_state=cfg.training.seed)
    val_scores = []
    fold_models = []
    X = full_df
    y = full_df[cfg.data.label_col]
    
    import albumentations as A
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/{cfg.training.num_folds}")
        train_df = full_df.iloc[train_idx]
        valid_df = full_df.iloc[valid_idx]
        
        train_transforms = A.Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.train.augs])
        valid_transforms = A.Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs])
        
        from torch.utils.data import DataLoader
        train_dataset = PatchClassificationDataset(train_df, cfg.data.folder_path, transforms=train_transforms)
        valid_dataset = PatchClassificationDataset(valid_df, cfg.data.folder_path, transforms=valid_transforms)
        train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
        
        from .models import build_classifier
        model = build_classifier(cfg, num_classes=num_classes)
        from .train_pipeline import LitClassifier
        lit_model = LitClassifier(cfg, model, num_classes)
        
        stage_id = f"{stage_name}_fold{fold+1}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        save_dir = os.path.join(cfg.general.save_dir, stage_id)
        os.makedirs(save_dir, exist_ok=True)
        logger = pl.loggers.TensorBoardLogger(save_dir=save_dir, name=f"{cfg.general.project_name}_{stage_name}_fold{fold+1}")
        
        callbacks = []
        if not suppress_metrics:
            callbacks.append(PlotMetricsCallback())
        callbacks.append(pl.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min"))
        callbacks.append(pl.callbacks.ModelCheckpoint(
            dirpath=save_dir, monitor="val_acc", mode="max",
            filename=f"{stage_name}_fold{fold+1}" + "-{{epoch:02d}}-{{val_acc:.4f}}"))
        if trial is not None:
            callbacks.append(OptunaReportingCallback(trial, metric_name="val_acc"))
            
        if stage_name == "detection":
            fold_epochs = cfg.training.tuning_epochs_detection
        else:
            fold_epochs = cfg.training.tuning_epochs_classification
        
        trainer = pl.Trainer(
            max_epochs=fold_epochs,
            devices=cfg.trainer.devices,
            accelerator=cfg.trainer.accelerator,
            precision=cfg.trainer.precision,
            logger=logger,
            callbacks=callbacks
        )
        trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        val_acc = trainer.callback_metrics.get("val_acc")
        score = val_acc.item() if val_acc is not None else 0.0
        print(f"Fold {fold+1} validation accuracy: {score:.4f}")
        val_scores.append(score)
        fold_models.append(lit_model)
    avg_score = np.mean(val_scores)
    print(f"Average cross-validation accuracy for {stage_name}: {avg_score:.4f}")
    best_idx = np.argmax(val_scores)
    return fold_models[best_idx], avg_score


def continue_training(lit_model: pl.LightningModule, cfg: DictConfig, csv_path: str,
                      num_classes: int, stage_name: str) -> pl.LightningModule:
    """
    Continue training an already-trained model for additional epochs.

    Args:
        lit_model (pl.LightningModule): The pre-trained model.
        cfg (DictConfig): The configuration.
        csv_path (str): Path to the CSV file.
        num_classes (int): Number of output classes.
        stage_name (str): Stage name.

    Returns:
        pl.LightningModule: The fine-tuned model.
    """
    full_df = pd.read_csv(csv_path)
    train_df, valid_df = train_test_split(
        full_df,
        test_size=cfg.data.valid_split,
        random_state=cfg.training.seed,
        stratify=full_df[cfg.data.label_col]
    )
    import albumentations as A
    train_transforms = A.Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.train.augs])
    valid_transforms = A.Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs])
    
    from torch.utils.data import DataLoader
    train_dataset = PatchClassificationDataset(train_df, cfg.data.folder_path, transforms=train_transforms)
    valid_dataset = PatchClassificationDataset(valid_df, cfg.data.folder_path, transforms=valid_transforms)
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    
    if stage_name == "detection":
        additional_epochs = cfg.training.additional_epochs_detection
    else:
        additional_epochs = cfg.training.additional_epochs_classification

    stage_id = f"{stage_name}_continued_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    save_dir = os.path.join(cfg.general.save_dir, stage_id)
    os.makedirs(save_dir, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(save_dir=save_dir, name=f"{cfg.general.project_name}_{stage_name}_continued")
    
    callbacks = [
        PlotMetricsCallback(),
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min"),
        pl.callbacks.ModelCheckpoint(
            dirpath=save_dir, monitor="val_acc", mode="max",
            filename=f"{stage_name}_continued" + "-{{epoch:02d}}-{{val_acc:.4f}}")
    ]
    
    trainer = pl.Trainer(
        max_epochs=additional_epochs,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        logger=logger,
        callbacks=callbacks
    )
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    return lit_model


def evaluate_model(model: pl.LightningModule, csv_path: str, cfg: DictConfig, stage: str) -> None:
    """
    Evaluate the model on a validation split and plot a confusion matrix.

    Args:
        model (pl.LightningModule): The trained model.
        csv_path (str): Path to the CSV file.
        cfg (DictConfig): The configuration.
        stage (str): Stage name for labeling the plot.
    """
    full_df = pd.read_csv(csv_path)
    _, valid_df = train_test_split(
        full_df,
        test_size=cfg.data.valid_split,
        random_state=cfg.training.seed,
        stratify=full_df[cfg.data.label_col]
    )
    import albumentations as A
    valid_transforms = A.Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs])
    valid_dataset = PatchClassificationDataset(valid_df, cfg.data.folder_path, transforms=valid_transforms)
    from torch.utils.data import DataLoader
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            images, labels = batch
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {stage}")
    plt.show()
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))


def evaluate_on_test(model: pl.LightningModule, test_csv: str, cfg: DictConfig) -> None:
    """
    Evaluate the model on a dedicated test set.

    Args:
        model (pl.LightningModule): The trained model.
        test_csv (str): Path to the test CSV file.
        cfg (DictConfig): The configuration.
    """
    if not os.path.exists(test_csv):
        print("Test CSV not found.")
        return
    import albumentations as A
    test_transforms = A.Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs])
    test_dataset = PatchClassificationDataset(test_csv, cfg.data.folder_path, transforms=test_transforms)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix: Test Set")
    plt.show()
    print("Test Set Classification Report:")
    print(classification_report(all_labels, all_preds))
