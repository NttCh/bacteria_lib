"""
Train pipeline module for bacteria_lib.

Contains functions for training (single split or cross-validation),
continuing training, hyperparameter tuning, and evaluation.
"""

import os
import datetime
import copy
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import optuna
import albumentations as A
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Tuple, Optional
from omegaconf import DictConfig, ListConfig
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

from torch.utils.data import DataLoader

from .callbacks import PlotMetricsCallback, OptunaReportingCallback
from .utils import load_obj
from .data import PatchClassificationDataset
from .models import build_classifier
from . import utils  # For set_seed if needed.


class LitClassifier(pl.LightningModule):
    """
    A PyTorch Lightning module for classification.
    """
    def __init__(self, cfg: DictConfig, model: Any, num_classes: int) -> None:
        """
        Initialize the Lightning module.

        Args:
            cfg (DictConfig): The configuration.
            model (Any): The model.
            num_classes (int): The number of output classes.
        """
        super().__init__()
        import torch.nn as nn
        import torch
        self.cfg = cfg
        self.model = model
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        import torch.optim as optim
        cfg_optimizer = load_obj(self.cfg.optimizer.class_name)
        optimizer_params = self.cfg.optimizer.params
        optimizer = cfg_optimizer(self.model.parameters(), **optimizer_params)

        cfg_scheduler = load_obj(self.cfg.scheduler.class_name)
        scheduler_params = self.cfg.scheduler.params
        scheduler = cfg_scheduler(optimizer, **scheduler_params)

        return [optimizer], [{
            "scheduler": scheduler, 
            "interval": self.cfg.scheduler.step, 
            "monitor": self.cfg.scheduler.monitor
        }]


def train_stage(cfg: DictConfig, csv_path: str, num_classes: int, stage_name: str,
                trial: Optional[optuna.trial.Trial] = None, suppress_metrics: bool = False
               ) -> Tuple[pl.LightningModule, Any]:
    """
    Train the model using a single train/validation split.
    """
    from torch.utils.data import DataLoader
    full_df = pd.read_csv(csv_path)
    train_df, valid_df = train_test_split(
        full_df,
        test_size=cfg.data.valid_split,
        random_state=cfg.training.seed,
        stratify=full_df[cfg.data.label_col]
    )
    train_transforms = A.Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.train.augs])
    valid_transforms = A.Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs])

    train_dataset = PatchClassificationDataset(train_df, cfg.data.folder_path, transforms=train_transforms)
    valid_dataset = PatchClassificationDataset(valid_df, cfg.data.folder_path, transforms=valid_transforms)
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    from .models import build_classifier
    model = build_classifier(cfg, num_classes=num_classes)
    lit_model = LitClassifier(cfg, model, num_classes)

    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    stage_id = f"{stage_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    save_dir = os.path.join("temp_logs", stage_id)  # You can override this in main if needed.
    os.makedirs(save_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=save_dir, name=f"{cfg.general.project_name}_{stage_name}")

    max_epochs = (cfg.training.tuning_epochs_detection 
                  if stage_name == "detection" 
                  else cfg.training.tuning_epochs_classification)

    callbacks = []
    if not suppress_metrics:
        callbacks.append(PlotMetricsCallback())
    callbacks.append(EarlyStopping(monitor="val_loss", patience=3, mode="min"))
    callbacks.append(ModelCheckpoint(
        dirpath=save_dir, 
        monitor="val_acc", 
        mode="max", 
        filename=f"{stage_name}" + "-{epoch:02d}-{val_acc:.4f}"
    ))
    if trial is not None:
        callbacks.append(OptunaReportingCallback(trial, metric_name="val_acc"))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        logger=logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=callbacks
    )
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    return lit_model, trainer.callback_metrics.get("val_acc")


def train_with_cross_validation(cfg: DictConfig, csv_path: str, num_classes: int, stage_name: str,
                                trial: Optional[optuna.trial.Trial] = None, suppress_metrics: bool = False
                               ) -> Tuple[pl.LightningModule, float]:
    """
    Train the model using k-fold stratified cross-validation.
    """
    from torch.utils.data import DataLoader
    full_df = pd.read_csv(csv_path)
    skf = StratifiedKFold(n_splits=cfg.training.num_folds, shuffle=True, random_state=cfg.training.seed)
    val_scores = []
    fold_models = []
    X = full_df
    y = full_df[cfg.data.label_col]

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/{cfg.training.num_folds}")
        train_df = full_df.iloc[train_idx]
        valid_df = full_df.iloc[valid_idx]

        train_transforms = A.Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.train.augs])
        valid_transforms = A.Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs])

        train_dataset = PatchClassificationDataset(train_df, cfg.data.folder_path, transforms=train_transforms)
        valid_dataset = PatchClassificationDataset(valid_df, cfg.data.folder_path, transforms=valid_transforms)
        train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

        from .models import build_classifier
        model = build_classifier(cfg, num_classes=num_classes)
        lit_model = LitClassifier(cfg, model, num_classes)

        import pytorch_lightning as pl
        from pytorch_lightning.loggers import TensorBoardLogger
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

        stage_id = f"{stage_name}_fold{fold+1}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        save_dir = os.path.join("temp_logs", stage_id)
        os.makedirs(save_dir, exist_ok=True)
        logger = TensorBoardLogger(save_dir=save_dir, name=f"{cfg.general.project_name}_{stage_name}_fold{fold+1}")

        callbacks = []
        if not suppress_metrics:
            callbacks.append(PlotMetricsCallback())
        callbacks.append(EarlyStopping(monitor="val_loss", patience=3, mode="min"))
        callbacks.append(ModelCheckpoint(
            dirpath=save_dir, 
            monitor="val_acc", 
            mode="max", 
            filename=f"{stage_name}_fold{fold+1}" + "-{epoch:02d}-{val_acc:.4f}"
        ))
        if trial is not None:
            callbacks.append(OptunaReportingCallback(trial, metric_name="val_acc"))

        fold_epochs = (cfg.training.tuning_epochs_detection 
                       if stage_name == "detection" 
                       else cfg.training.tuning_epochs_classification)

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
    """
    from torch.utils.data import DataLoader
    full_df = pd.read_csv(csv_path)
    train_df, valid_df = train_test_split(
        full_df,
        test_size=cfg.data.valid_split,
        random_state=cfg.training.seed,
        stratify=full_df[cfg.data.label_col]
    )
    train_transforms = A.Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.train.augs])
    valid_transforms = A.Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs])

    train_dataset = PatchClassificationDataset(train_df, cfg.data.folder_path, transforms=train_transforms)
    valid_dataset = PatchClassificationDataset(valid_df, cfg.data.folder_path, transforms=valid_transforms)
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    additional_epochs = (cfg.training.additional_epochs_detection 
                         if stage_name == "detection" 
                         else cfg.training.additional_epochs_classification)

    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from .callbacks import PlotMetricsCallback

    stage_id = f"{stage_name}_continued_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    save_dir = os.path.join("temp_logs", stage_id)
    os.makedirs(save_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=save_dir, name=f"{cfg.general.project_name}_{stage_name}_continued")

    callbacks = [
        PlotMetricsCallback(),
        EarlyStopping(monitor="val_loss", patience=2, mode="min"),
        ModelCheckpoint(
            dirpath=save_dir, 
            monitor="val_acc", 
            mode="max", 
            filename=f"{stage_name}_continued" + "-{epoch:02d}-{val_acc:.4f}"
        )
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


def objective_stage(trial: optuna.trial.Trial, stage: str) -> float:
    """
    The Optuna objective function for hyperparameter tuning.
    """
    # Implementation remains the same; see your original logic
    # ...
    pass


def evaluate_model(model: pl.LightningModule, csv_path: str, cfg: DictConfig, stage: str) -> None:
    """
    Evaluate the model on a validation split and plot the confusion matrix.
    """
    # Implementation remains the same; see your original logic
    # ...
    pass


def evaluate_on_test(model: pl.LightningModule, test_csv: str, cfg: DictConfig) -> None:
    """
    Evaluate the model on a dedicated test set.
    """
    # Implementation remains the same; see your original logic
    # ...
    pass
