"""
Callbacks module for bacteria_lib.

This module provides custom callbacks for metric plotting and Optuna reporting.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch_lightning as pl
from typing import List
import optuna


class PlotMetricsCallback(pl.Callback):
    """
    Callback to plot training and validation metrics at the end of training.
    """
    def __init__(self) -> None:
        super().__init__()
        self.epochs: List[int] = []
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Log metrics at the end of each validation epoch.
        """
        epoch = trainer.current_epoch
        self.epochs.append(epoch)
        train_loss = trainer.callback_metrics.get("train_loss")
        val_loss = trainer.callback_metrics.get("val_loss")
        train_acc = trainer.callback_metrics.get("train_acc")
        val_acc = trainer.callback_metrics.get("val_acc")
        self.train_losses.append(train_loss.item() if train_loss is not None else np.nan)
        self.val_losses.append(val_loss.item() if val_loss is not None else np.nan)
        self.train_accs.append(train_acc.item() if train_acc is not None else np.nan)
        self.val_accs.append(val_acc.item() if val_acc is not None else np.nan)
        print(f"Epoch {epoch}: train_loss={train_loss}, val_loss={val_loss}, train_acc={train_acc}, val_acc={val_acc}")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Plot and save the training metrics after training ends.
        """
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        axs[0].plot(self.epochs, self.train_losses, label="Train Loss", marker="o")
        axs[0].plot(self.epochs, self.val_losses, label="Validation Loss", marker="o")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Loss vs. Epoch")
        axs[0].legend()
        axs[0].grid(True)
        
        axs[1].plot(self.epochs, self.train_accs, label="Train Accuracy", marker="o")
        axs[1].plot(self.epochs, self.val_accs, label="Validation Accuracy", marker="o")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        axs[1].set_title("Accuracy vs. Epoch")
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(trainer.logger.log_dir, "metrics_plot.png")
        plt.savefig(save_path)
        print(f"Saved metrics plot to {save_path}")
        plt.show()


class OptunaReportingCallback(pl.Callback):
    """
    Callback to report validation metrics to an Optuna trial.
    """
    def __init__(self, trial: optuna.trial.Trial, metric_name: str = "val_acc") -> None:
        super().__init__()
        self.trial = trial
        self.metric_name = metric_name

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Report the specified metric to the Optuna trial.
        """
        val_metric = trainer.callback_metrics.get(self.metric_name)
        if val_metric is not None:
            self.trial.report(val_metric.item(), step=trainer.current_epoch)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
