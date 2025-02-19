# bacteria_lib

**bacteria_lib** is a Python library for bacteria detection and classification using deep learning.  
It provides core modules for data loading, model building, training pipelines, custom transforms, callbacks, and utility functions.  
This library is designed to be used as a dependency in other projects (such as a main training application).

## Features

- **Data Module:**  
  Provides `PatchClassificationDataset` for loading image data from CSV files.

- **Model Building:**  
  Contains functions such as `build_classifier` to build models with various backbones (e.g., ResNet50).

- **Custom Transforms:**  
  Includes Albumentations transforms like `ToGray3` to convert images to grayscale and replicate channels.

- **Training Pipeline:**  
  Offers functions for single-split and cross-validation training (`train_stage`, `train_with_cross_validation`), continuing training, and evaluation (`evaluate_model`, `evaluate_on_test`).

- **Callbacks:**  
  Custom PyTorch Lightning callbacks for plotting metrics (`PlotMetricsCallback`) and reporting metrics to Optuna (`OptunaReportingCallback`).

- **Utilities:**  
  Helper functions such as `load_obj` and `set_seed`.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/bacteria_lib.git
   cd bacteria_lib
