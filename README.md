# Weather Prediction with a Transformer-Based MVP Model

This repository contains a complete end-to-end project for weather forecasting using a custom transformer-based architecture (MVPModel). The project preprocesses ERA5 weather data stored in Zarr format, constructs PyTorch datasets for both vertical and surface meteorological variables, and trains a model that leverages transformer blocks to predict future weather conditions.

## Table of Contents

- [Overview](#overview)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Preprocessing and Dataset Creation](#preprocessing-and-dataset-creation)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
- [Tracking and Logging](#tracking-and-logging)
- [Results](#results)
- [License](#license)

## Overview

The project uses ERA5 reanalysis data (accessed via a public Zarr store on Google Cloud Storage) to create datasets for both surface and vertical atmospheric variables. A custom PyTorch dataset (`PanguEraDataset`) is defined to create sequences for forecasting. The model uses patch embedding modules (both 3D for vertical fields and 2D for surface fields), transformer blocks with self-attention, and patch recovery layers to reconstruct predictions at the original resolution.

## Data

- **ERA5 Zarr Store:**  
  The data is loaded from a public Google Cloud Storage bucket using `xarray` and `gcsfs`. The project uses a subset of ERA5 variables such as:
  - **Surface Variables:** `2m_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, `surface_pressure`
  - **Vertical Variables:** `temperature`, `u_component_of_wind`, `v_component_of_wind` (selected levels)

Data is normalized on a per-variable basis before being stacked into NumPy arrays.

## Model Architecture

The architecture consists of three main parts:

1. **Patch Embedding:**  
   - A 3D convolution is applied to the vertical input fields.
   - A 2D convolution is applied to the surface input fields.
   - The resulting tokens are concatenated and flattened for transformer processing.

2. **Transformer Blocks:**  
   - The embedded tokens are passed through one or more transformer blocks (each consisting of multi-head self-attention and MLP layers with skip connections and layer normalization).

3. **Patch Recovery:**  
   - The tokens are reshaped and passed through deconvolution (transpose convolution) layers to reconstruct the original spatial resolution for both the 3D (vertical) branch and the 2D (surface) branch.

## Installation

Ensure you have Python 3 installed. Then install the required packages:

`pip install numpy xarray gcsfs torch torchvision wandb`

*Note:* The project was developed and tested on Google Colab using a GPU (T4). Make sure your environment has GPU support if available.

## Usage

### Preprocessing and Dataset Creation

The notebook demonstrates how to:
- Load ERA5 data from the Zarr store.
- Select a specific time slice and spatial region.
- Normalize the data per variable.
- Stack the normalized data into NumPy arrays.

A custom dataset (`PanguEraDataset`) is defined which takes the vertical and surface arrays and returns input sequences (of length 6) along with corresponding targets (with a configurable lead time).

### Model Training

The training loop is implemented using PyTorch. Key details include:

- **Loss Function:**  
  Mean Squared Error (MSE) is computed for both branches. The final loss is the sum of the square root of the MSE for the vertical and surface branches.

- **Optimizer:**  
  Adam with a learning rate of 1e-4.

- **Logging:**  
  Training and test metrics (e.g., RMSE for each variable) are logged using Weights & Biases (wandb).

Example snippet from the training loop:

```python
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        (vert_seq, surf_seq), (vert_target, surf_target) = batch
        # Use the last time-step as input for prediction
        vert_input = vert_seq[:, -1]
        surf_input = surf_seq[:, -1]
        out_3d, out_2d = model(vert_input, surf_input)
        loss = torch.sqrt(criterion(out_3d, vert_target)) + torch.sqrt(criterion(out_2d, surf_target))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    ...
```
### Evaluation

After training, the model is evaluated on both a held-out test set and on an additional October data slice. The evaluation computes per-variable RMSE for the surface predictions.

### Tracking and Logging

Weights & Biases (wandb) is used to log training and evaluation metrics. The project logs the train loss and test RMSE values for the various surface variables.

### Results

Training logs (as shown in the notebook) report progressive improvement in RMSE values over epochs. For example, after 200 epochs the RMSE for surface pressure, temperature, and wind components reach values around 0.1–0.4 (unit-dependent). Detailed logs and visualizations are available on the wandb dashboard.

### License
