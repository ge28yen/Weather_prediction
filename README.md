# Weather Prediction with Transformer-Based MVP Model

This project forecasts weather using ERA5 reanalysis data and a transformer-based model (MVPModel). It preprocesses data from a public Zarr store, creates PyTorch datasets for both vertical and surface variables, and trains a model to predict future weather conditions.

## Requirements

- Python 3
- numpy
- xarray
- gcsfs
- torch
- torchvision
- wandb

## Installation

Install the required packages with:

`pip install numpy xarray gcsfs torch torchvision wandb`

## Usage

1. **Data Preprocessing:**  
   Load ERA5 data from the Zarr store, select a time and spatial region, normalize each variable, and stack the data into arrays.

2. **Dataset & Model:**  
   A custom PyTorch dataset (`PanguEraDataset`) creates sequences of inputs and targets. The MVPModel embeds 3D vertical and 2D surface patches, processes them with transformer blocks, and recovers the original resolution.

3. **Training & Evaluation:**  
   The model is trained using Adam (lr=1e-4) with an MSE-based loss and evaluated on both a held-out test set and an October data slice. Metrics are logged with Weights & Biases.

## License

This project is for educational purposes. (Add your license details here.)
