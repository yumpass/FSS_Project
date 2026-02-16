# ViT_specklegram
Explainable Vision Transformer Based Temperature Sensing with Fiber Specklegram Sensors

This repository contains the official PyTorch implementation for the paper: **"Explainable Vision Transformer Based Temperature Sensing with Fiber Specklegram Sensors"** (Discover Sensors, 2025).

This work proposes and evaluates a Vision Transformer (ViT) model for high-accuracy temperature regression from Fiber Specklegram Sensor (FSS) images. We also provide a robust analysis of the model's decision-making process using Explainable AI (XAI) techniques.

############################################
Please cite the related article.

@article{Sadik2025,
   author = {Serif Ali Sadik},
   doi = {10.1007/s44397-025-00027-9},
   issn = {3005-1851},
   issue = {1},
   journal = {Discover Sensors},
   pages = {25},
   title = {Explainable vision transformer based temperature sensing with fiber specklegram sensors},
   volume = {1},
   url = {https://doi.org/10.1007/s44397-025-00027-9},
   year = {2025}
}

https://link.springer.com/article/10.1007/s44397-025-00027-9
############################################

## Key Findings
* Our model achieves **state-of-the-art (SotA) performance** on the benchmark dataset (RMSE: 0.592, R²: 0.9997), significantly outperforming previous ANN, CNN, and other Transformer-based approaches.
* An XAI analysis using Integrated Gradients (IG) validates that the model learns to focus on **physically relevant features** (the speckle blobs) rather than spurious correlations.
* A detailed error analysis identifies the model's primary limitation: a proportional bias at the highest temperature range (>105°C), providing a clear direction for future work.

## Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/sheriff013/ViT_specklegram.git
    cd ViT_specklegram
    ```

2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    .\venv\Scripts\activate    # On Windows
    ```

3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

This study uses the synthetic FSS dataset

* **Publication:** "Dataset of synthetic specklegram images of a multimode fiber for temperature sensing"
* **DOI:** https://doi.org/10.1016/j.dib.2023.109134

### Data Preparation
1.  Download the dataset from the source above (e.g., via Mendeley Data).
2.  Create a `data/` directory in the root of this project.
3.  Extract the images into a subdirectory named `Original dataset`.
4.  The final expected file structure should be:
    ```
    ViT_specklegram/
    ├── data/
    │   └── Original dataset/
    │       ├── 0.4°C.tiff
    │       ├── 0.6°C.tiff
    │       └── ...
    ├── train.py
    └── ...
    ```

## Usage

### 1. Model Training

To train the ViT regressor from scratch and save the final model weights:

```bash
python train.py --data_dir "./data/Original dataset/" --epochs 100 --batch_size 16 --lr 1e-4
```

The trained model will be saved as `vit_speckle_regressor.pt` by default.

To see all available options: `python train.py --help`

### 2. Model Evaluation & Analysis

To load the trained model, run the full evaluation on the test set, and generate all analysis figures (Regression plot, Residuals, Bland-Altman, MAE per Bin, XAI):

```bash
python evaluate.py --model_path "vit_speckle_regressor.pt" --data_dir "./data/Original dataset/"
```


