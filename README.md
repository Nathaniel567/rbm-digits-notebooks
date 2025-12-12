# Simple RBM Experiments in Python

This repo shows  Restricted Boltzmann Machine (RBM) experiments using only NumPy, SciPy, scikit-learn, and Matplotlib inside Jupyter notebooks. It uses the built-in scikit-learn digits dataset (8×8) upsampled to 14×14 and binarized—no external downloads needed.

## Contents
- `rbm_utils.py` — small helper module for loading data, CD-1 training, Gibbs steps, corruption, Hamming error, and save/load.
- `01_experiment_baseline.ipynb` — trains a 196×50 RBM, plots train/test error, and shows original vs reconstructed digits; saves `models/rbm_digits_50.npz`.
- `02_experiment_features.ipynb` — visualizes hidden-unit weight vectors as 14×14 filters from the saved model.
- `03_experiment_denoising.ipynb` — denoises corrupted digits at several noise levels and plots error vs noise.
- `04_experiment_sampling_ablation.ipynb` — samples from the trained RBM and compares hidden sizes 20/50/100 with train/test curves and tiny sample grids.
- `requirements.txt` — minimal Python dependencies.
- `docs/MATH385_Final_Report.pdf` — final project report.

## Quickstart
1. Install deps (Python 3.12+):
   ```bash
   python3 -m pip install -r requirements.txt
   ```
2. Launch Jupyter:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
3. Run notebooks in order:
   - `01_experiment_baseline.ipynb` (trains and saves `models/rbm_digits_50.npz`)
   - `02_experiment_features.ipynb`
   - `03_experiment_denoising.ipynb`
   - `04_experiment_sampling_ablation.ipynb`

