UniWave-2: A Hybrid Model for Nucleic Acid Waveform Feature Extraction Enhanced by Fourier and Wavelet Transforms
https://img.shields.io/badge/python-3.8+-blue.svg
https://img.shields.io/badge/PyTorch-2.0+-red.svg
https://img.shields.io/badge/license-MIT-green.svg

https://img.shields.io/badge/paper-Bioinformatics%2520Advances-blue
https://img.shields.io/badge/github-UniWave--2-ff69b4

This repository contains the official implementation of UniWave-2, a lightweight feature extraction framework that transforms nucleic acid sequences into continuous waveform signals using physicochemical properties (hydrophobicity), Fourier interpolation, wavelet compression, and a multi‑scale GRU architecture. UniWave‑2 achieves competitive performance across diverse genomic tasks (promoter recognition, m6A site prediction, SARS‑CoV‑2 variant classification, dengue serotyping) while maintaining high interpretability and low computational cost.

📌 Table of Contents
Features

Requirements

Installation

Data Preparation

Preprocessing: Waveform Encoding

Training the Model

Evaluation & Visualization

Reproducing Paper Results

Project Structure

Citation

Contact

✨ Features
Physicochemical Encoding – Maps nucleotides to continuous values derived from Ghose–Crippen logP hydrophobicity.

Fourier Interpolation – Increases signal density via zero‑padding in the frequency domain (6× oversampling).

Wavelet Compression – Decomposes and reconstructs signals with sym6 wavelet to reduce length while preserving critical features (3× downsampling).

MultiScaleGRU – Lightweight hybrid network (≈0.33M parameters) combining convolutional embedding, attention‑guided positional encoding, inception blocks, and bidirectional GRU.

Interpretability – Supports in silico saturation mutagenesis (ISM) for gradient‑free attribution.

Reproducibility – Fixed random seed, documented splits, and provided scripts.

📦 Requirements
Python 3.8 or higher

PyTorch 2.0+

CUDA (optional, but recommended for training)

Additional packages listed in requirements.txt

Install all dependencies using:

bash
pip install -r requirements.txt
requirements.txt
text
numpy==1.24.3
scipy==1.10.1
torch==2.0.1
torchvision==0.15.2
h5py==3.8.0
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
pywavelets==1.4.1
pandas==2.0.1
tqdm==4.65.0
PyYAML==6.0
🛠 Installation
Clone the repository:

bash
git clone https://github.com/qcy7226-create/Uniwave.2.git
cd Uniwave.2
Create a conda environment (optional but recommended):

bash
conda env create -f environment.yml
conda activate uniwave2
Or install directly with pip:

bash
pip install -r requirements.txt
📂 Data Preparation
UniWave‑2 supports multiple tasks. Below we list the benchmark datasets used in the paper and their sources.

Task	Dataset / Source	Sequences length
Promoter recognition	Deepromclass (five species: Drosophila, yeast, mouse, C. elegans, human) – source	80, 150, 300 bp
Rice m6A	Experimentally validated m6A sites in rice – source	101 bp (context)
SARS‑CoV‑2 variants	Nine variants (Alpha, Beta, Gamma, Delta, BA.1, BA.2, BA.4, BA.5, XBB.1.5) from CISAID / DNABERT‑2 benchmark – source	1000 bp
Dengue serotypes	Four DENV serotypes – source	1000 bp
Place your FASTA files in a directory (e.g., data/raw/). For each task, each class should have its own FASTA file (e.g., Alpha.fasta, Beta.fasta, …).

🔧 Preprocessing: Waveform Encoding
The preprocessing pipeline converts raw FASTA sequences into waveform features and stores them in an HDF5 file with train/validation/test splits (70/20/10).

Usage
bash
python scripts/preprocess.py \
    --fasta_files data/raw/Alpha.fasta data/raw/Beta.fasta ... \
    --output data/processed/uniwave2_data.h5 \
    --test_size 0.1 \
    --val_size 0.2 \
    --seed 726
Key parameters (can be adjusted in the script or via command line):

Parameter	Description	Default
--fasta_files	List of FASTA files (one per class)	required
--output	Output HDF5 file path	required
--test_size	Proportion of test set	0.1
--val_size	Proportion of validation set (from remaining)	0.2
--seed	Random seed for reproducibility	726
--max_len	Target sequence length (after downsampling)	2000
--interp_factor	Fourier interpolation factor	6
--downsample_factor	Wavelet downsampling factor	3
--wavelet	Wavelet type	sym6
--wavelet_level	Decomposition level	3
The script performs:

Reading FASTA – extracts sequences of desired length (random cropping for longer reads).

Global deduplication across classes to avoid overlapping samples.

Stratified split (by class) into train/val/test.

Hydrophobicity mapping (A=-1.07, T=-0.36, C=-0.76, G=-1.36, derived from Ghose–Crippen logP).

Fourier interpolation (zero‑padding in frequency domain).

Adaptive signal enhancement (local gain).

Wavelet compression (sym6, 3‑level decomposition, energy‑based thresholding, reconstruction, 3× downsampling).

FIR filtering (bidirectional mean filter) for final noise reduction.

Z‑score normalization using mean and std from training set.

Saving to HDF5 with groups train, val, test, each containing class_0, class_1, … datasets.

The generated HDF5 file will also store metadata (creation date, split ratios, wavelet parameters).

🧠 Training the Model
We provide a flexible training script scripts/train.py. It loads the preprocessed HDF5, builds the MultiScaleGRU model, and trains it using the hyperparameters defined in a YAML configuration file (or inside the code).

Example command
bash
python scripts/train.py \
    --data data/processed/uniwave2_data.h5 \
    --config configs/default.yaml \
    --output_dir results/sars_cov2 \
    --device cuda \
    --seed 726
Configuration file (configs/default.yaml)
Example content (customize as needed):

yaml
# Model
num_classes: 9
seq_length: 2000
embedding:
  enable: true
  dim: 16
  kernel_size: 3

# Training
train_batch_size: 256
val_batch_size: 512
epochs: 100
max_lr: 1e-3
weight_decay: 5e-4
label_smoothing: 0.15
dropout_rate: 0.4
grad_clip: 5.0
accumulation_steps: 2
early_stop_patience: 14

# Scheduler (three‑phase)
div_factor: 15.0
init_lr: 1e-4
final_lr: 2e-5

# Evaluation
metrics:
  average_type: macro
  show_confusion_matrix: true
  plot_roc_curve: true
  class_names: ["Alpha","Beta","Gamma","Delta","BA.1","BA.2","BA.4","BA.5","XBB.1.5"]
If you prefer to use the hard‑coded configuration from the original code, simply run:

bash
python scripts/train.py --data data/processed/uniwave2_data.h5
The script will:

Automatically detect the number of classes from the HDF5.

Set random seeds for reproducibility.

Train with mixed precision (AMP) and gradient accumulation.

Apply three‑phase learning rate scheduling (warm‑up, cosine annealing, linear decay).

Monitor validation F1 and save the best model as best_model.pth in the output directory.

Stop early if no improvement for early_stop_patience epochs.

📊 Evaluation & Visualization
After training, the script will automatically evaluate on the validation and test sets. Evaluation metrics include:

Balanced Accuracy (with 95% bootstrap confidence intervals)

Macro‑F1

AUC‑ROC (macro‑averaged)

Confusion matrix (plotted)

ROC curves (per class, plotted)

To manually evaluate a saved model:

bash
python scripts/evaluate.py \
    --model results/sars_cov2/best_model.pth \
    --data data/processed/uniwave2_data.h5 \
    --split test \
    --output results/sars_cov2/eval_results.json
Visual outputs
confusion_matrix.png – heatmap

roc_curve.png – multi‑class ROC

training_curves.png – loss & accuracy over epochs (if logging enabled)

🔬 Reproducing Paper Results
We provide a shell script that reproduces all main experiments from the paper.

bash
bash scripts/run_all_experiments.sh
This script sequentially runs preprocessing and training for each task (promoter, m6A, SARS‑CoV‑2, dengue) using the same hyperparameters as reported. All results (metrics, plots, model weights) will be saved under results/.

Important: Before running, ensure that all raw FASTA files are placed in data/raw/ with the names matching those used in the script. Edit the paths if necessary.

For promoter recognition (different species and sequence lengths), the script will handle each combination automatically.

📁 Project Structure
text
UniWave-2/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/                 # Place your FASTA files here
│   └── processed/           # Generated HDF5 files
├── src/
│   ├── preprocessing/
│   │   ├── encode.py        # Waveform encoding functions
│   │   └── utils.py         # FASTA reading, deduplication, split
│   ├── models/
│   │   ├── inception_time.py # MultiScaleGRU model definition
│   │   └── trainer.py       # BioTrainer class (training loop)
│   └── utils/
│       ├── data_loader.py   # BioDataset (HDF5 loader)
│       └── metrics.py       # Bootstrap CI, plots, etc.
├── scripts/
│   ├── preprocess.py        # Main preprocessing executable
│   ├── train.py             # Main training executable
│   ├── evaluate.py          # Evaluation script
│   └── run_all_experiments.sh
├── notebooks/
│   └── ism_analysis.ipynb   # In silico mutagenesis interpretability
├── tests/                   # Unit tests (if any)
└── results/                 # Outputs (models, logs, plots)
📝 Citation
If you use UniWave‑2 in your research, please cite our paper:

bibtex
@article{Zhang2026UniWave2,
  author = {Zhang, Hao and Qi, Yujun and Feng, Yulan and Wang, Lili},
  title = {UniWave-2: A Hybrid Model for Nucleic Acid Waveform Feature Extraction Enhanced by Fourier and Wavelet Transforms},
  journal = {Bioinformatics Advances},
  year = {2026},
  volume = {XX},
  number = {X},
  pages = {0--0},
  doi = {10.1093/bioadv/xxxxx}
}
📧 Contact
For questions, bug reports, or suggestions, please open an issue on GitHub or contact the corresponding author:

Lili Wang
College of Physics and Electronic Engineering, Northwest Normal University
Email: wanglili@nwnu.edu.cn

📜 License
This project is licensed under the MIT License – see the LICENSE file for details.
