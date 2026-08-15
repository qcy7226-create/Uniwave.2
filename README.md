# UniWave-2: A Hybrid Model for Nucleic Acid Waveform Feature Extraction

[![python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of UniWave-2.

UniWave‑2 is a lightweight, interpretable framework that transforms nucleic acid sequences into continuous waveform signals via physicochemical encoding (Ghose–Crippen logP hydrophobicity), Fourier interpolation, wavelet compression, and a MultiScaleGRU architecture. It achieves competitive performance on promoter recognition, m6A site prediction, SARS‑CoV‑2 variant classification, and dengue serotyping with only **0.33M parameters**—no large‑scale pretraining required.

---

## 📌 Table of Contents

- [✨ Key Features](#-key-features)
- [⚡ Efficiency Benchmarking](#-efficiency-benchmarking)
- [💻 System Requirements](#-system-requirements)
- [🛠 Installation](#-installation)
- [📂 Data Preparation](#-data-preparation)
- [⚙️ Preprocessing](#️-preprocessing-fasta--waveform-hdf5)
- [🧠 Training](#-training-the-model)
- [📊 Evaluation & Visualization](#-evaluation--visualization)
- [🔬 Reproducing Paper Results](#-reproducing-paper-results)
- [🔍 Interpretability](#-interpretability-in-silico-mutagenesis)
- [📁 Project Structure](#-project-structure)
- [📝 Citation](#-citation)
- [📧 Contact](#-contact)

---

## ✨ Key Features

- **Physicochemical grounding** – Uses Ghose–Crippen logP values (A=-1.07, T=-0.36, C=-0.76, G=-1.36) for biologically meaningful encoding.
- **Signal processing pipeline** – Fourier zero‑padding interpolation (6× oversampling) → adaptive enhancement → wavelet compression (sym6, 3 levels) → FIR filtering.
- **MultiScaleGRU** – Lightweight hybrid network (≈0.33M parameters) with convolutional embedding, attention‑guided positional encoding, inception blocks, and bidirectional GRU.
- **Gradient‑free interpretability** – Supports in silico saturation mutagenesis (ISM) to map predictions back to known motifs (e.g., TATA‑box, DPE).
- **Reproducibility** – Fixed random seeds, strict train/val/test splits (70/20/10), and fully parameterized command‑line scripts.

---

## ⚡ Efficiency Benchmarking

We established a comprehensive efficiency benchmarking protocol for UniWave-2 to evaluate its deployment feasibility in resource-constrained settings. The evaluation measured practical metrics including inference time, GPU memory usage, FLOPs, and sequence-length scalability.

**Benchmark setup:**
- **Dataset**: SARS-CoV-2 variant classification (9 variants: Alpha, Beta, Gamma, Delta, BA.1, BA.2, BA.4, BA.5, XBB.1.5), 1,000 bp sequences
- **Hardware**: NVIDIA RTX 3090
- **Protocol**: 20 warm-up runs followed by 100 inference runs; mean values reported
- **Model variants**: Full model, w/o GRU, w/o WaveEncoder, w/o Dilated Conv

**Results summary:**

| Variant | Params (M) | FLOPs (G) | Inference Time (ms) | Peak GPU Memory (MB) |
|:--------|:----------:|:---------:|:--------------------:|:--------------------:|
| Full    | 0.327      | 0.319     | 4.43                 | 81.56                |
| w/o GRU | 0.186      | 0.192     | 2.78                 | 46.13                |
| w/o WaveEncoder | 0.265 | 0.195 | 3.86 | 71.30 |
| w/o Dilated Conv | 0.327 | 0.319 | 4.31 | 82.61 |

**Comparison with large models (DNABERT-2, Nucleotide Transformer):**

| Model | Params (M) | Rel. FLOPs | Inference Time (ms) | Peak GPU Memory (MB) |
|:------|:----------:|:---------:|:--------------------:|:--------------------:|
| DNABERT (3-mer) | 86 | 3.27 | ~15.0 | ~4,800 |
| DNABERT (4-mer) | 86 | 3.26 | ~15.0 | ~4,800 |
| DNABERT (5-mer) | 87 | 3.26 | ~15.0 | ~4,800 |
| DNABERT (6-mer) | 89 | 3.25 | ~15.0 | ~4,800 |
| NT-500M-human | 480 | 3.19 | ~30.0 | ~12,000 |
| NT-500M-1000g | 480 | 3.19 | ~30.0 | ~12,000 |
| NT-2500M-1000g | 2537 | 19.44 | ~45.0 | ~20,000 |
| NT-2500M-multi | 2537 | 19.44 | ~45.0 | ~20,000 |
| DNABERT-2 | 117 | 1.00 | ~15.0 | ~4,800 |
| **UniWave‑2 (ours)** | **0.327** | **0.32** | **4.43** | **81.56** |

*Relative FLOPs are normalized to DNABERT-2 (1.00×); larger values indicate greater computational cost.*

**Sequence-length scalability (500–5,000 bp):**

| Length (bp) | Inference Time (ms) | Peak GPU Memory (MB) | Time Scaling | Memory Scaling |
|:-----------:|:--------------------:|:--------------------:|:------------:|:--------------:|
| 500 | 3.74 | 51.33 | 1.00× | 1.00× |
| 1,000 | 3.97 | 62.50 | 1.06× | 1.22× |
| 2,000 | 4.43 | 81.56 | 1.18× | 1.59× |
| 5,000 | 5.16 | 135.19 | 1.38× | 2.63× |

> 📁 **Complete benchmarking tables** are available in the `results/benchmark/` folder.

---

## 💻 System Requirements

- **OS**: Linux (Ubuntu 20.04+) or macOS (≥12.0). Windows via WSL2 is supported.
- **Hardware**: GPU with ≥8 GB VRAM recommended (e.g., RTX 2070/3060). CPU training is possible but slower.
- **Python**: 3.8, 3.9, or 3.10.

---

## 🛠 Installation

**Option 1: Conda (recommended)**

```bash
git clone https://github.com/qcy7226-create/Uniwave.2.git
cd Uniwave.2
conda env create -f environment.yml
conda activate uniwave2
Option 2: Pip

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
requirements.txt:

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
📂 Data Preparation
Download the benchmark datasets and place each class's FASTA file in a directory (e.g., data/raw/).

Task	Source	Length	Classes
Promoter	Deepromclass	80, 150, 300 bp	5 species (binary)
Rice m6A	m6A-Rice	101 bp context	2 (m6A / non‑m6A)
SARS‑CoV‑2	GISAID / DNABERT‑2	1000 bp	9 variants
Dengue	NCBI Virus Variation	1000 bp	4 serotypes
⚠️ The preprocessing script performs global deduplication and splits by sequence identity (not random windows) to prevent data leakage.

⚙️ Preprocessing: FASTA → Waveform HDF5
preprocess.py implements the full waveform encoding pipeline.

Basic usage:

bash
python preprocess.py \
    --fasta_files data/raw/Alpha.fasta data/raw/Beta.fasta ... data/raw/XBB.1.5.fasta \
    --output data/processed/sars_cov2.h5
Parameters:

Argument	Type	Default	Description
--fasta_files	list	required	FASTA files (one per class)
--output	str	required	Output HDF5 file path
--test_size	float	0.1	Proportion for test set
--val_size	float	0.2	Proportion for validation (from remaining)
--seed	int	726	Random seed
--max_len	int	1000	Original sequence length
What happens inside:

Reads sequences, filters non‑ACGT characters, crops to max_len

Removes identical sequences across all classes (global deduplication)

Splits data into Train (70%), Validation (20%), Test (10%)

Maps nucleotides to Ghose–Crippen logP values

Applies Fourier interpolation (6×) → adaptive enhancement → wavelet compression (sym6, 3 levels) → FIR filtering

Z‑score normalizes using training set mean/std

Saves as HDF5 with groups train, val, test

🧠 Training the Model
train.py trains the MultiScaleGRU model using the preprocessed HDF5 file.

Basic usage:

bash
python train.py --data data/processed/sars_cov2.h5 --output_dir results/sars_cov2
Parameters:

Argument	Type	Default	Description
--data	str	required	HDF5 file path
--output_dir	str	./results	Output directory
--seed	int	726	Random seed
--batch_size	int	256	Training batch size
--epochs	int	100	Maximum epochs
--device	str	cuda	cuda or cpu
Training features:

Mixed precision (AMP) and gradient accumulation (2 steps)

Three‑phase learning rate scheduler: warm‑up (15%) → cosine annealing (70%) → linear decay (15%)

Monitors validation Macro‑F1; saves best model as best_model.pth

Early stopping (patience = 14 epochs)

📊 Evaluation & Visualization
After training, the script automatically evaluates the best model on the held‑out test set.

Metrics reported:

Balanced Accuracy (with 95% bootstrap confidence interval)

Macro‑F1

AUC‑ROC (macro‑averaged)

Confusion matrix and multi‑class ROC curves (evaluation_plots.png)

Manual evaluation:

bash
python evaluate.py --model results/best_model.pth --data data.h5 --split test
🔬 Reproducing Paper Results
Run the following scripts to reproduce Tables 1–4 from the paper:

bash
bash scripts/run_promoter_exp.sh      # Table 1 (5 species × 3 lengths)
bash scripts/run_m6a_exp.sh           # Table 2 (Rice m6A)
bash scripts/run_sars_cov2_exp.sh     # Table 3 (SARS‑CoV‑2)
bash scripts/run_dengue_exp.sh        # Table 4 (Dengue robustness)
All results (metrics, plots, model weights) are saved under results/.

🔍 Interpretability: In Silico Mutagenesis
A Jupyter notebook is provided for gradient‑free attribution analysis via in silico saturation mutagenesis (ISM).

bash
jupyter notebook notebooks/ism_interpretability.ipynb
What it does:

Loads the trained model and promoter sequences from five species

Iteratively mutates each position in sliding windows

Plots importance profiles revealing species‑specific peaks (TATA‑box at 170 bp in yeast; DPE at ~250 bp in higher eukaryotes)

No prior motif input required

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
│   ├── raw/                     # FASTA files
│   └── processed/               # HDF5 files
├── src/
│   ├── preprocessing/
│   │   ├── encode.py
│   │   └── utils.py
│   ├── models/
│   │   ├── inception_time.py
│   │   └── trainer.py
│   └── utils/
│       ├── data_loader.py
│       └── metrics.py
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_*_exp.sh
├── notebooks/
│   └── ism_interpretability.ipynb
├── results/
│   ├── benchmark/                # Efficiency tables
│   └── ...
└── .
📝 Citation
If you use UniWave‑2 in your research, please cite:

bibtex
@article{Zhang2026UniWave2,
  title  = {UniWave-2: A Hybrid Model for Nucleic Acid Waveform Feature Extraction Enhanced by Fourier and Wavelet Transforms},
  author = {Zhang, Hao and Qi, Yujun and Feng, Yulan and Wang, Lili},
  year   = {2026}
}
📧 Contact
For questions, bug reports, or suggestions, please open a GitHub Issue.

⭐ If you find this work useful, please consider giving a star to the repository!
