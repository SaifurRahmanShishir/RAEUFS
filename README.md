# RAEUFS

Official implementation of **RAEUFS (Robust Autoencoder-based Unsupervised Feature Selection)**, introduced in the paper:

> **Unsupervised Feature Selection via Robust Autoencoder and Adaptive Graph Learning**  
> Feng Yu, MD Saifur R. Mazumder, Ying Su and Oscar Contreras Velasco  
> 2026 6th International Conference on Electrical, Computer and Energy Technologies (ICECET), 2026  
> DOI: `10.1109/ICECET65726.2026.11632773`  
> IEEE Xplore Document ID: `11632773`

---

## Overview

RAEUFS is an unsupervised feature-selection framework that combines:

- a learnable feature-selection matrix $begin:math:text$W$end:math:text$,
- a deep autoencoder for nonlinear representation learning,
- a Robust Subspace Recovery (RSR) layer for robustness to outliers,
- a pseudo-label matrix $begin:math:text$F$end:math:text$,
- and adaptive graph learning through the similarity matrix $begin:math:text$S$end:math:text$.

The model is designed to identify discriminative features while preserving the latent clustering structure of the data.

The overall transformation is

$begin:math:display$
X\_s \= XW\,
$end:math:display$

$begin:math:display$
Z \= E\(X\_s\)\,
$end:math:display$

$begin:math:display$
\\tilde\{Z\} \= ZA\,
$end:math:display$

$begin:math:display$
\\tilde\{X\} \= D\(\\tilde\{Z\}\)\,
$end:math:display$

where:

- $begin:math:text$X$end:math:text$ is the original data matrix,
- $begin:math:text$W$end:math:text$ is the feature-selection matrix,
- $begin:math:text$E$end:math:text$ is the encoder,
- $begin:math:text$A$end:math:text$ is the RSR transformation,
- $begin:math:text$D$end:math:text$ is the decoder,
- $begin:math:text$Z$end:math:text$ is the encoder representation,
- and $begin:math:text$\\tilde\{Z\}$end:math:text$ is the RSR latent representation.

---

## Training Strategy

RAEUFS is optimized using an alternating strategy.

### Within each epoch

The neural-network-related parameters are updated **batch by batch**:

1. Encoder and decoder parameters
2. RSR transformation $begin:math:text$A$end:math:text$
3. Feature-selection matrix $begin:math:text$W$end:math:text$

### At the end of each epoch

After all mini-batches have been processed:

1. The complete dataset is passed through the current model.
2. The full latent representation $begin:math:text$\\tilde\{Z\}$end:math:text$ is computed.
3. The pseudo-label matrix $begin:math:text$F$end:math:text$ is updated using **Generalized Power Iteration (GPI)**.
4. The adaptive similarity matrix $begin:math:text$S$end:math:text$ is updated using the new $begin:math:text$F$end:math:text$.

Conceptually:

```text
Epoch k
│
├── Batch 1 ──► Update neural-network parameters / A / W
├── Batch 2 ──► Update neural-network parameters / A / W
├── ...
├── Batch N ──► Update neural-network parameters / A / W
│
└── End of epoch
      │
      ├── Compute full latent representation Z~
      ├── Update F using GPI
      └── Update S
             │
             ▼
          Epoch k+1
```

Thus, **$begin:math:text$F$end:math:text$ and $begin:math:text$S$end:math:text$ are not updated at the mini-batch level**. They are updated once after each complete training epoch.

---

## Repository Structure

```text
├── data/
│   └── jaffe.mat
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── evaluation.py
│   └── train.py
│
├── results/
│   └── .gitkeep
│
├── .gitignore
├── README.md
└── requirements.txt
```

### `src/model.py`

Contains the core RAEUFS implementation, including:

- feature-selection layer,
- encoder,
- RSR layer,
- decoder,
- reconstruction loss,
- RSR loss,
- $begin:math:text$\\ell\_\{2\,1\}$end:math:text$ feature sparsity,
- orthogonality handling,
- GPI update for $begin:math:text$F$end:math:text$,
- adaptive graph update for $begin:math:text$S$end:math:text$,
- and the PyTorch Lightning training procedure.

### `src/data.py`

Contains:

- data loading,
- Min-Max scaling,
- Robust scaling,
- and PyTorch dataset utilities.

### `src/evaluation.py`

Contains clustering evaluation utilities, including:

- K-means clustering,
- clustering accuracy (ACC),
- normalized mutual information (NMI),
- Hungarian matching,
- and baseline evaluation.

### `src/train.py`

Main experiment script used to:

- load the dataset,
- initialize RAEUFS,
- train the model,
- perform repeated experiments,
- and evaluate the selected features.



## Installation

Clone the repository:

```bash
git clone <RAEUFS >
```

Create or activate your Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include:

- NumPy
- SciPy
- pandas
- scikit-learn
- PyTorch
- PyTorch Lightning

---

## Dataset

The current implementation includes experiments using the **JAFFE** dataset.

For JAFFE:

| Property | Value |
|---|---:|
| Samples | 213 |
| Features | 676 |
| Classes / Clusters | 10 |
| Selected features in the main experiment | 200 |

The expected directory structure is:

```text
data/
└── jaffe.mat
```

The `.mat` file is expected to contain:

```python
X = data["fea"]
y = data["gnd"].flatten()
```

where:

- `fea` contains the feature matrix,
- `gnd` contains the ground-truth labels used only for evaluation.

The labels are **not used during unsupervised model training**.

---

## Running the Model

From the repository root, run:

```bash
python src/train.py
```

If command-line arguments are enabled in `train.py`, experiments can also be configured using options such as:

```bash
python src/train.py --epochs 500 --runs 50
```

For debugging, it is recommended to first use a small experiment:

```bash
python src/train.py --epochs 10 --runs 1
```

and then increase the number of epochs and independent runs after confirming that training is stable.

---

## Feature Selection

After training, feature importance is obtained from the learned feature-selection matrix

$begin:math:display$
W \\in \\mathbb\{R\}\^\{D \\times p\}\.
$end:math:display$

The importance of each original feature is determined from the norm of the corresponding row of $begin:math:text$W$end:math:text$.

Features with larger row norms have greater influence in the learned feature representation.

The selected features can subsequently be evaluated using clustering algorithms such as K-means.

---

## Evaluation

The implementation evaluates feature-selection quality using:

### Clustering Accuracy (ACC)

Cluster assignments are aligned with the ground-truth classes using the Hungarian assignment algorithm before computing accuracy.

### Normalized Mutual Information (NMI)

NMI measures the agreement between the predicted clustering structure and the true class structure while being invariant to the numerical cluster labels.

K-means clustering is repeated multiple times to account for random initialization.


## Hyperparameters

The main RAEUFS hyperparameters include:

| Parameter | Description |
|---|---|
| `lambda1` | RSR reconstruction regularization |
| `lambda2` | RSR orthogonality regularization |
| `alpha` | Row-sparsity regularization for $begin:math:text$W$end:math:text$ |
| `beta` | Adaptive graph entropy parameter |
| `gamma` | Graph regularization strength |
| `eta` | Latent pseudo-label fitting strength |
| `lr` | Learning rate |
| `batch_size` | Mini-batch size |
| `epochs` | Number of alternating optimization epochs |

Hyperparameters can be adjusted in `src/train.py`.

---

## Apple Silicon Note

On some Apple Silicon systems, certain NumPy matrix multiplications may produce warnings such as:

```text
RuntimeWarning: divide by zero encountered in matmul
RuntimeWarning: overflow encountered in matmul
RuntimeWarning: invalid value encountered in matmul
```

These warnings can occur even for numerically valid matrix multiplications.

Before assuming model instability, verify that the relevant arrays contain finite values:

```python
print(np.isfinite(X).all())
print(np.isfinite(W).all())
```

and inspect their magnitudes:

```python
print(np.abs(X).max())
print(np.abs(W).max())
```

The implementation also avoids unsupported MPS operations where appropriate when performing QR/SVD-based calculations.

---

## Reproducibility

Because RAEUFS contains several stochastic components, including:

- neural-network initialization,
- feature-selection matrix initialization,
- mini-batch ordering,
- and K-means initialization,

results should be evaluated over multiple independent runs.

For controlled experiments, set random seeds for NumPy, PyTorch, and PyTorch Lightning.

Example:

```python
import numpy as np
import torch
import pytorch_lightning as pl

seed = 21

np.random.seed(seed)
torch.manual_seed(seed)
pl.seed_everything(seed, workers=True)
```

---

## Citation

If you use RAEUFS or this implementation in your research, please cite:

```bibtex
@INPROCEEDINGS{11632773,
  author={Yu, Feng and Mazumder, MD Saifur R. and Su, Ying and Velasco, Oscar Contreras},
  booktitle={2026 6th International Conference on Electrical, Computer and Energy Technologies (ICECET)}, 
  title={Unsupervised Feature Selection via Robust Autoencoder and Adaptive Graph Learning}, 
  year={2026},
  volume={},
  number={},
  pages={1-8},
  keywords={Printing;Feature extraction;Matrices;Modeling;Labeling;Algorithms;Timing;Optimization;Equations;Conferences;unsupervised feature selection;autoencoder;Graph learning},
  doi={10.1109/ICECET65726.2026.11632773}}

```

---

## Paper

**Unsupervised Feature Selection via Robust Autoencoder and Adaptive Graph Learning**

- Conference: 6th International Conference on Electrical, Computer and Energy Technologies (ICECET 2026)
- Year: 2026
- DOI: `10.1109/ICECET65726.2026.11632773`
- IEEE Xplore Document ID: `11632773`