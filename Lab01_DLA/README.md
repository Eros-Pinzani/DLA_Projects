# DLA Lab 1 — Exploiting Pre-trained Models for Traffic Sign Classification

Laboratory notebook for the **Deep Learning Applications** course.
The notebook explores multiple strategies to adapt pre-trained convolutional
networks to the task of traffic sign recognition.

---

## Overview

The laboratory is organized around three main blocks:

1. **Exploratory Data Analysis and Baselines** — dataset inspection, statistical analysis, feature extraction with a frozen backbone, and SVM classification.
2. **Pipeline Consolidation** — a configurable, reproducible training pipeline with Weights & Biases experiment tracking.
3. **Extended Exercise** — a training-free retrieval classification (Nearest-Mean Classifier).

---

## Contents

| Exercise | Description |
|---|---|
| **1.1** | Exploratory Data Analysis and dataset setup |
| **1.2** | Feature extraction with a frozen backbone + SVM classifier |
| **1.3** | Full fine-tuning baseline (ResNet-18) |
| **2** | Pipeline consolidation — reproducible training loop with configurable hyperparameters |
| **3.2** | Retrieval as Training-free Classification (Nearest-Mean Classifier + mAP evaluation) |

---

## Dataset

**GTSRB — German Traffic Sign Recognition Benchmark**

- 43 classes of road signs
- ~39,000 training images, ~12,600 test images
- Downloaded automatically via `torchvision.datasets.GTSRB`

---

## Requirements

```
torch
torchvision
numpy
matplotlib
scikit-learn
scipy
tqdm
```

Install all dependencies with:

```bash
pip install torch torchvision numpy matplotlib scikit-learn scipy tqdm
```

A CUDA-capable GPU is recommended for feature extraction and fine-tuning,
but all cells can also run on CPU (with longer runtimes).

---

## Key Results

### Exercise 2 — Fine-tuned ResNet-18

| Metric | Value |
|---|---|
| Test Accuracy | **94.86%** |

### Exercise 3.2 — Nearest-Mean Classifier (training-free)

| Backbone | NMC Accuracy |
|---|---|
| ResNet-18 | ~39% |
| ResNet-50 | ~41% |
| EfficientNet-B0 | **~45%** |
| ViT-B/16 | ~38% |

| Metric | Value |
|---|---|
| Best backbone | EfficientNet-B0 |
| NMC Accuracy | **~45%** |
| Retrieval mAP | **~16%** |

All results are obtained without any task-specific training.
Random chance on this 43-class dataset would yield approximately 2.3%.

---

## Exercise 3.2 — Method Overview

The exercise treats traffic sign classification as a **retrieval problem**,
avoiding fine-tuning entirely.

**Pipeline:**

1. **Feature extraction** — a pretrained CNN is used as a pure feature extractor
   by replacing its classification head with `nn.Identity()`. Both the training
   set (gallery) and the test set (queries) are embedded into a common feature
   space. All feature vectors are L2-normalized so that cosine similarity reduces
   to a dot product.

2. **Nearest-Mean Classifier (NMC)** — one prototype vector is computed per class
   by averaging all training embeddings belonging to that class. Each test image
   is then assigned to the class whose prototype has the highest cosine similarity.
   This reduces classification cost from O(N\_train) to O(num\_classes) per query.

3. **Retrieval evaluation (mAP)** — the full gallery is ranked by similarity for
   each query and Average Precision is computed per class. The mean over all classes
   gives the final mAP score.

4. **t-SNE visualization** — a 2D projection of the test embeddings is used to
   inspect cluster structure. Speed Limit signs (classes 0–4) collapse into a
   single overlapping region, which directly explains their consistently low F1
   scores across all backbones.

5. **Error analysis** — misclassified test images are displayed alongside their
   true and predicted labels, with a per-class confusion summary for the most
   problematic sign categories.

---

## Notebook Structure

```
DLA-Lab1.ipynb
|
|-- Exercise 1
|   |-- 1.1  EDA and dataset loading
|   |-- 1.2  Frozen backbone + SVM
|   |-- 1.3  Fine-tuning ResNet-18
|
|-- Exercise 2
|   |-- Reproducible training pipeline (build_transforms, build_model, run, ...)
|
|-- Exercise 3.2
    |-- Setup and imports
    |-- Data preparation
    |-- Feature extraction (extract_features)
    |-- Backbone comparison (ResNet-18/50, EfficientNet-B0, ViT-B/16)
    |-- NMC implementation (compute_class_means, nmc_classify)
    |-- NMC evaluation across backbones
    |-- Retrieval evaluation (compute_map)
    |-- Results visualization
    |-- Results and Discussion (markdown)
    |-- t-SNE feature space visualization
    |-- t-SNE analysis (markdown)
    |-- Error analysis
```

---

## Notes

- Feature extraction for ViT-B/16 uses a dedicated 224x224 dataloader, as
  required by the architecture, while all CNN backbones use 64x64 inputs.
- t-SNE is run on a random subsample of 1500 test images for computational
  efficiency (`random_state=42` for reproducibility).
