# DLA Lab 2 — Transformers and the Hugging Face Ecosystem

Laboratory notebook (+ standalone application) for the **Deep Learning
Applications** course. The lab moves from a training-free baseline for
sentiment analysis to full fine-tuning of a pretrained Transformer, and
finally to a practical application: a text-to-image retrieval system built
with CLIP.

---

## Overview

The lab is split across two locations:
- **Exercises 1 and 2** are implemented in `DLA-Lab2.ipynb`.
- **Exercise 3.3** is implemented as a standalone application in
  [`TextToImageApp/`](./TextToImageApp/), since it is explicitly meant to be
  built outside of a notebook. See
  [`TextToImageApp/README.md`](./TextToImageApp/README.md) for details on how
  it works and how to run it.

---

## Repository Structure

```
Lab02_DLA/
├── DLA-Lab2.ipynb        # Exercises 1 and 2 (Sentiment Analysis + Fine-tuning)
├── TextToImageApp/         # Exercise 3.3: Text-to-Image Retrieval app (see its own README)
│   ├── app.py
│   ├── build_index.py
│   ├── requirements.txt
│   └── README.md
└── README.md
```

---

## Contents

| Exercise | Description |
|---|---|
| **1.1** | Dataset exploration (Rotten Tomatoes) |
| **1.2** | Tokenizer & DistilBERT exploration (role of the attention mask) |
| **1.3** | Stable baseline: frozen DistilBERT `[CLS]` features + Linear SVM, with a log-spaced grid search over `C` |
| **2.1** | Token preprocessing (`Dataset.map`, dynamic padding) |
| **2.2** | Model setup: `DistilBertForSequenceClassification` |
| **2.3** | Fine-tuning with the `Trainer` API, including a 3-configuration hyperparameter sweep |
| **3.3** | Text-to-Image Retrieval app (CLIP + Gradio) — see [`TextToImageApp/`](./TextToImageApp/) |

---

## Dataset

**Rotten Tomatoes** ([`cornell-movie-review-data/rotten_tomatoes`](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes))

- Short movie review snippets, binary sentiment label (0 = negative, 1 = positive)
- 8,530 training / 1,066 validation / 1,066 test examples
- Perfectly balanced (50/50) across all splits

---

## Requirements

```
torch
transformers
datasets
scikit-learn
numpy
pandas
matplotlib
```

For a quick setup:

```bash
pip install torch transformers datasets scikit-learn numpy pandas matplotlib
```

For an exact reproduction of the tested environment (pinned versions),
install from [`requirements-freeze.txt`](./requirements-freeze.txt) instead:

```bash
pip install -r requirements-freeze.txt
```

A CUDA-capable GPU is recommended for fine-tuning (Exercise 2.3), but the
SVM baseline (Exercise 1.3) runs comfortably on CPU as well.

---

## Key Results

### Exercise 1.3 — Stable Baseline (frozen DistilBERT + SVM)

| Configuration | Validation Accuracy | Test Accuracy |
|---|---|---|
| Default `C` | 0.82 | 0.80 |
| Best `C` (≈0.089, log-spaced grid search) | 0.8246 | 0.80 |

Tuning `C` does not meaningfully change performance — confirming that the
bottleneck of this baseline is the (frozen, task-agnostic) feature quality
itself, not the classifier's regularization strength. This motivates
fine-tuning DistilBERT directly in Exercise 2.

### Exercise 2.3 — Fine-tuning sweep (best validation checkpoint per config)

| Configuration | Best epoch (of max) | Test Accuracy | Test F1 (positive class) |
|---|---|---|---|
| lr=2e-5, 10 epochs | 6 / 10 | 0.8358 | 0.8357 |
| lr=2e-5, 3 epochs | 3 / 3 | 0.8452 | 0.8471 |
| lr=5e-5, 10 epochs | 1 / 10 | 0.8405 | 0.8471 |

All three configurations overfit the training loss quickly (down to
0.0001–0.003), and validation accuracy plateaus within the first few epochs.
`epochs=3` and `lr=5e-5` reach an almost identical F1 score via different
paths (stopping early vs. converging faster and peaking at epoch 1), while
the standard 10-epoch/lr=2e-5 run is the weakest of the three even at its own
best checkpoint — training longer with a conservative learning rate does not
pay off here. All configurations clearly outperform the frozen-feature SVM
baseline (test accuracy ~0.80).

---

## Method Overview

### Exercise 1.3 — Stable Baseline
1. **Feature extraction** — DistilBERT's `[CLS]` token embedding (last
   hidden layer) is extracted for every sentence via a Hugging Face
   `feature-extraction` pipeline, with the model completely frozen (no
   fine-tuning at all).
2. **Classification** — a `LinearSVC` is trained on top of these fixed
   features.
3. **Hyperparameter search** — `C` is searched over a logarithmic grid
   (`np.logspace(-4, 4, 20)`), selecting the value that maximizes accuracy on
   the validation split.

### Exercise 2.3 — Fine-tuning
1. **Preprocessing** — text is tokenized once via `Dataset.map`, with
   dynamic padding handled by `DataCollatorWithPadding` at batch time (rather
   than fixed-length padding).
2. **Model** — `DistilBertForSequenceClassification` attaches a new,
   randomly-initialized classification head on top of the `[CLS]` token.
3. **Training** — the Hugging Face `Trainer` API fine-tunes the whole model
   end-to-end. A `finetune_and_test` wrapper function re-instantiates a
   *fresh* model for every configuration — essential for a fair comparison,
   since reusing the same model object would mean each new configuration
   continues training from the previous run's already fine-tuned weights.
4. **Model selection** — `load_best_model_at_end=True` (with
   `metric_for_best_model='f1'`) ensures the final test evaluation uses each
   configuration's best validation checkpoint, not just whichever epoch
   happens to be last.

---

## Notebook Structure

```
DLA-Lab2.ipynb
|
|-- Exercise 1
|   |-- 1.1  Dataset exploration
|   |-- 1.2  Tokenizer & DistilBERT exploration (attention masking)
|   |-- 1.3  Stable baseline (frozen features + SVM, C grid search)
|
|-- Exercise 2
    |-- 2.1  Token preprocessing
    |-- 2.2  Model setup (DistilBertForSequenceClassification)
    |-- 2.3  Fine-tuning with Trainer + 3-configuration hyperparameter sweep
```
