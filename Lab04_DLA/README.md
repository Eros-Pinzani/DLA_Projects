# Lab 4 — Adversarial Learning and OOD Detection

Laboratory notebook for the **Deep Learning Applications** course. The lab
builds a full out-of-distribution (OOD) detection pipeline, evaluates it
with standard threshold-free metrics, studies adversarial robustness via
FGSM, and closes with ODIN as an improved, training-free OOD detector.

---

## Repository Structure

```
Lab04_DLA/
├── DLA-Lab4.ipynb        # All exercises (1, 2, 3.1 - ODIN)
├── checkpoints/            # Trained model weights + training history
├── data/                     # CIFAR-10 / CIFAR-100 (downloaded automatically, gitignored)
├── .gitignore
└── README.md
```

---

## Contents

| Exercise | Description |
|---|---|
| **1.1** | OOD detection pipeline: CIFAR-10 (ID) vs. a semantically far subset of CIFAR-100 (OOD), across 4 model variants |
| **1.2** | Threshold-free evaluation: ROC/AUROC and both Precision-Recall curves (AUPR-In, AUPR-Out) |
| **2.1** | FGSM implementation, qualitative and quantitative evaluation, dependence on epsilon |
| **2.2** | On-the-fly adversarial training, re-evaluated with the Exercise 1 OOD pipeline |
| **3.1** | ODIN (temperature scaling + input perturbation), with a grid search over T/epsilon and a sanity check on raw logits |


---

## Dataset

- **In-distribution (ID):** CIFAR-10 — 45,000 train / 5,000 validation / 10,000 test images, 10 classes.
- **Out-of-distribution (OOD):** a 15-class subset of CIFAR-100's test set (1,500 images), deliberately chosen with **no semantic overlap** with any CIFAR-10 class (CIFAR-10 consists only of vehicles and animals; the OOD subset uses flowers, food containers, and man-made outdoor structures instead). This pool is itself split 50/50 into a **search** subset (750 images, paired with `id_val` for every model/hyperparameter comparison) and a **final** subset (750 images, paired with `id_test`, touched only once for the metrics reported as "final" below) — mirroring the ID train/val/test split, so no data is reused between choosing a model/hyperparameters and reporting how well that choice performs.

Both datasets are downloaded automatically via `torchvision.datasets` on first run.

---

## Requirements

```
torch
torchvision
numpy
matplotlib
scikit-learn
```

For a quick setup:

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

For an exact reproduction of the tested environment (pinned versions),
install from [`requirements-freeze.txt`](./requirements-freeze.txt) instead:

```bash
pip install -r requirements-freeze.txt
```

A CUDA-capable GPU is recommended (training 5 model variants for up to 100 epochs each), but everything also runs on CPU, just considerably slower.

---

## Key Results

### Exercise 1 — OOD detection across four model variants

| Model | Val Accuracy | AUROC (search split) |
|---|---|---|
| baseline (no dropout, no augmentation) | 63.4% | 0.634 |
| dropout | 66.5% | 0.718 |
| baseline + augmentation | 78.6% | 0.714 |
| dropout + augmentation | **79.6%** | **0.770** |

Both dropout and data augmentation independently improve OOD separability;
combining them (`dropout_aug`) gives the best result on both classification
accuracy and OOD detection — but only once trained for enough epochs (an
earlier 50-epoch run showed an apparent trade-off between "best classifier"
and "best OOD detector" that disappeared with adequate training).

The table above uses `id_val` + the OOD *search* split — the same data used
to pick `dropout_aug` as the best model, useful for comparison but not an
unbiased estimate. Re-evaluated once on the held-out `id_test` + OOD *final*
split, `dropout_aug` scores **0.789 AUROC**, close to (if anything slightly
above) the search-split figure.

### Exercise 2 — Adversarial robustness and its effect on OOD detection

| | Clean accuracy | FGSM accuracy (eps=0.05) | OOD AUROC (search split) |
|---|---|---|---|
| `dropout_aug` (no adversarial training) | 79.6% | 7.6% | 0.770 |
| `adv_trained` (on-the-fly FGSM training) | 73.5% | **43.1%** | 0.727 |

Adversarial training achieves its primary goal — much stronger FGSM
robustness — at the cost of ~6pp clean accuracy and a *decrease* in OOD
AUROC. This shows that adversarial robustness (to small, targeted
perturbations) and OOD detection (via a confidence-based score) are not
automatically aligned goals.

### Exercise 3 — ODIN

| Method | AUROC (search split) |
|---|---|
| plain max-softmax (`dropout_aug`) | 0.770 |
| ODIN, T=10, eps=0.005 (practical choice) | 0.826 |
| ODIN, T=1000, eps=0.005 (grid maximum) | 0.830 |

ODIN improves AUROC by ~0.06 over plain max-softmax. The gain comes almost
entirely from temperature scaling (which saturates by T=10 — going higher
adds negligible benefit); input perturbation contributes a smaller,
consistent additional gain. Re-evaluated once on the held-out `id_test` +
OOD *final* split with the grid's chosen (T=1000, eps=0.005), ODIN scores
**0.835 AUROC** — consistent with the search-split result, so the
20-combination grid search doesn't show obvious overfitting to the data it
was tuned on.

---

## Method Overview

### Exercise 1 — OOD pipeline
A custom CNN (5 conv + 3 fc layers) is trained in four configurations
(dropout on/off × augmentation on/off), each cached to disk after training
so the notebook can be re-run without retraining. For each model,
max-softmax confidence scores are computed on ID validation data and on the
OOD *search* subset, visualized as histograms, and evaluated with ROC and
both Precision-Recall curves (ID-positive / OOD-positive), following the
metrics described in the ODIN paper. The selected model is then re-evaluated
once on the untouched ID test / OOD *final* subsets for an unbiased final
AUROC.

### Exercise 2 — Adversarial robustness
FGSM perturbs an input in the direction that most increases the model's
loss (`x + epsilon * sign(grad)`), evaluated both qualitatively (visual
examples) and quantitatively (accuracy across a range of epsilon values).
Adversarial training then augments each training batch with an FGSM
counterpart generated **on the fly**, using the model's current weights at
that point in training rather than a fixed pretrained model. The resulting
model is re-evaluated with the exact same OOD pipeline from Exercise 1.

### Exercise 3 — ODIN
ODIN combines two training-free tricks: dividing logits by a temperature T
before the softmax (counteracting softmax saturation), and perturbing the
input slightly in the direction of *higher* confidence in the model's own
predicted class (the mirror image of FGSM, which perturbs toward *lower*
confidence). Both hyperparameters are tuned via a small grid search against
AUROC on the OOD *search* split only; the final reported AUROC uses the
separate, untouched *final* split.

---

## Notebook Structure

```
DLA-Lab4.ipynb
|
|-- Exercise 1: OOD Detection and Performance Evaluation
|   |-- 1.1  Build the OOD detection pipeline (CNN, ID/OOD datasets, 4 model variants, scores/histograms)
|   |-- 1.2  ROC / PR curve evaluation across all four variants, plus a held-out final evaluation of the best model
|
|-- Exercise 2: Enhancing Robustness to Adversarial Attack
|   |-- 2.1  FGSM implementation, qualitative + quantitative evaluation, epsilon dependence
|   |-- 2.2  On-the-fly adversarial training, re-evaluated on the Exercise 1 OOD pipeline
|
|-- Exercise 3: Wildcard
    |-- 3.1  ODIN (temperature scaling + input perturbation, grid search, held-out final evaluation, raw-logit sanity check)
```
