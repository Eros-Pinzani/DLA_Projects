# DLA Projects

This repository contains the lab exercises for the **Deep Learning
Applications** course.

## Labs

| Lab | Topic |
|---|---|
| [`Lab01_DLA`](./Lab01_DLA) | Transfer learning, fine-tuning, and retrieval-based classification on GTSRB |
| [`Lab02_DLA`](./Lab02_DLA) | Transformers and the Hugging Face ecosystem: DistilBERT fine-tuning for sentiment analysis, plus a standalone CLIP-based text-to-image retrieval app |
| [`Lab04_DLA`](./Lab04_DLA) | OOD detection, adversarial robustness (FGSM), and ODIN |

Each lab has its own `README.md` with a detailed breakdown of its exercises, dataset, method, and key results.

---

## Repository Structure

```
DLA_Projects/
├── Lab01_DLA/
│   ├── DLA-Lab1.ipynb
│   ├── README.md
│   └── requirements-freeze.txt
├── Lab02_DLA/
│   ├── DLA-Lab2.ipynb
│   ├── README.md
│   ├── requirements-freeze.txt
│   └── TextToImageApp/              # standalone CLIP retrieval app (Exercise 3.3)
│       ├── app.py
│       ├── build_index.py
│       ├── requirements.txt
│       ├── requirements-freeze.txt
│       └── README.md
├── Lab04_DLA/
│   ├── DLA-Lab4.ipynb
│   ├── README.md
│   └── requirements-freeze.txt
└── README.md
```

---

## Environment Setup

Each lab manages its own dependencies independently -- they don't all need
the same libraries (e.g. Lab02 needs `transformers`/`datasets`, Lab04 does
not). Clone the repo and set up a virtual environment first:

```bash
git clone https://github.com/Eros-Pinzani/DLA_Projects.git
cd DLA_Projects
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\Activate.ps1
```

Then, from inside the specific lab folder you want to run:

```bash
pip install -r requirements-freeze.txt   # exact pinned versions
# or, for a quicker/unpinned setup:
pip install -r requirements.txt          # where available (see that lab's README)
```

See each lab's own `README.md` for its exact list of dependencies and any
lab-specific setup notes (dataset download, checkpoints, etc.).

Requires Python 3.10+.
