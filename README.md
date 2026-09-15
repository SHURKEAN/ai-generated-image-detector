# AI-Generated Image Detector

This repository contains the research code and a local Gradio demonstration for detecting whether an image is likely real or AI-generated.

The published CIFAKE study evaluates ResNet-50, EfficientNetV2-S, ViT-B/16, Xception, and a weighted soft-voting ensemble. The application also includes a normal-resolution route intended for more realistic photographs. An AI-image detector is probabilistic: its output should be treated as an estimate, not proof of an image's origin.

## Run the demo locally

Use Python 3.11 and install the application requirements:

```powershell
python -m pip install -r requirements-app.txt
python app\app.py
```

Model weights are deliberately not stored in Git. Place the required checkpoints in `models/` as described by the project documentation before starting the app.

## Repository layout

- `app/` — Gradio interface and inference pipeline
- `realworld_v2/` — normal-resolution detector components
- `scripts/` and `tests/` — utilities and tests

The paper, competition submission, research figures, presentations, and working documentation are retained locally and are not included in the public repository.

## What is not uploaded

Datasets, checkpoints, virtual environments, logs, credentials, prediction caches, package installers, papers, competition material, research figures, presentations, and working documentation are ignored by `.gitignore`. In particular, never upload `kaggle.json`.

## Before publishing

Run the test suite and review exactly what will be committed:

```powershell
python -m pytest -q
git status
git add -n .
```

This repository does not currently grant an open-source licence. Copyright remains with the repository owner.
# ai-generated-image-detector
