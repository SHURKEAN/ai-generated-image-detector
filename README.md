# AI-Generated Image Detector

A research-to-application project for detecting AI-generated images, developed from the CIFAKE experiments associated with **IEEE CITDS 2026** and subsequently extended with a practical Gradio interface and a separate normal-resolution detector.

**Research result: 99.07% accuracy on the balanced 20,000-image CIFAKE test set.** This is not 99.07% universal real-world accuracy. The later application and cross-generator work are separate from the experiments reported in the paper.

## Associated Research

**Paper:** *Hyperparameter-Aware Evaluation of Deep Learning Architectures for AI-Generated Image Detection*

**Authors:** Ilham Gafarov, Zakarya Qasem, Kerem Düzenli, Balázs Harangi, and Mokhaled N. A. Al-Hamadani.

**Conference:** [2026 IEEE 4th Conference on Information Technology and Data Science (CITDS 2026)](https://citds.inf.unideb.hu/), Debrecen, Hungary, 27-28 August 2026.

**Publication status:** The [University of Debrecen publication database](https://tudoster.unideb.hu/en/szerzok/5404) currently lists the proceedings paper as **"Accepted by Publisher"**. This is not a claim that the final IEEE Xplore record is already available.

**IEEE Xplore:** Coming soon

**DOI:** Coming soon

## Research foundation and application extension

The academic study compared ResNet-50, EfficientNetV2-S, ViT-B/16, and Xception under baseline and hyperparameter-tuned configurations, followed by weighted soft-voting evaluation on CIFAKE. The later application adds image upload, reusable inference, transparent detector selection, and normal-resolution screening.

| Layer | Purpose | Evidence and scope |
|---|---|---|
| Academic research | Architecture comparison, tuning, and ensemble evaluation | CIFAKE experiments in `notebooks/` and saved results in `outputs/`. |
| Published detector | Preserve the research ensemble and checkpoint identity | CIFAKE-only benchmark; checkpoint hashes in the frozen baseline manifest. |
| Application extension | Analyze new user-provided images through a Gradio interface | A later engineering extension, not a result from the original paper. |
| Broader evaluation | Test unseen generators, camera sources, and post-processing | Separate ongoing work; no universal accuracy claim. |

The interface reports the predicted class, model-estimated confidence, AI probability, detector version, interpretation note, and individual model scores where available.

The detector is intended for research and screening. Its output is probabilistic and must not be treated as forensic proof of an image's origin.

## Current detector design

The application supports two main inference routes:

- **Normal-resolution images:** the Community Forensics ViT-S/16 detector is used for broader cross-generator screening.
- **CIFAKE-scale images:** images whose longest side is at most 64 pixels can use the published four-model CIFAKE ensemble.

All four published models are evaluated and displayed. The saved tuned voting weights are **0 for ResNet-50 and 1/3 each for EfficientNetV2-S, ViT-B/16, and Xception**, so ResNet-50 does not contribute to the final ensemble probability. See [research reproduction guidance](REPRODUCIBILITY.md) for the exact configuration and limitations.

In `auto` mode, Community Forensics is preferred for normal-resolution inputs when its assets are present. Without them, the factory may select the local experimental detector or fall back to the published CIFAKE detector. Use explicit `community` mode for the public setup below rather than relying on an unavailable fallback.

## Features

- Upload and analyze one image at a time
- Real or AI-generated prediction
- Model-estimated confidence and AI probability
- Transparent detector-version and routing information
- Individual model scores when using the published ensemble
- Automatic CUDA use when available, with CPU support
- Lazy model loading and reuse between requests
- 20 MB upload limit and periodic temporary-file cleanup

## Requirements

- Python 3.11
- Windows, Linux, or macOS
- A CUDA-capable GPU is optional; CPU inference is supported
- Sufficient disk space for the selected model assets

## Installation

From the repository root, create a Python 3.11 virtual environment and install the application dependencies.

First clone the public repository if you do not already have a checkout:

```text
git clone https://github.com/SHURKEAN/ai-generated-image-detector.git
cd ai-generated-image-detector
```

### PowerShell

```powershell
py -3.11 -m venv .venv-app
.\.venv-app\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-app.txt
```

## Model assets

Model weights are intentionally excluded from Git because they are large. The normal-resolution reference assets can be downloaded and SHA-256 verified with:

```powershell
.\scripts\download_realworld_assets.ps1
```

The script stores the assets under `models/realworld_v2/upstream/`. It currently downloads both the Community Forensics checkpoint/configuration and the CLIP backbone used by the experimental reference modes (approximately 1 GB in total). Downloading assets is explicit; the app itself does not fetch them automatically. Review the upstream model licences and intended research usage before redistribution or deployment.

The published CIFAKE route additionally requires these locally retained checkpoints:

```text
models/tuned/resnet50_hyperparameter_tuned_best.pth
models/tuned/efficientnetv2s_hyperparameter_tuned_best.pth
models/tuned/vit16_hyperparameter_tuned_best.pth
models/tuned/xception_hyperparameter_tuned_best.pth
```

Their expected sizes and SHA-256 hashes are recorded in `models/published_ctds_v1/manifest.json`. The application does not retrain or silently replace them.

The project's four original research checkpoints are **not currently distributed through a public download link**. A public source checkout alone therefore cannot reproduce their exact scores. Training workflows are described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md); a verified artifact release remains pending redistribution review.

## Running the application

For the public setup using the downloadable Community Forensics model:

```powershell
$env:AI_DETECTOR_MODE = "community"
python -m app.app
```

Open the local Gradio address shown in the terminal, upload an image, and select **Analyze**.

If all local model assets are available, the default adaptive mode can be used:

```powershell
Remove-Item Env:AI_DETECTOR_MODE -ErrorAction SilentlyContinue
python -m app.app
```

### Runtime settings

| Variable | Value | Effect |
|---|---|---|
| `AI_DETECTOR_MODE` | `auto` | Default adaptive routing between normal-resolution and CIFAKE-scale inputs. |
| `AI_DETECTOR_MODE` | `community` | Always use the Community Forensics detector. |
| `AI_DETECTOR_MODE` | `published` | Always use the four-model published CIFAKE ensemble. |
| `AI_DETECTOR_DEVICE` | `cpu` | Force CPU inference. |
| `GRADIO_SHARE` | `true` | Deliberately create a temporary public Gradio share link. Disabled by default. |

Additional experimental modes remain available in the code but are not the default competition route.

## Testing

Run the automated tests from the repository root:

```powershell
python -m pytest -q
```

Run the optional real-checkpoint integration test only when all four published checkpoints and the CIFAKE `test/FAKE` and `test/REAL` JPEG samples are present:

```powershell
$env:RUN_MODEL_INTEGRATION = "1"
python -m pytest -q
```

Unit tests validate application behavior, routing, preprocessing, and voting logic. They do not establish real-world detection accuracy. The optional integration test checks one real and one AI-generated CIFAKE sample, not the full benchmark.

## Research reproduction and versioning

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the dataset layout, experiment commands, preprocessing, ensemble weights, saved metrics, and checkpoint verification.

The original research model identity is protected by [the baseline manifest](models/published_ctds_v1/manifest.json). Application development continues on `main` without modifying those checkpoint files.

A dedicated `v1.0-citds2026` academic tag/release is **not yet created**. The existing initial Git commit already includes the later application, so it should not be represented as an exact paper-era snapshot. An appropriate research snapshot must be identified and reviewed before that tag is assigned. A future application release can be versioned separately.

## Repository layout

| Path | Purpose |
|---|---|
| `app/` | Gradio interface, detector selection, preprocessing, inference, and ensemble logic. |
| `realworld_v2/` | Normal-resolution detector loading, evaluation, and training utilities. |
| `models/` | Trackable manifests and documentation; large checkpoint files remain ignored. |
| `notebooks/` | Baseline and hyperparameter-tuned research notebooks. |
| `scripts/` | Asset download, validation, benchmarking, and experiment utilities. |
| `tests/` | Automated unit and routing tests. |
| `third_party/` | Vendored academic reference code with upstream sources and licences documented separately. |
| `REPRODUCIBILITY.md` | Public reproduction instructions and the remaining artifact/release gaps. |
| `CITATION.cff` | Repository citation metadata and the associated paper's author list. |

Private papers, PDFs, competition documents, presentations, figures, working documentation, datasets, checkpoints, credentials, logs, environments, and generated caches remain local and are excluded through `.gitignore`.

## Evaluation roadmap

Broader evaluation must be reported separately from the CIFAKE result. Priorities include unseen generators, genuine camera photographs from independent sources, JPEG compression, resizing, crops, screenshots, light editing, and social-media re-encoding. Report class-balanced performance, genuine-image false-positive rate, precision, recall, F1, ROC-AUC, confusion matrices, and calibration rather than accuracy alone.

## Citation

Refer to [CITATION.cff](CITATION.cff) for repository and paper citation metadata. The paper attribution includes all five authors in the official publication-record order; the application repository maintainer is Ilham Gafarov. Final DOI, page numbers, and IEEE Xplore identifiers are intentionally omitted until verified. Citation does not grant permission to reuse restricted project code.

## Responsible use and limitations

- A high confidence score does not prove that an image is AI-generated or real.
- Performance can change with resizing, compression, screenshots, editing, camera pipelines, and previously unseen generators.
- Do not use the result as the sole basis for legal, disciplinary, journalistic, or moderation decisions.
- Validate the detector on data representative of the intended deployment environment.

## Copyright

Copyright (c) 2026 Ilham Gafarov. All rights reserved.

This repository is publicly visible for viewing and evaluation. No open-source licence is granted for the original project code. Permission to copy, modify, distribute, or reuse that code must be obtained in writing from the copyright holder. Third-party components remain governed by their respective licences.
