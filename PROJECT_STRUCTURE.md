# Project Folder Structure

This repository is arranged so that the shareable code, papers, slides, and competition material are distinct from local-only datasets, checkpoints, environments, and caches. The local-only material is preserved on this computer but excluded by `.gitignore`.

## Main Folders

| Folder | Purpose |
|---|---|
| `app/` | Gradio application and inference code. |
| `realworld_v2/` | Normal-resolution detector implementation and evaluation configuration. |
| `notebooks/baseline/` | Baseline experiment notebooks `01` to `05`. |
| `notebooks/tuned/` | Fine-tuned experiment notebooks `01` to `05`. |
| `scripts/` | Python and PowerShell runners for executing notebooks. |
| `docs/` | Local-only project notes, audit material, and technical documentation. |
| `research/papers/` | Local-only LaTeX sources, published paper, drafts, and build artifacts. |
| `presentations/` | Local-only PowerPoint presentations and preview images. |
| `competition_submission/` | Local-only competition project-description document. |
| `figures/` | Local-only figures used by the papers and presentations. |
| `archive/legacy_notebooks/` | Older notebook copies kept for reference. |
| `CIFAKE_FULL/` | Official CIFAKE train/test dataset. |
| `models/` | Saved baseline and fine-tuned model checkpoints. |
| `outputs/` | Metrics, plots, JSON files, predictions, and hybrid outputs. |
| `run_logs/` | Terminal and notebook execution logs. |
| `wheelhouse/` | Local Python wheel files, including CUDA/PyTorch installers. |
| `.python-3.10.11/` | Portable Python base used by the project environment. |
| `.venv-tuned/` | Project-local Python environment with CUDA and notebook packages. |

## Local-Only Folders

`CIFAKE_FULL/`, `data/`, `models/`, `outputs/`, `run_logs/`, `wheelhouse/`, `.python-3.10.11/`, `.venv-tuned/`, `competition_submission/`, `presentations/`, `research/papers/`, `docs/`, `figures/`, and `archive/` stay available locally but are not intended for GitHub.

## Important Files Kept In The Project Root

| File | Purpose |
|---|---|
| `kaggle.json` | Kaggle API credentials used by notebooks when needed; never upload this file. |
| `cifake-real-and-ai-generated-synthetic-images.zip` | Original downloaded CIFAKE dataset archive; kept local only. |
| `PROJECT_STRUCTURE.md` | This structure guide. |
| `README.md` | GitHub entry point and local run instructions. |
| `requirements.txt` | Reproducible list of direct Python dependencies. |
| `docs/project_audit_2026-08-11.md` | Latest full validation report, repairs, and unresolved research warnings. |

## Running The Existing Scripts

Run scripts from the project root. The PowerShell launcher automatically repairs
the copied virtual environment path and uses the project-local Python:

```powershell
.\scripts\run_all_gpu.ps1
.\scripts\run_all_gpu.ps1 -SkipHybrid
```

For individual or tuned runs, use the same project-local Python explicitly:

```powershell
.\.venv-tuned\Scripts\python.exe scripts\run_tuned_01_04_05.py
.\.venv-tuned\Scripts\python.exe scripts\run_tuned_models_live.py
.\.venv-tuned\Scripts\python.exe scripts\run_all_live.py --start-at 01
```

To validate the notebooks, dataset, results, checkpoints, and saved media without
retraining the models:

```powershell
.\.venv-tuned\Scripts\python.exe scripts\validate_project.py --deep
```

The runners create a relocatable project-local `tdk-gpu` Jupyter kernel. They use
the project root for `CIFAKE_FULL`, `models`, `outputs`, and `run_logs`, so existing
experiment results remain in the same places.
