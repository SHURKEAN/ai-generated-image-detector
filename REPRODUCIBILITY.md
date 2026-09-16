# Research reproduction and artifact provenance

The CIFAKE study and the later application are distinct stages of this project. The application is not the sole reproduction path for the paper, and normal-resolution detector results must not be substituted for CIFAKE benchmark results.

## Scope and current availability

The tuned research ensemble's saved result is **99.07% accuracy on the balanced 20,000-image official CIFAKE test split**. The saved results are available in `outputs/tuned/hybrid/hybrid_hyperparameter_tuned_final_results.json`.

Training notebooks, selected configurations, and summary results are public. Datasets and trained weights are excluded from Git. The four exact research checkpoints are retained locally and described by `models/published_ctds_v1/manifest.json`, but a public checkpoint download has not yet been released. Reproducing those exact artifact hashes is not guaranteed by retraining.

The dependency files describe the current supported code environment. They should not be interpreted as a complete reconstruction of every package installed during the original experiments. Bit-for-bit retraining and fully independent benchmark reproduction have not been established by the application unit tests.

## Dataset

Obtain the [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) and retain its official split:

```text
CIFAKE_FULL/
  train/
    FAKE/   # 50,000 images
    REAL/   # 50,000 images
  test/
    FAKE/   # 10,000 images
    REAL/   # 10,000 images
```

Hyperparameter selection uses training-split subsets: 4,000 tuning-training images and 1,000 tuning-validation images, balanced by class. The official test set must not be used to choose configurations, ensemble weights, or thresholds. Check the individual notebooks for the sampling and final-training implementation.

The class ordering is `0 = FAKE / AI-generated` and `1 = REAL`. Review the dataset's attribution and usage conditions before obtaining or redistributing it. Do not commit `kaggle.json`, source image archives, or private photographs.

## Experiment locations

| Material | Location |
|---|---|
| Baseline models and ensemble | `notebooks/baseline/`, notebooks 01-05. |
| Tuned models and ensemble | `notebooks/tuned/`, notebooks 01-05. |
| Selected model configurations | `outputs/tuned/*_hyperparameter_tuned_best_config.json`. |
| Model tuning histories | `outputs/tuned/*_hyperparameter_tuned_tuning_results.json`. |
| Ensemble configuration | `outputs/tuned/hybrid/hybrid_hyperparameter_tuned_best_config.json`. |
| Final ensemble metrics | `outputs/tuned/hybrid/hybrid_hyperparameter_tuned_final_results.json`. |
| Frozen checkpoint identity | `models/published_ctds_v1/manifest.json`. |

The tuned notebooks set random seed 42. GPU implementation details and package changes may still affect reproducibility.

## Preprocessing and ensemble

The current tuned detector uses EXIF-aware RGB conversion, bilinear resize to 224 x 224 with antialiasing, conversion to a tensor, and ImageNet normalization (mean `0.485, 0.456, 0.406`; standard deviation `0.229, 0.224, 0.225`). See `app/preprocessing.py` and the selected training configurations rather than assuming a different resize for Xception.

For each of the four models, two-class logits are converted to probabilities with softmax. The saved tuned soft-voting weights are:

| Model | Normalized weight |
|---|---:|
| ResNet-50 | 0 |
| EfficientNetV2-S | 1/3 |
| ViT-B/16 | 1/3 |
| Xception | 1/3 |

ResNet-50 is still evaluated and shown but does not influence the final ensemble score. Weights are selected on training-split validation data, not the official test set. The implementation is in `app/ensemble.py`.

### Existing methodology caveats

The repository audit (`scripts/validate_project.py`) flags potential dependence in ensemble-weight selection: the validation subset comes from the training pool already seen by the final member checkpoints. This is not an independent, unseen ensemble-validation set, even though it is separate from the official test set. It also flags duplicate tuning trials and probability caches without checkpoint/dataset fingerprints. Review these issues before interpreting the tuning-validation scores or claiming a fully independent reproduction. Future methodology improvements must be recorded separately rather than altering the retained publication artifacts.

Machine-specific paths in public saved outputs have been normalized to relative paths or anonymized home markers. Notebook source code and numeric result values are preserved; this is a privacy cleanup, not a new experiment.

## Verify retained research checkpoints without training

When the four local checkpoint files are present:

```text
python scripts/verify_published_baseline.py
```

This checks file sizes and SHA-256 hashes against the protected manifest. It confirms artifact identity, **not** detection accuracy. Do not overwrite or train into the four files referenced by that manifest.

## Reproduce experiments in an isolated checkout

Use a separate clone and environment so training cannot overwrite the retained publication artifacts. The research dependency file targets CUDA 12.8 PyTorch; it is separate from `requirements-app.txt`, which is the application setup.

```text
python -m pip install -r requirements.txt
python scripts/run_all_gpu.py
python scripts/run_tuned_models_live.py
python scripts/run_tuned_01_04_05.py
```

The first runner executes baseline notebooks 01-05 and requires CUDA. The second executes tuned EfficientNetV2-S and ViT-B/16; the third executes tuned ResNet-50, Xception, and the final ensemble. These commands perform training, save checkpoints/results, and modify executed notebooks. They can take substantial time and generate large logs/caches. Initial backbone acquisition may need network access.

If matching trained checkpoints and the official dataset are already available, the tuned ensemble notebook can be executed separately to recompute its validation weight search and test evaluation:

```text
python scripts/run_notebook_live.py notebooks/tuned/05_hybrid_ensemble_hyperparameter_tuned.ipynb
```

That notebook can write ensemble configuration, caches, and metrics. Run it in the isolated checkout, inspect the dataset counts and class mapping, and compare its `official_test_metrics` with the saved reference. Keep any changed result distinct from the paper's reported result rather than silently replacing it.

## Separate application extension

Community Forensics provides the preferred normal-resolution route when its verified assets are available. The local CLIP-head experiment and UniversalFakeDetect reference are additional academic modes, not the published CIFAKE ensemble. Their weights, datasets, transformations, and measurements must be recorded separately.

The model source URLs, revisions, hashes, and usage tracks are in `models/realworld_v2/upstream_manifest.json`; vendored source attribution is in `third_party/SOURCES.md`. The application asset-download script does not supply the project's original four research checkpoints.

## Release readiness

- [x] Research/application distinction documented.
- [x] Official CIFAKE split, class order, seed, preprocessing, and ensemble weights documented.
- [x] Research notebooks, selected configurations, and summary metrics retained.
- [x] Original checkpoint paths, file sizes, and SHA-256 hashes recorded.
- [x] Paper author list and repository citation metadata included.
- [ ] Public release of the exact research checkpoints, subject to redistribution review.
- [ ] Independently verified end-to-end benchmark reproduction.
- [ ] A reviewed paper-era code snapshot for `v1.0-citds2026`.
- [ ] Verified final IEEE Xplore article URL, DOI, and final bibliographic details.

No paper-era tag is claimed yet: the earliest existing Git commit already contains the later application. Preserve that history and identify an appropriate research snapshot before assigning the academic tag. An application release can be prepared separately once its acceptance criteria are met.
