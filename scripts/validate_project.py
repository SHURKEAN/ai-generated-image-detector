from __future__ import annotations

import argparse
import ast
import gc
import importlib.util
import json
import math
import re
import shutil
import sys
import zipfile
from pathlib import Path

import nbformat
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from runtime_utils import configure_console_output


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
EXPECTED_DATA_COUNTS = {
    ("train", "FAKE"): 50_000,
    ("train", "REAL"): 50_000,
    ("test", "FAKE"): 10_000,
    ("test", "REAL"): 10_000,
}
REQUIRED_MODULES = [
    "PIL",
    "jupyter_client",
    "kaggle",
    "matplotlib",
    "nbclient",
    "nbformat",
    "numpy",
    "seaborn",
    "sklearn",
    "timm",
    "torch",
    "torchvision",
    "tqdm",
]


class Audit:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        print(f"[OK]   {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"[FAIL] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[WARN] {message}")


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def check_dependencies(audit: Audit) -> None:
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        audit.fail("Missing Python modules: " + ", ".join(missing))
        return
    audit.ok(f"All {len(REQUIRED_MODULES)} required third-party modules import")
    if not torch.cuda.is_available():
        audit.fail("CUDA is not available to PyTorch")
        return
    audit.ok(f"CUDA available: {torch.cuda.get_device_name(0)} ({torch.__version__})")


def check_source_files(audit: Audit) -> None:
    scripts = sorted((ROOT / "scripts").glob("*.py"))
    for path in scripts:
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except Exception as exc:
            audit.fail(f"Python syntax error in {path.relative_to(ROOT)}: {exc}")
    if not any(str(path.relative_to(ROOT)) in failure for path in scripts for failure in audit.failures):
        audit.ok(f"Python syntax valid in {len(scripts)} scripts")

    active = sorted((ROOT / "notebooks").rglob("*.ipynb"))
    for path in active:
        try:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code" and cell.source.strip():
                    ast.parse(cell.source, filename=f"{path}:cell_{index}")
            saved_errors = sum(
                output.get("output_type") == "error"
                for cell in notebook.cells
                if cell.cell_type == "code"
                for output in cell.get("outputs", [])
            )
            if saved_errors:
                audit.fail(f"{path.relative_to(ROOT)} contains {saved_errors} saved error output(s)")
        except Exception as exc:
            audit.fail(f"Invalid notebook {path.relative_to(ROOT)}: {exc}")
    if not any("notebook" in failure.lower() for failure in audit.failures):
        audit.ok(f"JSON, nbformat, and code syntax valid in {len(active)} active notebooks")

    archive_errors = 0
    for path in (ROOT / "archive").rglob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        archive_errors += sum(
            output.get("output_type") == "error"
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
            for output in cell.get("outputs", [])
        )
    if archive_errors:
        audit.warn(f"Archived notebooks retain {archive_errors} historical saved error output(s)")


def count_images(directory: Path) -> int:
    return sum(path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES for path in directory.iterdir())


def check_dataset(audit: Audit, deep: bool) -> None:
    data_root = ROOT / "CIFAKE_FULL"
    all_images: list[Path] = []
    for (split, label), expected in EXPECTED_DATA_COUNTS.items():
        directory = data_root / split / label
        if not directory.is_dir():
            audit.fail(f"Missing dataset directory: {directory.relative_to(ROOT)}")
            continue
        actual = count_images(directory)
        if actual != expected:
            audit.fail(f"{directory.relative_to(ROOT)} has {actual:,} images; expected {expected:,}")
        all_images.extend(
            path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if not any("dataset" in failure.lower() or "images; expected" in failure for failure in audit.failures):
        audit.ok("CIFAKE class counts: 100,000 train and 20,000 test")

    zero_length = [path for path in all_images if path.stat().st_size == 0]
    if zero_length:
        audit.fail(f"Dataset contains {len(zero_length)} zero-length image files")
    else:
        audit.ok("No zero-length dataset images")

    if deep:
        corrupt: list[tuple[Path, str]] = []
        for path in all_images:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                corrupt.append((path, str(exc)))
        if corrupt:
            audit.fail(f"Pillow verification failed for {len(corrupt)} dataset images; first: {corrupt[0]}")
        else:
            audit.ok(f"Pillow verified all {len(all_images):,} authoritative dataset images")

    duplicate_root = ROOT / "notebooks" / "baseline" / "CIFAKE_FULL"
    if duplicate_root.exists():
        duplicate_total = sum(
            count_images(duplicate_root / split / label)
            for split, label in EXPECTED_DATA_COUNTS
            if (duplicate_root / split / label).is_dir()
        )
        audit.warn(f"A duplicate dataset tree exists under notebooks/baseline ({duplicate_total:,} images)")

    archive_path = ROOT / "cifake-real-and-ai-generated-synthetic-images.zip"
    try:
        with zipfile.ZipFile(archive_path) as archive:
            bad_member = archive.testzip()
            if bad_member:
                audit.fail(f"Dataset ZIP CRC failed at {bad_member}")
            else:
                audit.ok(f"Dataset ZIP CRC valid ({len(archive.infolist()):,} entries)")
    except Exception as exc:
        audit.fail(f"Dataset ZIP is invalid: {exc}")


def matrix_accuracy(matrix: object) -> float | None:
    try:
        array = np.asarray(matrix, dtype=np.int64)
    except Exception:
        return None
    if array.shape != (2, 2) or array.sum() <= 0:
        return None
    return float(np.trace(array) / array.sum())


def check_result_json(audit: Audit) -> None:
    paths = sorted((ROOT / "outputs").rglob("*.json"))
    parsed: dict[Path, object] = {}
    for path in paths:
        try:
            parsed[path] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            audit.fail(f"Invalid JSON {path.relative_to(ROOT)}: {exc}")
    if len(parsed) == len(paths):
        audit.ok(f"All {len(paths)} output JSON files parse")

    final_files = [
        ROOT / "outputs" / "resnet50_final_metrics.json",
        ROOT / "outputs" / "efficientnetv2s_final_metrics_official.json",
        ROOT / "outputs" / "vit_b16_final_metrics_official.json",
        ROOT / "outputs" / "xception_final_metrics_official.json",
        ROOT / "outputs" / "hybrid" / "hybrid_final_metrics_official.json",
        *sorted((ROOT / "outputs" / "tuned").glob("*_final_results.json")),
        ROOT / "outputs" / "tuned" / "hybrid" / "hybrid_hyperparameter_tuned_final_results.json",
    ]
    checked = 0
    for path in final_files:
        data = parsed.get(path)
        if not isinstance(data, dict):
            audit.fail(f"Missing final result JSON: {path.relative_to(ROOT)}")
            continue
        if "final_test_metrics" in data:
            metrics = data["final_test_metrics"]
            matrix = metrics.get("confusion_matrix")
        elif "hybrid_test_metrics" in data:
            metrics = data["hybrid_test_metrics"]
            matrix = metrics.get("confusion_matrix")
        else:
            metrics = data.get("official_test_metrics", {})
            matrix = data.get("confusion_matrix")
        actual_accuracy = matrix_accuracy(matrix)
        saved_accuracy = metrics.get("accuracy") if isinstance(metrics, dict) else None
        if actual_accuracy is None or not isinstance(saved_accuracy, (int, float)):
            audit.fail(f"Missing usable accuracy/confusion matrix in {path.relative_to(ROOT)}")
            continue
        if not math.isclose(actual_accuracy, float(saved_accuracy), abs_tol=1e-12):
            audit.fail(
                f"Accuracy/confusion-matrix mismatch in {path.relative_to(ROOT)}: "
                f"{saved_accuracy} vs {actual_accuracy}"
            )
            continue
        checked += 1
    if checked == len(final_files):
        audit.ok(f"Accuracy matches confusion matrices in all {checked} final result files")


def check_npz_files(audit: Audit) -> None:
    paths = sorted((ROOT / "outputs").rglob("*.npz"))
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as archive:
                if not archive.files:
                    raise ValueError("archive has no arrays")
                for name in archive.files:
                    array = archive[name]
                    if array.dtype.kind in "fc" and not np.isfinite(array).all():
                        raise ValueError(f"array {name} contains non-finite values")
        except Exception as exc:
            audit.fail(f"Invalid NPZ {path.relative_to(ROOT)}: {exc}")
    if not any("NPZ" in failure for failure in audit.failures):
        audit.ok(f"All {len(paths)} NPZ files load and contain finite numeric arrays")

    prediction_specs = [
        (
            ROOT / "outputs" / "hybrid" / "hybrid_test_predictions_official.npz",
            ROOT / "outputs" / "hybrid" / "hybrid_final_metrics_official.json",
            "hybrid_test_metrics",
        ),
        (
            ROOT / "outputs" / "tuned" / "hybrid" / "hybrid_hyperparameter_tuned_predictions.npz",
            ROOT / "outputs" / "tuned" / "hybrid" / "hybrid_hyperparameter_tuned_final_results.json",
            "official_test_metrics",
        ),
    ]
    for predictions_path, metrics_path, key in prediction_specs:
        with np.load(predictions_path, allow_pickle=False) as predictions:
            y_true = predictions["y_true"]
            y_pred = predictions["y_pred"]
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))[key]
        calculated = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_fake_positive": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
            "recall_fake_positive": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            "f1_fake_positive": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        }
        for metric, value in calculated.items():
            if not math.isclose(float(saved[metric]), float(value), abs_tol=1e-12):
                audit.fail(f"{metrics_path.relative_to(ROOT)} {metric} does not match saved predictions")
        if len(y_true) != 20_000:
            audit.fail(f"{predictions_path.relative_to(ROOT)} has {len(y_true):,} predictions, expected 20,000")
    if not any("saved predictions" in failure for failure in audit.failures):
        audit.ok("Baseline and tuned hybrid metrics reproduce from saved test predictions")


def check_checkpoints(audit: Audit) -> None:
    paths = sorted((ROOT / "models").rglob("*.pth"))
    for path in paths:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            if not isinstance(checkpoint, dict) or not checkpoint:
                raise ValueError("checkpoint is not a non-empty state dictionary")
            tensor_count = sum(isinstance(value, torch.Tensor) for value in checkpoint.values())
            if tensor_count == 0:
                raise ValueError("checkpoint contains no tensors")
            non_finite = [
                name
                for name, value in checkpoint.items()
                if isinstance(value, torch.Tensor)
                and (value.is_floating_point() or value.is_complex())
                and not torch.isfinite(value).all()
            ]
            if non_finite:
                preview = ", ".join(non_finite[:3])
                raise ValueError(f"checkpoint contains non-finite tensor values: {preview}")
            del checkpoint
            gc.collect()
        except Exception as exc:
            audit.fail(f"Invalid checkpoint {path.relative_to(ROOT)}: {exc}")
    if not any("checkpoint" in failure.lower() for failure in audit.failures):
        audit.ok(f"All {len(paths)} model checkpoints deserialize safely and contain finite tensors")


def check_media_and_tex(audit: Audit) -> None:
    media = [
        path
        for base in (ROOT / "outputs", ROOT / "figures")
        for path in base.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    for path in media:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            audit.fail(f"Invalid image artifact {path.relative_to(ROOT)}: {exc}")
    if not any("image artifact" in failure for failure in audit.failures):
        audit.ok(f"All {len(media)} plot/figure images pass Pillow verification")

    for path in ROOT.glob("*.pdf"):
        data = path.read_bytes()
        if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-2048:]:
            audit.fail(f"Invalid PDF envelope: {path.name}")
    if not any("PDF" in failure for failure in audit.failures):
        audit.ok("Root PDF files have valid PDF headers and EOF markers")

    for path in ROOT.glob("*.tex"):
        source = path.read_text(encoding="utf-8", errors="replace")
        for reference in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source):
            candidate = ROOT / reference
            choices = [candidate] if candidate.suffix else [candidate.with_suffix(ext) for ext in (".pdf", ".png", ".jpg", ".jpeg")]
            if not any(choice.exists() for choice in choices) and f"\\IfFileExists{{{reference}}}" not in source:
                audit.fail(f"{path.name} references missing graphic: {reference}")
    if shutil.which("pdflatex") is None and shutil.which("latexmk") is None:
        audit.warn("No LaTeX compiler is installed; existing PDFs were checked, but TeX was not rebuilt")


def extract_grid(path: Path) -> list[dict] | None:
    tree = ast.parse(notebook_source(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "HYPERPARAMETER_GRID" for target in node.targets):
            value = ast.literal_eval(node.value)
            return value if isinstance(value, list) else None
    return None


def check_methodology(audit: Audit) -> None:
    for path in sorted((ROOT / "notebooks" / "tuned").glob("0[1-4]_*.ipynb")):
        grid = extract_grid(path)
        if grid is None:
            audit.warn(f"Could not statically read HYPERPARAMETER_GRID in {path.name}")
            continue
        normalized = [json.dumps(row, sort_keys=True) for row in grid]
        if len(set(normalized)) != len(normalized):
            audit.warn(f"{path.name} contains duplicate hyperparameter trials")

    baseline_hybrid = notebook_source(ROOT / "notebooks" / "baseline" / "05_hybrid_ensemble.ipynb")
    tuned_hybrid = notebook_source(ROOT / "notebooks" / "tuned" / "05_hybrid_ensemble_hyperparameter_tuned.ipynb")
    if "Subset(dataset, VAL_INDICES)" in baseline_hybrid:
        audit.warn(
            "Baseline hybrid tunes weights on training images already seen by the final member checkpoints (validation leakage)"
        )
    if 'get_probabilities(model_name, "val"' in tuned_hybrid and "load_tuned_model" in tuned_hybrid:
        audit.warn(
            "Tuned hybrid tunes weights on training images already seen by the final member checkpoints (validation leakage)"
        )
    if "if cache_path.exists():" in tuned_hybrid and "checkpoint" not in tuned_hybrid.split("if cache_path.exists():", 1)[1].split("return", 1)[0]:
        audit.warn("Tuned hybrid probability caches are not fingerprinted against checkpoints or datasets")

    credentials = ROOT / "kaggle.json"
    if credentials.exists():
        try:
            value = json.loads(credentials.read_text(encoding="utf-8"))
            if not value.get("username") or not value.get("key"):
                audit.fail("kaggle.json does not contain both username and key")
            else:
                audit.warn("kaggle.json contains a plaintext API credential; keep the workspace private and rotate if exposed")
        except Exception as exc:
            audit.fail(f"Invalid kaggle.json: {exc}")


def main() -> int:
    configure_console_output()
    parser = argparse.ArgumentParser(description="Validate the complete CIFAKE notebook project.")
    parser.add_argument("--deep", action="store_true", help="Open and verify all 120,000 dataset images.")
    args = parser.parse_args()

    audit = Audit()
    check_dependencies(audit)
    check_source_files(audit)
    check_dataset(audit, args.deep)
    check_result_json(audit)
    check_npz_files(audit)
    check_checkpoints(audit)
    check_media_and_tex(audit)
    check_methodology(audit)

    print()
    print(f"Validation complete: {len(audit.failures)} failure(s), {len(audit.warnings)} warning(s).")
    return 1 if audit.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
