from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from runtime_utils import configure_console_output


PROJECT_ROOT = Path(__file__).resolve().parents[1]
configure_console_output()
TUNED_NOTEBOOK_DIR = PROJECT_ROOT / "notebooks" / "tuned"
NOTEBOOKS = [
    TUNED_NOTEBOOK_DIR / "01_resnet50_hyperparameter_tuned.ipynb",
    TUNED_NOTEBOOK_DIR / "04_xception_hyperparameter_tuned.ipynb",
    TUNED_NOTEBOOK_DIR / "05_hybrid_ensemble_hyperparameter_tuned.ipynb",
]
EXPECTED_OUTPUTS = {
    "01_resnet50_hyperparameter_tuned.ipynb": [
        PROJECT_ROOT / "outputs" / "tuned" / "resnet50_hyperparameter_tuned_final_results.json",
        PROJECT_ROOT / "models" / "tuned" / "resnet50_hyperparameter_tuned_best.pth",
    ],
    "04_xception_hyperparameter_tuned.ipynb": [
        PROJECT_ROOT / "outputs" / "tuned" / "xception_hyperparameter_tuned_final_results.json",
        PROJECT_ROOT / "models" / "tuned" / "xception_hyperparameter_tuned_best.pth",
    ],
    "05_hybrid_ensemble_hyperparameter_tuned.ipynb": [
        PROJECT_ROOT / "outputs" / "tuned" / "hybrid" / "hybrid_hyperparameter_tuned_final_results.json",
    ],
}


def print_line(message: str = "") -> None:
    print(message, flush=True)


def import_required_modules() -> object:
    required = ["torch", "torchvision", "numpy", "sklearn", "PIL", "tqdm", "timm"]
    missing = []
    loaded = {}
    for module_name in required:
        try:
            loaded[module_name] = importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        print_line("Missing required Python packages:")
        for module_name in missing:
            print_line(f"  - {module_name}")
        print_line()
        print_line("Run this script from the same Python environment you use for GPU training.")
        print_line("For example, after activating that environment:")
        print_line("  python scripts\\run_tuned_01_04_05.py")
        raise SystemExit(1)

    return loaded["torch"]


def require_cuda() -> None:
    torch = import_required_modules()
    if not torch.cuda.is_available():
        print_line("CUDA GPU was not detected by PyTorch.")
        print_line("Stopping so the notebooks do not accidentally run on CPU.")
        print_line()
        print_line("Check that you launched this with a CUDA-enabled PyTorch environment.")
        print_line("Quick check:")
        print_line('  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"')
        raise SystemExit(1)

    print_line("CUDA GPU detected:")
    print_line(f"  {torch.cuda.get_device_name(0)}")
    print_line(f"  CUDA version reported by PyTorch: {torch.version.cuda}")


def read_notebook(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def code_cells(path: Path) -> list[tuple[int, str]]:
    notebook = read_notebook(path)
    cells = []
    for cell_number, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.strip():
            cells.append((cell_number, source))
    return cells


def cell_preview(source: str, max_lines: int = 3, max_chars: int = 110) -> str:
    useful_lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped:
            useful_lines.append(stripped)
        if len(useful_lines) >= max_lines:
            break
    preview = " | ".join(useful_lines) if useful_lines else "<empty cell>"
    if len(preview) > max_chars:
        return preview[: max_chars - 3] + "..."
    return preview


def execute_notebook(path: Path) -> None:
    require_cuda()

    os.chdir(PROJECT_ROOT)
    cells = code_cells(path)
    total = len(cells)
    if total == 0:
        raise RuntimeError(f"No code cells found in {path}")

    print_line()
    print_line("=" * 100)
    print_line(f"Executing notebook: {path.name}")
    print_line(f"Path: {path}")
    print_line(f"Code cells: {total}")
    print_line("=" * 100)

    namespace = {
        "__name__": "__main__",
        "__file__": str(path),
    }
    start_time = time.time()

    for index, (cell_number, source) in enumerate(cells, start=1):
        before_percent = ((index - 1) / total) * 100
        after_percent = (index / total) * 100
        print_line()
        print_line(
            f"[{path.name}] cell {index}/{total} "
            f"(notebook cell {cell_number}) starting - {before_percent:6.2f}% complete"
        )
        print_line(f"Code: {cell_preview(source)}")
        cell_start = time.time()

        try:
            compiled = compile(source, f"{path.name}:cell_{cell_number}", "exec")
            exec(compiled, namespace)
        except Exception:
            print_line()
            print_line(f"FAILED in {path.name}, notebook cell {cell_number}")
            print_line("-" * 100)
            traceback.print_exc()
            print_line("-" * 100)
            raise

        elapsed = time.time() - cell_start
        print_line(
            f"[{path.name}] cell {index}/{total} finished in {elapsed / 60:.1f} min "
            f"- {after_percent:6.2f}% complete"
        )

    total_elapsed = time.time() - start_time
    print_line()
    print_line(f"Finished {path.name} in {total_elapsed / 60:.1f} minutes.")

    namespace.clear()
    gc.collect()

    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def check_expected_outputs(notebook_path: Path) -> None:
    missing = []
    for output_path in EXPECTED_OUTPUTS.get(notebook_path.name, []):
        if not output_path.exists():
            missing.append(output_path)

    if missing:
        print_line()
        print_line(f"{notebook_path.name} finished, but expected output files are missing:")
        for output_path in missing:
            print_line(f"  - {output_path}")
        raise SystemExit(1)

    print_line()
    print_line(f"Verified expected outputs for {notebook_path.name}:")
    for output_path in EXPECTED_OUTPUTS.get(notebook_path.name, []):
        print_line(f"  - {output_path}")


def run_child_process(notebook_path: Path, sequence_index: int, total_notebooks: int, log_file) -> None:
    overall_before = ((sequence_index - 1) / total_notebooks) * 100
    overall_after = (sequence_index / total_notebooks) * 100

    print_line()
    print_line("#" * 100)
    print_line(
        f"Starting {sequence_index}/{total_notebooks}: {notebook_path.name} "
        f"- overall {overall_before:6.2f}% complete"
    )
    print_line("#" * 100)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--run-one",
        str(notebook_path),
    ]
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
    )

    assert process.stdout is not None
    for chunk in process.stdout:
        print(chunk, end="", flush=True)
        log_file.write(chunk)
        log_file.flush()

    return_code = process.wait()
    if return_code != 0:
        raise SystemExit(f"{notebook_path.name} failed with exit code {return_code}")

    check_expected_outputs(notebook_path)
    print_line(
        f"Completed {sequence_index}/{total_notebooks}: {notebook_path.name} "
        f"- overall {overall_after:6.2f}% complete"
    )


def run_sequence() -> None:
    require_cuda()

    missing_notebooks = [path for path in NOTEBOOKS if not path.exists()]
    if missing_notebooks:
        print_line("Missing notebooks:")
        for path in missing_notebooks:
            print_line(f"  - {path}")
        raise SystemExit(1)

    log_dir = PROJECT_ROOT / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_tuned_01_04_05_{timestamp}.log"

    print_line()
    print_line("Run order:")
    for index, notebook_path in enumerate(NOTEBOOKS, start=1):
        print_line(f"  {index}. {notebook_path.name}")
    print_line()
    print_line(f"Live log will be saved to: {log_path}")

    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Python: {sys.executable}\n")
        log_file.write(f"Project root: {PROJECT_ROOT}\n\n")
        for index, notebook_path in enumerate(NOTEBOOKS, start=1):
            run_child_process(notebook_path, index, len(NOTEBOOKS), log_file)

    elapsed = time.time() - started
    print_line()
    print_line("=" * 100)
    print_line(f"All requested tuned notebooks finished in {elapsed / 3600:.2f} hours.")
    print_line(f"Log file: {log_path}")
    print_line("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tuned notebooks 01, 04, then 05 with CUDA and terminal progress."
    )
    parser.add_argument(
        "--run-one",
        type=Path,
        help="Internal option used by the parent process to execute one notebook.",
    )
    args = parser.parse_args()

    if args.run_one:
        execute_notebook(args.run_one.resolve())
    else:
        run_sequence()


if __name__ == "__main__":
    main()
