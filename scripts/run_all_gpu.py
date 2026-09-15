import argparse
import importlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from runtime_utils import configure_console_output, ensure_project_kernel


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "run_logs"
LOG_DIR.mkdir(exist_ok=True)
configure_console_output()

NOTEBOOKS = [
    Path("notebooks") / "baseline" / "01_resnet50_baseline.ipynb",
    Path("notebooks") / "baseline" / "02_efficientnetv2s_baseline(1)_(1).ipynb",
    Path("notebooks") / "baseline" / "03_vit16.ipynb",
    Path("notebooks") / "baseline" / "04_xception.ipynb",
    Path("notebooks") / "baseline" / "05_hybrid_ensemble.ipynb",
]


def run(command, log_path=None):
    if log_path is None:
        return subprocess.run(command, check=True)

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=ROOT,
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main():
    parser = argparse.ArgumentParser(description="Run all baseline notebooks with CUDA.")
    parser.add_argument("--skip-hybrid", action="store_true", help="Skip notebook 05.")
    parser.add_argument(
        "--kernel",
        default=os.getenv("TDK_KERNEL", "tdk-gpu"),
        help="Jupyter kernel name (default: project-local tdk-gpu)",
    )
    args = parser.parse_args()

    os.environ["TDK_PROJECT_ROOT"] = str(ROOT)
    os.environ["CIFAKE_ROOT"] = str(ROOT / "CIFAKE_FULL")
    os.environ["LOCAL_NUM_WORKERS"] = "0"

    print("Python:", sys.executable)
    print("Workspace:", ROOT)
    print("Dataset:", os.environ["CIFAKE_ROOT"])

    run([
        sys.executable,
        "-c",
        (
            "import torch, sys; "
            "print('torch:', torch.__version__); "
            "print('cuda:', torch.cuda.is_available()); "
            "print('cuda build:', torch.version.cuda); "
            "print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); "
            "sys.exit(0 if torch.cuda.is_available() else 10)"
        ),
    ])

    required = ["nbconvert", "nbclient", "nbformat", "ipykernel"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Missing notebook packages: "
            + ", ".join(missing)
            + ". Install them in this Python environment before running notebooks."
        )

    kernel_name = args.kernel
    if kernel_name == "tdk-gpu":
        kernel_name = ensure_project_kernel(ROOT, kernel_name)

    notebooks = NOTEBOOKS[:-1] if args.skip_hybrid else NOTEBOOKS

    for notebook in notebooks:
        notebook_path = ROOT / notebook
        if not notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in notebook_path.name)
        log_path = LOG_DIR / f"{stamp}_{safe_name}.log"

        print()
        print("=" * 60)
        print("Starting:", notebook_path.relative_to(ROOT))
        print("Log:", log_path)
        print("=" * 60)

        run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=-1",
                f"--ExecutePreprocessor.kernel_name={kernel_name}",
                str(notebook_path),
            ],
            log_path=log_path,
        )

        print("Finished:", notebook_path.relative_to(ROOT))

    print()
    print("All requested notebooks completed.")


if __name__ == "__main__":
    main()
