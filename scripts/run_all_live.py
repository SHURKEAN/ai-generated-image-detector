import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

ALL_NOTEBOOKS = [
    Path("notebooks") / "baseline" / "01_resnet50_baseline.ipynb",
    Path("notebooks") / "baseline" / "02_efficientnetv2s_baseline(1)_(1).ipynb",
    Path("notebooks") / "baseline" / "03_vit16.ipynb",
    Path("notebooks") / "baseline" / "04_xception.ipynb",
    Path("notebooks") / "baseline" / "05_hybrid_ensemble.ipynb",
]


def main():
    parser = argparse.ArgumentParser(description="Run project notebooks one at a time with live notebook saves.")
    parser.add_argument(
        "--start-at",
        default="01",
        choices=["01", "02", "03", "04", "05"],
        help="First notebook number to run.",
    )
    parser.add_argument(
        "--kernel",
        default=os.getenv("TDK_KERNEL", "tdk-gpu"),
        help="Jupyter kernel name (default: project-local tdk-gpu)",
    )
    args = parser.parse_args()

    start_index = next(index for index, name in enumerate(ALL_NOTEBOOKS) if name.name.startswith(args.start_at))
    notebooks = ALL_NOTEBOOKS[start_index:]

    for notebook in notebooks:
        print()
        print("#" * 72)
        print("RUNNING:", notebook)
        print("#" * 72)

        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "run_notebook_live.py"),
                str(notebook),
                "--kernel",
                args.kernel,
            ],
            cwd=ROOT,
            check=True,
        )

    print()
    print("All selected notebooks completed.")


if __name__ == "__main__":
    main()
