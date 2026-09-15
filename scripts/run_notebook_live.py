import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from runtime_utils import configure_console_output, ensure_project_kernel


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "run_logs"
LOG_DIR.mkdir(exist_ok=True)
configure_console_output()


def output_text(output):
    output_type = output.get("output_type")
    if output_type == "stream":
        return output.get("text", "")
    if output_type in {"display_data", "execute_result"}:
        data = output.get("data", {})
        text = data.get("text/plain", "")
        return "".join(text) if isinstance(text, list) else text
    if output_type == "error":
        traceback_lines = output.get("traceback", [])
        return "\n".join(traceback_lines) + "\n"
    return ""


def save_notebook(nb, notebook_path):
    with notebook_path.open("w", encoding="utf-8") as file:
        nbformat.write(nb, file)


def main():
    parser = argparse.ArgumentParser(description="Run a notebook cell-by-cell and save progress after each cell.")
    parser.add_argument("notebook", help="Notebook file to execute, for example notebooks/baseline/01_resnet50_baseline.ipynb")
    parser.add_argument(
        "--kernel",
        default=os.getenv("TDK_KERNEL", "tdk-gpu"),
        help="Jupyter kernel name (default: project-local tdk-gpu)",
    )
    args = parser.parse_args()

    notebook_arg = Path(args.notebook)
    notebook_path = notebook_arg if notebook_arg.is_absolute() else ROOT / notebook_arg
    notebook_path = notebook_path.resolve()
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    os.environ["TDK_PROJECT_ROOT"] = str(ROOT)
    os.environ["CIFAKE_ROOT"] = str(ROOT / "CIFAKE_FULL")
    os.environ["LOCAL_NUM_WORKERS"] = "0"

    if args.kernel == "tdk-gpu":
        ensure_project_kernel(ROOT, args.kernel)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in notebook_path.name)
    log_path = LOG_DIR / f"{stamp}_live_{safe_name}.log"

    print("Python:", sys.executable)
    print("Notebook:", notebook_path)
    print("Kernel:", args.kernel)
    print("Log:", log_path)

    nb = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(nb, timeout=None, kernel_name=args.kernel)

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        def write(message=""):
            print(message)
            log.write(message + "\n")
            log.flush()

        try:
            with client.setup_kernel():
                for index, cell in enumerate(nb.cells):
                    if cell.cell_type != "code":
                        continue

                    first_line = cell.source.strip().splitlines()[0] if cell.source.strip() else "<empty code cell>"
                    write("")
                    write("=" * 60)
                    write(f"Running cell {index}: {first_line}")
                    write("=" * 60)

                    cell.outputs = []
                    client.execute_cell(cell, index, store_history=True)
                    save_notebook(nb, notebook_path)

                    for output in cell.get("outputs", []):
                        text = output_text(output).rstrip()
                        if text:
                            write(text)

                    write(f"Saved after cell {index}.")

        except Exception:
            save_notebook(nb, notebook_path)
            write("")
            write("Notebook failed. Saved progress before exiting.")
            error_text = traceback.format_exc()
            print(error_text)
            log.write(error_text)
            raise

    print()
    print("Notebook completed.")


if __name__ == "__main__":
    main()
