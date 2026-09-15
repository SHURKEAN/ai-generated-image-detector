import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from queue import Empty

import nbformat
from jupyter_client import KernelManager

from runtime_utils import configure_console_output, ensure_project_kernel


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "run_logs"
LOG_DIR.mkdir(exist_ok=True)
configure_console_output()

TUNED_NOTEBOOKS = [
    ROOT / "notebooks" / "tuned" / "02_efficientnetv2s_hyperparameter_tuned_FIXED.ipynb",
    ROOT / "notebooks" / "tuned" / "03_vit16_hyperparameter_tuned.ipynb",
]


def save_notebook(nb, notebook_path):
    with notebook_path.open("w", encoding="utf-8") as file:
        nbformat.write(nb, file)


def safe_log_name(notebook_path):
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in notebook_path.name)


def text_from_data(data):
    text = data.get("text/plain", "")
    return "".join(text) if isinstance(text, list) else text


def write_both(log, message="", end="\n"):
    print(message, end=end, flush=True)
    log.write(str(message) + end)
    log.flush()


def execute_notebook(notebook_path, kernel_name):
    notebook_path = notebook_path.resolve()
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    os.environ["TDK_PROJECT_ROOT"] = str(ROOT)
    os.environ["CIFAKE_ROOT"] = str(ROOT / "CIFAKE_FULL")
    os.environ["LOCAL_NUM_WORKERS"] = "0"
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{stamp}_tuned_live_{safe_log_name(notebook_path)}.log"

    nb = nbformat.read(notebook_path, as_version=4)
    km = KernelManager(kernel_name=kernel_name)
    kc = None

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        write_both(log, "")
        write_both(log, "#" * 72)
        write_both(log, f"RUNNING: {notebook_path}")
        write_both(log, f"Kernel: {kernel_name}")
        write_both(log, f"Project root: {ROOT}")
        write_both(log, f"CIFAKE root: {ROOT / 'CIFAKE_FULL'}")
        write_both(log, "Batch size: 16 for all tuned trials and final training")
        write_both(log, f"Log: {log_path}")
        write_both(log, "#" * 72)

        try:
            km.start_kernel(cwd=str(ROOT), env=os.environ.copy())
            kc = km.client()
            kc.start_channels()
            kc.wait_for_ready(timeout=90)

            for index, cell in enumerate(nb.cells):
                if cell.cell_type != "code":
                    continue

                first_line = cell.source.strip().splitlines()[0] if cell.source.strip() else "<empty code cell>"
                write_both(log, "")
                write_both(log, "=" * 72)
                write_both(log, f"Running cell {index}: {first_line}")
                write_both(log, "=" * 72)

                cell.outputs = []
                cell.execution_count = None
                msg_id = kc.execute(cell.source, store_history=True)
                cell_failed = False
                error_text = ""

                last_wait_message = datetime.now()
                while True:
                    try:
                        msg = kc.get_iopub_msg(timeout=1)
                    except Empty:
                        now = datetime.now()
                        if (now - last_wait_message).total_seconds() >= 60:
                            write_both(log, "Still running... waiting for notebook output.")
                            last_wait_message = now
                        continue

                    if msg.get("parent_header", {}).get("msg_id") != msg_id:
                        continue

                    msg_type = msg["header"]["msg_type"]
                    content = msg["content"]

                    if msg_type == "status" and content.get("execution_state") == "idle":
                        break

                    if msg_type == "execute_input":
                        cell.execution_count = content.get("execution_count")

                    elif msg_type == "stream":
                        text = content.get("text", "")
                        cell.outputs.append(
                            nbformat.v4.new_output(
                                output_type="stream",
                                name=content.get("name", "stdout"),
                                text=text,
                            )
                        )
                        write_both(log, text, end="")

                    elif msg_type in {"display_data", "execute_result"}:
                        output = nbformat.v4.new_output(
                            output_type=msg_type,
                            data=content.get("data", {}),
                            metadata=content.get("metadata", {}),
                        )
                        if msg_type == "execute_result":
                            output["execution_count"] = content.get("execution_count")
                        cell.outputs.append(output)
                        text = text_from_data(content.get("data", {}))
                        if text:
                            write_both(log, text)

                    elif msg_type == "error":
                        cell_failed = True
                        traceback_lines = content.get("traceback", [])
                        error_text = "\n".join(traceback_lines)
                        cell.outputs.append(
                            nbformat.v4.new_output(
                                output_type="error",
                                ename=content.get("ename", ""),
                                evalue=content.get("evalue", ""),
                                traceback=traceback_lines,
                            )
                        )
                        write_both(log, error_text)

                    elif msg_type == "clear_output":
                        cell.outputs = []

                save_notebook(nb, notebook_path)
                write_both(log, f"\nSaved after cell {index}.")

                reply = kc.get_shell_msg(timeout=30)
                if reply.get("parent_header", {}).get("msg_id") == msg_id:
                    status = reply.get("content", {}).get("status")
                    if status == "error":
                        cell_failed = True

                if cell_failed:
                    raise RuntimeError(f"Notebook failed in cell {index}.\n{error_text}")

        except Exception:
            save_notebook(nb, notebook_path)
            write_both(log, "")
            write_both(log, "Notebook failed. Progress was saved before exiting.")
            error = traceback.format_exc()
            write_both(log, error)
            raise
        finally:
            if kc is not None:
                kc.stop_channels()
            if km.has_kernel:
                km.shutdown_kernel(now=True)

    print()
    print(f"Notebook completed: {notebook_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run both tuned model notebooks with live terminal output.")
    parser.add_argument(
        "--kernel",
        default=os.getenv("TDK_KERNEL", "tdk-gpu"),
        help="Jupyter kernel name (default: project-local tdk-gpu)",
    )
    parser.add_argument(
        "--only",
        choices=["02", "03", "all"],
        default="all",
        help="Run only one tuned notebook or both.",
    )
    args = parser.parse_args()

    if args.kernel == "tdk-gpu":
        ensure_project_kernel(ROOT, args.kernel)

    if args.only == "02":
        notebooks = [TUNED_NOTEBOOKS[0]]
    elif args.only == "03":
        notebooks = [TUNED_NOTEBOOKS[1]]
    else:
        notebooks = TUNED_NOTEBOOKS

    print("Python:", sys.executable, flush=True)
    print("Running tuned notebooks in order:", flush=True)
    for notebook in notebooks:
        print(" -", notebook, flush=True)

    for notebook in notebooks:
        execute_notebook(notebook, args.kernel)

    print()
    print("All selected tuned notebooks completed.", flush=True)


if __name__ == "__main__":
    main()
