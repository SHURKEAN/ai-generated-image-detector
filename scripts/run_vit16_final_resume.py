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
NOTEBOOK_PATH = ROOT / "notebooks" / "tuned" / "03_vit16_hyperparameter_tuned.ipynb"
LOG_DIR = ROOT / "run_logs"
LOG_DIR.mkdir(exist_ok=True)
configure_console_output()


def write_both(log, message="", end="\n"):
    print(message, end=end, flush=True)
    log.write(str(message) + end)
    log.flush()


def text_from_data(data):
    text = data.get("text/plain", "")
    return "".join(text) if isinstance(text, list) else text


def execute_source(kc, nb, log, source, label):
    write_both(log, "")
    write_both(log, "=" * 72)
    write_both(log, f"Running: {label}")
    write_both(log, "=" * 72)

    msg_id = kc.execute(source, store_history=True)
    last_wait_message = datetime.now()
    cell_failed = False
    error_text = ""

    while True:
        try:
            msg = kc.get_iopub_msg(timeout=1)
        except Empty:
            now = datetime.now()
            if (now - last_wait_message).total_seconds() >= 60:
                write_both(log, "Still running... waiting for output.")
                last_wait_message = now
            continue

        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue

        msg_type = msg["header"]["msg_type"]
        content = msg["content"]

        if msg_type == "status" and content.get("execution_state") == "idle":
            break

        if msg_type == "stream":
            write_both(log, content.get("text", ""), end="")
        elif msg_type in {"display_data", "execute_result"}:
            text = text_from_data(content.get("data", {}))
            if text:
                write_both(log, text)
        elif msg_type == "error":
            cell_failed = True
            error_text = "\n".join(content.get("traceback", []))
            write_both(log, error_text)

    reply = kc.get_shell_msg(timeout=30)
    if reply.get("parent_header", {}).get("msg_id") == msg_id:
        if reply.get("content", {}).get("status") == "error":
            cell_failed = True

    if cell_failed:
        raise RuntimeError(f"Execution failed in {label}.\n{error_text}")


def main():
    parser = argparse.ArgumentParser(description="Resume only the final ViT-B/16 training stage.")
    parser.add_argument(
        "--kernel",
        default=os.getenv("TDK_KERNEL", "tdk-gpu"),
        help="Jupyter kernel name (default: project-local tdk-gpu)",
    )
    args = parser.parse_args()

    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_PATH}")

    os.environ["TDK_PROJECT_ROOT"] = str(ROOT)
    os.environ["CIFAKE_ROOT"] = str(ROOT / "CIFAKE_FULL")
    os.environ["LOCAL_NUM_WORKERS"] = "0"
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{stamp}_resume_vit16_final.log"

    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code_cells = [(index, cell.source) for index, cell in enumerate(nb.cells) if cell.cell_type == "code"]
    if not code_cells:
        raise RuntimeError(f"No code cells found in {NOTEBOOK_PATH}")
    setup_cells = code_cells[:-1]
    final_cell_index, final_cell_source = code_cells[-1]
    tuning_call = "\nbest_trial, all_trials = run_hyperparameter_tuning()\nfinal_results = train_final_model(best_trial[\"hyperparameters\"])"
    if tuning_call not in final_cell_source:
        raise RuntimeError("Could not find the final tuning call to replace.")
    final_definition_source = final_cell_source.replace(tuning_call, "\n")

    resume_source = """
with open(BEST_CONFIG_PATH, "r", encoding="utf-8") as f:
    best_trial = json.load(f)
print("Loaded saved best ViT16 hyperparameters from:", BEST_CONFIG_PATH)
print(json.dumps(best_trial["hyperparameters"], indent=2))
final_results = train_final_model(best_trial["hyperparameters"])
"""

    kernel_name = args.kernel
    if kernel_name == "tdk-gpu":
        kernel_name = ensure_project_kernel(ROOT, kernel_name)
    km = KernelManager(kernel_name=kernel_name)
    kc = None

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        write_both(log, "#" * 72)
        write_both(log, "RESUMING VIT16 FINAL TRAINING ONLY")
        write_both(log, f"Notebook: {NOTEBOOK_PATH}")
        write_both(log, f"Project root: {ROOT}")
        write_both(log, f"Log: {log_path}")
        write_both(log, "#" * 72)

        try:
            km.start_kernel(cwd=str(ROOT), env=os.environ.copy())
            kc = km.client()
            kc.start_channels()
            kc.wait_for_ready(timeout=90)

            for index, source in setup_cells:
                first_line = source.strip().splitlines()[0] if source.strip() else "<empty>"
                execute_source(kc, nb, log, source, f"setup cell {index}: {first_line}")

            execute_source(kc, nb, log, final_definition_source, f"final definition cell {final_cell_index}")
            execute_source(kc, nb, log, resume_source, "resume final ViT16 training/evaluation")

        except Exception:
            write_both(log, "")
            write_both(log, "Resume run failed.")
            write_both(log, traceback.format_exc())
            raise
        finally:
            if kc is not None:
                kc.stop_channels()
            if km.has_kernel:
                km.shutdown_kernel(now=True)

    print(f"Resume run completed. Log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
