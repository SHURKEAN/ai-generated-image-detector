from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def configure_console_output() -> None:
    """Keep Unicode notebook output from crashing Windows console runners."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def ensure_project_kernel(project_root: Path, kernel_name: str = "tdk-gpu") -> str:
    """Create a relocatable kernelspec that always uses the current Python."""
    project_root = Path(project_root).resolve()
    jupyter_root = project_root / ".jupyter"
    kernel_dir = jupyter_root / "kernels" / kernel_name
    ipython_dir = jupyter_root / "ipython"
    matplotlib_dir = jupyter_root / "matplotlib"
    torch_dir = jupyter_root / "torch"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    ipython_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    torch_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "argv": [
            str(Path(sys.executable).resolve()),
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ],
        "display_name": "TDK GPU Python",
        "language": "python",
        "metadata": {"debugger": True},
    }
    kernel_path = kernel_dir / "kernel.json"
    rendered = json.dumps(spec, indent=2) + "\n"
    if not kernel_path.exists() or kernel_path.read_text(encoding="utf-8") != rendered:
        kernel_path.write_text(rendered, encoding="utf-8")

    existing_path = os.environ.get("JUPYTER_PATH")
    path_parts = [str(jupyter_root)]
    if existing_path:
        path_parts.append(existing_path)
    os.environ["JUPYTER_PATH"] = os.pathsep.join(path_parts)
    os.environ.setdefault("JUPYTER_DATA_DIR", str(jupyter_root))
    os.environ.setdefault("IPYTHONDIR", str(ipython_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(project_root))
    os.environ.setdefault("TORCH_HOME", str(torch_dir))
    return kernel_name
