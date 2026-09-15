# AI Image Detector

This Gradio application classifies one uploaded image as **Real** or
**AI-generated**. In `auto` mode, native CIFAKE-scale images (at most 64 pixels
on the longest side) use the published four-model detector, while ordinary
images use the newer Community Forensics cross-generator detector. The selected
route is displayed with every answer. Neither path modifies the published
checkpoints.

## Python 3.11 setup

From the project root, create and activate a Python 3.11 virtual environment,
then install the application dependencies:

```powershell
py -3.11 -m venv .venv-app
.\.venv-app\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-app.txt
```

Start the interface from the project root:

```powershell
python -m app.app
```

For an immediate run in this workspace, Gradio has also been installed into the
existing validated environment:

```powershell
.\.venv-tuned\Scripts\python.exe -m app.app
```

The first analysis request loads the selected detector. Set
`AI_DETECTOR_MODE=community` to require the CVPR 2025 model,
`AI_DETECTOR_MODE=realworld` to inspect the locally trained experimental head,
`AI_DETECTOR_MODE=universal` for the CVPR 2023 CLIP reference, or
`AI_DETECTOR_MODE=published` to demonstrate the exact paper baseline. The
default is `auto`, which uses Community Forensics for normal-resolution images
because it passed the current acceptance comparison. Merely creating a new
checkpoint never promotes it into the automatic route.
Set `AI_DETECTOR_DEVICE=cpu` before launching to force CPU inference; otherwise
CUDA is selected automatically when available.

Public Gradio sharing is disabled by default. For a temporary Colab/share link,
set `GRADIO_SHARE=true` deliberately before starting the app. Uploaded files are
limited to 20 MB and temporary upload data is periodically removed.

Run all tests, including the real-checkpoint smoke test, with:

```powershell
$env:RUN_MODEL_INTEGRATION = "1"
python -m pytest -q tests
```

The percentage shown by the application is a model-estimated score, not
forensic proof. The interface displays the selected detector version and its
scope limitation with every result.
