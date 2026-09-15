"""Gradio interface for the trained AI-image detector."""

from __future__ import annotations

import math
import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image


LOGGER = logging.getLogger(__name__)

if __package__:
    from .detector_factory import get_detector
else:  # Allow `python app/app.py` as well as `python -m app.app`.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.detector_factory import get_detector


def _number(value: Any, field_name: str) -> float:
    """Convert an inference value to a finite float with a useful error."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is not numeric.") from exc

    if not math.isfinite(number):
        raise ValueError(f"{field_name} is not finite.")
    return number


def _probability_text(value: Any, field_name: str) -> str:
    probability = _number(value, field_name)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return f"{probability * 100:.2f}%"


def analyze_image(
    image: Image.Image | None,
) -> tuple[str, str, str, str, str, list[list[str]]]:
    """Run the detector and format its result for the Gradio components."""
    if image is None:
        raise gr.Error("Please upload an image before selecting Analyze.")

    try:
        # get_detector() owns the singleton, so the models are loaded only on
        # the first analysis request and reused for later requests.
        result = get_detector().predict(image)
        if not isinstance(result, Mapping):
            raise ValueError("The detector returned an invalid result.")

        label = str(result["label"])
        confidence = _number(result["confidence_percent"], "Confidence")
        if not 0.0 <= confidence <= 100.0:
            raise ValueError("Confidence must be between 0 and 100.")

        ensemble_score = _probability_text(
            result["ensemble_ai_probability"], "Ensemble AI probability"
        )
        individual = result["individual_ai_probabilities"]
        if not isinstance(individual, Mapping):
            raise ValueError("Individual model scores are invalid.")

        score_rows = [
            [str(model_name), _probability_text(score, f"{model_name} AI probability")]
            for model_name, score in individual.items()
        ]
        detector_version = str(result.get("detector_version", "unknown"))
        reliability_note = str(result.get("reliability_note", ""))
        return (
            label,
            f"{confidence:.2f}%",
            ensemble_score,
            detector_version,
            reliability_note,
            score_rows,
        )
    except gr.Error:
        raise
    except Exception as exc:
        LOGGER.exception("Image analysis failed")
        raise gr.Error(
            "Analysis failed. Check that the image is valid and that all four "
            "model checkpoints are available; see the server log for details."
        ) from exc


def _clear() -> tuple[None, str, str, str, str, str, list[list[str]]]:
    return None, "", "", "", "", "", []


def _clear_results() -> tuple[str, str, str, str, str, list[list[str]]]:
    """Remove scores whenever the selected image changes."""

    return "", "", "", "", "", []


def _set_busy() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Prevent an image change from being paired with an in-flight result."""

    disabled = gr.update(interactive=False)
    return disabled, disabled, disabled


def _set_ready() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    enabled = gr.update(interactive=True)
    return enabled, enabled, enabled


def build_app() -> gr.Blocks:
    """Create the Gradio application without loading any model checkpoints."""
    with gr.Blocks(
        title="AI Image Detector",
        # Uploaded files are temporary and should not remain on disk forever.
        delete_cache=(3600, 3600),
    ) as demo:
        gr.Markdown(
            "# AI Image Detector\n"
            "Upload one image to estimate whether it is real or AI-generated. "
            "Confidence is the model's estimate, not forensic certainty."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="Image",
                    sources=["upload"],
                )
                with gr.Row():
                    analyze_button = gr.Button("Analyze", variant="primary")
                    clear_button = gr.Button("Clear")

            with gr.Column():
                prediction_output = gr.Textbox(
                    label="Prediction (Real / AI-generated)", interactive=False
                )
                confidence_output = gr.Textbox(
                    label="Model-estimated confidence", interactive=False
                )
                ensemble_output = gr.Textbox(
                    label="AI probability", interactive=False
                )
                version_output = gr.Textbox(
                    label="Detector version", interactive=False
                )
                reliability_output = gr.Textbox(
                    label="Interpretation", interactive=False
                )
                individual_output = gr.Dataframe(
                    headers=["Model", "AI probability"],
                    datatype=["str", "str"],
                    label="Individual model scores",
                    interactive=False,
                    value=[],
                )

        image_input.change(
            fn=_clear_results,
            inputs=None,
            outputs=[
                prediction_output,
                confidence_output,
                ensemble_output,
                version_output,
                reliability_output,
                individual_output,
            ],
            queue=False,
        )

        busy_event = analyze_button.click(
            fn=_set_busy,
            inputs=None,
            outputs=[image_input, analyze_button, clear_button],
            queue=False,
        )
        analysis_event = busy_event.then(
            fn=analyze_image,
            inputs=image_input,
            outputs=[
                prediction_output,
                confidence_output,
                ensemble_output,
                version_output,
                reliability_output,
                individual_output,
            ],
        )
        analysis_event.then(
            fn=_set_ready,
            inputs=None,
            outputs=[image_input, analyze_button, clear_button],
            queue=False,
        )
        clear_button.click(
            fn=_clear,
            inputs=None,
            outputs=[
                image_input,
                prediction_output,
                confidence_output,
                ensemble_output,
                version_output,
                reliability_output,
                individual_output,
            ],
            queue=False,
        )

    return demo


demo = build_app()


if __name__ == "__main__":
    share_enabled = os.getenv("GRADIO_SHARE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    demo.launch(
        share=share_enabled,
        max_file_size="20mb",
        run_history=False,
    )
