"""Print per-model detector scores for an image and controlled variants."""

from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import ImageDetector  # noqa: E402


def make_variants(image: Image.Image) -> dict[str, Image.Image]:
    """Create in-memory variants that expose resolution/compression shortcuts."""

    base = ImageOps.exif_transpose(image).convert("RGB")
    compressed_buffer = BytesIO()
    base.save(compressed_buffer, format="JPEG", quality=50)
    compressed_buffer.seek(0)
    with Image.open(compressed_buffer) as compressed_source:
        compressed = compressed_source.convert("RGB").copy()

    return {
        f"original_{base.width}x{base.height}": base.copy(),
        "resized_512x512": base.resize((512, 512), Image.Resampling.LANCZOS),
        "resized_224x224": base.resize((224, 224), Image.Resampling.LANCZOS),
        "cifake_like_32x32": base.resize((32, 32), Image.Resampling.LANCZOS),
        "jpeg_quality_50": compressed,
        "center_square_crop": ImageOps.fit(
            base,
            (min(base.size), min(base.size)),
            method=Image.Resampling.LANCZOS,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")

    with Image.open(args.image) as source:
        variants = make_variants(source)

    detector = ImageDetector(device=args.device)
    headers = [
        "variant",
        "ensemble_ai",
        "resnet50",
        "efficientnetv2s",
        "vit16",
        "xception",
        "label",
    ]
    print(",".join(headers))
    for name, variant in variants.items():
        result = detector.predict(variant)
        scores = result["individual_ai_probabilities"]
        row = [
            name,
            f"{result['ensemble_ai_probability']:.6f}",
            f"{scores['resnet50']:.6f}",
            f"{scores['efficientnetv2s']:.6f}",
            f"{scores['vit16']:.6f}",
            f"{scores['xception']:.6f}",
            result["label"],
        ]
        print(",".join(row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
