"""Image preprocessing shared by every detector model.

The tuned checkpoints were trained with 224 x 224 RGB inputs normalized with
ImageNet statistics.  Keeping this pipeline in one module prevents subtle
differences between models at inference time.
"""

from __future__ import annotations

from PIL import Image, ImageOps
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            IMAGE_SIZE,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def preprocess_image(image: Image.Image) -> Tensor:
    """Return a normalized ``(3, 224, 224)`` tensor for one PIL image.

    EXIF orientation is applied before resizing so photographs taken on phones
    are analyzed in the same orientation in which users see them.  All modes,
    including grayscale and RGBA, are converted to RGB.
    """

    if not isinstance(image, Image.Image):
        raise TypeError(
            "preprocess_image expects a PIL.Image.Image instance; "
            f"received {type(image).__name__}."
        )
    if image.width < 1 or image.height < 1:
        raise ValueError("The image must have a non-zero width and height.")

    try:
        oriented_image = ImageOps.exif_transpose(image)
        rgb_image = oriented_image.convert("RGB")
        return INFERENCE_TRANSFORM(rgb_image)
    except (OSError, SyntaxError, ValueError) as exc:
        raise ValueError("The uploaded image could not be decoded or processed.") from exc


__all__ = [
    "IMAGE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "INFERENCE_TRANSFORM",
    "preprocess_image",
]
