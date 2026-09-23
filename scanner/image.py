"""Stable image decoding and two views for catalog and shelf photographs."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageChops, ImageOps


SIDE = 384
PREPROCESSING_VERSION = "center-views-v1"


def decode_image(source: bytes | Path | str | Image.Image) -> Image.Image:
    """Decode by file contents, honor EXIF, and composite transparent product photos."""
    if isinstance(source, bytes):
        image = Image.open(BytesIO(source))
    elif isinstance(source, (Path, str)):
        image = Image.open(source)
    else:
        image = source.copy()
    try:
        image.load()
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA", "P"}:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            background.alpha_composite(rgba)
            return background.convert("RGB")
        return image.convert("RGB")
    finally:
        image.close()


def _reference_bounds(image: Image.Image) -> Image.Image:
    """Trim empty white borders without guessing where the label is."""
    difference = ImageChops.difference(image, Image.new("RGB", image.size, "white"))
    mask = difference.convert("L").point(lambda value: 255 if value > 18 else 0)
    bounds = mask.getbbox()
    if not bounds:
        return image
    left, top, right, bottom = bounds
    if (right - left) * (bottom - top) < 0.08 * image.width * image.height:
        return image
    return image.crop(bounds)


def _crop_fraction(image: Image.Image, left: float, top: float,
                   right: float, bottom: float) -> Image.Image:
    width, height = image.size
    return image.crop((round(left * width), round(top * height),
                       round(right * width), round(bottom * height)))


def _square(image: Image.Image) -> Image.Image:
    return ImageOps.pad(image, (SIDE, SIDE), method=Image.Resampling.BICUBIC,
                        color="white", centering=(0.5, 0.5))


def reference_views(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    image = _reference_bounds(image)
    full = _square(image)
    label = _square(_crop_fraction(image, 0.06, 0.28, 0.94, 0.84))
    return full, label


def query_views(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Favor the central bottle while retaining a full-scene fallback."""
    full = _square(image)
    center = _square(_crop_fraction(image, 0.22, 0.16, 0.78, 0.88))
    return full, center
