#!/usr/bin/env python3
"""Prepare an RGB/RGBA image for the native Pixal3D condition encoder.

This mirrors ``Pixal3DImageTo3DPipeline.preprocess_image`` from the reference
implementation: resize to at most 1024px, remove the background when the input
does not contain a meaningful alpha channel, crop to a square around pixels
with alpha > 0.8, add a 10% margin, and composite onto black.  The C++ image
loader intentionally has no hidden matting dependency, so this is an explicit
boundary step rather than a silent approximation.

The default rembg model is BiRefNet when supported by the installed rembg
version.  ``u2net`` is a smaller fallback for environments where BiRefNet has
not been downloaded yet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _parse_rgb(value: str) -> tuple[int, int, int]:
    try:
        channels = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("background must be R,G,B") from exc
    if len(channels) != 3 or any(channel < 0 or channel > 255 for channel in channels):
        raise argparse.ArgumentTypeError("background must be R,G,B in [0,255]")
    return channels  # type: ignore[return-value]


def _matte(image: Image.Image, model: str) -> Image.Image:
    try:
        from rembg import new_session, remove
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "background removal requires rembg; install it in the Python "
            "environment or pass --no-matting for an image with alpha"
        ) from exc
    try:
        session = new_session(model)
    except Exception as exc:  # pragma: no cover - model/download dependent
        raise RuntimeError(
            f"cannot initialize rembg model {model!r}; try --model u2net"
        ) from exc
    result = remove(image.convert("RGB"), session=session)
    if not isinstance(result, Image.Image):
        result = Image.fromarray(np.asarray(result))
    return result.convert("RGBA")


def preprocess(input_path: Path, output_path: Path, model: str,
               no_matting: bool, background: tuple[int, int, int]) -> None:
    image = Image.open(input_path)
    max_size = max(image.size)
    scale = min(1.0, 1024.0 / max_size)
    if scale < 1.0:
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )

    has_alpha = image.mode == "RGBA" and not np.all(
        np.asarray(image.getchannel("A")) == 255
    )
    if has_alpha:
        rgba = image
    elif no_matting:
        raise RuntimeError(
            "--no-matting requires an RGBA input with a non-opaque alpha channel"
        )
    else:
        rgba = _matte(image, model)

    rgba_np = np.asarray(rgba)
    alpha = rgba_np[:, :, 3]
    active = np.argwhere(alpha > 0.8 * 255)
    if active.size == 0:
        raise RuntimeError("matting produced an empty foreground")
    y0, x0 = active.min(axis=0)
    y1, x1 = active.max(axis=0)
    center_x = (float(x0) + float(x1)) / 2.0
    center_y = (float(y0) + float(y1)) / 2.0
    size = max(float(x1 - x0), float(y1 - y0)) * 1.1
    half = size / 2.0
    crop_box = (
        int(center_x - half), int(center_y - half),
        int(center_x + half), int(center_y + half),
    )
    cropped = rgba.crop(crop_box)
    cropped_np = np.asarray(cropped).astype(np.float32) / 255.0
    rgb = cropped_np[:, :, :3]
    a = cropped_np[:, :, 3:4]
    bg = np.asarray(background, dtype=np.float32).reshape(1, 1, 3) / 255.0
    composited = np.clip(rgb * a + bg * (1.0 - a), 0.0, 1.0)
    output = Image.fromarray((composited * 255.0).round().astype(np.uint8), "RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)
    print(
        f"input={input_path} output={output_path} size={output.size} "
        f"bbox=({int(x0)},{int(y0)},{int(x1)},{int(y1)}) model={model}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="birefnet-general",
                        help="rembg model (default: birefnet-general; fallback: u2net)")
    parser.add_argument("--no-matting", action="store_true",
                        help="only accept an input with a non-opaque alpha channel")
    parser.add_argument("--background", type=_parse_rgb, default=(0, 0, 0),
                        help="composite background as R,G,B (default: 0,0,0)")
    args = parser.parse_args()
    preprocess(args.input, args.output, args.model, args.no_matting, args.background)


if __name__ == "__main__":
    main()
