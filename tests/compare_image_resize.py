#!/usr/bin/env python3
"""Compare the native Lanczos image resize against Pillow's LANCZOS path."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    fixture = ROOT / "build" / "bin" / "pixal3d_image_resize_numeric_fixture"
    if not fixture.is_file():
        raise RuntimeError(f"missing {fixture}; build the project first")
    with tempfile.TemporaryDirectory(prefix="pixal3d-image-resize-") as tmp:
        path = Path(tmp) / "input.ppm"
        height, width = 5, 7
        rgb = np.asarray(
            [((x * 31 + y * 17 + c * 53) % 256)
             for y in range(height) for x in range(width) for c in range(3)],
            dtype=np.uint8,
        ).reshape(height, width, 3)
        with path.open("wb") as file:
            file.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            file.write(rgb.tobytes())
        size = 11
        native = subprocess.run(
            [str(fixture), str(path), str(size)],
            check=True, capture_output=True, text=True,
        ).stdout.split()
        actual = np.asarray([float(value) for value in native[3:]], dtype=np.float32)
        expected_hwc = np.asarray(
            Image.open(path).resize((size, size), Image.Resampling.LANCZOS),
            dtype=np.float32,
        ) / 255.0
        expected = np.transpose(expected_hwc, (2, 0, 1)).reshape(-1)
        diff = np.abs(actual - expected)
        maximum = float(diff.max())
        # Pillow writes an 8-bit image after filtering; the native boundary
        # keeps F32, so one quantization step plus floating-point noise is the
        # expected difference.
        if maximum > 0.065:
            raise AssertionError(f"Lanczos mismatch max_abs={maximum:.6g}")
        print(f"image_resize PASS: max_abs={maximum:.6g}, values={actual.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
