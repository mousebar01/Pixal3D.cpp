#!/usr/bin/env python3
"""Compare the native NAF implementation with the released NAF checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_naf import load_reference_naf, make_inputs, parse_native  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "weights/NAF/naf_release.pth"),
                        type=Path)
    parser.add_argument("--gguf", default=str(ROOT / "weights/NAF/naf_release-f32.gguf"),
                        type=Path)
    parser.add_argument("--naf-source", default="/app/ComfyUI/comfy/image_encoders/naf.py",
                        type=Path)
    parser.add_argument("--fixture", default=str(ROOT / "build/bin/pixal3d_naf_numeric_fixture"),
                        type=Path)
    args = parser.parse_args()
    for path in (args.checkpoint, args.gguf, args.naf_source, args.fixture):
        if not path.exists():
            raise RuntimeError("missing {}".format(path))

    NAF = load_reference_naf(args.naf_source)
    try:
        state = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(args.checkpoint), map_location="cpu")
    model = NAF(operations=torch.nn).eval()
    model.load_state_dict(state, strict=True)
    image, low = make_inputs()
    with torch.no_grad():
        expected = model(image, low, output_size=(4, 6))[0].permute(1, 2, 0).numpy()

    import subprocess
    native = subprocess.run([str(args.fixture), str(args.gguf)], check=True,
                            capture_output=True, text=True)
    actual = parse_native(native.stdout).reshape(4, 6, 8)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    diff = actual - expected
    print("naf_real PASS: max_abs={:.3g}, rel_l2={:.3g}, values={}".format(
        np.max(np.abs(diff)),
        np.linalg.norm(diff) / max(np.linalg.norm(expected), 1e-12),
        actual.size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
