#!/usr/bin/env python3
"""Compare the real DINOv3 checkpoint at a small dynamic image size.

Using 32x32 keeps the attention sequence short while exercising all 415
production tensors and the dynamic patch/RoPE path.  This is a weight-mapping
check in addition to the tiny synthetic oracle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DINOv3ViTModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_dino_vit import parse_native  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", default="build/weights/dinov3-vitl16-pretrain-lvd1689m-f32.gguf")
    parser.add_argument("--vision-dir", required=True)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--fixture", default="build/bin/pixal3d_dino_vit_numeric_fixture")
    args = parser.parse_args()

    size = args.image_size
    pixels = np.asarray(
        [0.15 * np.sin(float(i + 1)) + 0.01 * float(i)
         for i in range(3 * size * size)], dtype=np.float32
    ).reshape(1, 3, size, size)
    model = DINOv3ViTModel.from_pretrained(args.vision_dir, local_files_only=True)
    model.eval()
    x = torch.from_numpy(pixels)
    with torch.no_grad():
        hidden = model.embeddings(x)
        position_embeddings = model.rope_embeddings(x)
        for layer in model.model.layer:
            hidden = layer(hidden, position_embeddings=position_embeddings)
        hidden = F.layer_norm(hidden, hidden.shape[-1:])
    prefix = 1 + int(model.config.num_register_tokens)
    expected_global = hidden[0, :prefix].numpy().reshape(-1)
    expected_patch = hidden[0, prefix:].numpy().reshape(-1)

    import subprocess
    native = subprocess.run([args.fixture, args.gguf, str(size)], check=True,
                            capture_output=True, text=True)
    actual_global, actual_patch = parse_native(native.stdout)
    np.testing.assert_allclose(actual_global, expected_global, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_patch, expected_patch, rtol=2e-5, atol=2e-5)
    print("dino_vit_real PASS: global max_abs={:.3g}, patch max_abs={:.3g}, "
          "global rel_l2={:.3g}, patch rel_l2={:.3g}".format(
              np.max(np.abs(actual_global - expected_global)),
              np.max(np.abs(actual_patch - expected_patch)),
              np.linalg.norm(actual_global - expected_global) /
              max(np.linalg.norm(expected_global), 1e-12),
              np.linalg.norm(actual_patch - expected_patch) /
              max(np.linalg.norm(expected_patch), 1e-12)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
