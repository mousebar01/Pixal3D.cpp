#!/usr/bin/env python3
"""Compare the native DINO+NAF stage bridge with released weights.

The 32x32 input keeps the real DINO sequence short while still exercising the
production 1024-channel checkpoint and the released NAF weights together.
"""

from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DINOv3ViTModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_naf import load_reference_naf  # noqa: E402
from compare_vision_condition import parse_native  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-dir", required=True, type=Path)
    parser.add_argument("--dino-gguf", default=str(ROOT / "build/weights/dinov3-vitl16-pretrain-lvd1689m-f32.gguf"), type=Path)
    parser.add_argument("--naf-gguf", default=str(ROOT / "weights/NAF/naf_release-f32.gguf"), type=Path)
    parser.add_argument("--naf-checkpoint", default=str(ROOT / "weights/NAF/naf_release.pth"), type=Path)
    parser.add_argument("--naf-source", default="/app/ComfyUI/comfy/image_encoders/naf.py", type=Path)
    parser.add_argument("--fixture", default=str(ROOT / "build/bin/pixal3d_vision_encoder_fixture"), type=Path)
    parser.add_argument("--image-size", type=int, default=32)
    args = parser.parse_args()
    for path in (args.vision_dir, args.dino_gguf, args.naf_gguf,
                 args.naf_checkpoint, args.naf_source, args.fixture):
        if not path.exists():
            raise RuntimeError(f"missing {path}")
    size = args.image_size
    raw = np.asarray(
        [0.1 + 0.8 * ((i * 7) % 23) / 22.0 for i in range(3 * size * size)],
        dtype=np.float32,
    ).reshape(3, size, size)

    dino_model = DINOv3ViTModel.from_pretrained(args.vision_dir, local_files_only=True)
    dino_model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    with torch.no_grad():
        x = (torch.from_numpy(raw).unsqueeze(0) - mean) / std
        hidden = dino_model.embeddings(x)
        position_embeddings = dino_model.rope_embeddings(x)
        for layer in dino_model.model.layer:
            hidden = layer(hidden, position_embeddings=position_embeddings)
        hidden = F.layer_norm(hidden, hidden.shape[-1:])
    prefix = 1 + int(dino_model.config.num_register_tokens)
    expected_global = hidden[0, :prefix].numpy().reshape(-1)
    patch_count = size // int(dino_model.config.patch_size)
    expected_dino = hidden[0, prefix:].numpy().reshape(patch_count, patch_count, -1)

    naf_reference = load_reference_naf(args.naf_source)
    try:
        state = torch.load(str(args.naf_checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(args.naf_checkpoint), map_location="cpu")
    naf_model = naf_reference(operations=torch.nn).eval()
    naf_model.load_state_dict(state, strict=True)
    with torch.no_grad():
        low = torch.from_numpy(expected_dino).permute(2, 0, 1).unsqueeze(0)
        expected_naf = naf_model(torch.from_numpy(raw).unsqueeze(0), low,
                                 output_size=(size, size))[0].permute(1, 2, 0).numpy()

    native = subprocess.run(
        [str(args.fixture), str(args.dino_gguf), str(args.naf_gguf), str(size)],
        check=True, capture_output=True, text=True,
    )
    actual_global, actual_dino, actual_naf = parse_native(native.stdout)
    expected_global = expected_global.astype(np.float32, copy=False)
    expected_dino = expected_dino.reshape(-1).astype(np.float32, copy=False)
    expected_naf = expected_naf.reshape(-1).astype(np.float32, copy=False)
    np.testing.assert_allclose(actual_global, expected_global, rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(actual_dino, expected_dino, rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(actual_naf, expected_naf, rtol=3e-4, atol=3e-4)
    diff = np.concatenate([
        actual_global - expected_global,
        actual_dino - expected_dino,
        actual_naf - expected_naf,
    ])
    expected = np.concatenate([expected_global, expected_dino, expected_naf])
    print("vision_condition_real PASS: max_abs={:.3g}, rel_l2={:.3g}, values={}".format(
        np.max(np.abs(diff)),
        np.linalg.norm(diff) / max(np.linalg.norm(expected), 1e-12),
        diff.size,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
