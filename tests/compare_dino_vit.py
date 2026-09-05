#!/usr/bin/env python3
"""Numerically compare the native DINOv3 ggml graph with Transformers.

The fixture uses the production tensor inventory (24 blocks, 415 tensors) but
small dimensions, so it runs quickly and exercises every serialized weight,
the prefix-only RoPE layout, GELU-erf, and the affine-free output LayerNorm.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel


ROOT = Path(__file__).resolve().parents[1]


def make_fixture(directory: Path) -> tuple[Path, dict, dict[str, torch.Tensor]]:
    config = {
        "model_type": "dinov3_vit",
        "architectures": ["DINOv3ViTModel"],
        "image_size": 4,
        "patch_size": 2,
        "num_channels": 3,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 24,
        "num_attention_heads": 2,
        "hidden_act": "gelu",
        "attention_dropout": 0.0,
        "layer_norm_eps": 1e-5,
        "layerscale_value": 1.0,
        "drop_path_rate": 0.0,
        "use_gated_mlp": False,
        "rope_theta": 100.0,
        "query_bias": True,
        "key_bias": False,
        "value_bias": True,
        "proj_bias": True,
        "mlp_bias": True,
        "num_register_tokens": 1,
    }
    generator = torch.Generator().manual_seed(20260904)

    def rand(*shape: int, scale: float = 0.1) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * scale

    h, i = config["hidden_size"], config["intermediate_size"]
    state: dict[str, torch.Tensor] = {
        "embeddings.cls_token": rand(1, 1, h),
        "embeddings.mask_token": torch.zeros(1, 1, h),
        "embeddings.register_tokens": rand(1, 1, h),
        "embeddings.patch_embeddings.weight": rand(h, 3, 2, 2),
        "embeddings.patch_embeddings.bias": rand(h),
        "norm.weight": rand(h),
        "norm.bias": rand(h),
    }
    for layer in range(config["num_hidden_layers"]):
        p = f"layer.{layer}"
        state.update({
            f"{p}.attention.k_proj.weight": rand(h, h),
            f"{p}.attention.o_proj.weight": rand(h, h),
            f"{p}.attention.o_proj.bias": rand(h),
            f"{p}.attention.q_proj.weight": rand(h, h),
            f"{p}.attention.q_proj.bias": rand(h),
            f"{p}.attention.v_proj.weight": rand(h, h),
            f"{p}.attention.v_proj.bias": rand(h),
            f"{p}.layer_scale1.lambda1": rand(h, scale=0.8),
            f"{p}.layer_scale2.lambda1": rand(h, scale=0.8),
            f"{p}.mlp.down_proj.weight": rand(h, i),
            f"{p}.mlp.down_proj.bias": rand(h),
            f"{p}.mlp.up_proj.weight": rand(i, h),
            f"{p}.mlp.up_proj.bias": rand(i),
            f"{p}.norm1.weight": rand(h, scale=0.8),
            f"{p}.norm1.bias": rand(h),
            f"{p}.norm2.weight": rand(h, scale=0.8),
            f"{p}.norm2.bias": rand(h),
        })
    source = directory / "tiny-dino.safetensors"
    config_path = directory / "tiny-dino.json"
    save_file(state, str(source))
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return source, config, state


def reference(config: dict, state: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hf_config = DINOv3ViTConfig(**config)
    model = DINOv3ViTModel(hf_config)
    mapped = {
        ("model." + key if key.startswith("layer.") else key): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"tiny DINO state mismatch: missing={missing} unexpected={unexpected}")
    model.eval()
    pixels = np.asarray(
        [0.15 * np.sin(float(i + 1)) + 0.01 * float(i)
         for i in range(3 * 4 * 4)], dtype=np.float32
    ).reshape(1, 3, 4, 4)
    x = torch.from_numpy(pixels)
    with torch.no_grad():
        hidden = model.embeddings(x)
        position_embeddings = model.rope_embeddings(x)
        for layer in model.model.layer:
            hidden = layer(hidden, position_embeddings=position_embeddings)
        hidden = F.layer_norm(hidden, hidden.shape[-1:])
    prefix = 1 + config["num_register_tokens"]
    return pixels, hidden[0, :prefix].numpy().reshape(-1), hidden[0, prefix:].numpy().reshape(-1)


def parse_native(text: str) -> tuple[np.ndarray, np.ndarray]:
    tokens = text.split()
    if len(tokens) < 4 or tokens[0] != "global":
        raise RuntimeError(f"unexpected native fixture output: {text[:200]}")
    global_count = int(tokens[1])
    global_values = np.asarray([float(v) for v in tokens[2:2 + global_count]], dtype=np.float32)
    marker = 2 + global_count
    if tokens[marker] != "patch":
        raise RuntimeError("native fixture omitted patch marker")
    patch_count = int(tokens[marker + 1])
    patch_values = np.asarray(
        [float(v) for v in tokens[marker + 2:marker + 2 + patch_count]], dtype=np.float32
    )
    if global_values.size != global_count or patch_values.size != patch_count:
        raise RuntimeError("native fixture output is truncated")
    return global_values, patch_values


def main() -> int:
    fixture = ROOT / "build" / "bin" / "pixal3d_dino_vit_numeric_fixture"
    if not fixture.is_file():
        raise RuntimeError(f"missing {fixture}; build the project first")
    with tempfile.TemporaryDirectory(prefix="pixal3d-dino-vs-") as tmp:
        directory = Path(tmp)
        source, config, state = make_fixture(directory)
        output = directory / "tiny-dino-f32.gguf"
        converter = ROOT / "scripts" / "convert_pixal3d_to_gguf.py"
        subprocess.run([
            sys.executable, str(converter), "--component", "dino",
            "--model", str(source), "--config", str(directory / "tiny-dino.json"),
            "--output", str(output), "--ftype", "0",
        ], check=True)
        expected_pixels, expected_global, expected_patch = reference(config, state)
        native = subprocess.run([str(fixture), str(output)], check=True,
                                capture_output=True, text=True)
        actual_global, actual_patch = parse_native(native.stdout)
        np.testing.assert_allclose(actual_global, expected_global, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(actual_patch, expected_patch, rtol=2e-5, atol=2e-5)
        print("dino_vit PASS: global max_abs={:.3g}, patch max_abs={:.3g}, "
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
