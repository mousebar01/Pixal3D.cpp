#!/usr/bin/env python3
"""Numerically compare the native NAF implementation with ComfyUI's NAF.

The fixture intentionally uses a small internal width but keeps the reference
architecture intact (two convolution stacks, learned RoPE, adaptive pooling,
and neighborhood cross-attention).  Its source state is serialized through the
production converter before the C++ implementation is exercised.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


def load_reference_naf(path: Path):
    """Import naf.py without loading ComfyUI's GPU-only runtime modules."""
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    ops = types.ModuleType("comfy.ops")
    ops.cast_to_input = lambda source, target: source.to(target)
    comfy.ops = ops
    sys.modules["comfy"] = comfy
    sys.modules["comfy.ops"] = ops
    spec = importlib.util.spec_from_file_location("pixal3d_reference_naf", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import reference NAF from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NAF


def make_fixture(directory: Path, naf_source: Path) -> tuple[Path, Path, dict, dict[str, torch.Tensor]]:
    config = {
        "model_type": "naf",
        "architectures": ["NAF"],
        "dim": 32,
        "in_channels": 3,
        "heads_attn": 4,
        "heads_rope": 4,
        "kernel_size": 3,
        "rope_base": 100.0,
        "img_layers": 1,
        "num_groups": 8,
    }
    NAF = load_reference_naf(naf_source)
    torch.manual_seed(20260904)
    model = NAF(
        operations=torch.nn,
        dim=config["dim"],
        heads_attn=config["heads_attn"],
        heads_rope=config["heads_rope"],
        kernel_size=config["kernel_size"],
        rope_base=config["rope_base"],
        img_layers=config["img_layers"],
    ).eval()
    state = {name: value.detach().float().cpu() for name, value in model.state_dict().items()}
    state["image_encoder.rope.periods"] = torch.linspace(
        1.25, 3.5, state["image_encoder.rope.periods"].numel(), dtype=torch.float32
    )
    source = directory / "tiny-naf.safetensors"
    config_path = directory / "tiny-naf.json"
    save_file(state, str(source))
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return source, config_path, config, state


def make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.tensor(
        [0.35 + 0.2 * ((i * 13) % 17) / 16.0 for i in range(3 * 5 * 7)],
        dtype=torch.float32,
    ).reshape(1, 3, 5, 7)
    low_hwc = torch.tensor(
        [-0.25 + 0.07 * ((i * 7) % 19) for i in range(2 * 3 * 8)],
        dtype=torch.float32,
    ).reshape(1, 2, 3, 8)
    low = low_hwc.permute(0, 3, 1, 2).contiguous()
    return image, low


def parse_native(text: str) -> np.ndarray:
    tokens = text.split()
    if len(tokens) < 2 or tokens[0] != "output":
        raise RuntimeError(f"unexpected native fixture output: {text[:200]}")
    count = int(tokens[1])
    values = np.asarray([float(v) for v in tokens[2:2 + count]], dtype=np.float32)
    if values.size != count:
        raise RuntimeError("native NAF fixture output is truncated")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--naf-source", default="/app/ComfyUI/comfy/image_encoders/naf.py",
                        type=Path)
    parser.add_argument("--fixture", default=str(ROOT / "build/bin/pixal3d_naf_numeric_fixture"),
                        type=Path)
    args = parser.parse_args()
    if not args.fixture.is_file():
        raise RuntimeError(f"missing {args.fixture}; build the project first")
    if not args.naf_source.is_file():
        raise RuntimeError(f"missing reference NAF source: {args.naf_source}")
    with tempfile.TemporaryDirectory(prefix="pixal3d-naf-vs-") as tmp:
        directory = Path(tmp)
        source, config_path, _config, state = make_fixture(directory, args.naf_source)
        del state
        output = directory / "tiny-naf-f32.gguf"
        converter = ROOT / "scripts" / "convert_pixal3d_to_gguf.py"
        subprocess.run([
            sys.executable, str(converter), "--component", "naf",
            "--model", str(source), "--config", str(config_path),
            "--output", str(output), "--ftype", "0",
        ], check=True)

        # Recreate the deterministic model from the serialized state so the
        # reference output is driven by precisely the same values as GGUF.
        NAF = load_reference_naf(args.naf_source)
        model = NAF(operations=torch.nn, dim=32, heads_attn=4, heads_rope=4,
                    kernel_size=3, rope_base=100.0, img_layers=1).eval()
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(source)), strict=True)
        image, low = make_inputs()
        with torch.no_grad():
            expected = model(image, low, output_size=(4, 6))[0].permute(1, 2, 0)
        native = subprocess.run([str(args.fixture), str(output)], check=True,
                                capture_output=True, text=True)
        actual = parse_native(native.stdout).reshape(4, 6, 8)
        expected_np = expected.numpy()
        np.testing.assert_allclose(actual, expected_np, rtol=2e-5, atol=2e-5)
        diff = actual - expected_np
        print("naf PASS: max_abs={:.3g}, rel_l2={:.3g}, values={}".format(
            np.max(np.abs(diff)),
            np.linalg.norm(diff) / max(np.linalg.norm(expected_np), 1e-12),
            actual.size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
