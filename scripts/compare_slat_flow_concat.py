#!/usr/bin/env python3
"""Check texture SLat concatenation conditioning against the Python model."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_slat_flow as base

from pixal3d.modules import sparse as sp
from pixal3d.models.structured_latent_flow import SLatFlowModel
from pixal3d.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler


INPUT_CHANNELS = 12
CONCAT_CHANNELS = 4
MODEL_INPUT_CHANNELS = INPUT_CHANNELS + CONCAT_CHANNELS


def coords() -> np.ndarray:
    return np.asarray([
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 1, 1], [0, 2, 2, 2],
        [1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1], [1, 2, 1, 2], [1, 3, 3, 3],
    ], dtype=np.int32)


def make_model():
    model = SLatFlowModel(
        resolution=4, in_channels=MODEL_INPUT_CHANNELS,
        model_channels=base.MODEL_CHANNELS, cond_channels=base.COND_CHANNELS,
        out_channels=base.OUT_CHANNELS, num_blocks=1, num_heads=base.HEADS,
        mlp_ratio=1.5, pe_mode="rope", share_mod=True, qk_rms_norm=True,
        qk_rms_norm_cross=True, image_attn_mode="proj",
        proj_in_channels=base.PROJ_CHANNELS, dtype="float32")
    with torch.no_grad():
        for key, parameter in model.state_dict().items():
            parameter.copy_(torch.from_numpy(
                base.tensor_values(key, parameter.numel()).reshape(parameter.shape)))
    model.eval()
    return model


def parse_fixture(text: str):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] in {"slat_flow_concat_output", "slat_flow_concat_sample"}:
            count = int(fields[1])
            values = np.asarray(fields[2:], dtype=np.float32)
            if values.size != count:
                raise ValueError(f"{fields[0]} count mismatch")
            result[fields[0]] = values
    if len(result) != 2:
        raise ValueError("concat fixture did not emit both outputs")
    return result


def fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()[:16]


def main() -> int:
    model = make_model()
    coord_np = coords()
    point_count = coord_np.shape[0]
    x_np = (-0.27 + 0.013 * np.arange(point_count * INPUT_CHANNELS,
                                        dtype=np.float32)).reshape(point_count, INPUT_CHANNELS)
    concat_np = (0.07 - 0.011 * np.arange(point_count * CONCAT_CHANNELS,
                                            dtype=np.float32)).reshape(point_count, CONCAT_CHANNELS)
    projected_np = (0.19 - 0.017 * np.arange(point_count * base.PROJ_CHANNELS,
                                               dtype=np.float32)).reshape(point_count, base.PROJ_CHANNELS)
    global_np = (-0.11 + 0.021 * np.arange(5 * base.COND_CHANNELS,
                                             dtype=np.float32)).reshape(5, base.COND_CHANNELS)
    x = sp.SparseTensor(torch.from_numpy(x_np), torch.from_numpy(coord_np))
    concat = sp.SparseTensor(torch.from_numpy(concat_np), torch.from_numpy(coord_np))
    projected = sp.SparseTensor(torch.from_numpy(projected_np), torch.from_numpy(coord_np))
    global_cond = sp.VarLenTensor.from_tensor_list([
        torch.from_numpy(global_np[:3]), torch.from_numpy(global_np[3:])])
    condition = (global_cond, projected)
    negative = (global_cond.replace(torch.zeros_like(global_cond.feats)),
                projected.replace(torch.zeros_like(projected.feats)))
    timesteps = torch.tensor([0.17, -0.23], dtype=torch.float32)
    with torch.no_grad():
        reference = model(x, timesteps, condition, concat_cond=concat)
    expected_forward = reference.feats.detach().cpu().numpy().astype(np.float32).reshape(-1)
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=0.031)
    with torch.no_grad():
        sampled = sampler.sample(model, x, condition, negative, concat_cond=concat,
                                 steps=3, rescale_t=2.7, guidance_strength=1.0,
                                 verbose=False)
    expected_sample = sampled.samples.feats.detach().cpu().numpy().astype(np.float32).reshape(-1)

    fixture = ROOT / "build" / "bin" / "pixal3d_slat_flow_concat_fixture"
    state = {key: value.detach().cpu().numpy().astype(np.float32)
             for key, value in model.state_dict().items()}
    with tempfile.TemporaryDirectory(prefix="pixal3d-slat-concat-") as directory:
        pack = Path(directory) / "texture-flow-test.gguf"
        base.write_gguf(pack, state, component="texture_flow_1024",
                        alias="tx1024", in_channels=MODEL_INPUT_CHANNELS)
        actual = parse_fixture(subprocess.run(
            [str(fixture), str(pack)], check=True, capture_output=True, text=True).stdout)
    pairs = [("forward", actual["slat_flow_concat_output"], expected_forward),
             ("sample", actual["slat_flow_concat_sample"], expected_sample)]
    report = []
    for name, got, expected in pairs:
        if got.shape != expected.shape or not np.isfinite(got).all():
            raise AssertionError(f"{name} shape or finiteness mismatch")
        delta = got.astype(np.float64) - expected.astype(np.float64)
        max_abs = float(np.max(np.abs(delta)))
        rel_l2 = float(np.linalg.norm(delta) /
                       max(np.linalg.norm(expected.astype(np.float64)), 1e-12))
        report.append({"name": name, "shape": list(expected.shape),
                       "reference_fingerprint": fingerprint(expected),
                       "cpp_fingerprint": fingerprint(got),
                       "max_abs_error": max_abs, "relative_l2_error": rel_l2})
        if max_abs > 5e-4 or rel_l2 > 5e-4:
            print(json.dumps(report, sort_keys=True))
            raise AssertionError(f"SLat concat {name} comparison exceeds tolerance")
    print(json.dumps(report, sort_keys=True))
    print("SLat concat conditioning comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
