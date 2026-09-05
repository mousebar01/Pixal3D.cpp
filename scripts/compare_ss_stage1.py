#!/usr/bin/env python3
"""Compare the composed SS-flow -> SS-decoder stage-1 boundary.

The two tiny packs are generated from the same deterministic PyTorch models
used by the individual graph tests.  This checks that the C++ model-bound
sampler, sparse-to-dense layout bridge, decoder, threshold, and max-pool all
compose into the same active-coordinate list as the reference pipeline.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_ss_decoder import make_model as make_decoder_model
from compare_ss_decoder import write_gguf as write_decoder_gguf
from compare_ss_flow import make_model as make_flow_model
from compare_ss_flow import write_gguf as write_flow_gguf

from pixal3d.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler


def parse_fixture(text: str):
    values = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] not in {"ss_stage1_sample", "ss_stage1_coords"}:
            continue
        count = int(fields[1])
        dtype = np.float32 if fields[0].endswith("sample") else np.int32
        data = np.asarray(fields[2:], dtype=dtype)
        if data.size != count:
            raise ValueError(f"{fields[0]} count mismatch")
        values[fields[0]] = data
    if set(values) != {"ss_stage1_sample", "ss_stage1_coords"}:
        raise ValueError("stage-1 fixture did not emit both outputs")
    return values


def fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()[:16]


def main() -> int:
    flow_model, flow_state, flow_config = make_flow_model()
    decoder_model, decoder_state, decoder_config = make_decoder_model()
    resolution = int(flow_config["resolution"])
    points = resolution ** 3
    x_np = (-0.27 + 0.031 * np.arange(
        int(flow_config["in_channels"]) * points, dtype=np.float32)).reshape(
            1, int(flow_config["in_channels"]), resolution, resolution, resolution)
    cond_np = (-0.19 + 0.017 * np.arange(
        3 * int(flow_config["cond_channels"]), dtype=np.float32)).reshape(
            1, 3, int(flow_config["cond_channels"])
    )
    projected_np = (0.11 - 0.013 * np.arange(
        points * int(flow_config["proj_in_channels"]), dtype=np.float32)).reshape(
            1, points, int(flow_config["proj_in_channels"])
    )
    condition = (torch.from_numpy(cond_np), torch.from_numpy(projected_np))
    negative = (torch.zeros_like(condition[0]), torch.zeros_like(condition[1]))
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=0.031)
    with torch.no_grad():
        sampled = sampler.sample(
            flow_model, torch.from_numpy(x_np), condition, negative,
            steps=4, rescale_t=2.7, guidance_strength=2.0,
            guidance_rescale=0.23, guidance_interval=[0.34, 0.91], verbose=False,
        )
        decoded = decoder_model(sampled.samples)
        pooled = F.max_pool3d((decoded > 0).float(), kernel_size=2,
                              stride=2, padding=0)
        reference_coords = torch.argwhere(pooled > 0.5)[:, [0, 2, 3, 4]]
    expected_sample = sampled.samples.detach().cpu().numpy().astype(np.float32)[0]
    expected_sample = expected_sample.reshape(
        int(flow_config["in_channels"]), -1).T.reshape(-1)
    expected_coords = reference_coords.detach().cpu().numpy().astype(np.int32).reshape(-1)

    fixture = ROOT / "build" / "bin" / "pixal3d_ss_stage1_fixture"
    with tempfile.TemporaryDirectory(prefix="pixal3d-ss-stage1-") as directory:
        directory = Path(directory)
        flow_pack = directory / "ss-flow.gguf"
        decoder_pack = directory / "ss-decoder.gguf"
        write_flow_gguf(flow_pack, flow_state, flow_config)
        write_decoder_gguf(decoder_pack, decoder_state, decoder_config)
        completed = subprocess.run(
            [str(fixture), str(flow_pack), str(decoder_pack)],
            check=True, capture_output=True, text=True,
        )
    actual = parse_fixture(completed.stdout)
    actual_sample = actual["ss_stage1_sample"]
    if actual_sample.shape != expected_sample.shape or not np.isfinite(actual_sample).all():
        raise AssertionError("stage-1 sample shape or finiteness mismatch")
    delta = actual_sample.astype(np.float64) - expected_sample.astype(np.float64)
    max_abs = float(np.max(np.abs(delta)))
    rel_l2 = float(np.linalg.norm(delta) /
                   max(np.linalg.norm(expected_sample.astype(np.float64)), 1e-12))
    if not np.array_equal(actual["ss_stage1_coords"], expected_coords):
        raise AssertionError(
            f"stage-1 coordinates mismatch: {actual['ss_stage1_coords'].tolist()} != "
            f"{expected_coords.tolist()}")
    print(json.dumps({
        "sample_shape": list(expected_sample.shape),
        "sample_reference_fingerprint": fingerprint(expected_sample),
        "sample_cpp_fingerprint": fingerprint(actual_sample),
        "sample_max_abs_error": max_abs,
        "sample_relative_l2_error": rel_l2,
        "active_coordinate_count": int(expected_coords.size // 4),
    }, sort_keys=True))
    if max_abs > 3e-3 or rel_l2 > 2e-3:
        raise AssertionError("SS stage-1 sample comparison exceeds tolerance")
    print("SS stage-1 numerical comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
