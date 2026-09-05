#!/usr/bin/env python3
"""Compare the production F16 SS decoder GGUF against the PyTorch model."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref" / "Pixal3D"))
from pixal3d.models.sparse_structure_vae import SparseStructureDecoder


def parse_fixture(text: str) -> np.ndarray:
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == "ss_decoder_output":
            count = int(fields[1])
            values = np.asarray(fields[2:], dtype=np.float32)
            if values.size != count:
                raise ValueError("fixture output count mismatch")
            return values
    raise ValueError("fixture did not emit ss_decoder_output")


def fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()[:16]


def main() -> int:
    checkpoint = ROOT / "weights" / "Pixal3D" / "ckpts" / "ss_dec_conv3d_16l8_fp16"
    pack = ROOT / "build" / "weights" / "pixal3d-shared-f16.gguf"
    fixture = ROOT / "build" / "bin" / "pixal3d_ss_decoder_fixture"
    with checkpoint.with_suffix(".json").open(encoding="utf-8") as handle:
        config = json.load(handle)["args"]
    # The legacy container's CPU F16 Conv3d kernel is effectively scalar and
    # can take several minutes.  Keep the exact serialized F16 values, but run
    # the Python oracle in F32 so this check remains practical.  The C++ side
    # still reads the production F16 GGUF and accumulates in its backend type.
    torch.set_num_threads(4)
    config = dict(config)
    config["use_fp16"] = False
    model = SparseStructureDecoder(**config)
    from safetensors.torch import load_file
    state = {key: value.float() for key, value in load_file(
        str(checkpoint.with_suffix(".safetensors"))).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint state mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()

    resolution = int(config.get("resolution", 16))
    latent_channels = int(config["latent_channels"])
    latent = (-0.21 + 0.037 * np.arange(latent_channels * resolution ** 3,
                                         dtype=np.float32)).reshape(
                                             1, latent_channels, resolution,
                                             resolution, resolution)
    with torch.no_grad():
        reference = model(torch.from_numpy(latent))
    reference_np = reference.detach().cpu().numpy().astype(np.float32).reshape(-1)

    completed = subprocess.run([str(fixture), str(pack)], check=True,
                               capture_output=True, text=True)
    actual = parse_fixture(completed.stdout)
    if actual.shape != reference_np.shape:
        raise AssertionError(f"shape mismatch: {actual.shape} != {reference_np.shape}")
    if not np.isfinite(actual).all():
        raise AssertionError("C++ output contains non-finite values")
    difference = actual.astype(np.float64) - reference_np.astype(np.float64)
    max_abs = float(np.max(np.abs(difference)))
    rel_l2 = float(np.linalg.norm(difference) /
                   max(np.linalg.norm(reference_np.astype(np.float64)), 1e-12))
    print(json.dumps({
        "shape": list(reference_np.shape),
        "reference_fingerprint": fingerprint(reference_np),
        "cpp_fingerprint": fingerprint(actual),
        "max_abs_error": max_abs,
        "relative_l2_error": rel_l2,
    }, sort_keys=True))
    # The source decoder intentionally stores its torso in FP16; the F32 oracle
    # above therefore differs only by expected half-vs-float accumulation.
    if max_abs > 5e-2 or rel_l2 > 3e-3:
        raise AssertionError("production SS-decoder numerical comparison exceeds tolerance")
    print("production ss-decoder numerical comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
