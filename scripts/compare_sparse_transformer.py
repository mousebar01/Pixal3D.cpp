#!/usr/bin/env python3
"""Compare one SLat modulated cross-transformer block with the Python oracle."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


# The block has no convolution; the reference sparse container can therefore
# use its dependency-free coordinate backend while attention uses PyTorch SDPA.
os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref" / "Pixal3D"))
from pixal3d.modules import sparse as sp
from pixal3d.modules.sparse.transformer.modulated import ModulatedSparseTransformerCrossBlock


CHANNELS = 12
HEADS = 2
HEAD_DIM = 6
MLP_HIDDEN = 18
CONTEXT_CHANNELS = 5
PROJECTION_CHANNELS = 7


def tensor_values(key: str, count: int) -> np.ndarray:
    stable = sum((index + 1) * ord(char) for index, char in enumerate(key))
    values = ((np.arange(count, dtype=np.float32) + stable) % 29.0 - 14.0) * 0.018
    if key.endswith("gamma"):
        values = np.abs(values) + 0.5
    return values.astype(np.float32)


def make_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray([
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 1, 1], [0, 2, 2, 2],
        [1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1], [1, 2, 1, 2], [1, 3, 3, 3],
    ], dtype=np.int32)
    x = -0.27 + 0.013 * np.arange(coords.shape[0] * CHANNELS, dtype=np.float32)
    projected = 0.19 - 0.017 * np.arange(coords.shape[0] * PROJECTION_CHANNELS,
                                           dtype=np.float32)
    global_context = -0.11 + 0.021 * np.arange(5 * CONTEXT_CHANNELS, dtype=np.float32)
    timestep_mod = 0.07 - 0.009 * np.arange(2 * 6 * CHANNELS, dtype=np.float32)
    return coords, x.reshape(coords.shape[0], CHANNELS), projected.reshape(
        coords.shape[0], PROJECTION_CHANNELS), global_context.reshape(5, CONTEXT_CHANNELS), timestep_mod.reshape(2, 6 * CHANNELS)


def make_model() -> ModulatedSparseTransformerCrossBlock:
    model = ModulatedSparseTransformerCrossBlock(
        CHANNELS,
        CONTEXT_CHANNELS,
        num_heads=HEADS,
        mlp_ratio=1.5,
        attn_mode="full",
        use_rope=True,
        rope_freq=(1.0, 10000.0),
        share_mod=True,
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
        image_attn_mode="proj",
        proj_in_channels=PROJECTION_CHANNELS,
    )
    with torch.no_grad():
        for key, parameter in model.state_dict().items():
            parameter.copy_(torch.from_numpy(tensor_values(key, parameter.numel()).reshape(parameter.shape)))
    model.eval()
    return model


def parse_fixture(text: str) -> np.ndarray:
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == "sparse_transformer_output":
            count = int(fields[1])
            values = np.asarray(fields[2:], dtype=np.float32)
            if values.size != count:
                raise ValueError("transformer fixture output count mismatch")
            return values
    raise ValueError("transformer fixture did not emit output")


def fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()[:16]


def main() -> int:
    coords, x, projected, global_context, timestep_mod = make_inputs()
    model = make_model()
    sparse_x = sp.SparseTensor(torch.from_numpy(x), torch.from_numpy(coords))
    sparse_projected = sp.SparseTensor(torch.from_numpy(projected), torch.from_numpy(coords))
    context = sp.VarLenTensor.from_tensor_list([
        torch.from_numpy(global_context[:3]), torch.from_numpy(global_context[3:])
    ])
    with torch.no_grad():
        reference = model(sparse_x, torch.from_numpy(timestep_mod), (context, sparse_projected))
    reference_np = reference.feats.detach().cpu().numpy().astype(np.float32).reshape(-1)

    fixture = ROOT / "build" / "bin" / "pixal3d_sparse_transformer_fixture"
    completed = subprocess.run([str(fixture)], check=True, capture_output=True, text=True)
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
    if max_abs > 5e-4 or rel_l2 > 5e-4:
        raise AssertionError("sparse transformer numerical comparison exceeds tolerance")
    print("sparse transformer numerical comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
