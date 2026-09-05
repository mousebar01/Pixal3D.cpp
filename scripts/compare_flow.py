#!/usr/bin/env python3
"""Compare the C++ timestep primitives with Pixal3D's Python oracle."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "ref" / "Pixal3D" / "pixal3d" / "models" / "sparse_structure_flow.py"
ABS_TOL = 3e-5
L2_TOL = 3e-6


def load_reference() -> Any:
    tree = ast.parse(REFERENCE.read_text(encoding="utf-8"), filename=str(REFERENCE))
    node = next(item for item in tree.body
                if isinstance(item, ast.ClassDef) and item.name == "TimestepEmbedder")
    # The class only needs torch/nn and does not depend on the rest of the
    # legacy package, so execute the original class body verbatim.
    module = ast.Module(body=[node], type_ignores=[])
    namespace: Dict[str, Any] = {"torch": torch, "nn": torch.nn, "np": np}
    exec(compile(module, str(REFERENCE), "exec"), namespace)
    return namespace["TimestepEmbedder"]


def parse_fixture(text: str) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        name, count = fields[0], int(fields[1])
        values = fields[2:]
        if len(values) != count:
            raise ValueError(f"{name} has {len(values)} values, expected {count}")
        result[name] = np.asarray(values, dtype=np.float32)
    return result


def fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()[:16]


def compare(name: str, reference: np.ndarray, actual: np.ndarray) -> None:
    if reference.shape != actual.shape:
        raise AssertionError(f"{name} shape mismatch: {actual.shape} != {reference.shape}")
    if not np.isfinite(actual).all():
        raise AssertionError(f"{name} contains non-finite C++ values")
    diff = actual.astype(np.float64) - reference.astype(np.float64)
    max_abs = float(np.max(np.abs(diff)))
    l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(reference), 1e-12))
    print(json.dumps({
        "name": name,
        "shape": list(reference.shape),
        "reference_fingerprint": fingerprint(reference),
        "cpp_fingerprint": fingerprint(actual),
        "max_abs_error": max_abs,
        "relative_l2_error": l2,
    }, sort_keys=True))
    if max_abs > ABS_TOL or l2 > L2_TOL:
        raise AssertionError(f"{name} exceeds tolerance: {max_abs}, {l2}")


def reference_outputs() -> Dict[str, np.ndarray]:
    cls = load_reference()
    timesteps = torch.tensor([-1.25, 0.0, 0.75, 2.5], dtype=torch.float32)
    frequency = cls.timestep_embedding(timesteps, 9).detach().numpy().reshape(-1)

    model = cls(hidden_size=6, frequency_embedding_size=8)
    with torch.no_grad():
        model.mlp[0].weight.copy_(torch.tensor([
            [0.10, -0.20, 0.30, -0.40, 0.50, -0.60, 0.70, -0.80],
            [-0.15, 0.25, -0.35, 0.45, -0.55, 0.65, -0.75, 0.85],
            [0.05, 0.15, -0.25, -0.35, 0.45, 0.55, -0.65, -0.75],
            [-0.08, 0.18, 0.28, -0.38, -0.48, 0.58, 0.68, -0.78],
            [0.12, 0.22, 0.32, 0.42, -0.52, -0.62, -0.72, 0.82],
            [-0.11, -0.21, 0.31, 0.41, 0.51, -0.61, -0.71, -0.81],
        ], dtype=torch.float32))
        model.mlp[0].bias.copy_(torch.tensor([0.03, -0.04, 0.05, -0.06, 0.07, -0.08]))
        model.mlp[2].weight.copy_(torch.tensor([
            [0.21, -0.31, 0.41, -0.51, 0.61, -0.71],
            [-0.17, 0.27, -0.37, 0.47, -0.57, 0.67],
            [0.13, 0.23, -0.33, -0.43, 0.53, 0.63],
            [-0.19, -0.29, 0.39, 0.49, -0.59, -0.69],
            [0.16, 0.26, 0.36, -0.46, -0.56, 0.66],
            [-0.14, 0.24, 0.34, 0.44, 0.54, -0.64],
        ], dtype=torch.float32))
        model.mlp[2].bias.copy_(torch.tensor([-0.02, 0.04, -0.06, 0.08, -0.10, 0.12]))
    mlp = model(timesteps).detach().numpy().reshape(-1)
    rows = torch.tensor([
        [0.15, -0.20, 0.35, 0.40, -0.55],
        [0.65, 0.70, -0.85, 0.90, 1.05],
        [-1.15, 1.20, 1.35, -1.40, 1.55],
        [1.65, -1.70, 1.85, 1.90, -2.05],
        [2.15, 2.20, -2.35, 2.40, 2.55],
        [-2.65, 2.70, 2.85, -2.90, 3.05],
    ], dtype=torch.float32).reshape(2, 3, 5)
    gamma = torch.tensor([0.90, 1.05, -0.80, 1.15, 0.70], dtype=torch.float32)
    beta = torch.tensor([-0.10, 0.20, 0.30, -0.40, 0.50], dtype=torch.float32)
    norm = torch.nn.functional.layer_norm(rows, (5,), weight=None, bias=None,
                                          eps=1e-6)
    affine = torch.nn.functional.layer_norm(rows, (5,), weight=gamma, bias=beta,
                                            eps=1e-6)
    shift_scale = torch.tensor([
        [0.01, -0.02, 0.03, -0.04, 0.05, 0.10, -0.20, 0.30, -0.40, 0.50],
        [-0.06, 0.07, -0.08, 0.09, -0.10, -0.15, 0.25, -0.35, 0.45, -0.55],
    ], dtype=torch.float32)
    shift, scale = shift_scale.chunk(2, dim=1)
    modulated = norm * (1.0 + scale[:, None, :]) + shift[:, None, :]
    return {
        "timestep_frequency": frequency,
        "timestep_mlp": mlp,
        "layer_norm": norm.detach().numpy().reshape(-1),
        "layer_norm_affine": affine.detach().numpy().reshape(-1),
        "adaln_modulated": modulated.detach().numpy().reshape(-1),
    }


def main() -> int:
    fixture = ROOT / "build" / "bin" / "pixal3d_flow_fixture"
    completed = subprocess.run([str(fixture)], check=True, capture_output=True, text=True)
    actual = parse_fixture(completed.stdout)
    reference = reference_outputs()
    for key in reference:
        compare(key, reference[key], actual[key])
    print("flow numerical comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
