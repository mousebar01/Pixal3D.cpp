#!/usr/bin/env python3
"""Compare the C++ multi-view condition projection with ProjGridMV."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch

from compare_projection import load_reference


ROOT = Path(__file__).resolve().parents[1]


def parse(text: str):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        result[fields[0]] = np.asarray(fields[2:], dtype=np.float32)
        if result[fields[0]].size != int(fields[1]):
            raise ValueError(f"malformed fixture line: {line}")
    return result


def reference_outputs():
    ns = load_reference()
    options = dict(grid_resolution=2, image_resolution=4)
    front = torch.tensor([[1.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, -1.0, -2.0,
                           0.0, 1.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 1.0]], dtype=torch.float32).reshape(1, 4, 4)
    second = front.clone()
    second[:, 0, 3] = 0.25
    transforms = torch.stack([front[0], second[0]], dim=0).unsqueeze(0)
    relative = ns["compute_relative_calc_mat"](
        transforms, torch.tensor([[2.0, 2.0]], dtype=torch.float32),
        front[0])
    angle = torch.tensor([0.9], dtype=torch.float32)
    distance = torch.tensor([2.0], dtype=torch.float32)
    mesh_scale = torch.tensor([1.0], dtype=torch.float32)
    mv = ns["ProjGridMV"](**options)

    def make_map(base: float):
        return torch.tensor([base, base + 1.0, base + 2.0, base + 3.0],
                            dtype=torch.float32).reshape(1, 2, 2, 1)

    projected = []
    for base in (0.0, 4.0):
        projected.append(mv(make_map(base), angle, distance, mesh_scale,
                            transform_matrix=relative[:, len(projected)]))
    naf_projected = []
    for base in (10.0, 14.0):
        naf_projected.append(mv(make_map(base), angle, distance, mesh_scale,
                                transform_matrix=relative[:, len(naf_projected)]))
    coords = [0, 7]  # [0,0,0] and [1,1,1] in the R^3 meshgrid order.
    dino = (projected[0] + projected[1]) / 2.0
    naf = (naf_projected[0] + naf_projected[1]) / 2.0
    projection = torch.cat([dino, naf], dim=-1)[:, coords, :]
    return {
        "mv_global": np.asarray([0.5, 1.5, 2.5, 3.5], dtype=np.float32),
        "mv_projection": projection.reshape(-1).detach().cpu().numpy(),
    }


def main() -> int:
    fixture = ROOT / "build" / "bin" / "pixal3d_multiview_condition_fixture"
    actual = parse(subprocess.run([str(fixture)], check=True,
                                  capture_output=True, text=True).stdout)
    reference = reference_outputs()
    for name, expected in reference.items():
        got = actual[name]
        if expected.shape != got.shape:
            raise AssertionError(f"{name} shape mismatch: {got.shape} != {expected.shape}")
        diff = got.astype(np.float64) - expected.astype(np.float64)
        max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
        relative_l2 = float(np.linalg.norm(diff) /
                            max(np.linalg.norm(expected.astype(np.float64)), 1e-12))
        print(f"{name}: max_abs={max_abs:.8g} relative_l2={relative_l2:.8g}")
        if max_abs > 2e-5 or relative_l2 > 2e-6:
            raise AssertionError(f"{name} exceeds tolerance")
    print("multi-view condition numerical comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
