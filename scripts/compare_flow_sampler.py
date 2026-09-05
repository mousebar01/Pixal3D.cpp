#!/usr/bin/env python3
"""Compare the CPU flow-Euler sampler with Pixal3D's Python sampler."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
sys.path.insert(0, str(ROOT / "ref" / "Pixal3D"))
# The minimal reference image omits easydict; the sampler only needs
# attribute access on its small return dictionary.
if "easydict" not in sys.modules:
    class _EasyDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    easydict = types.ModuleType("easydict")
    easydict.EasyDict = _EasyDict
    sys.modules["easydict"] = easydict
from pixal3d.modules import sparse as sp
from pixal3d.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler


def parse_fixture(text: str):
    outputs = {}
    wanted = {
        "flow_sampler_samples", "flow_sampler_pred_x_t_last",
        "flow_sampler_pred_x_0_last",
    }
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] not in wanted:
            continue
        count = int(fields[1])
        values = np.asarray(fields[2:], dtype=np.float32)
        if values.size != count:
            raise ValueError(f"{fields[0]} output count mismatch")
        outputs[fields[0]] = values
    if set(outputs) != wanted:
        raise ValueError(f"fixture outputs {set(outputs)} != {wanted}")
    return outputs


def make_noise() -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.tensor([
        [0, 0, 0, 0], [0, 0, 1, 0], [0, 2, 1, 1],
        [1, 0, 0, 1], [1, 1, 2, 0],
    ], dtype=torch.int32)
    values = -0.73 + 0.091 * torch.arange(coords.shape[0] * 3, dtype=torch.float32)
    return coords, values.reshape(coords.shape[0], 3)


class OracleModel:
    def __call__(self, state, timestep, cond):
        conditional = bool(cond[0, 0].item() > 0.5)
        condition = 0.19 if conditional else -0.07
        channel = torch.arange(state.feats.shape[1], dtype=state.feats.dtype,
                              device=state.feats.device).view(1, -1)
        values = (0.043 * state.feats + 0.00037 * timestep[0] +
                  0.011 * channel + condition)
        return state.replace(values)


def reference():
    coords, values = make_noise()
    noise = sp.SparseTensor(values, coords)
    cond = torch.ones((2, 1), dtype=torch.float32)
    neg_cond = torch.zeros_like(cond)
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=0.031)
    with torch.no_grad():
        result = sampler.sample(
            OracleModel(), noise, cond, neg_cond,
            steps=6, rescale_t=2.7, guidance_strength=2.35,
            guidance_rescale=0.41, guidance_interval=[0.34, 0.91],
            verbose=False,
        )
    return {
        "flow_sampler_samples": result.samples.feats.detach().cpu().numpy().reshape(-1),
        "flow_sampler_pred_x_t_last": result.pred_x_t[-1].feats.detach().cpu().numpy().reshape(-1),
        "flow_sampler_pred_x_0_last": result.pred_x_0[-1].feats.detach().cpu().numpy().reshape(-1),
    }


def main() -> int:
    fixture = ROOT / "build" / "bin" / "pixal3d_flow_sampler_fixture"
    actual = parse_fixture(subprocess.run([str(fixture)], check=True,
                                          capture_output=True, text=True).stdout)
    expected = reference()
    report = []
    for name, target in expected.items():
        got = actual[name]
        if got.shape != target.shape or not np.isfinite(got).all():
            raise AssertionError(f"{name} shape or finiteness mismatch")
        delta = got.astype(np.float64) - target.astype(np.float64)
        max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
        rel_l2 = float(np.linalg.norm(delta) /
                       max(np.linalg.norm(target.astype(np.float64)), 1e-12))
        if max_abs > 5e-6 or rel_l2 > 5e-6:
            raise AssertionError(f"{name} error too large: {max_abs}, {rel_l2}")
        report.append({"name": name, "shape": list(target.shape),
                       "max_abs_error": max_abs, "relative_l2_error": rel_l2})
    print(json.dumps(report, sort_keys=True))
    print("flow Euler sampler numerical comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
