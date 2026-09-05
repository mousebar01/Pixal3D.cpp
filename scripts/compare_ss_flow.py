#!/usr/bin/env python3
"""Compare the compact Pixal3D SS-flow ggml graph with the Python oracle.

The test creates a tiny one-block Pixal3D GGUF in a temporary directory.  It
uses the real SparseStructureFlowModel implementation with deterministic
weights, then invokes the C++ graph on the same x/t/global/projection inputs.
This validates graph layout and operation order without loading the 22 GiB
production checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
# The reference image may default to flash-attn even when that optional
# package is not installed.  The naive backend is mathematically identical
# for this tiny fixture and keeps the oracle deterministic.
os.environ.setdefault("ATTN_BACKEND", "naive")
sys.path.insert(0, str(ROOT / "ref" / "Pixal3D"))
# The reference image used for numerical checks does not always include
# easydict, while the sampler returns an attribute-style dictionary.
if "easydict" not in sys.modules:
    class _EasyDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    easydict = types.ModuleType("easydict")
    easydict.EasyDict = _EasyDict
    sys.modules["easydict"] = easydict
from pixal3d.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
from pixal3d.models.sparse_structure_flow import SparseStructureFlowModel


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32
VT_UINT32 = 4
VT_FLOAT32 = 6
VT_BOOL = 7
VT_STRING = 8
ABS_TOL = 3e-3
L2_TOL = 2e-3


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def kv(key: str, kind: int, payload: bytes) -> bytes:
    return gguf_string(key) + struct.pack("<I", kind) + payload


def kv_u32(key: str, value: int) -> bytes:
    return kv(key, VT_UINT32, struct.pack("<I", value))


def kv_f32(key: str, value: float) -> bytes:
    return kv(key, VT_FLOAT32, struct.pack("<f", value))


def kv_bool(key: str, value: bool) -> bytes:
    return kv(key, VT_BOOL, struct.pack("<?", value))


def kv_str(key: str, value: str) -> bytes:
    return kv(key, VT_STRING, gguf_string(value))


def tensor_values(key: str, shape: Sequence[int]) -> np.ndarray:
    count = int(np.prod(shape))
    stable = sum(ord(char) for char in key)
    # Keep values small enough that the unquantized tiny graph stays in the
    # same numerical regime as the float32 production reference.
    values = ((np.arange(count, dtype=np.float32) + stable) % 23.0 - 11.0) * 0.0125
    if key.endswith("q_rms_norm.gamma") or key.endswith("k_rms_norm.gamma"):
        values = np.abs(values) + 0.5
    return values.reshape(tuple(shape)).astype(np.float32)


def make_model() -> Tuple[SparseStructureFlowModel, Dict[str, np.ndarray], Dict[str, int]]:
    config = dict(
        resolution=2,
        in_channels=2,
        model_channels=12,
        cond_channels=5,
        out_channels=2,
        num_blocks=1,
        num_heads=2,
        mlp_ratio=1.5,
        pe_mode="rope",
        rope_freq=(1.0, 10000.0),
        dtype="float32",
        share_mod=True,
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
        image_attn_mode="proj",
        proj_in_channels=3,
    )
    model = SparseStructureFlowModel(**config)
    state: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for key, parameter in model.state_dict().items():
            if key == "rope_phases":
                continue
            data = tensor_values(key, tuple(parameter.shape))
            parameter.copy_(torch.from_numpy(data))
            state[key] = data
    model.eval()
    return model, state, config


def write_gguf(path: Path, state: Dict[str, np.ndarray], config: Dict[str, int]) -> None:
    metadata = [
        kv_str("general.architecture", "pixal3d"),
        kv_str("general.name", "ss-flow-numeric-test"),
        kv_u32("general.file_type", 0),
        kv_u32("general.alignment", ALIGNMENT),
        kv_u32("pixal3d.bundle_format", 1),
        kv_str("pixal3d.tensor_name_scheme", "compact-v1"),
        kv_str("pixal3d.bundle_kind", "component"),
        kv_u32("pixal3d.component_count", 1),
        kv_str("pixal3d.component.0", "ss_flow"),
        kv_str("pixal3d.ss_flow.model_class", "SparseStructureFlowModel"),
        kv_str("pixal3d.ss_flow.checkpoint", "ss-flow-numeric-test"),
        kv_str("pixal3d.ss_flow.variant", "base"),
        kv_u32("pixal3d.ss_flow.resolution", config["resolution"]),
        kv_u32("pixal3d.ss_flow.in_channels", config["in_channels"]),
        kv_u32("pixal3d.ss_flow.out_channels", config["out_channels"]),
        kv_u32("pixal3d.ss_flow.model_channels", config["model_channels"]),
        kv_u32("pixal3d.ss_flow.cond_channels", config["cond_channels"]),
        kv_u32("pixal3d.ss_flow.num_blocks", config["num_blocks"]),
        kv_u32("pixal3d.ss_flow.num_heads", config["num_heads"]),
        kv_f32("pixal3d.ss_flow.mlp_ratio", config["mlp_ratio"]),
        kv_str("pixal3d.ss_flow.pe_mode", "rope"),
        kv_bool("pixal3d.ss_flow.share_mod", True),
        kv_bool("pixal3d.ss_flow.qk_rms_norm", True),
        kv_bool("pixal3d.ss_flow.qk_rms_norm_cross", True),
        kv_f32("pixal3d.ss_flow.rope_freq_min", 1.0),
        kv_f32("pixal3d.ss_flow.rope_freq_base", 10000.0),
        kv_str("pixal3d.ss_flow.image_attn_mode", "proj"),
        kv_u32("pixal3d.ss_flow.proj_in_channels", config["proj_in_channels"]),
    ]
    tensors: List[Tuple[str, Tuple[int, ...], int, bytes]] = []
    for key in sorted(state):
        value = state[key]
        name = "ss." + key
        raw = np.asarray(value, dtype="<f4").tobytes(order="C")
        tensors.append((name, tuple(reversed(value.shape)), 0, raw))
    infos = bytearray()
    offsets: List[int] = []
    offset = 0
    for name, dims, kind, raw in tensors:
        offsets.append(offset)
        infos += gguf_string(name) + struct.pack("<I", len(dims))
        for dim in dims:
            infos += struct.pack("<Q", int(dim))
        infos += struct.pack("<I", kind) + struct.pack("<Q", offset)
        offset = align(offset + len(raw))
    header = bytearray(GGUF_MAGIC) + struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(metadata))
    for item in metadata:
        header += item
    data_offset = align(len(header) + len(infos))
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(infos)
        handle.write(b"\0" * (data_offset - handle.tell()))
        for (_, _, _, raw), wanted in zip(tensors, offsets):
            if handle.tell() - data_offset != wanted:
                raise RuntimeError("internal GGUF offset failure")
            handle.write(raw)
            handle.write(b"\0" * (align(len(raw)) - len(raw)))


def parse_fixture(text: str) -> np.ndarray:
    outputs = {}
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] in {"ss_flow_output", "ss_flow_sample",
                                   "ss_flow_sample_x0_last"}:
            name = fields[0]
            count = int(fields[1])
            values = np.asarray(fields[2:], dtype=np.float32)
            if values.size != count:
                raise ValueError(f"{name} output count mismatch")
            outputs[name] = values
    if "ss_flow_output" not in outputs:
        raise ValueError("fixture did not emit ss_flow_output")
    return outputs


def fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()[:16]


def main() -> int:
    model, state, config = make_model()
    points = config["resolution"] ** 3
    cond_tokens = 3
    x_np = (-0.27 + 0.031 * np.arange(config["in_channels"] * points,
                                       dtype=np.float32)).reshape(
                                           1, config["in_channels"], config["resolution"],
                                           config["resolution"], config["resolution"])
    cond_np = (-0.19 + 0.017 * np.arange(cond_tokens * config["cond_channels"],
                                         dtype=np.float32)).reshape(
                                             1, cond_tokens, config["cond_channels"])
    projected_np = (0.11 - 0.013 * np.arange(points * config["proj_in_channels"],
                                              dtype=np.float32)).reshape(
                                                  1, points, config["proj_in_channels"])
    x = torch.from_numpy(x_np)
    cond = torch.from_numpy(cond_np)
    projected = torch.from_numpy(projected_np)
    condition = (cond, projected)
    negative_condition = (torch.zeros_like(cond), torch.zeros_like(projected))
    with torch.no_grad():
        reference = model(x, torch.tensor([0.75]), condition)
    reference_np = reference.detach().numpy().astype(np.float32).reshape(-1)

    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=0.031)
    with torch.no_grad():
        sampled = sampler.sample(
            model, x, condition, negative_condition, steps=4, rescale_t=2.7,
            guidance_strength=2.0, guidance_rescale=0.23,
            guidance_interval=[0.34, 0.91], verbose=False,
        )
    # C++ exposes sparse rows as [point, channel], whereas the dense Python
    # model stores [channel, x, y, z].  Convert both expected arrays to the
    # C++ row order before comparing.
    expected_sample = sampled.samples.detach().numpy().astype(np.float32)[0]
    expected_sample = expected_sample.reshape(config["in_channels"], -1).T.reshape(-1)
    expected_x0 = sampled.pred_x_0[-1].detach().numpy().astype(np.float32)[0]
    expected_x0 = expected_x0.reshape(config["in_channels"], -1).T.reshape(-1)

    fixture = ROOT / "build" / "bin" / "pixal3d_ss_flow_fixture"
    with tempfile.TemporaryDirectory(prefix="pixal3d-ss-flow-") as directory:
        pack = Path(directory) / "ss-flow-numeric-test.gguf"
        write_gguf(pack, state, config)
        completed = subprocess.run([str(fixture), str(pack), "--sample"], check=True,
                                   capture_output=True, text=True)
    actuals = parse_fixture(completed.stdout)
    comparisons = [
        ("forward", actuals["ss_flow_output"], reference_np),
        ("sample", actuals["ss_flow_sample"], expected_sample),
        ("sample_x0_last", actuals["ss_flow_sample_x0_last"], expected_x0),
    ]
    report = []
    for name, actual, expected in comparisons:
        if actual.shape != expected.shape:
            raise AssertionError(f"{name} shape mismatch: {actual.shape} != {expected.shape}")
        if not np.isfinite(actual).all():
            raise AssertionError(f"{name} C++ output contains non-finite values")
        difference = actual.astype(np.float64) - expected.astype(np.float64)
        max_abs = float(np.max(np.abs(difference)))
        rel_l2 = float(np.linalg.norm(difference) /
                       max(np.linalg.norm(expected.astype(np.float64)), 1e-12))
        report.append({"name": name, "shape": list(expected.shape),
                       "reference_fingerprint": fingerprint(expected),
                       "cpp_fingerprint": fingerprint(actual),
                       "max_abs_error": max_abs,
                       "relative_l2_error": rel_l2})
        if max_abs > ABS_TOL or rel_l2 > L2_TOL:
            print(json.dumps(report, sort_keys=True))
            raise AssertionError(f"SS-flow {name} numerical comparison exceeds tolerance")
    print(json.dumps(report, sort_keys=True))
    print("ss-flow numerical comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
