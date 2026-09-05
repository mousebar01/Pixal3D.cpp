#!/usr/bin/env python3
"""Compare the complete CPU SLat flow stack with the Pixal3D oracle."""

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

import numpy as np
import torch

os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref" / "Pixal3D"))
# The minimal legacy image omits easydict; the sampler only uses its
# dictionary return object through attributes.
if "easydict" not in sys.modules:
    class _EasyDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    easydict = types.ModuleType("easydict")
    easydict.EasyDict = _EasyDict
    sys.modules["easydict"] = easydict
from pixal3d.modules import sparse as sp
from pixal3d.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
from pixal3d.models.structured_latent_flow import SLatFlowModel


IN_CHANNELS = 12
OUT_CHANNELS = 12
MODEL_CHANNELS = 12
COND_CHANNELS = 5
PROJ_CHANNELS = 7
HEADS = 2
MLP_HIDDEN = 18
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32
GGUF_VT_UINT32 = 4
GGUF_VT_FLOAT32 = 6
GGUF_VT_BOOL = 7
GGUF_VT_STRING = 8


def _gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, value_type: int, payload: bytes) -> bytes:
    return _gguf_string(key) + struct.pack("<I", value_type) + payload


def _kv_u32(key: str, value: int) -> bytes:
    return _kv(key, GGUF_VT_UINT32, struct.pack("<I", int(value)))


def _kv_f32(key: str, value: float) -> bytes:
    return _kv(key, GGUF_VT_FLOAT32, struct.pack("<f", float(value)))


def _kv_bool(key: str, value: bool) -> bytes:
    return _kv(key, GGUF_VT_BOOL, struct.pack("<?", bool(value)))


def _kv_str(key: str, value: str) -> bytes:
    return _kv(key, GGUF_VT_STRING, _gguf_string(value))


def _align(value: int) -> int:
    return (value + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT * GGUF_ALIGNMENT


def write_gguf(path: Path, state: dict[str, np.ndarray], *,
               component: str = "shape_flow_512",
               alias: str = "sh512", in_channels: int = IN_CHANNELS) -> None:
    metadata = [
        _kv_str("general.architecture", "pixal3d"),
        _kv_str("general.name", "slat-flow-loader-test"),
        _kv_u32("general.file_type", 0),
        _kv_u32("general.alignment", GGUF_ALIGNMENT),
        _kv_u32("pixal3d.bundle_format", 1),
        _kv_str("pixal3d.tensor_name_scheme", "compact-v1"),
        _kv_str("pixal3d.bundle_kind", "component"),
        _kv_u32("pixal3d.component_count", 1),
        _kv_str("pixal3d.component.0", component),
        _kv_str(f"pixal3d.{component}.model_class", "ElasticSLatFlowModel"),
        _kv_u32(f"pixal3d.{component}.resolution", 4),
        _kv_u32(f"pixal3d.{component}.in_channels", in_channels),
        _kv_u32(f"pixal3d.{component}.out_channels", OUT_CHANNELS),
        _kv_u32(f"pixal3d.{component}.model_channels", MODEL_CHANNELS),
        _kv_u32(f"pixal3d.{component}.cond_channels", COND_CHANNELS),
        _kv_u32(f"pixal3d.{component}.num_blocks", 1),
        _kv_u32(f"pixal3d.{component}.num_heads", HEADS),
        _kv_f32(f"pixal3d.{component}.mlp_ratio", 1.5),
        _kv_str(f"pixal3d.{component}.pe_mode", "rope"),
        _kv_bool(f"pixal3d.{component}.share_mod", True),
        _kv_bool(f"pixal3d.{component}.qk_rms_norm", True),
        _kv_bool(f"pixal3d.{component}.qk_rms_norm_cross", True),
        _kv_f32(f"pixal3d.{component}.rope_freq_min", 1.0),
        _kv_f32(f"pixal3d.{component}.rope_freq_base", 10000.0),
        _kv_str(f"pixal3d.{component}.image_attn_mode", "proj"),
        _kv_u32(f"pixal3d.{component}.proj_in_channels", PROJ_CHANNELS),
    ]
    tensors = []
    offsets = []
    offset = 0
    for key in sorted(state):
        value = np.asarray(state[key], dtype=np.float32)
        raw = value.astype("<f4", copy=False).tobytes(order="C")
        name = alias + "." + key
        dims = tuple(reversed(value.shape))
        offsets.append(offset)
        tensors.append((name, dims, raw))
        offset = _align(offset + len(raw))
    infos = bytearray()
    for (name, dims, _), tensor_offset in zip(tensors, offsets):
        infos += _gguf_string(name) + struct.pack("<I", len(dims))
        infos += b"".join(struct.pack("<Q", int(dim)) for dim in dims)
        infos += struct.pack("<I", 0) + struct.pack("<Q", tensor_offset)
    header = bytearray(GGUF_MAGIC) + struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(metadata))
    for item in metadata:
        header += item
    data_offset = _align(len(header) + len(infos))
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(infos)
        handle.write(b"\0" * (data_offset - handle.tell()))
        for (_, _, raw), tensor_offset in zip(tensors, offsets):
            if handle.tell() - data_offset != tensor_offset:
                raise RuntimeError("internal GGUF offset failure")
            handle.write(raw)
            handle.write(b"\0" * (_align(len(raw)) - len(raw)))


def tensor_values(key: str, count: int) -> np.ndarray:
    stable = sum((index + 1) * ord(char) for index, char in enumerate(key))
    values = ((np.arange(count, dtype=np.float32) + stable) % 29.0 - 14.0) * 0.018
    if key.endswith("gamma"):
        values = np.abs(values) + 0.5
    return values.astype(np.float32)


def inputs():
    coords = np.asarray([
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 1, 1], [0, 2, 2, 2],
        [1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1], [1, 2, 1, 2], [1, 3, 3, 3],
    ], dtype=np.int32)
    x = (-0.27 + 0.013 * np.arange(coords.shape[0] * IN_CHANNELS, dtype=np.float32)).reshape(
        coords.shape[0], IN_CHANNELS)
    projected = (0.19 - 0.017 * np.arange(coords.shape[0] * PROJ_CHANNELS,
                                           dtype=np.float32)).reshape(coords.shape[0], PROJ_CHANNELS)
    global_context = (-0.11 + 0.021 * np.arange(5 * COND_CHANNELS,
                                                  dtype=np.float32)).reshape(5, COND_CHANNELS)
    return coords, x, projected, global_context


def model():
    result = SLatFlowModel(
        resolution=4, in_channels=IN_CHANNELS, model_channels=MODEL_CHANNELS,
        cond_channels=COND_CHANNELS, out_channels=OUT_CHANNELS, num_blocks=1,
        num_heads=HEADS, mlp_ratio=1.5, pe_mode="rope", share_mod=True,
        qk_rms_norm=True, qk_rms_norm_cross=True, image_attn_mode="proj",
        proj_in_channels=PROJ_CHANNELS, dtype="float32")
    with torch.no_grad():
        for key, parameter in result.state_dict().items():
            parameter.copy_(torch.from_numpy(tensor_values(key, parameter.numel()).reshape(parameter.shape)))
    result.eval()
    return result


def parse_fixture(text: str, include_stage: bool = False):
    outputs = {}
    wanted = {"slat_flow_output", "slat_flow_probe", "slat_flow_sample",
              "slat_flow_sample_x0_last", "slat_flow_sample_direct",
              "slat_flow_sample_direct_x0_last"}
    if include_stage:
        wanted.add("slat_flow_stage_latent")
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] in wanted:
            name = fields[0]
            count = int(fields[1])
            values = np.asarray(fields[2:], dtype=np.float32)
            if values.size != count:
                raise ValueError(f"{name} count mismatch")
            outputs[name] = values
    if "slat_flow_output" not in outputs:
        raise ValueError("SLat flow fixture did not emit output")
    return outputs


def fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()[:16]


def main() -> int:
    coords, x, projected, global_context = inputs()
    sparse_x = sp.SparseTensor(torch.from_numpy(x), torch.from_numpy(coords))
    sparse_projected = sp.SparseTensor(torch.from_numpy(projected), torch.from_numpy(coords))
    context = sp.VarLenTensor.from_tensor_list([
        torch.from_numpy(global_context[:3]), torch.from_numpy(global_context[3:])
    ])
    with torch.no_grad():
        reference = model()(sparse_x, torch.tensor([0.17, -0.23]), (context, sparse_projected))
    expected = reference.feats.detach().cpu().numpy().astype(np.float32).reshape(-1)
    fixture = ROOT / "build" / "bin" / "pixal3d_slat_flow_fixture"
    direct = parse_fixture(subprocess.run(
        [str(fixture)], check=True, capture_output=True, text=True).stdout)
    actuals = [("direct", direct["slat_flow_output"], expected)]
    loader_fixture = ROOT / "build" / "bin" / "pixal3d_slat_flow_loader_fixture"
    state = {}
    reference_model = model()
    for key, parameter in reference_model.state_dict().items():
        state[key] = parameter.detach().cpu().numpy().astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="pixal3d-slat-flow-") as directory:
        pack = Path(directory) / "slat-flow-loader-test.gguf"
        write_gguf(pack, state)
        loaded = parse_fixture(subprocess.run([str(loader_fixture), str(pack)], check=True,
                                              capture_output=True, text=True).stdout)
        actuals.append(("loader-f32", loaded["slat_flow_output"], expected))

        # Exercise the model-bound sampler adapter as well as the standalone
        # sampler.  Its negative branch is zeros_like for both conditions.
        negative_context = context.replace(torch.zeros_like(context.feats))
        negative_projected = sparse_projected.replace(torch.zeros_like(sparse_projected.feats))
        sampler = FlowEulerGuidanceIntervalSampler(sigma_min=0.031)
        probe_t = 2.7 * 0.75 / (1.0 + (2.7 - 1.0) * 0.75)
        with torch.no_grad():
            probe = reference_model(
                sparse_x, torch.tensor([1000.0 * probe_t, 1000.0 * probe_t]),
                (context, sparse_projected))
        expected_probe = probe.feats.detach().cpu().numpy().astype(np.float32).reshape(-1)
        with torch.no_grad():
            sampled = sampler.sample(
                reference_model, sparse_x, (context, sparse_projected),
                (negative_context, negative_projected), steps=4, rescale_t=2.7,
                guidance_strength=2.0, guidance_rescale=0.23,
                guidance_interval=[0.34, 0.91], verbose=False,
            )
        expected_sample = sampled.samples.feats.detach().cpu().numpy().astype(np.float32).reshape(-1)
        expected_x0 = sampled.pred_x_0[-1].feats.detach().cpu().numpy().astype(np.float32).reshape(-1)
        actuals.append(("direct-sampler", direct["slat_flow_sample_direct"], expected_sample))
        actuals.append(("direct-sampler-x0", direct["slat_flow_sample_direct_x0_last"], expected_x0))
        sampled_loaded = parse_fixture(subprocess.run(
            [str(loader_fixture), str(pack), "--sample"], check=True,
            capture_output=True, text=True).stdout)
        actuals.append(("loader-probe", sampled_loaded["slat_flow_probe"], expected_probe))
        actuals.append(("loader-sampler", sampled_loaded["slat_flow_sample"], expected_sample))
        actuals.append(("loader-sampler-x0", sampled_loaded["slat_flow_sample_x0_last"], expected_x0))
        normalization_mean = 0.041 + 0.007 * np.arange(OUT_CHANNELS, dtype=np.float32)
        normalization_std = 0.73 + 0.013 * np.arange(OUT_CHANNELS, dtype=np.float32)
        expected_stage = (expected_sample.reshape(-1, OUT_CHANNELS) * normalization_std +
                          normalization_mean).reshape(-1)
        staged_loaded = parse_fixture(subprocess.run(
            [str(loader_fixture), str(pack), "--stage"], check=True,
            capture_output=True, text=True).stdout, include_stage=True)
        actuals.append(("loader-stage-denormalized",
                        staged_loaded["slat_flow_stage_latent"], expected_stage))
    report = []
    for source, actual, target in actuals:
        if actual.shape != target.shape or not np.isfinite(actual).all():
            raise AssertionError(f"{source} output shape or finiteness mismatch")
        delta = actual.astype(np.float64) - target.astype(np.float64)
        max_abs = float(np.max(np.abs(delta)))
        rel_l2 = float(np.linalg.norm(delta) / max(np.linalg.norm(target.astype(np.float64)), 1e-12))
        report.append({"source": source, "shape": list(target.shape),
                       "reference_fingerprint": fingerprint(target),
                       "cpp_fingerprint": fingerprint(actual),
                       "max_abs_error": max_abs,
                       "relative_l2_error": rel_l2})
        if max_abs > 5e-4 or rel_l2 > 5e-4:
            print(json.dumps(report, sort_keys=True))
            raise AssertionError(f"{source} SLat flow numerical comparison exceeds tolerance")
    print(json.dumps(report, sort_keys=True))
    print("SLat flow numerical comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
