#!/usr/bin/env python3
"""Compare the sparse SLat decoder blocks with the Pixal3D Python oracle."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref" / "Pixal3D"))

# The reference container does not always ship flex_gemm/torchsparse.  This
# tiny backend spells out the documented submanifold contract so the oracle
# remains the actual Pixal3D module graph rather than a second implementation.
from pixal3d.modules import sparse as sp

sp.config.CONV = "pixal3d_cpu_fixture"
fallback = types.ModuleType("pixal3d.modules.sparse.conv.conv_pixal3d_cpu_fixture")


def sparse_conv3d_init(module: nn.Module, in_channels: int, out_channels: int,
                       kernel_size: int, stride=1, dilation=1, padding=None,
                       bias=True, indice_key=None) -> None:
    del stride, dilation, padding, indice_key
    module.in_channels = in_channels
    module.out_channels = out_channels
    module.kernel_size = (kernel_size,) * 3 if isinstance(kernel_size, int) else tuple(kernel_size)
    module.weight = nn.Parameter(torch.empty((out_channels, *module.kernel_size, in_channels)))
    module.bias = nn.Parameter(torch.empty(out_channels)) if bias else None


def sparse_conv3d_forward(module: nn.Module, x: sp.SparseTensor) -> sp.SparseTensor:
    index = {tuple(int(value) for value in coord): point
             for point, coord in enumerate(x.coords.tolist())}
    output = torch.zeros((x.coords.shape[0], module.out_channels),
                         dtype=x.feats.dtype, device=x.feats.device)
    if module.bias is not None:
        output += module.bias
    for point, center in enumerate(x.coords.tolist()):
        for kd in range(3):
            for kh in range(3):
                for kw in range(3):
                    neighbor = (center[0], center[1] + kd - 1,
                                center[2] + kh - 1, center[3] + kw - 1)
                    neighbor_point = index.get(neighbor)
                    if neighbor_point is None:
                        continue
                    output[point] += module.weight[:, kd, kh, kw, :] @ x.feats[neighbor_point]
    return x.replace(output)


def sparse_inverse_conv3d_init(*args, **kwargs):
    raise NotImplementedError


def sparse_inverse_conv3d_forward(*args, **kwargs):
    raise NotImplementedError


fallback.sparse_conv3d_init = sparse_conv3d_init
fallback.sparse_conv3d_forward = sparse_conv3d_forward
fallback.sparse_inverse_conv3d_init = sparse_inverse_conv3d_init
fallback.sparse_inverse_conv3d_forward = sparse_inverse_conv3d_forward
sys.modules[fallback.__name__] = fallback

from pixal3d.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder


LATENT_CHANNELS = 3
OUT_CHANNELS = 3
LEVEL_CHANNELS = [8, 8, 4]
NUM_BLOCKS = [1, 1, 1]
N_LEVELS = len(LEVEL_CHANNELS)
N_SUBDIVISIONS = N_LEVELS - 1
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32
GGUF_VT_UINT32 = 4
GGUF_VT_FLOAT32 = 6
GGUF_VT_BOOL = 7
GGUF_VT_STRING = 8
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


def tensor_values(key: str, shape: Sequence[int]) -> np.ndarray:
    count = int(np.prod(shape))
    stable = sum((index + 1) * ord(char) for index, char in enumerate(key))
    values = ((np.arange(count, dtype=np.float32) + stable) % 29.0 - 14.0) * 0.018
    if key.endswith("norm.weight"):
        values += 1.0
    return values.reshape(tuple(shape)).astype(np.float32)


def make_model() -> Tuple[SparseUnetVaeDecoder, Dict[str, np.ndarray]]:
    model = SparseUnetVaeDecoder(
        OUT_CHANNELS,
        LEVEL_CHANNELS,
        LATENT_CHANNELS,
        NUM_BLOCKS,
        ["SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d"],
        ["SparseResBlockC2S3d", "SparseResBlockC2S3d"],
        [{}, {}, {}],
        use_fp16=False,
        pred_subdiv=True,
    )
    state: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for key, parameter in model.state_dict().items():
            value = tensor_values(key, tuple(parameter.shape))
            parameter.copy_(torch.from_numpy(value))
            state[key] = value
    model.eval()
    return model, state


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


def _serial_tensor(key: str, value: np.ndarray, ggml_type: int) -> Tuple[Tuple[int, ...], bytes]:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 5:
        out_channels, kd, kh, kw, in_channels = value.shape
        if (kd, kh, kw) != (3, 3, 3):
            raise ValueError(f"unexpected kernel shape for {key}: {value.shape}")
        value = value.transpose(1, 2, 3, 0, 4).reshape(27, out_channels, in_channels)
    if ggml_type == GGML_TYPE_F32:
        raw = value.astype("<f4", copy=False).tobytes(order="C")
    elif ggml_type == GGML_TYPE_F16:
        raw = value.astype("<f2", copy=False).tobytes(order="C")
    else:
        raise ValueError(f"unsupported fixture GGML type: {ggml_type}")
    return tuple(reversed(value.shape)), raw


def write_gguf(path: Path, state: Dict[str, np.ndarray], ggml_type: int = GGML_TYPE_F32) -> None:
    metadata = [
        _kv_str("general.architecture", "pixal3d"),
        _kv_str("general.name", "slat-decoder-loader-test"),
        _kv_u32("general.file_type", 0 if ggml_type == GGML_TYPE_F32 else 1),
        _kv_u32("general.alignment", GGUF_ALIGNMENT),
        _kv_u32("pixal3d.bundle_format", 1),
        _kv_str("pixal3d.tensor_name_scheme", "compact-v1"),
        _kv_str("pixal3d.bundle_kind", "component"),
        _kv_u32("pixal3d.component_count", 1),
        _kv_str("pixal3d.component.0", "shape_decoder"),
        _kv_str("pixal3d.shape_decoder.model_class", "FlexiDualGridVaeDecoder"),
        _kv_str("pixal3d.shape_decoder.checkpoint", "loader-test"),
        _kv_str("pixal3d.shape_decoder.variant", "fixture"),
        _kv_u32("pixal3d.shape_decoder.resolution", 2),
        _kv_u32("pixal3d.shape_decoder.out_channels", OUT_CHANNELS),
        _kv_u32("pixal3d.shape_decoder.latent_channels", LATENT_CHANNELS),
        _kv_f32("pixal3d.shape_decoder.norm_eps", 1e-6),
        _kv_u32("pixal3d.shape_decoder.n_levels", len(LEVEL_CHANNELS)),
        _kv_bool("pixal3d.shape_decoder.pred_subdiv", True),
        _kv_str("pixal3d.shape_decoder.conv_weight_source_layout", "out,kd,kh,kw,in"),
        _kv_str("pixal3d.shape_decoder.conv_weight_gguf_layout", "kernel_volume,out,in"),
        _kv_u32("pixal3d.shape_decoder.conv_kernel_size", 3),
        _kv_u32("pixal3d.shape_decoder.conv_kernel_volume", 27),
    ]
    for index, channels in enumerate(LEVEL_CHANNELS):
        metadata.append(_kv_u32(f"pixal3d.shape_decoder.model_channels.{index}", channels))
        metadata.append(_kv_u32(f"pixal3d.shape_decoder.num_blocks.{index}", NUM_BLOCKS[index]))

    tensors = []
    offsets = []
    offset = 0
    for key in sorted(state):
        dims, raw = _serial_tensor(key, state[key], ggml_type)
        name = "shape_decoder." + key
        offsets.append(offset)
        tensors.append((name, dims, raw))
        offset = _align(offset + len(raw))
    infos = bytearray()
    for (name, dims, raw), tensor_offset in zip(tensors, offsets):
        infos += _gguf_string(name) + struct.pack("<I", len(dims))
        for dim in dims:
            infos += struct.pack("<Q", int(dim))
        infos += struct.pack("<I", ggml_type) + struct.pack("<Q", tensor_offset)
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


def parse_fixture(text: str, include_upsample: bool = False) -> Dict[str, np.ndarray]:
    outputs: Dict[str, np.ndarray] = {}
    wanted = {
        "slat_decoder_input_coords", "slat_decoder_coords", "slat_decoder_output",
    }
    for level in range(N_SUBDIVISIONS):
        wanted.update({
            f"slat_decoder_subdiv_{level}_coords",
            f"slat_decoder_subdiv_{level}_output",
            f"slat_decoder_subdiv_{level}_active",
        })
    if include_upsample:
        for level in range(N_LEVELS):
            wanted.add(f"slat_decoder_upsample_coords_{level}")
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
        raise ValueError("fixture did not emit all SLat decoder outputs")
    return outputs


def fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()[:16]


def model_with_state(state: Dict[str, np.ndarray]) -> SparseUnetVaeDecoder:
    model, _ = make_model()
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            parameter.copy_(torch.from_numpy(state[key]))
    model.eval()
    return model


def reference_outputs(model: SparseUnetVaeDecoder,
                      coords: torch.Tensor,
                      latent: torch.Tensor,
                      include_upsample: bool = False) -> Dict[str, np.ndarray]:
    sparse_input = sp.SparseTensor(latent, coords)
    with torch.no_grad():
        reference, subdivisions = model(sparse_input, return_subs=True)
    values: Dict[str, np.ndarray] = {
        "slat_decoder_input_coords": coords.detach().cpu().numpy().astype(np.float32).reshape(-1),
        "slat_decoder_coords": reference.coords.detach().cpu().numpy().astype(np.float32).reshape(-1),
        "slat_decoder_output": reference.feats.detach().cpu().numpy().astype(np.float32).reshape(-1),
    }
    for level, subdivision in enumerate(subdivisions):
        logits = subdivision.feats.detach().cpu().numpy().astype(np.float32)
        values[f"slat_decoder_subdiv_{level}_coords"] = (
            subdivision.coords.detach().cpu().numpy().astype(np.float32).reshape(-1)
        )
        values[f"slat_decoder_subdiv_{level}_output"] = logits.reshape(-1)
        values[f"slat_decoder_subdiv_{level}_active"] = (logits > 0.0).astype(np.float32).reshape(-1)
    if include_upsample:
        for level in range(N_LEVELS):
            upsampled = model.upsample(sparse_input, upsample_times=level)
            values[f"slat_decoder_upsample_coords_{level}"] = (
                upsampled.detach().cpu().numpy().astype(np.float32).reshape(-1)
            )
    return values


def main() -> int:
    model, state = make_model()
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 1]], dtype=torch.int32)
    latent = (-0.21 + 0.037 * torch.arange(2 * LATENT_CHANNELS, dtype=torch.float32)).reshape(
        2, LATENT_CHANNELS
    )
    reference_values = reference_outputs(model, coords, latent, include_upsample=True)
    fixture = ROOT / "build" / "bin" / "pixal3d_slat_decoder_fixture"
    completed = subprocess.run([str(fixture), "--upsample"], check=True, capture_output=True, text=True)
    actuals = [("direct", parse_fixture(completed.stdout, include_upsample=True), reference_values)]
    loader_fixture = ROOT / "build" / "bin" / "pixal3d_slat_decoder_loader_fixture"
    with tempfile.TemporaryDirectory(prefix="pixal3d-slat-decoder-") as directory:
        pack = Path(directory) / "slat-decoder-loader-test.gguf"
        write_gguf(pack, state)
        loaded = subprocess.run([str(loader_fixture), str(pack), "--upsample"], check=True,
                                capture_output=True, text=True)
        actuals.append(("loader-f32", parse_fixture(loaded.stdout, include_upsample=True), reference_values))
        state_f16 = {key: value.astype(np.float16).astype(np.float32)
                     for key, value in state.items()}
        pack_f16 = Path(directory) / "slat-decoder-loader-test-f16.gguf"
        write_gguf(pack_f16, state_f16, GGML_TYPE_F16)
        loaded_f16 = subprocess.run([str(loader_fixture), str(pack_f16), "--upsample"], check=True,
                                    capture_output=True, text=True)
        loaded_f16_expected = reference_outputs(model_with_state(state_f16), coords, latent,
                                                include_upsample=True)
        actuals.append(("loader-f16", parse_fixture(loaded_f16.stdout, include_upsample=True),
                        loaded_f16_expected))
    report = []
    for source, actual, expected_values in actuals:
        for name, expected in expected_values.items():
            value = actual[name]
            if value.shape != expected.shape:
                raise AssertionError(f"{source}/{name} shape mismatch: {value.shape} != {expected.shape}")
            if not np.isfinite(value).all():
                raise AssertionError(f"{source}/{name} contains non-finite values")
            difference = value.astype(np.float64) - expected.astype(np.float64)
            max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
            rel_l2 = float(np.linalg.norm(difference) /
                           max(np.linalg.norm(expected.astype(np.float64)), 1e-12))
            report.append({
                "source": source,
                "name": name,
                "shape": list(expected.shape),
                "reference_fingerprint": fingerprint(expected),
                "cpp_fingerprint": fingerprint(value),
                "max_abs_error": max_abs,
                "relative_l2_error": rel_l2,
            })
            # The CPU fallback accumulates each sparse kernel dot product in a
            # scalar loop, while the reference backend uses vectorized torch
            # reductions.  The absolute error remains 2e-5; the relative metric
            # is correspondingly dominated by the small signed fixture outputs.
            if max_abs > 3e-5 or rel_l2 > 1e-4:
                raise AssertionError(f"{source}/{name} SLat decoder comparison exceeds tolerance")
    print(json.dumps(report, sort_keys=True))
    print("SLat decoder numerical comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
