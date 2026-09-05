#!/usr/bin/env python3
"""Compare the compact Pixal3D SS decoder graph with its PyTorch oracle.

The fixture intentionally uses a small 2^3 latent grid and two decoder levels.
It exercises Conv3d, channel-wise LayerNorm, residual blocks, and the exact
3D pixel-shuffle ordering used by the released 16^3 -> 64^3 decoder.
"""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref" / "Pixal3D"))
from pixal3d.models.sparse_structure_vae import SparseStructureDecoder


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32
VT_UINT32 = 4
VT_FLOAT32 = 6
VT_BOOL = 7
VT_STRING = 8
ABS_TOL = 3e-4
L2_TOL = 2e-4


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def kv(key: str, kind: int, payload: bytes) -> bytes:
    return gguf_string(key) + struct.pack("<I", kind) + payload


def kv_u32(key: str, value: int) -> bytes:
    return kv(key, VT_UINT32, struct.pack("<I", int(value)))


def kv_f32(key: str, value: float) -> bytes:
    return kv(key, VT_FLOAT32, struct.pack("<f", float(value)))


def kv_str(key: str, value: str) -> bytes:
    return kv(key, VT_STRING, gguf_string(value))


def tensor_values(key: str, shape: Sequence[int]) -> np.ndarray:
    count = int(np.prod(shape))
    stable = sum((index + 1) * ord(char) for index, char in enumerate(key))
    values = ((np.arange(count, dtype=np.float32) + stable) % 29.0 - 14.0) * 0.018
    if key.endswith("norm1.weight") or key.endswith("norm2.weight") or key.endswith("out_layer.0.weight"):
        values = values + 1.0
    return values.reshape(tuple(shape)).astype(np.float32)


def make_model() -> Tuple[SparseStructureDecoder, Dict[str, np.ndarray], Dict[str, object]]:
    config: Dict[str, object] = dict(
        out_channels=1,
        latent_channels=2,
        num_res_blocks=1,
        channels=[4, 3],
        num_res_blocks_middle=1,
        norm_type="layer",
        use_fp16=False,
    )
    model = SparseStructureDecoder(**config)
    state: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for key, parameter in model.state_dict().items():
            data = tensor_values(key, tuple(parameter.shape))
            parameter.copy_(torch.from_numpy(data))
            state[key] = data
    model.eval()
    return model, state, config


def serial_tensor(key: str, value: np.ndarray) -> Tuple[Tuple[int, ...], bytes]:
    # Compact-v1 flattens [OC, IC, KD, KH, KW] to [OC*IC, KD, KH, KW].
    shape = tuple(int(dim) for dim in value.shape)
    if key.endswith(".weight") and len(shape) == 5:
        oc, ic, kd, kh, kw = shape
        value = value.reshape(oc * ic, kd, kh, kw)
        shape = tuple(int(dim) for dim in value.shape)
    return tuple(reversed(shape)), np.asarray(value, dtype="<f4").tobytes(order="C")


def write_gguf(path: Path, state: Dict[str, np.ndarray], config: Dict[str, object]) -> None:
    channels = [int(value) for value in config["channels"]]
    metadata = [
        kv_str("general.architecture", "pixal3d"),
        kv_str("general.name", "ss-decoder-numeric-test"),
        kv_u32("general.file_type", 0),
        kv_u32("general.alignment", ALIGNMENT),
        kv_u32("pixal3d.bundle_format", 1),
        kv_str("pixal3d.tensor_name_scheme", "compact-v1"),
        kv_str("pixal3d.bundle_kind", "component"),
        kv_u32("pixal3d.component_count", 1),
        kv_str("pixal3d.component.0", "ss_decoder"),
        kv_str("pixal3d.ss_decoder.model_class", "SparseStructureDecoder"),
        kv_str("pixal3d.ss_decoder.checkpoint", "ss-decoder-numeric-test"),
        kv_str("pixal3d.ss_decoder.variant", "fixture"),
        kv_u32("pixal3d.ss_decoder.resolution", 2),
        kv_u32("pixal3d.ss_decoder.out_channels", int(config["out_channels"])),
        kv_u32("pixal3d.ss_decoder.latent_channels", int(config["latent_channels"])),
        kv_u32("pixal3d.ss_decoder.num_res_blocks", int(config["num_res_blocks"])),
        kv_u32("pixal3d.ss_decoder.num_res_blocks_middle", int(config["num_res_blocks_middle"])),
        kv_u32("pixal3d.ss_decoder.n_levels", len(channels)),
        kv_str("pixal3d.ss_decoder.norm_type", "layer"),
        kv_f32("pixal3d.ss_decoder.norm_eps", 1e-5),
        kv_str("pixal3d.ss_decoder.conv_weight_source_layout", "out,in,kd,kh,kw"),
        kv_str("pixal3d.ss_decoder.conv_weight_gguf_layout", "out_mul_in,kd,kh,kw"),
    ]
    for index, channel_count in enumerate(channels):
        metadata.append(kv_u32(f"pixal3d.ss_decoder.channels.{index}", channel_count))

    tensors: List[Tuple[str, Tuple[int, ...], bytes]] = []
    for key in sorted(state):
        dims, raw = serial_tensor(key, state[key])
        tensors.append(("ss_decoder." + key, dims, raw))

    infos = bytearray()
    offsets: List[int] = []
    offset = 0
    for name, dims, raw in tensors:
        offsets.append(offset)
        infos += gguf_string(name) + struct.pack("<I", len(dims))
        for dim in dims:
            infos += struct.pack("<Q", int(dim))
        infos += struct.pack("<I", 0) + struct.pack("<Q", offset)
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
        for (_, _, raw), wanted in zip(tensors, offsets):
            if handle.tell() - data_offset != wanted:
                raise RuntimeError("internal GGUF offset failure")
            handle.write(raw)
            handle.write(b"\0" * (align(len(raw)) - len(raw)))


def parse_fixture(text: str) -> Tuple[np.ndarray, np.ndarray]:
    output = None
    coords = None
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == "ss_decoder_output":
            count = int(fields[1])
            values = np.asarray(fields[2:], dtype=np.float32)
            if values.size != count:
                raise ValueError("fixture output count mismatch")
            output = values
        elif fields and fields[0] == "ss_decoder_coords":
            count = int(fields[1])
            values = np.asarray(fields[2:], dtype=np.int32)
            if values.size != count:
                raise ValueError("fixture coordinate count mismatch")
            coords = values
    if output is None or coords is None:
        raise ValueError("fixture did not emit decoder output and coordinates")
    return output, coords


def fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()[:16]


def main() -> int:
    model, state, config = make_model()
    resolution = 2
    latent = (-0.21 + 0.037 * np.arange(int(config["latent_channels"]) * resolution ** 3,
                                         dtype=np.float32)).reshape(
                                             1, int(config["latent_channels"]), resolution,
                                             resolution, resolution)
    with torch.no_grad():
        reference = model(torch.from_numpy(latent))
    reference_np = reference.detach().numpy().astype(np.float32).reshape(-1)
    reference_coords = torch.argwhere(reference > 0)[:, [0, 2, 3, 4]]
    reference_coords = reference_coords.detach().cpu().numpy().astype(np.int32).reshape(-1)

    fixture = ROOT / "build" / "bin" / "pixal3d_ss_decoder_fixture"
    with tempfile.TemporaryDirectory(prefix="pixal3d-ss-decoder-") as directory:
        pack = Path(directory) / "ss-decoder-numeric-test.gguf"
        write_gguf(pack, state, config)
        completed = subprocess.run([str(fixture), str(pack), "--coords"], check=True,
                                   capture_output=True, text=True)
    actual, actual_coords = parse_fixture(completed.stdout)
    if actual.shape != reference_np.shape:
        raise AssertionError(f"shape mismatch: {actual.shape} != {reference_np.shape}")
    if not np.isfinite(actual).all():
        raise AssertionError("C++ output contains non-finite values")
    if not np.array_equal(actual_coords, reference_coords):
        raise AssertionError(
            f"occupancy coordinates mismatch: {actual_coords.tolist()} != "
            f"{reference_coords.tolist()}")
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
        "active_coordinate_count": int(actual_coords.size // 4),
    }, sort_keys=True))
    if max_abs > ABS_TOL or rel_l2 > L2_TOL:
        raise AssertionError("SS-decoder numerical comparison exceeds tolerance")
    print("ss-decoder numerical comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
