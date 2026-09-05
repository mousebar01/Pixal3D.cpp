#!/usr/bin/env python3
"""Compare the CPU sparse primitives with Pixal3D's flex_gemm reference."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_fixture(text: str) -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] not in ("sparse_linear_output", "sparse_conv_output",
                                           "sparse_norm_output", "sparse_downsample_coords",
                                           "sparse_downsample_output", "sparse_attention_output",
                                           "sparse_rope_output", "sparse_rms_output",
                                           "sparse_cross_output",
                                           "sparse_channel_to_spatial_coords",
                                           "sparse_channel_to_spatial_output"):
            continue
        count = int(fields[1])
        values = np.asarray(fields[2:], dtype=np.float32)
        if values.size != count:
            raise ValueError(f"{fields[0]} output count mismatch")
        outputs[fields[0]] = values
    if len(outputs) != 11:
        raise ValueError("fixture did not emit all sparse outputs")
    return outputs


def fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()[:16]


def main() -> int:
    in_channels, out_channels = 3, 4
    coords = np.asarray([
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 1, 1], [0, 2, 2, 2],
        [1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1], [1, 2, 1, 2], [1, 3, 3, 3],
    ], dtype=np.int32)
    points = coords.shape[0]
    features = (-0.31 + 0.047 * np.arange(points * in_channels,
                                             dtype=np.float32)).reshape(points, in_channels)
    linear_weight = (-0.11 + 0.013 * np.arange(out_channels * in_channels,
                                                 dtype=np.float32)).reshape(out_channels, in_channels)
    linear_bias = np.asarray([-0.07, 0.02, 0.11, 0.20], dtype=np.float32)
    conv_serial = (-0.021 + 0.0017 * np.arange(27 * out_channels * in_channels,
                                                 dtype=np.float32)).reshape(27, out_channels, in_channels)
    conv_bias = np.asarray([-0.03, 0.01, 0.05, 0.09], dtype=np.float32)

    linear_reference = features @ linear_weight.T + linear_bias
    # Mirror the reference flex_gemm submanifold contract.  The optional
    # flex_gemm wheel is not installed in every legacy image, so this oracle
    # deliberately spells out its documented active-coordinate accumulation.
    index = {tuple(coord): point for point, coord in enumerate(coords.tolist())}
    conv_reference = np.zeros((points, out_channels), dtype=np.float32)
    for point, center in enumerate(coords):
        for kd in range(3):
            for kh in range(3):
                for kw in range(3):
                    neighbor = (int(center[0]), int(center[1] + kd - 1),
                                int(center[2] + kh - 1), int(center[3] + kw - 1))
                    neighbor_index = index.get(neighbor)
                    if neighbor_index is None:
                        continue
                    kernel = conv_serial[kd * 9 + kh * 3 + kw]
                    conv_reference[point] += kernel @ features[neighbor_index]
        conv_reference[point] += conv_bias

    norm_gamma = np.asarray([0.8, 1.1, 1.4], dtype=np.float32)
    norm_beta = np.asarray([-0.2, 0.03, 0.17], dtype=np.float32)
    mean = features.mean(axis=1, keepdims=True)
    variance = ((features - mean) ** 2).mean(axis=1, keepdims=True)
    norm_reference = (features - mean) / np.sqrt(variance + np.float32(1e-5))
    norm_reference = norm_reference * norm_gamma + norm_beta

    groups = {}
    for point, coord in enumerate(coords):
        key = (int(coord[0]), int(coord[1] // 2), int(coord[2] // 2), int(coord[3] // 2))
        if key not in groups:
            groups[key] = []
        groups[key].append(point)
    ordered_groups = sorted(groups)
    downsample_coords = np.asarray(ordered_groups, dtype=np.float32).reshape(-1)
    downsample_reference = np.asarray([
        features[groups[key]].mean(axis=0) for key in ordered_groups
    ], dtype=np.float32).reshape(-1)

    attention_channels, attention_heads, attention_head_dim = 12, 2, 6
    query_features = -0.23 + 0.017 * np.arange(points * attention_channels, dtype=np.float32)
    query_features = query_features.reshape(points, attention_channels)
    key_features = 0.17 - 0.021 * np.arange(points * attention_channels, dtype=np.float32)
    key_features = key_features.reshape(points, attention_channels)
    value_features = -0.13 + 0.031 * np.arange(points * attention_channels, dtype=np.float32)
    value_features = value_features.reshape(points, attention_channels)
    attention_reference = np.zeros_like(value_features)
    for point in range(points):
        batch_mask = coords[:, 0] == coords[point, 0]
        for head in range(attention_heads):
            q = query_features[point, head * attention_head_dim:(head + 1) * attention_head_dim]
            k = key_features[batch_mask, head * attention_head_dim:(head + 1) * attention_head_dim]
            scores = (k @ q) * np.float32(0.37)
            scores = np.exp(scores - scores.max())
            scores /= scores.sum()
            attention_reference[point, head * attention_head_dim:(head + 1) * attention_head_dim] = (
                scores @ value_features[batch_mask, head * attention_head_dim:(head + 1) * attention_head_dim])

    rope_reference = np.zeros_like(query_features)
    pair_count = attention_head_dim // 2
    frequency_dim = pair_count // 3
    frequencies = np.asarray([
        1.0 / (10000.0 ** (index / frequency_dim))
        for index in range(frequency_dim)
    ], dtype=np.float32)
    for point, coord in enumerate(coords):
        for head in range(attention_heads):
            head_offset = head * attention_head_dim
            for pair in range(pair_count):
                phase = 0.0
                if pair < 3 * frequency_dim:
                    phase = float(coord[1 + pair // frequency_dim]) * frequencies[pair % frequency_dim]
                cosine, sine = np.cos(phase), np.sin(phase)
                offset = head_offset + pair * 2
                even, odd = query_features[point, offset:offset + 2]
                rope_reference[point, offset] = even * cosine - odd * sine
                rope_reference[point, offset + 1] = even * sine + odd * cosine

    rms_gamma = 0.7 + 0.013 * np.arange(attention_channels, dtype=np.float32)
    rms_reference = np.zeros_like(query_features)
    for point in range(points):
        for head in range(attention_heads):
            offset = head * attention_head_dim
            values = query_features[point, offset:offset + attention_head_dim]
            inverse = 1.0 / max(float(np.sqrt(np.sum(values * values))), 1e-12)
            rms_reference[point, offset:offset + attention_head_dim] = (
                values * inverse * np.sqrt(attention_head_dim) *
                rms_gamma[offset:offset + attention_head_dim])

    context_key = 0.09 - 0.014 * np.arange(5 * attention_channels, dtype=np.float32)
    context_key = context_key.reshape(5, attention_channels)
    context_value = -0.18 + 0.022 * np.arange(5 * attention_channels, dtype=np.float32)
    context_value = context_value.reshape(5, attention_channels)
    context_offsets = [0, 3, 5]
    cross_reference = np.zeros_like(query_features)
    for point in range(points):
        batch = int(coords[point, 0])
        key_begin, key_end = context_offsets[batch], context_offsets[batch + 1]
        for head in range(attention_heads):
            offset = head * attention_head_dim
            q = query_features[point, offset:offset + attention_head_dim]
            k = context_key[key_begin:key_end, offset:offset + attention_head_dim]
            scores = (k @ q) * np.float32(0.29)
            scores = np.exp(scores - scores.max())
            scores /= scores.sum()
            cross_reference[point, offset:offset + attention_head_dim] = (
                scores @ context_value[key_begin:key_end, offset:offset + attention_head_dim])

    packed_coords = np.asarray([
        [0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0]
    ], dtype=np.int32)
    packed_features = (-0.17 + 0.011 * np.arange(3 * 16, dtype=np.float32)).reshape(3, 16)
    subdivision = np.asarray([
        [1, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 1, 0],
    ], dtype=np.float32)
    spatial_coords = []
    spatial_features = []
    for point, coord in enumerate(packed_coords):
        for child in range(8):
            if subdivision[point, child] <= 0:
                continue
            child_xyz = np.asarray([child % 2, (child // 2) % 2, child // 4], dtype=np.int32)
            spatial_coords.append([int(coord[0]), *(2 * coord[1:] + child_xyz)])
            spatial_features.append(packed_features[point, child * 2:(child + 1) * 2])
    spatial_coords_reference = np.asarray(spatial_coords, dtype=np.float32).reshape(-1)
    spatial_reference = np.asarray(spatial_features, dtype=np.float32).reshape(-1)

    fixture = ROOT / "build" / "bin" / "pixal3d_sparse_fixture"
    completed = subprocess.run([str(fixture)], check=True, capture_output=True, text=True)
    actual = parse_fixture(completed.stdout)
    references = {
        "sparse_linear_output": linear_reference.reshape(-1),
        "sparse_conv_output": conv_reference.reshape(-1),
        "sparse_norm_output": norm_reference.reshape(-1),
        "sparse_downsample_coords": downsample_coords,
        "sparse_downsample_output": downsample_reference,
        "sparse_attention_output": attention_reference.reshape(-1),
        "sparse_rope_output": rope_reference.reshape(-1),
        "sparse_rms_output": rms_reference.reshape(-1),
        "sparse_cross_output": cross_reference.reshape(-1),
        "sparse_channel_to_spatial_coords": spatial_coords_reference,
        "sparse_channel_to_spatial_output": spatial_reference,
    }
    report = []
    for name, reference in references.items():
        output = actual[name]
        if output.shape != reference.shape:
            raise AssertionError(f"{name} shape mismatch: {output.shape} != {reference.shape}")
        if not np.isfinite(output).all():
            raise AssertionError(f"{name} contains non-finite values")
        difference = output.astype(np.float64) - reference.astype(np.float64)
        max_abs = float(np.max(np.abs(difference)))
        rel_l2 = float(np.linalg.norm(difference) /
                       max(np.linalg.norm(reference.astype(np.float64)), 1e-12))
        report.append({
            "name": name,
            "shape": list(reference.shape),
            "reference_fingerprint": fingerprint(reference),
            "cpp_fingerprint": fingerprint(output),
            "max_abs_error": max_abs,
            "relative_l2_error": rel_l2,
        })
        if max_abs > 2e-5 or rel_l2 > 2e-6:
            raise AssertionError(f"{name} sparse numerical comparison exceeds tolerance")
    print(json.dumps(report, sort_keys=True))
    print("sparse numerical comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
