#!/usr/bin/env python3
"""Compare the CPU Flexible Dual Grid extraction with its reference contract."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_fixture(text: str):
    outputs = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or not (fields[0].startswith("dual_grid_") or
                              fields[0].startswith("flexi_decode_")):
            continue
        count = int(fields[1])
        values = np.asarray(fields[2:], dtype=np.float32 if "vertices" in fields[0] else np.int32)
        if values.size != count:
            raise ValueError(f"{fields[0]} output count mismatch")
        outputs[fields[0]] = values
    wanted = {"dual_grid_vertices", "dual_grid_weighted_faces", "dual_grid_auto_faces",
              "flexi_decode_vertices", "flexi_decode_faces"}
    if set(outputs) != wanted:
        raise ValueError(f"fixture outputs {set(outputs)} != {wanted}")
    return outputs


def reference():
    coords = np.asarray([(x, y, z) for x in range(2) for y in range(2) for z in range(2)],
                        dtype=np.int32)
    indices = {tuple(coord): index for index, coord in enumerate(coords.tolist())}
    count = len(coords)
    base = np.float32(0.11) + np.float32(0.037) * np.arange(1, count + 1, dtype=np.float32)
    dual = np.stack([base, base + np.float32(0.07), base + np.float32(0.13)], axis=1)
    vertices = (coords.astype(np.float32) + dual) - np.float32(1.0)
    weights = np.float32(0.2) + np.float32(0.17) * np.arange(1, count + 1, dtype=np.float32)
    offsets = np.asarray([
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ], dtype=np.int32)
    quads = []
    for index, coord in enumerate(coords):
        for axis in range(3):
            neighbors = [tuple(coord + offsets[axis, corner]) for corner in range(4)]
            if all(neighbor in indices for neighbor in neighbors):
                quads.append([indices[neighbor] for neighbor in neighbors])

    weighted_faces = []
    auto_faces = []
    for quad in quads:
        use_weighted_split_1 = weights[quad[0]] * weights[quad[2]] > weights[quad[1]] * weights[quad[3]]
        if use_weighted_split_1:
            weighted_faces.extend([quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]])
        else:
            weighted_faces.extend([quad[0], quad[1], quad[3], quad[3], quad[1], quad[2]])

        a, b, c, d = [vertices[index] for index in quad]
        align_1 = abs(float(np.dot(np.cross(b - a, c - a), np.cross(c - a, d - a))))
        align_2 = abs(float(np.dot(np.cross(b - a, d - a), np.cross(d - a, c - a))))
        if align_1 > align_2:
            auto_faces.extend([quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]])
        else:
            auto_faces.extend([quad[0], quad[1], quad[3], quad[3], quad[1], quad[2]])
    decode_logits = np.stack([
        np.float32(-0.31) + np.float32(0.09) * np.arange(count, dtype=np.float32),
        np.float32(0.17) - np.float32(0.06) * np.arange(count, dtype=np.float32),
        np.float32(-0.05) + np.float32(0.04) * np.arange(count, dtype=np.float32),
    ], axis=1)
    decoded_dual = np.float32(2.0) / (np.float32(1.0) + np.exp(-decode_logits)) - np.float32(0.5)
    decoded_vertices = (coords.astype(np.float32) + decoded_dual) * np.float32(0.5) - np.float32(0.5)
    decoded_weights = np.logaddexp(
        np.float32(0.0), np.float32(-0.4) + np.float32(0.13) * np.arange(count, dtype=np.float32))
    decoded_faces = []
    for quad in quads:
        if decoded_weights[quad[0]] * decoded_weights[quad[2]] > decoded_weights[quad[1]] * decoded_weights[quad[3]]:
            decoded_faces.extend([quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]])
        else:
            decoded_faces.extend([quad[0], quad[1], quad[3], quad[3], quad[1], quad[2]])
    return {
        "dual_grid_vertices": vertices.reshape(-1),
        "dual_grid_weighted_faces": np.asarray(weighted_faces, dtype=np.int32),
        "dual_grid_auto_faces": np.asarray(auto_faces, dtype=np.int32),
        "flexi_decode_vertices": decoded_vertices.reshape(-1),
        "flexi_decode_faces": np.asarray(decoded_faces, dtype=np.int32),
    }


def main() -> int:
    fixture = ROOT / "build" / "bin" / "pixal3d_dual_grid_fixture"
    actual = parse_fixture(subprocess.run([str(fixture)], check=True,
                                          capture_output=True, text=True).stdout)
    expected = reference()
    report = []
    for name, value in expected.items():
        got = actual[name]
        if got.shape != value.shape:
            raise AssertionError(f"{name} shape mismatch: {got.shape} != {value.shape}")
        if np.issubdtype(value.dtype, np.floating):
            delta = got.astype(np.float64) - value.astype(np.float64)
            max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
            if max_abs > 2e-6:
                raise AssertionError(f"{name} max error {max_abs} exceeds tolerance")
        else:
            max_abs = float(np.max(np.abs(got.astype(np.int64) - value.astype(np.int64)))) if got.size else 0.0
            if max_abs != 0.0:
                raise AssertionError(f"{name} face indices differ")
        report.append({"name": name, "count": int(value.size), "max_abs_error": max_abs})
    print(json.dumps(report, sort_keys=True))
    print("Flexible Dual Grid numerical comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
