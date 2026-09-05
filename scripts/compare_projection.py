#!/usr/bin/env python3
"""Compare the C++ projection fixture with the original Python reference."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "ref" / "Pixal3D" / "pixal3d" / "trainers" / \
    "flow_matching" / "mixins" / "image_conditioned_proj.py"

# These tolerances are intentionally tight for float32 CPU projection.  A
# mismatch above them must be explained before the module is considered
# numerically aligned.
FEATURE_ABS_TOL = 2e-5
FEATURE_L2_TOL = 2e-6
MATRIX_ABS_TOL = 2e-5
MATRIX_L2_TOL = 2e-6


def load_reference() -> Dict[str, Any]:
    """Load only the projection definitions from the Python oracle.

    The complete reference module imports DINOv3 and image dependencies that
    are unrelated to this fixture.  Extracting these definitions by AST keeps
    the executed implementation verbatim while making the comparison test
    independent of the optional backbone packages.
    """
    tree = ast.parse(REFERENCE.read_text(encoding="utf-8"), filename=str(REFERENCE))
    wanted = {
        "project_points_to_image_batch",
        "sample_features",
        "ProjGrid",
        "ProjGridMV",
        "compute_relative_calc_mat",
    }
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    nodes.sort(key=lambda node: (0 if node.name == "project_points_to_image_batch" else
                                 1 if node.name == "sample_features" else
                                 2 if node.name == "ProjGrid" else
                                 3 if node.name == "ProjGridMV" else 4))
    # The class body contains annotations for optional visualization types
    # (PIL.Image, List, ...).  Strip annotations so only the numerical
    # implementation is evaluated; method bodies are unchanged.
    for node in ast.walk(ast.Module(body=nodes, type_ignores=[])):
        if isinstance(node, (ast.arg, ast.AnnAssign)):
            if isinstance(node, ast.arg):
                node.annotation = None
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None
    namespace: Dict[str, Any] = {
        "torch": torch,
        "nn": torch.nn,
        "F": torch.nn.functional,
    }
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(REFERENCE), "exec"), namespace)
    return namespace


def make_feature_map(base: float) -> np.ndarray:
    values = [base + 0.125 * i + 0.03125 * (i % 7)
              for i in range(5 * 4 * 3)]
    return np.asarray(values, dtype=np.float32).reshape(1, 5, 4, 3)


def parse_fixture(text: str) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        name, count = fields[0], int(fields[1])
        values = fields[2:]
        if len(values) != count:
            raise ValueError("{} has {} values, expected {}".format(name, len(values), count))
        dtype = np.uint8 if name.endswith("_valid") else np.float32
        result[name] = np.asarray(values, dtype=dtype)
    return result


def fingerprint(values: np.ndarray) -> str:
    data = np.asarray(values).astype(values.dtype.newbyteorder("<"), copy=False).tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def compare(name: str, reference: np.ndarray, actual: np.ndarray,
            abs_tol: float, l2_tol: float) -> None:
    if reference.shape != actual.shape:
        raise AssertionError("{} shape mismatch: {} != {}".format(
            name, actual.shape, reference.shape))
    if not np.isfinite(actual.astype(np.float64)).all():
        raise AssertionError("{} contains non-finite C++ values".format(name))
    diff = actual.astype(np.float64) - reference.astype(np.float64)
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    ref_norm = float(np.linalg.norm(reference.astype(np.float64)))
    l2 = float(np.linalg.norm(diff) / max(ref_norm, 1e-12))
    print(json.dumps({
        "name": name,
        "shape": list(reference.shape),
        "finite": bool(np.isfinite(reference.astype(np.float64)).all() and
                       np.isfinite(actual.astype(np.float64)).all()),
        "reference_fingerprint": fingerprint(reference),
        "cpp_fingerprint": fingerprint(actual),
        "max_abs_error": max_abs,
        "relative_l2_error": l2,
    }, sort_keys=True))
    if max_abs > abs_tol or l2 > l2_tol:
        raise AssertionError("{} exceeds tolerance: max_abs={} (tol {}), relative_l2={} (tol {})".format(
            name, max_abs, abs_tol, l2, l2_tol))


def reference_outputs() -> Dict[str, np.ndarray]:
    ns = load_reference()
    torch.set_grad_enabled(False)
    options = dict(grid_resolution=4, image_resolution=16)
    map_a = torch.from_numpy(make_feature_map(-0.5))
    map_b = torch.from_numpy(make_feature_map(0.75))
    angle = torch.tensor([0.8], dtype=torch.float32)
    distance = torch.tensor([2.25], dtype=torch.float32)
    mesh_scale = torch.tensor([1.5], dtype=torch.float32)

    grid = ns["ProjGrid"](**options)
    single = grid(map_a, angle, distance, mesh_scale, BHWC=True)

    translated = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    translated[:, 0, 3] = 0.5
    grid_points = grid.grid_points.unsqueeze(0) / mesh_scale[:, None, None] / 2
    front = grid.front_view_transform_matrix.unsqueeze(0).clone()
    front[:, 1, 3] = -distance
    _, _, single_valid = ns["project_points_to_image_batch"](
        grid_points, front, angle, options["image_resolution"])

    mv = ns["ProjGridMV"](**options)
    first = mv(map_a, angle, distance, mesh_scale, BHWC=True)
    second = mv(map_b, angle, distance, mesh_scale,
                transform_matrix=translated, BHWC=True)
    average = (first + second) / 2

    _, _, first_valid = ns["project_points_to_image_batch"](
        grid_points, front, angle, options["image_resolution"])
    _, _, second_valid = ns["project_points_to_image_batch"](
        grid_points, translated, angle, options["image_resolution"])

    identity = torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(2, -1, -1).clone()
    identity[1] = translated[0]
    relative = ns["compute_relative_calc_mat"](
        identity.unsqueeze(0),
        torch.tensor([[2.25, 2.25]], dtype=torch.float32),
        grid.front_view_transform_matrix,
    )

    return {
        "single_features": single.detach().cpu().numpy().reshape(-1),
        "single_valid": single_valid.detach().cpu().numpy().reshape(-1).astype(np.uint8),
        "average_features": average.detach().cpu().numpy().reshape(-1),
        "average_valid": (first_valid | second_valid).detach().cpu().numpy().reshape(-1).astype(np.uint8),
        "relative_matrices": relative.detach().cpu().numpy().reshape(-1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path,
                        default=ROOT / "build" / "bin" / "pixal3d_projection_fixture")
    args = parser.parse_args()
    completed = subprocess.run([str(args.fixture)], check=True,
                               capture_output=True, text=True)
    actual = parse_fixture(completed.stdout)
    reference = reference_outputs()
    for key in ("single_features", "average_features", "relative_matrices"):
        tolerance = (MATRIX_ABS_TOL, MATRIX_L2_TOL) if key == "relative_matrices" else \
            (FEATURE_ABS_TOL, FEATURE_L2_TOL)
        compare(key, reference[key], actual[key], *tolerance)
    for key in ("single_valid", "average_valid"):
        if not np.array_equal(reference[key], actual[key]):
            raise AssertionError("{} differs: {} mismatched entries".format(
                key, int(np.count_nonzero(reference[key] != actual[key]))))
        print(json.dumps({"name": key, "shape": list(reference[key].shape),
                          "sign_agreement": 1.0,
                          "reference_fingerprint": fingerprint(reference[key]),
                          "cpp_fingerprint": fingerprint(actual[key])}, sort_keys=True))
    print("projection numerical comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
