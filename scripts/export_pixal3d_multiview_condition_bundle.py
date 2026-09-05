#!/usr/bin/env python3
"""Pack per-view DINO/NAF features and camera poses into ``P3DMVCON``.

The input manifest keeps the vision encoder outside the ggml runtime while
preserving the reference ``DinoV3ProjMultiViewFeatureExtractor`` contract.
Each stage has a ``views`` list.  A view contains a global token ``.npy``, a
DINO feature map, an optional NAF map, and an absolute row-major 4x4 c2w
transform.  The C++ side computes ``F @ inv(C0) @ Ci`` and averages all views.

Example::

  {
    "stages": [{
      "name": "ss",
      "views": [
        {"global": "ss_v0_global.npy", "dino_map": "ss_v0_dino.npy",
         "image_resolution": 512,
         "camera": {"camera_angle_x": 0.86, "distance": 2.0,
                    "mesh_scale": 1.0,
                    "transform_matrix": [1,0,0,0, 0,0,-1,-2,
                                          0,1,0,0, 0,0,0,1]}},
        {"global": "ss_v1_global.npy", "dino_map": "ss_v1_dino.npy",
         "image_resolution": 512,
         "camera": {"camera_angle_x": 0.86, "distance": 2.0,
                    "mesh_scale": 1.0,
                    "transform_matrix": "cam_v1.npy"}}
      ]
    }]
  }

Global arrays are ``[tokens,C]`` or ``[1,tokens,C]``.  Maps are ``[H,W,C]``
or ``[C,H,W]`` with an optional per-map ``layout`` field.  All stages must
contain the same view count; fusion is currently the reference ``average``
mode because the ggml flow blocks consume one projected row per voxel.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from export_pixal3d_condition_bundle import (
    ConditionError,
    STAGE_ORDER,
    _global_array,
    _load_array,
    _map_spec,
    _path,
)


MAGIC = b"P3DMVCON"
VERSION = 1
MAX_VIEWS = 64


def _u32(value: int) -> bytes:
    if value < 0 or value > 0xFFFFFFFF:
        raise ConditionError(f"uint32 out of range: {value}")
    return struct.pack("<I", int(value))


def _f32(value: Any, description: str) -> bytes:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConditionError(f"{description} must be a finite number") from exc
    if not np.isfinite(number):
        raise ConditionError(f"{description} must be finite")
    return struct.pack("<f", number)


def _write_u32(handle, value: int) -> None:
    handle.write(_u32(value))


def _write_array(handle, value: np.ndarray) -> None:
    handle.write(np.asarray(value, dtype="<f4", order="C").tobytes(order="C"))


def _camera(view: Mapping[str, Any], base: Path) -> tuple[float, float, float, np.ndarray]:
    raw = view.get("camera")
    if not isinstance(raw, Mapping):
        raise ConditionError("every multi-view entry needs a camera object")
    try:
        fov = float(raw["camera_angle_x"])
        distance = float(raw["distance"])
        mesh_scale = float(raw["mesh_scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConditionError("camera needs camera_angle_x, distance, and mesh_scale") from exc
    if not (np.isfinite(fov) and 0.0 < fov < np.pi):
        raise ConditionError("camera_angle_x must be in (0, pi)")
    if not (np.isfinite(distance) and distance > 0.0):
        raise ConditionError("camera distance must be positive")
    if not (np.isfinite(mesh_scale) and mesh_scale > 0.0):
        raise ConditionError("camera mesh_scale must be positive")
    transform = raw.get("transform_matrix")
    if isinstance(transform, str):
        transform = _load_array(_path(base, transform, "camera transform"),
                                "camera transform")
    try:
        matrix = np.asarray(transform, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ConditionError("camera transform_matrix must be a 4x4 array or .npy path") from exc
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ConditionError("camera transform_matrix must be finite 4x4")
    return fov, distance, mesh_scale, np.ascontiguousarray(matrix.reshape(-1))


def _view_payload(view: Mapping[str, Any], base: Path):
    if not isinstance(view, Mapping):
        raise ConditionError("each multi-view entry must be an object")
    image_resolution = int(view.get("image_resolution", 0))
    if image_resolution <= 0:
        raise ConditionError("view image_resolution must be positive")
    global_array = _global_array(
        _path(base, view.get("global"), "multi-view global token array"))
    dino_array, dino_resolution = _map_spec(view, "dino_map", base, image_resolution)
    naf_array, naf_resolution = _map_spec(view, "naf_map", base, image_resolution)
    fov, distance, mesh_scale, transform = _camera(view, base)
    return (global_array, dino_array, naf_array, dino_resolution, naf_resolution,
            fov, distance, mesh_scale, transform)


def _load_manifest(path: Path):
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConditionError(f"cannot read manifest {path}: {exc}") from exc
    stages = manifest.get("stages") if isinstance(manifest, Mapping) else None
    if not isinstance(stages, list) or not stages:
        raise ConditionError("manifest must contain a non-empty 'stages' list")
    payloads = []
    seen = set()
    expected_views = None
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise ConditionError("each stage entry must be an object")
        name = stage.get("name")
        if not isinstance(name, str) or name not in STAGE_ORDER:
            raise ConditionError(f"unsupported stage name {name!r}; expected one of {STAGE_ORDER}")
        if name in seen:
            raise ConditionError(f"duplicate stage: {name}")
        seen.add(name)
        views = stage.get("views")
        if not isinstance(views, list) or not 1 <= len(views) <= MAX_VIEWS:
            raise ConditionError(f"stage {name} needs between one and {MAX_VIEWS} views")
        if expected_views is None:
            expected_views = len(views)
        elif len(views) != expected_views:
            raise ConditionError("all stages must contain the same number of views")
        payloads.append((name, [_view_payload(view, path.parent) for view in views]))
    payloads.sort(key=lambda item: STAGE_ORDER.index(item[0]))

    # Validate the channel contract before writing anything.  Different
    # stages may legitimately use different map widths (for example, DINO
    # only versus DINO+NAF), but all views within one stage must agree.
    for name, views in payloads:
        first_global = views[0][0]
        first_dino = views[0][1]
        first_naf_array = views[0][2]
        first_naf = bool(first_naf_array.size)
        for payload in views:
            global_array, dino_array, naf_array = payload[:3]
            if global_array.shape != first_global.shape:
                raise ConditionError(f"global shape differs in stage {name}")
            if dino_array.shape != first_dino.shape:
                raise ConditionError(f"DINO channel count differs in stage {name}")
            if bool(naf_array.size) != first_naf:
                raise ConditionError("all multi-view entries must consistently include NAF maps")
            if first_naf and naf_array.shape != first_naf_array.shape:
                raise ConditionError("NAF channel count differs between views")
    return payloads


def write_bundle(manifest_path: Path, output_path: Path) -> None:
    payloads = _load_manifest(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(MAGIC)
            _write_u32(handle, VERSION)
            _write_u32(handle, len(payloads))
            for name, views in payloads:
                encoded_name = name.encode("utf-8")
                _write_u32(handle, len(encoded_name))
                handle.write(encoded_name)
                _write_u32(handle, len(views))
                for (global_array, dino_array, naf_array, dino_resolution, naf_resolution,
                     fov, distance, mesh_scale, transform) in views:
                    handle.write(_f32(fov, "camera_angle_x"))
                    handle.write(_f32(distance, "camera distance"))
                    handle.write(_f32(mesh_scale, "camera mesh_scale"))
                    _write_u32(handle, 1)  # absolute c2w is required by P3DMVCON
                    for value in transform:
                        handle.write(_f32(value, "camera transform"))
                    _write_u32(handle, global_array.shape[0])
                    _write_u32(handle, global_array.shape[1])
                    _write_u32(handle, 2 if naf_array.size else 1)
                    _write_array(handle, global_array)
                    _write_u32(handle, 0)
                    _write_u32(handle, dino_resolution)
                    _write_u32(handle, dino_array.shape[0])
                    _write_u32(handle, dino_array.shape[1])
                    _write_u32(handle, dino_array.shape[2])
                    _write_array(handle, dino_array)
                    if naf_array.size:
                        _write_u32(handle, 1)
                        _write_u32(handle, naf_resolution)
                        _write_u32(handle, naf_array.shape[0])
                        _write_u32(handle, naf_array.shape[1])
                        _write_u32(handle, naf_array.shape[2])
                        _write_array(handle, naf_array)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        payloads = _load_manifest(args.manifest)
        if args.dry_run:
            print(json.dumps({"stages": [
                {"name": name, "views": [
                    {"global": list(item[0].shape),
                     "dino_map": list(item[1].shape),
                     "naf_map": None if not item[2].size else list(item[2].shape),
                     "dino_image_resolution": item[3],
                     "naf_image_resolution": None if not item[2].size else item[4]}
                    for item in views]
                } for name, views in payloads]}, sort_keys=True))
        else:
            write_bundle(args.manifest, args.output)
            print(f"wrote {args.output}")
        return 0
    except (ConditionError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
