#!/usr/bin/env python3
"""Pack external DINO/NAF feature maps into the P3DCOND format.

The encoder is intentionally outside this repository.  The manifest points
at NumPy arrays produced by the original Pixal3D Python pipeline (or another
encoder with the same feature contract): one global token array and one DINO
HWC map per stage, plus an optional NAF HWC map.  The C++ runtime projects the
2D maps at the requested sparse coordinates, so this file never stores a
multi-gigabyte 3D feature grid.

Manifest example::

  {
    "stages": [
      {"name": "ss", "global": "ss_global.npy",
       "dino_map": "ss_dino.npy", "image_resolution": 512},
      {"name": "shape_512", "global": "shape_global.npy",
       "dino_map": "shape_dino.npy", "naf_map": "shape_naf.npy",
       "image_resolution": 512}
    ]
  }

Arrays may be [tokens,C] / [1,tokens,C] for global and [H,W,C] / [1,H,W,C]
for maps.  ``--map-layout CHW`` can be used per map object when a producer
saved NAF output in [C,H,W] form; the default is HWC.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


MAGIC = b"P3DCOND\0"
VERSION = 1
STAGE_ORDER = ("ss", "shape_512", "shape_1024", "tex_1024")


class ConditionError(RuntimeError):
    pass


def _u32(value: int) -> bytes:
    if value < 0 or value > 0xFFFFFFFF:
        raise ConditionError(f"uint32 out of range: {value}")
    return struct.pack("<I", int(value))


def _load_array(path: Path, description: str) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ConditionError(f"cannot load {description} {path}: {exc}") from exc
    if not isinstance(value, np.ndarray):
        raise ConditionError(f"{description} must be a NumPy ndarray: {path}")
    if value.dtype.kind not in "fc":
        raise ConditionError(f"{description} must have a floating dtype: {path}")
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ConditionError(f"{description} contains non-finite values: {path}")
    return np.ascontiguousarray(value)


def _global_array(path: Path) -> np.ndarray:
    value = _load_array(path, "global token array")
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise ConditionError(f"global token array must have batch 1: {path}")
        value = value[0]
    if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ConditionError(f"global token array must be [tokens,channels]: {path}")
    return np.ascontiguousarray(value)


def _map_array(path: Path, layout: str) -> np.ndarray:
    value = _load_array(path, "projection feature map")
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ConditionError(f"projection feature map must have batch 1: {path}")
        value = value[0]
    if value.ndim != 3:
        raise ConditionError(f"projection feature map must be [H,W,C] or [C,H,W]: {path}")
    layout = layout.upper()
    if layout == "CHW":
        value = value.transpose(1, 2, 0)
    elif layout != "HWC":
        raise ConditionError(f"unsupported map layout {layout!r}; use HWC or CHW")
    if min(value.shape) <= 0:
        raise ConditionError(f"projection feature map dimensions must be positive: {path}")
    return np.ascontiguousarray(value, dtype=np.float32)


def _path(base: Path, value: Any, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConditionError(f"{description} path is missing")
    path = Path(value)
    return path if path.is_absolute() else base / path


def _map_spec(stage: Mapping[str, Any], key: str, base: Path,
              image_resolution: int) -> tuple[np.ndarray, int]:
    raw = stage.get(key)
    if raw is None:
        if key == "naf_map":
            return np.empty((0, 0, 0), dtype=np.float32), image_resolution
        raise ConditionError(f"stage {stage.get('name', '<unnamed>')} lacks {key}")
    if isinstance(raw, str):
        path = _path(base, raw, f"stage {stage.get('name', '<unnamed>')} {key}")
        layout = str(stage.get(f"{key}_layout", "HWC"))
        map_resolution = image_resolution
    elif isinstance(raw, Mapping):
        path = _path(base, raw.get("path"),
                     f"stage {stage.get('name', '<unnamed>')} {key}")
        layout = str(raw.get("layout", "HWC"))
        map_resolution = int(raw.get("image_resolution", image_resolution))
    else:
        raise ConditionError(f"stage {stage.get('name', '<unnamed>')} {key} must be a path or object")
    if map_resolution <= 0:
        raise ConditionError(f"{key} image_resolution must be positive")
    return _map_array(path, layout), map_resolution


def _stage_payload(stage: Mapping[str, Any], base: Path) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, int, int]:
    name = stage.get("name")
    if not isinstance(name, str) or not name:
        raise ConditionError("every condition stage needs a non-empty name")
    if name not in STAGE_ORDER:
        raise ConditionError(f"unsupported stage name {name!r}; expected one of {STAGE_ORDER}")
    image_resolution = int(stage.get("image_resolution", 0))
    if image_resolution <= 0:
        raise ConditionError(f"stage {name} image_resolution must be positive")
    global_array = _global_array(_path(base, stage.get("global"), f"stage {name} global"))
    dino_array, dino_resolution = _map_spec(stage, "dino_map", base, image_resolution)
    naf_array, naf_resolution = _map_spec(stage, "naf_map", base, image_resolution)
    if naf_array.size and naf_resolution <= 0:
        raise ConditionError(f"stage {name} NAF image_resolution must be positive")
    return name, global_array, dino_array, naf_array, dino_resolution, naf_resolution


def _write_u32(handle, value: int) -> None:
    handle.write(_u32(value))


def _write_array(handle, value: np.ndarray) -> None:
    handle.write(np.asarray(value, dtype="<f4", order="C").tobytes(order="C"))


def write_bundle(manifest_path: Path, output_path: Path) -> None:
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        raise ConditionError(f"cannot read manifest {manifest_path}: {exc}") from exc
    stages_raw = manifest.get("stages") if isinstance(manifest, Mapping) else None
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ConditionError("manifest must contain a non-empty 'stages' list")
    seen = set()
    payloads = []
    for raw in stages_raw:
        if not isinstance(raw, Mapping):
            raise ConditionError("each stage entry must be an object")
        payload = _stage_payload(raw, manifest_path.parent)
        if payload[0] in seen:
            raise ConditionError(f"duplicate stage: {payload[0]}")
        seen.add(payload[0])
        payloads.append(payload)
    payloads.sort(key=lambda item: STAGE_ORDER.index(item[0]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(MAGIC)
            _write_u32(handle, VERSION)
            _write_u32(handle, len(payloads))
            for name, global_array, dino_array, naf_array, dino_resolution, naf_resolution in payloads:
                encoded_name = name.encode("utf-8")
                _write_u32(handle, len(encoded_name))
                handle.write(encoded_name)
                _write_u32(handle, global_array.shape[0])
                _write_u32(handle, global_array.shape[1])
                _write_u32(handle, 2 if naf_array.size else 1)
                _write_array(handle, global_array)

                _write_u32(handle, 0)  # DINO role
                _write_u32(handle, dino_resolution)
                _write_u32(handle, dino_array.shape[0])
                _write_u32(handle, dino_array.shape[1])
                _write_u32(handle, dino_array.shape[2])
                _write_array(handle, dino_array)
                if naf_array.size:
                    _write_u32(handle, 1)  # NAF role
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
    parser.add_argument("manifest", type=Path, help="JSON feature-map manifest")
    parser.add_argument("output", type=Path, help="output .p3dcond path")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate arrays and print the planned stages without writing")
    args = parser.parse_args()
    try:
        if args.dry_run:
            manifest = json.loads(args.manifest.read_text())
            stages = manifest.get("stages", [])
            payloads = [_stage_payload(stage, args.manifest.parent) for stage in stages]
            print(json.dumps({"stages": [
                {"name": name, "global": list(global_array.shape),
                 "dino_map": list(dino_array.shape),
                 "naf_map": None if not naf_array.size else list(naf_array.shape),
                 "dino_image_resolution": dino_resolution,
                 "naf_image_resolution": None if not naf_array.size else naf_resolution}
                for name, global_array, dino_array, naf_array, dino_resolution, naf_resolution in payloads
            ]}, sort_keys=True))
        else:
            write_bundle(args.manifest, args.output)
            print(f"wrote {args.output}")
        return 0
    except (ConditionError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
