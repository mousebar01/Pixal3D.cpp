#!/usr/bin/env python3
"""Run a bounded real image-to-asset smoke test against the native CLI.

The fixture is deliberately generated at runtime, so no image asset is added
to the repository.  This test checks the complete image boundary (PNM loader,
DINO/NAF condition encoding, four-stage cascade, and the default GLB writer).
It is not a quality benchmark: two Euler steps, a diagnostic occupancy threshold, and a
bounded structure cap keep it usable on a development GPU.  Those diagnostic
settings are intentionally different from the production defaults.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_fixture(path: Path, size: int = 64) -> None:
    pixels = bytearray()
    for y in range(size):
        for x in range(size):
            # A deterministic RGB gradient exercises all channels without
            # relying on a model-specific image or an external decoder.
            pixels.extend((x * 255 // (size - 1),
                           y * 255 // (size - 1),
                           (x + y) * 255 // (2 * (size - 1))))
    path.write_bytes(b"P6\n%d %d\n255\n" % (size, size) + pixels)


def validate_obj(path: Path) -> tuple[int, int]:
    vertices = 0
    triangles = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v":
                if len(fields) != 4 or not all(math.isfinite(float(value))
                                                for value in fields[1:]):
                    raise RuntimeError("OBJ contains an invalid vertex")
                vertices += 1
            elif fields[0] == "f":
                if len(fields) != 4:
                    raise RuntimeError("OBJ contains a non-triangular face")
                triangles += 1
    if vertices == 0 or triangles == 0:
        raise RuntimeError("OBJ contains no mesh geometry")
    return vertices, triangles


def validate_glb(path: Path) -> None:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF" or data[4:8] != b"\x02\x00\x00\x00":
        raise RuntimeError("default output is not a glTF 2.0 binary asset")
    json_length = int.from_bytes(data[12:16], "little")
    if 20 + json_length > len(data):
        raise RuntimeError("GLB JSON chunk exceeds file size")
    json_chunk = data[20:20 + json_length].decode("utf-8").rstrip(" \x00")
    required = (
        '"baseColorTexture"',
        '"metallicRoughnessTexture"',
        '"mimeType":"image/png"',
        '"doubleSided":true',
        '"alphaMode":"OPAQUE"',
    )
    if any(field not in json_chunk for field in required):
        raise RuntimeError("GLB is missing the expected PBR material contract")
    if b"\x89PNG\r\n\x1a\n" not in data[20 + json_length:]:
        raise RuntimeError("GLB does not contain an embedded PNG texture")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/bin/pixal3d")
    parser.add_argument("--shared", type=Path,
                        default=ROOT / "build/weights/pixal3d-shared-f16.gguf")
    parser.add_argument("--flow", type=Path,
                        default=ROOT / "build/weights/pixal3d-base-flow-f16.gguf",
                        help="Use the experimental F16 flow pack for a GPU smoke test.")
    parser.add_argument("--dino", type=Path,
                        default=ROOT / "build/weights/dinov3-vitl16-pretrain-lvd1689m-f32.gguf")
    parser.add_argument("--naf", type=Path,
                        default=ROOT / "weights/NAF/naf_release-f32.gguf")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--vision-resolution", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-structure-points", type=int, default=4096)
    parser.add_argument(
        "--occupancy-threshold", type=float, default=-100.0,
        help="Diagnostic threshold; production uses the trained default 0.",
    )
    args = parser.parse_args()

    required = (args.binary, args.shared, args.flow, args.dino, args.naf)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print("missing required smoke-test input(s):", *missing, file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="pixal3d-run-image-") as directory:
        directory = Path(directory)
        image = directory / "gradient.ppm"
        default_output = directory / "output.glb"
        explicit_obj = directory / "gradient.obj"
        write_fixture(image)
        common = [
            "--resolution", str(args.resolution), "--steps", str(args.steps),
            "--vision-resolution", str(args.vision_resolution),
            "--max-structure-points", str(args.max_structure_points),
            "--occupancy-threshold", str(args.occupancy_threshold),
            "--texture-size", "64",
        ]
        default_command = [
            str(args.binary), "run-image", str(args.shared), str(args.flow),
            str(args.dino), str(args.naf), str(image), *common,
        ]
        print("$", " ".join(default_command), flush=True)
        result = subprocess.run(default_command, text=True, capture_output=True,
                                cwd=directory)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode != 0:
            print("default run-image smoke failed with exit code", result.returncode,
                  file=sys.stderr)
            return result.returncode
        if not default_output.is_file():
            print("default run-image did not create output.glb", file=sys.stderr)
            return 1
        validate_glb(default_output)
        print("validated default GLB: {} bytes".format(default_output.stat().st_size))

        obj_command = [
            str(args.binary), "run-image", str(args.shared), str(args.flow),
            str(args.dino), str(args.naf), str(image), str(explicit_obj), *common,
        ]
        print("$", " ".join(obj_command), flush=True)
        result = subprocess.run(obj_command, text=True, capture_output=True)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode != 0:
            print("explicit OBJ smoke failed with exit code", result.returncode,
                  file=sys.stderr)
            return result.returncode
        vertices, triangles = validate_obj(explicit_obj)
        print("validated explicit OBJ: vertices={} triangles={}".format(
            vertices, triangles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
