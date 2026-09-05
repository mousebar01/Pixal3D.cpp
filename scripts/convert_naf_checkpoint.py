#!/usr/bin/env python3
"""Convert the official valeoai/NAF torch checkpoint to Pixal3D GGUF.

The upstream release is a bare ``naf_release.pth`` state_dict rather than a
safetensors file with a JSON sidecar.  This adapter materializes one temporary
F32 safetensors file, then delegates inventory/layout/metadata verification to
the repository converter.  The intermediate file is retained by default so a
conversion can be resumed without loading the PyTorch checkpoint again.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def load_state(path: Path):
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("PyTorch and safetensors are required: {}".format(exc))
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        # The project Docker image supports weights_only; this fallback keeps
        # the adapter usable with an older torch image without changing the
        # serialized output contract.
        state = torch.load(str(path), map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError("NAF checkpoint must contain a state_dict mapping")
    converted = {}
    for name, value in state.items():
        if not isinstance(name, str) or not hasattr(value, "detach") or \
                not value.is_floating_point():
            raise RuntimeError("NAF checkpoint contains a non-floating tensor: {}".format(name))
        converted[name] = value.detach().cpu().float().contiguous()
    if not converted:
        raise RuntimeError("NAF checkpoint contains no tensors")
    return converted, save_file


def atomic_save(state, save_file, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        raise RuntimeError("refusing to overwrite {} (use --force)".format(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix="." + destination.name + ".", suffix=".part", dir=str(destination.parent))
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        save_file(state, str(temporary_path), metadata={"source": "valeoai/NAF"})
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="official naf_release.pth")
    parser.add_argument("--safetensors", type=Path,
                        help="intermediate F32 safetensors path (default: beside output)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Pixal3D NAF GGUF output")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        raise RuntimeError("checkpoint not found: {}".format(args.input))
    safe_path = args.safetensors or args.output.with_suffix(".safetensors")
    state, save_file = load_state(args.input)
    atomic_save(state, save_file, safe_path, args.force)
    converter = Path(__file__).with_name("convert_pixal3d_to_gguf.py")
    command = [sys.executable, str(converter), "--component", "naf",
               "--model", str(safe_path), "--output", str(args.output),
               "--ftype", "0"]
    if args.force:
        command.append("--force")
    subprocess.run(command, check=True)
    print("[naf-adapter] source : {}".format(args.input))
    print("[naf-adapter] F32 safetensors : {}".format(safe_path))
    print("[naf-adapter] GGUF : {}".format(args.output))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
