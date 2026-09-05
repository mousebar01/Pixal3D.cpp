#!/usr/bin/env python3
"""Generate a deterministic full-size SS-flow Python reference fixture."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

# Use PyTorch's reference SDPA implementation.  The optional flash-attn
# package is not part of every legacy image, while the C++ graph implements
# the same scaled dot-product operation.
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_CONV_BACKEND", "none")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref" / "Pixal3D"))
from pixal3d.models.sparse_structure_flow import SparseStructureFlowModel


MAGIC = b"PXSSREF1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "weights" / "Pixal3D" / "ckpts" /
                    "ss_flow_img_dit_1_3B_64_bf16"),
        help="checkpoint stem without .json/.safetensors",
    )
    parser.add_argument("--out", default=str(ROOT / "build" / "ss_flow_ref.bin"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--t", type=float, default=500.0)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    with checkpoint.with_suffix(".json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    model_args = dict(config["args"])
    model_args["dtype"] = "float32"
    model = SparseStructureFlowModel(**model_args)
    model.convert_to(torch.float32)
    model.eval()

    from safetensors.torch import load_file
    # rope_phases is a computed complex buffer.  The C++ graph recreates it
    # from the resolution/frequency metadata, so do not load the serialized
    # C64 copy from the checkpoint into the model.
    state = {key: value.float() for key, value in load_file(
        str(checkpoint.with_suffix(".safetensors"))).items()
             if key != "rope_phases"}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or missing != ["rope_phases"]:
        raise RuntimeError(f"unexpected checkpoint keys: missing={missing}, unexpected={unexpected}")

    resolution = int(model_args["resolution"])
    in_channels = int(model_args["in_channels"])
    out_channels = int(model_args["out_channels"])
    cond_channels = int(model_args["cond_channels"])
    proj_channels = int(model_args.get("proj_in_channels", cond_channels))
    points = resolution ** 3
    cond_tokens = 1029

    rng = np.random.default_rng(args.seed)
    x_np = rng.standard_normal((1, in_channels, resolution, resolution, resolution),
                               dtype=np.float32)
    cond_np = rng.standard_normal((1, cond_tokens, cond_channels), dtype=np.float32)
    projected_np = rng.standard_normal((1, points, proj_channels), dtype=np.float32)
    x = torch.from_numpy(x_np)
    cond = torch.from_numpy(cond_np)
    projected = torch.from_numpy(projected_np)
    timestep = torch.tensor([args.t], dtype=torch.float32)
    with torch.no_grad():
        output = model(x, timestep, (cond, projected))
    output_np = output.detach().cpu().numpy().astype(np.float32)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<6i", resolution, in_channels, out_channels,
                                 cond_tokens, cond_channels, proj_channels))
        handle.write(struct.pack("<f", float(args.t)))
        handle.write(x_np.reshape(-1).astype("<f4").tobytes())
        handle.write(cond_np.reshape(-1).astype("<f4").tobytes())
        handle.write(projected_np.reshape(-1).astype("<f4").tobytes())
        handle.write(output_np.reshape(-1).astype("<f4").tobytes())
    print(json.dumps({
        "checkpoint": str(checkpoint),
        "output": str(output_path),
        "resolution": resolution,
        "input_shape": list(x_np.shape),
        "cond_shape": list(cond_np.shape),
        "projected_shape": list(projected_np.shape),
        "output_shape": list(output_np.shape),
        "output_min": float(output_np.min()),
        "output_max": float(output_np.max()),
        "output_l2": float(np.linalg.norm(output_np.astype(np.float64))),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
