#!/usr/bin/env python3
"""Extract per-view DINO/NAF features for a P3DMVCON manifest.

The ggml runtime keeps the vision encoder outside its graph: the C++ side
consumes pre-projection per-view features (DINO global tokens, the DINO patch
map, and the optional NAF upsampled map) plus each view's absolute c2w pose,
and re-implements the reference projection/fusion itself.  This script plays
the role of DinoV3ProjMultiViewFeatureExtractor.forward's extraction half for a
transform view directory, producing the manifest that
``export_pixal3d_multiview_condition_bundle.py`` packs into P3DMVCON.

Views are read exactly like the reference ``inference_mv.load_views``: RGBA
frames with alpha (LANCZOS resize, alpha premultiply, no crop), camera_angle_x
from the frame or the meta block, distance derived from the pose translation
norm, and mesh_scale from the meta block.

Example::

  python scripts/extract_pixal3d_multiview_features.py \
      ref/Pixal3D/assets/mv_images/example build/mv-example \
      --device cuda

The stage configs mirror inference_mv.IMAGE_COND_CONFIGS: ss at 512 without
NAF, shape_512 at 512 with NAF 512, shape_1024 at 1024 with NAF 512, and
tex_1024 at 1024 with NAF 1024.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

STAGE_CONFIGS = {
    "ss": {"image_size": 512, "grid_resolution": 16},
    "shape_512": {"image_size": 512, "grid_resolution": 32,
                  "use_naf_upsample": True, "naf_target_size": 512},
    "shape_1024": {"image_size": 1024, "grid_resolution": 64,
                   "use_naf_upsample": True, "naf_target_size": 512},
    "tex_1024": {"image_size": 1024, "grid_resolution": 64,
                 "use_naf_upsample": True, "naf_target_size": 1024},
}
DINO_MODEL_NAME = "camenduru/dinov3-vitl16-pretrain-lvd1689m"


def load_rgba(path: Path) -> Image.Image:
    """Load one view, requiring an alpha channel (see load_views in the reference)."""
    image = Image.open(path)
    if image.mode != "RGBA":
        raise SystemExit(
            f"{path} has no alpha channel; this tool does not mat, matte the view first")
    return image


def to_cond_tensor(image: Image.Image, image_size: int) -> np.ndarray:
    """Reference to_cond_tensor: LANCZOS resize then alpha premultiply, [3,H,W] float32."""
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    alpha = np.asarray(image.getchannel(3), dtype=np.float32) / 255.0
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
    return rgb * alpha[None, :, :]


def fingerprint(array: np.ndarray) -> str:
    digest = hashlib.sha256(np.ascontiguousarray(array, dtype="<f4").tobytes())
    return digest.hexdigest()[:16]


def install_chunked_naf_attention() -> None:
    """Bound NAF cross-attention's peak memory by evaluating heads separately.

    NA2d neighborhood attention is independent per head, so calling the same
    reference kernels one head at a time keeps intermediate buffers at
    [1, 1, H, W, ...] instead of [1, heads, H, W, ...].  The cross-attention
    forward is replicated per head as well because the reference _resize
    materializes the full-channel k/v at the output resolution first, which
    alone exceeds a 24 GiB GPU at the 1024 tex stage.
    """
    import importlib

    import torch
    import torch.nn.functional as F
    from einops import rearrange
    from natten.functional import na2d_av, na2d_qk

    def chunked_cross_forward(self, q, k, v, image=None, return_weights=False, **kwargs):
        if return_weights:
            raise ValueError("return_weights is not supported by the chunked path")
        hq, wq = q.shape[-2:]
        hk, wk = k.shape[-2:]
        dilation = (hq // hk, wq // wk)
        q = rearrange(q, "b (n d) h w -> b n h w d", n=self.num_heads)
        d_k = k.shape[1] // self.num_heads
        d_v = v.shape[1] // self.num_heads
        outs = []
        for head in range(self.num_heads):
            kh = F.interpolate(k[:, head * d_k:(head + 1) * d_k], size=(hq, wq),
                               mode="nearest-exact")
            vh = F.interpolate(v[:, head * d_v:(head + 1) * d_v], size=(hq, wq),
                               mode="nearest-exact")
            kh = rearrange(kh, "b (n d) h w -> b n h w d", n=1).to(q.dtype)
            vh = rearrange(vh, "b (n d) h w -> b n h w d", n=1).to(q.dtype)
            scores = na2d_qk(q[:, head:head + 1].contiguous(), kh,
                             kernel_size=self.kernel_size,
                             dilation=dilation) * self.scale
            weights = scores.softmax(dim=-1)
            del scores
            outs.append(na2d_av(weights, vh, kernel_size=self.kernel_size,
                                dilation=dilation))
            del weights, kh, vh
        out = torch.cat(outs, dim=1)
        del outs
        return rearrange(out, "b n h w d -> b (n d) h w")

    attentions = importlib.import_module("src.layers.attentions")
    attentions.CrossAttention.forward = chunked_cross_forward


def extract_features(model, image):
    """Reference DinoV3ProjFeatureExtractor.extract_features, verbatim.

    The reference module iterates ``self.model.layer`` and closes with a
    non-affine ``F.layer_norm``; transformers 5.x moved the layer stack to
    ``model.model.layer`` and its own forward applies an affine final norm,
    so the reference sequence is replayed here against the native module tree.
    """
    import torch
    import torch.nn.functional as F

    vit = model.model
    image = image.to(vit.embeddings.patch_embeddings.weight.dtype)
    hidden_states = vit.embeddings(image, bool_masked_pos=None)
    position_embeddings = vit.rope_embeddings(image)
    for layer_module in vit.model.layer:
        hidden_states = layer_module(hidden_states,
                                     position_embeddings=position_embeddings)
    return F.layer_norm(hidden_states, hidden_states.shape[-1:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--stages", default="ss,shape_512,shape_1024,tex_1024",
                        help="comma-separated subset of " + ",".join(STAGE_CONFIGS))
    parser.add_argument("--num-views", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    stage_names = [name for name in args.stages.split(",") if name]
    for name in stage_names:
        if name not in STAGE_CONFIGS:
            raise SystemExit(f"unknown stage {name!r}; expected one of {sorted(STAGE_CONFIGS)}")

    meta = json.loads((args.views_dir / "transforms.json").read_text(encoding="utf-8"))
    frames = meta["frames"]
    if args.num_views is not None:
        frames = frames[:args.num_views]
    mesh_scale = float(meta.get("mesh_scale", 1.0))

    cameras = []
    images_rgba = []
    for frame in frames:
        fov = float(frame.get("camera_angle_x", meta["camera_angle_x"]))
        transform = np.asarray(frame["transform_matrix"], dtype=np.float32)
        if transform.shape != (4, 4):
            raise SystemExit(f"frame {frame.get('file_path')} transform must be 4x4")
        distance = float(np.linalg.norm(transform[:3, 3]))
        cameras.append({"camera_angle_x": fov, "distance": distance,
                        "mesh_scale": mesh_scale,
                        "transform_matrix": transform.reshape(-1).tolist()})
        images_rgba.append(load_rgba(args.views_dir / frame["file_path"]))
    print(f"[Views] V={len(frames)} from {args.views_dir}, mesh_scale={mesh_scale}")
    print(f"[Camera] fov={math.degrees(cameras[0]['camera_angle_x']):.2f}deg, "
          f"distance={cameras[0]['distance']:.4f}")

    import torch
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjMultiViewFeatureExtractor,
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    manifest = {"stages": []}
    for name in stage_names:
        config = dict(STAGE_CONFIGS[name])
        image_size = config["image_size"]
        use_naf = bool(config.pop("use_naf_upsample", False))
        naf_target = config.pop("naf_target_size", None)
        grid_resolution = config.pop("grid_resolution")

        model = DinoV3ProjMultiViewFeatureExtractor(
            model_name=DINO_MODEL_NAME, image_size=image_size,
            grid_resolution=grid_resolution,
            use_naf_upsample=use_naf, naf_target_size=naf_target,
            multiview_fusion="average",
        ).to(device).eval()
        num_reg = getattr(model.model.config, "num_register_tokens", 4)
        print(f"[{name}] image_size={image_size} grid={grid_resolution} "
              f"naf={naf_target if use_naf else None} register_tokens={num_reg}")

        cond = np.stack([to_cond_tensor(image, image_size) for image in images_rgba])
        cond_t = torch.from_numpy(cond).to(device)
        with torch.no_grad():
            z = extract_features(model, model.transform(cond_t))
            z_global = torch.cat([z[:, 0:1], z[:, 1:1 + num_reg]], dim=1)  # [V,1+reg,D]
            patch_number = model.patch_number
            z_map = z[:, 1 + num_reg:].reshape(
                len(frames), patch_number, patch_number, -1)               # [V,h,w,D]
            if use_naf:
                model._load_naf()
                install_chunked_naf_attention()
                lr_bchw = z_map.permute(0, 3, 1, 2).contiguous()          # [V,D,h,w]
                # NAF's neighborhood attention at the 1024 guide resolution is
                # sized for the reference's high-VRAM cards; keep the batch at
                # one view and ship each output to host memory immediately so
                # the peak stays within a 24 GiB GPU.
                hr_parts = []
                for index in range(len(frames)):
                    hr_parts.append(model.naf_model(
                        cond_t[index:index + 1].clone(),
                        lr_bchw[index:index + 1],
                        model.naf_target_size).float().cpu())
                    torch.cuda.empty_cache()
                hr = torch.stack(hr_parts).numpy()                        # [V,D,H,W]
                del hr_parts
        z_global = z_global.float().cpu().numpy()
        z_map = z_map.float().cpu().numpy()

        views = []
        for index, camera in enumerate(cameras):
            prefix = f"{name}_v{index}"
            global_path = out_dir / f"{prefix}_global.npy"
            dino_path = out_dir / f"{prefix}_dino.npy"
            np.save(global_path, z_global[index])
            np.save(dino_path, z_map[index])
            view = {
                "global": os.path.relpath(global_path, out_dir),
                "dino_map": os.path.relpath(dino_path, out_dir),
                "image_resolution": image_size,
                "camera": camera,
            }
            print(f"  {prefix}: global{z_global[index].shape} "
                  f"dino{z_map[index].shape} "
                  f"sha={fingerprint(z_global[index])[:8]}/{fingerprint(z_map[index])[:8]}")
            if use_naf:
                naf_path = out_dir / f"{prefix}_naf.npy"
                np.save(naf_path, hr[index])
                view["naf_map"] = {"path": os.path.relpath(naf_path, out_dir),
                                   "layout": "CHW",
                                   "image_resolution": model.naf_target_size[0]}
                print(f"    naf{hr[index].shape} sha={fingerprint(hr[index])[:8]}")
            views.append(view)
        manifest["stages"].append({"name": name, "views": views})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest_path = out_dir / "manifest.json"
    manifest = {"stages": manifest["stages"]}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(previous, dict) and isinstance(previous.get("stages"), list):
                merged = {stage.get("name"): stage for stage in previous["stages"]
                          if isinstance(stage, dict)}
                for stage in manifest["stages"]:
                    merged[stage["name"]] = stage
                manifest["stages"] = [merged[name] for name in STAGE_CONFIGS
                                      if name in merged]
        except (OSError, ValueError):
            pass
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[Manifest] {manifest_path} ({len(manifest['stages'])} stages: "
          f"{', '.join(stage['name'] for stage in manifest['stages'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
