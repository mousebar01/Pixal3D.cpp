#!/usr/bin/env python3
"""Compare the native raw-image DINO/NAF stage bridge with both references.

The test builds tiny versions of the exact DINOv3 and NAF graphs, serializes
them through the production GGUF converter, and checks that
encode_pixal3d_condition_stage_f32 applies the raw [0,1] image guide,
ImageNet normalization, HWC map layout, and NAF concatenation correctly.
"""

from __future__ import annotations

import importlib.util
import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import compare_dino_vit  # noqa: E402
import compare_naf  # noqa: E402


def dino_reference(config: dict, state: dict[str, torch.Tensor], raw: np.ndarray):
    model = DINOv3ViTModel(DINOv3ViTConfig(**config))
    mapped = {
        ("model." + key if key.startswith("layer.") else key): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"tiny DINO state mismatch: missing={missing} unexpected={unexpected}")
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    with torch.no_grad():
        x = (torch.from_numpy(raw).unsqueeze(0) - mean) / std
        hidden = model.embeddings(x)
        position_embeddings = model.rope_embeddings(x)
        for layer in model.model.layer:
            hidden = layer(hidden, position_embeddings=position_embeddings)
        hidden = F.layer_norm(hidden, hidden.shape[-1:])
    prefix = 1 + config["num_register_tokens"]
    return hidden[0, :prefix].numpy().reshape(-1), hidden[0, prefix:].numpy().reshape(2, 2, -1)


def parse_native(text: str):
    tokens = text.split()
    index = 0
    if tokens[index] != "stage":
        raise RuntimeError(f"unexpected native output: {text[:200]}")
    index += 5  # stage label, name, H, W, C

    def read(label: str):
        nonlocal index
        if tokens[index] != label:
            raise RuntimeError(f"expected {label}, got {tokens[index]}")
        count = int(tokens[index + 1])
        values = np.asarray(tokens[index + 2:index + 2 + count], dtype=np.float32)
        if values.size != count:
            raise RuntimeError(f"truncated native {label} output")
        index += 2 + count
        return values

    return read("global"), read("dino"), read("naf")


def main() -> int:
    fixture = ROOT / "build" / "bin" / "pixal3d_vision_encoder_fixture"
    if not fixture.is_file():
        raise RuntimeError(f"missing {fixture}; build the project first")
    naf_source = Path("/app/ComfyUI/comfy/image_encoders/naf.py")
    if not naf_source.is_file():
        raise RuntimeError(f"missing reference NAF source: {naf_source}")
    with tempfile.TemporaryDirectory(prefix="pixal3d-vision-vs-") as tmp:
        directory = Path(tmp)
        dino_source, dino_config, dino_state = compare_dino_vit.make_fixture(directory)
        naf_source_weights, naf_config_path, naf_config, _ = compare_naf.make_fixture(
            directory, naf_source
        )
        dino_output = directory / "tiny-dino-f32.gguf"
        naf_output = directory / "tiny-naf-f32.gguf"
        converter = ROOT / "scripts" / "convert_pixal3d_to_gguf.py"
        subprocess.run([
            sys.executable, str(converter), "--component", "dino",
            "--model", str(dino_source), "--config", str(directory / "tiny-dino.json"),
            "--output", str(dino_output), "--ftype", "0",
        ], check=True)
        subprocess.run([
            sys.executable, str(converter), "--component", "naf",
            "--model", str(naf_source_weights), "--config", str(naf_config_path),
            "--output", str(naf_output), "--ftype", "0",
        ], check=True)

        raw = np.asarray(
            [0.1 + 0.8 * ((i * 7) % 23) / 22.0 for i in range(3 * 4 * 4)],
            dtype=np.float32,
        ).reshape(3, 4, 4)
        expected_global, expected_dino = dino_reference(dino_config, dino_state, raw)
        naf_reference = compare_naf.load_reference_naf(naf_source)
        naf_model = naf_reference(
            operations=torch.nn, dim=naf_config["dim"],
            heads_attn=naf_config["heads_attn"], heads_rope=naf_config["heads_rope"],
            kernel_size=naf_config["kernel_size"], rope_base=naf_config["rope_base"],
            img_layers=naf_config["img_layers"],
        ).eval()
        naf_model.load_state_dict(load_file(str(naf_source_weights)), strict=True)
        with torch.no_grad():
            low = torch.from_numpy(expected_dino).permute(2, 0, 1).unsqueeze(0)
            expected_naf = naf_model(torch.from_numpy(raw).unsqueeze(0), low,
                                     output_size=(4, 4))[0].permute(1, 2, 0).numpy()

        native = subprocess.run(
            [str(fixture), str(dino_output), str(naf_output)],
            check=True, capture_output=True, text=True,
        )
        actual_global, actual_dino, actual_naf = parse_native(native.stdout)
        expected_global = expected_global.astype(np.float32, copy=False)
        expected_dino = expected_dino.reshape(-1).astype(np.float32, copy=False)
        expected_naf = expected_naf.reshape(-1).astype(np.float32, copy=False)
        np.testing.assert_allclose(actual_global, expected_global, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(actual_dino, expected_dino, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(actual_naf, expected_naf, rtol=2e-5, atol=2e-5)
        diff = np.concatenate([
            actual_global - expected_global,
            actual_dino - expected_dino,
            actual_naf - expected_naf,
        ])
        expected = np.concatenate([expected_global, expected_dino, expected_naf])
        print("vision_condition PASS: max_abs={:.3g}, rel_l2={:.3g}, values={}".format(
            np.max(np.abs(diff)),
            np.linalg.norm(diff) / max(np.linalg.norm(expected), 1e-12),
            diff.size,
        ))

        # Exercise the public image-to-stage CLI on a real P6 input as well as
        # the raw in-memory fixture above.  A 4x4 target avoids any resize
        # ambiguity while still traversing decode -> DINO -> NAF -> P3DCOND.
        ppm = directory / "input.ppm"
        image_u8 = np.asarray(
            [int(round(255.0 * (0.1 + 0.8 * ((i * 7) % 23) / 22.0)))
             for i in range(3 * 4 * 4)], dtype=np.uint8
        ).reshape(3, 4, 4)
        with ppm.open("wb") as file:
            file.write(b"P6\n4 4\n255\n")
            file.write(np.transpose(image_u8, (1, 2, 0)).tobytes())
        cli_bundle = directory / "cli-condition.p3dcond"
        cli = fixture.parent / "pixal3d"
        subprocess.run([
            str(cli), "encode-condition-stage", str(dino_output), str(naf_output),
            str(ppm), "shape_512", str(cli_bundle),
            "--resolution", "4", "--naf-resolution", "4",
        ], check=True, capture_output=True, text=True)
        inspected = subprocess.run(
            [str(cli), "inspect-condition", str(cli_bundle)],
            check=True, capture_output=True, text=True,
        ).stdout
        if "stage         : shape_512" not in inspected or "naf_map" not in inspected:
            raise RuntimeError(f"CLI condition bundle inspection mismatch: {inspected}")
        print("vision_condition_cli PASS: P6 decode and stage bundle write")

        full_bundle = directory / "full-condition.p3dcond"
        subprocess.run([
            str(cli), "encode-condition-bundle", str(dino_output), str(naf_output),
            str(ppm), str(full_bundle),
            "--ss-resolution", "4", "--shape-512-resolution", "4",
            "--shape-1024-resolution", "4", "--tex-1024-resolution", "4",
            "--ss-naf-resolution", "0", "--shape-512-naf-resolution", "4",
            "--shape-1024-naf-resolution", "4", "--tex-1024-naf-resolution", "4",
        ], check=True, capture_output=True, text=True)
        full_inspected = subprocess.run(
            [str(cli), "inspect-condition", str(full_bundle)],
            check=True, capture_output=True, text=True,
        ).stdout
        for name in ("ss", "shape_512", "shape_1024", "tex_1024"):
            if f"stage         : {name}" not in full_inspected:
                raise RuntimeError(f"full CLI bundle is missing {name}: {full_inspected}")
        print("vision_condition_bundle_cli PASS: four-stage image bundle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
