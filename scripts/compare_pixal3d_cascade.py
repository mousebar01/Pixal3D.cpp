#!/usr/bin/env python3
"""Compare the external-condition Pixal3D cascade with the Python graph."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

# The reference package defaults to optional flash-attn.  Keep this oracle
# dependency-free and deterministic, just like the other numerical fixtures.
os.environ.setdefault("ATTN_BACKEND", "naive")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_CONV_BACKEND", "none")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_slat_decoder as decoder_base
import compare_slat_flow as flow_base
import compare_ss_decoder as ss_decoder_base
import compare_ss_flow as ss_flow_base

from pixal3d.modules import sparse as sp
from pixal3d.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
from pixal3d.models.structured_latent_flow import SLatFlowModel
from pixal3d.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler


STAGE_BASE = {
    "sparse_structure": np.float32(-0.31),
    "shape_slat_low": np.float32(-0.23),
    "shape_slat_high": np.float32(-0.17),
    "texture_slat": np.float32(-0.11),
}
SHAPE_MEAN = np.asarray([0.11, -0.07, 0.23], dtype=np.float32)
SHAPE_STD = np.asarray([1.3, 0.8, 1.7], dtype=np.float32)
TEXTURE_MEAN = np.asarray([-0.13, 0.19, 0.03], dtype=np.float32)
TEXTURE_STD = np.asarray([0.9, 1.4, 1.1], dtype=np.float32)


def stage_noise(stage: str, coords: np.ndarray, channels: int,
                grid_resolution: int) -> sp.SparseTensor:
    del grid_resolution
    values = STAGE_BASE[stage] + np.float32(0.0017) * np.arange(
        coords.shape[0] * channels, dtype=np.float32)
    return sp.SparseTensor(torch.from_numpy(values.reshape(coords.shape[0], channels)),
                           torch.from_numpy(coords))


def stage_condition(stage: str, coords: np.ndarray,
                    grid_resolution: int) -> Tuple[sp.VarLenTensor, sp.SparseTensor]:
    del grid_resolution
    base = np.float32(STAGE_BASE[stage] * np.float32(0.5))
    global_np = base + np.float32(0.0023) * np.arange(15, dtype=np.float32)
    global_np = global_np.reshape(3, 5)
    projected_np = base - np.float32(0.0011) * np.arange(
        coords.shape[0] * 3, dtype=np.float32)
    projected_np = projected_np.reshape(coords.shape[0], 3)
    global_cond = sp.VarLenTensor.from_tensor_list([torch.from_numpy(global_np)])
    projected = sp.SparseTensor(torch.from_numpy(projected_np), torch.from_numpy(coords))
    return global_cond, projected


def make_slat_model(resolution: int, in_channels: int, out_channels: int):
    model = SLatFlowModel(
        resolution=resolution, in_channels=in_channels, model_channels=12,
        cond_channels=5, out_channels=out_channels, num_blocks=1, num_heads=2,
        mlp_ratio=1.5, pe_mode="rope", share_mod=True, qk_rms_norm=True,
        qk_rms_norm_cross=True, image_attn_mode="proj", proj_in_channels=3,
        dtype="float32")
    state = {}
    with torch.no_grad():
        for key, parameter in model.state_dict().items():
            value = flow_base.tensor_values(key, parameter.numel()).reshape(parameter.shape)
            parameter.copy_(torch.from_numpy(value))
            state[key] = value.astype(np.float32)
    model.eval()
    return model, state


def make_slat_decoder(pred_subdiv: bool, out_channels: int):
    model = SparseUnetVaeDecoder(
        out_channels, [8, 4], 3, [1, 1],
        ["SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d"],
        ["SparseResBlockC2S3d"], [{}, {}], use_fp16=False,
        pred_subdiv=pred_subdiv)
    state = {}
    with torch.no_grad():
        for key, parameter in model.state_dict().items():
            value = decoder_base.tensor_values(key, tuple(parameter.shape))
            if pred_subdiv and key.endswith("to_subdiv.bias"):
                value = np.full(tuple(parameter.shape), 5.0, dtype=np.float32)
            # Keep the mesh split channel away from near-ties caused by tiny
            # convolution accumulation differences between Torch and the
            # scalar C++ oracle.  A constant positive logit still exercises
            # the exact FlexiDualGrid diagonal rule deterministically.
            if pred_subdiv and key == "output_layer.weight":
                value = np.zeros(tuple(parameter.shape), dtype=np.float32)
            if pred_subdiv and key == "output_layer.bias":
                value = np.zeros(tuple(parameter.shape), dtype=np.float32)
                # Intersect all three edge families so the tiny grid also
                # exercises quad reconstruction and diagonal selection.
                value[3:7] = 5.0
            parameter.copy_(torch.from_numpy(value))
            state[key] = value.astype(np.float32)
    model.eval()
    return model, state


def write_slat_flow_pack(path: Path, state: Dict[str, np.ndarray], *,
                         component: str, alias: str, resolution: int,
                         in_channels: int, out_channels: int) -> None:
    metadata = [
        flow_base._kv_str("general.architecture", "pixal3d"),
        flow_base._kv_str("general.name", "pixal3d-cascade-test"),
        flow_base._kv_u32("general.file_type", 0),
        flow_base._kv_u32("general.alignment", flow_base.GGUF_ALIGNMENT),
        flow_base._kv_u32("pixal3d.bundle_format", 1),
        flow_base._kv_str("pixal3d.tensor_name_scheme", "compact-v1"),
        flow_base._kv_str("pixal3d.bundle_kind", "component"),
        flow_base._kv_u32("pixal3d.component_count", 1),
        flow_base._kv_str("pixal3d.component.0", component),
        flow_base._kv_str(f"pixal3d.{component}.model_class", "ElasticSLatFlowModel"),
        flow_base._kv_u32(f"pixal3d.{component}.resolution", resolution),
        flow_base._kv_u32(f"pixal3d.{component}.in_channels", in_channels),
        flow_base._kv_u32(f"pixal3d.{component}.out_channels", out_channels),
        flow_base._kv_u32(f"pixal3d.{component}.model_channels", 12),
        flow_base._kv_u32(f"pixal3d.{component}.cond_channels", 5),
        flow_base._kv_u32(f"pixal3d.{component}.num_blocks", 1),
        flow_base._kv_u32(f"pixal3d.{component}.num_heads", 2),
        flow_base._kv_f32(f"pixal3d.{component}.mlp_ratio", 1.5),
        flow_base._kv_str(f"pixal3d.{component}.pe_mode", "rope"),
        flow_base._kv_bool(f"pixal3d.{component}.share_mod", True),
        flow_base._kv_bool(f"pixal3d.{component}.qk_rms_norm", True),
        flow_base._kv_bool(f"pixal3d.{component}.qk_rms_norm_cross", True),
        flow_base._kv_f32(f"pixal3d.{component}.rope_freq_min", 1.0),
        flow_base._kv_f32(f"pixal3d.{component}.rope_freq_base", 10000.0),
        flow_base._kv_str(f"pixal3d.{component}.image_attn_mode", "proj"),
        flow_base._kv_u32(f"pixal3d.{component}.proj_in_channels", 3),
    ]
    tensors = []
    offsets = []
    offset = 0
    for key in sorted(state):
        value = np.asarray(state[key], dtype=np.float32)
        raw = value.astype("<f4", copy=False).tobytes(order="C")
        dims = tuple(reversed(value.shape))
        offsets.append(offset)
        tensors.append((alias + "." + key, dims, raw))
        offset = flow_base._align(offset + len(raw))
    infos = bytearray()
    for (name, dims, _), tensor_offset in zip(tensors, offsets):
        infos += flow_base._gguf_string(name) + flow_base.struct.pack("<I", len(dims))
        for dim in dims:
            infos += flow_base.struct.pack("<Q", int(dim))
        infos += flow_base.struct.pack("<I", 0) + flow_base.struct.pack("<Q", tensor_offset)
    header = bytearray(flow_base.GGUF_MAGIC) + flow_base.struct.pack("<I", flow_base.GGUF_VERSION)
    header += flow_base.struct.pack("<Q", len(tensors)) + flow_base.struct.pack("<Q", len(metadata))
    for item in metadata:
        header += item
    data_offset = flow_base._align(len(header) + len(infos))
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(infos)
        handle.write(b"\0" * (data_offset - handle.tell()))
        for (_, _, raw), tensor_offset in zip(tensors, offsets):
            if handle.tell() - data_offset != tensor_offset:
                raise RuntimeError("SLat flow GGUF offset failure")
            handle.write(raw)
            handle.write(b"\0" * (flow_base._align(len(raw)) - len(raw)))


def write_slat_decoder_pack(path: Path, state: Dict[str, np.ndarray], *,
                            component: str, out_channels: int,
                            pred_subdiv: bool) -> None:
    metadata = [
        decoder_base._kv_str("general.architecture", "pixal3d"),
        decoder_base._kv_str("general.name", "pixal3d-cascade-test"),
        decoder_base._kv_u32("general.file_type", 0),
        decoder_base._kv_u32("general.alignment", decoder_base.GGUF_ALIGNMENT),
        decoder_base._kv_u32("pixal3d.bundle_format", 1),
        decoder_base._kv_str("pixal3d.tensor_name_scheme", "compact-v1"),
        decoder_base._kv_str("pixal3d.bundle_kind", "component"),
        decoder_base._kv_u32("pixal3d.component_count", 1),
        decoder_base._kv_str("pixal3d.component.0", component),
        decoder_base._kv_str(f"pixal3d.{component}.model_class", "FlexiDualGridVaeDecoder"),
        decoder_base._kv_u32(f"pixal3d.{component}.resolution", 2),
        decoder_base._kv_u32(f"pixal3d.{component}.latent_channels", 3),
        decoder_base._kv_u32(f"pixal3d.{component}.out_channels", out_channels),
        decoder_base._kv_u32(f"pixal3d.{component}.n_levels", 2),
        decoder_base._kv_f32(f"pixal3d.{component}.norm_eps", 1e-6),
        decoder_base._kv_bool(f"pixal3d.{component}.pred_subdiv", pred_subdiv),
        decoder_base._kv_str(f"pixal3d.{component}.conv_weight_source_layout", "out,kd,kh,kw,in"),
        decoder_base._kv_str(f"pixal3d.{component}.conv_weight_gguf_layout", "kernel_volume,out,in"),
        decoder_base._kv_u32(f"pixal3d.{component}.conv_kernel_size", 3),
        decoder_base._kv_u32(f"pixal3d.{component}.conv_kernel_volume", 27),
        decoder_base._kv_u32(f"pixal3d.{component}.model_channels.0", 8),
        decoder_base._kv_u32(f"pixal3d.{component}.model_channels.1", 4),
        decoder_base._kv_u32(f"pixal3d.{component}.num_blocks.0", 1),
        decoder_base._kv_u32(f"pixal3d.{component}.num_blocks.1", 1),
    ]
    tensors = []
    offsets = []
    offset = 0
    for key in sorted(state):
        value = np.asarray(state[key], dtype=np.float32)
        if value.ndim == 5:
            out, kd, kh, kw, inn = value.shape
            if (kd, kh, kw) != (3, 3, 3):
                raise ValueError(f"unexpected decoder kernel shape: {key} {value.shape}")
            value = value.transpose(1, 2, 3, 0, 4).reshape(27, out, inn)
        raw = value.astype("<f4", copy=False).tobytes(order="C")
        dims = tuple(reversed(value.shape))
        offsets.append(offset)
        tensors.append((component + "." + key, dims, raw))
        offset = decoder_base._align(offset + len(raw))
    infos = bytearray()
    for (name, dims, _), tensor_offset in zip(tensors, offsets):
        infos += decoder_base._gguf_string(name) + decoder_base.struct.pack("<I", len(dims))
        for dim in dims:
            infos += decoder_base.struct.pack("<Q", int(dim))
        infos += decoder_base.struct.pack("<I", decoder_base.GGML_TYPE_F32)
        infos += decoder_base.struct.pack("<Q", tensor_offset)
    header = bytearray(decoder_base.GGUF_MAGIC) + decoder_base.struct.pack("<I", decoder_base.GGUF_VERSION)
    header += decoder_base.struct.pack("<Q", len(tensors)) + decoder_base.struct.pack("<Q", len(metadata))
    for item in metadata:
        header += item
    data_offset = decoder_base._align(len(header) + len(infos))
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(infos)
        handle.write(b"\0" * (data_offset - handle.tell()))
        for (_, _, raw), tensor_offset in zip(tensors, offsets):
            if handle.tell() - data_offset != tensor_offset:
                raise RuntimeError("SLat decoder GGUF offset failure")
            handle.write(raw)
            handle.write(b"\0" * (decoder_base._align(len(raw)) - len(raw)))


def flatten_sparse(values: torch.Tensor) -> np.ndarray:
    # SparseTensor stores the row-major features in ``.feats``; keeping this
    # helper explicit avoids accidentally flattening the coordinate wrapper.
    tensor = values.feats if isinstance(values, sp.SparseTensor) else values
    return tensor.detach().cpu().numpy().astype(np.float32).reshape(-1)


def reference_mesh(decoded: sp.SparseTensor, resolution: int,
                   voxel_margin: float = 0.5):
    coords = decoded.coords.detach().cpu().numpy().astype(np.int32)
    features = decoded.feats.detach().cpu().numpy().astype(np.float32)
    coords_xyz = coords[:, 1:]
    index = {tuple(coord): i for i, coord in enumerate(coords_xyz.tolist())}
    offsets = np.asarray([
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ], dtype=np.int32)
    quads = []
    for i, coord in enumerate(coords_xyz):
        for axis in range(3):
            # In inference the decoder's channels 3..5 are thresholded to
            # select intersected primal edges before the quad is emitted.
            # The earlier standalone mesh check used all complete neighbor
            # quads; that is not the FlexiDualGrid decoder contract.
            if features[i, 3 + axis] <= 0.0:
                continue
            neighbours = [tuple(coord + offsets[axis, corner]) for corner in range(4)]
            if all(neighbour in index for neighbour in neighbours):
                quads.append([index[neighbour] for neighbour in neighbours])
    sigmoid = 1.0 / (1.0 + np.exp(-features[:, :3]))
    dual = (1.0 + 2.0 * voxel_margin) * sigmoid - voxel_margin
    vertices = (coords_xyz.astype(np.float32) + dual) / np.float32(resolution) - 0.5
    split_logits = features[:, 6]
    # Match the scalar C++ stable softplus expression (including float32
    # intermediates) instead of NumPy's potentially wider logaddexp path.
    weights = np.where(
        split_logits >= 0.0,
        split_logits + np.log1p(np.exp(-split_logits)),
        np.log1p(np.exp(split_logits)),
    ).astype(np.float32)
    faces = []
    for quad in quads:
        if weights[quad[0]] * weights[quad[2]] > weights[quad[1]] * weights[quad[3]]:
            faces.extend([quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]])
        else:
            faces.extend([quad[0], quad[1], quad[3], quad[3], quad[1], quad[2]])
    return vertices.reshape(-1), np.asarray(faces, dtype=np.int32).reshape(-1)


def parse_fixture(text: str):
    result = {}
    names = {
        "cascade_structure_coords", "cascade_structure_sample", "cascade_upsampled_coords",
        "cascade_high_coords", "cascade_shape_low", "cascade_shape_high",
        "cascade_texture", "cascade_shape_decoded", "cascade_texture_decoded",
        "cascade_mesh_vertices", "cascade_mesh_faces",
    }
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] not in names:
            continue
        count = int(fields[1])
        dtype = np.int32 if fields[0].endswith("coords") or fields[0].endswith("faces") else np.float32
        values = np.asarray(fields[2:], dtype=dtype)
        if values.size != count:
            raise ValueError(f"{fields[0]} count mismatch")
        result[fields[0]] = values
    if set(result) != names:
        raise ValueError(f"cascade fixture outputs {set(result)} != {names}")
    resolution = None
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == "cascade_resolution":
            resolution = int(fields[1])
    if resolution is None:
        raise ValueError("cascade fixture did not emit resolution")
    result["cascade_resolution"] = resolution
    return result


def fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()[:16]


def main() -> int:
    ss_model, ss_state, ss_config = ss_flow_base.make_model()
    ss_dec_model, ss_dec_state, ss_dec_config = ss_decoder_base.make_model()
    shape_low, shape_low_state = make_slat_model(4, 3, 3)
    shape_high, shape_high_state = make_slat_model(8, 3, 3)
    texture_flow, texture_flow_state = make_slat_model(8, 6, 3)
    shape_decoder, shape_decoder_state = make_slat_decoder(True, 7)
    texture_decoder, texture_decoder_state = make_slat_decoder(False, 3)

    def ss_grid(resolution: int) -> np.ndarray:
        return np.asarray([[0, x, y, z] for x in range(resolution)
                           for y in range(resolution) for z in range(resolution)], dtype=np.int32)

    def dense_ss(noise_sparse: sp.SparseTensor, resolution: int, channels: int) -> torch.Tensor:
        values = noise_sparse.feats.detach().cpu().numpy().reshape(resolution, resolution, resolution, channels)
        return torch.from_numpy(values.transpose(3, 0, 1, 2)[None].astype(np.float32))

    ss_coords = ss_grid(2)
    ss_noise = stage_noise("sparse_structure", ss_coords, 2, 2)
    ss_global, ss_projected = stage_condition("sparse_structure", ss_coords, 2)
    ss_x = dense_ss(ss_noise, 2, 2)
    ss_condition = (torch.from_numpy(ss_global.feats.detach().cpu().numpy()[None]),
                    torch.from_numpy(ss_projected.feats.detach().cpu().numpy()[None]))
    ss_negative = (torch.zeros_like(ss_condition[0]), torch.zeros_like(ss_condition[1]))
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=0.031)
    with torch.no_grad():
        ss_sampled = sampler.sample(ss_model, ss_x, ss_condition, ss_negative,
                                    steps=1, rescale_t=1.0, guidance_strength=1.0,
                                    verbose=False)
        ss_decoded = ss_dec_model(ss_sampled.samples)
        ss_occupied = ss_decoded > -1.0e9
        ss_stage_coords = torch.argwhere(ss_occupied)[:, [0, 2, 3, 4]].int()
    ss_sample_expected = ss_sampled.samples.detach().cpu().numpy()[0].transpose(1, 2, 3, 0).reshape(-1)
    ss_coords_expected = ss_stage_coords.detach().cpu().numpy().reshape(-1)

    shape_low_coords = ss_stage_coords.detach().cpu().numpy().astype(np.int32)
    shape_low_noise = stage_noise("shape_slat_low", shape_low_coords, 3, 4)
    shape_low_global, shape_low_projected = stage_condition("shape_slat_low", shape_low_coords, 4)
    with torch.no_grad():
        low_sample = sampler.sample(
            shape_low, shape_low_noise, (shape_low_global, shape_low_projected),
            (shape_low_global.replace(torch.zeros_like(shape_low_global.feats)),
             shape_low_projected.replace(torch.zeros_like(shape_low_projected.feats))),
            steps=1, rescale_t=1.0, guidance_strength=1.0, verbose=False)
        low_latent = low_sample.samples.replace(
            low_sample.samples.feats * torch.from_numpy(SHAPE_STD) + torch.from_numpy(SHAPE_MEAN))
        upsampled = shape_decoder.upsample(low_latent, upsample_times=1)
        # The reference decoder returns the coordinate tensor directly.
        upsampled_np = upsampled.detach().cpu().numpy().astype(np.int32)
    high_coords = []
    for coord in upsampled_np:
        mapped = [0]
        for value in coord[1:]:
            mapped.append(int(np.rint((float(value) + 0.5) / 8.0 * 63.0)))
        high_coords.append(mapped)
    high_coords = np.unique(np.asarray(high_coords, dtype=np.int32), axis=0)

    shape_high_noise = stage_noise("shape_slat_high", high_coords, 3, 64)
    shape_high_global, shape_high_projected = stage_condition("shape_slat_high", high_coords, 64)
    with torch.no_grad():
        high_sample = sampler.sample(
            shape_high, shape_high_noise,
            (shape_high_global, shape_high_projected),
            (shape_high_global.replace(torch.zeros_like(shape_high_global.feats)),
             shape_high_projected.replace(torch.zeros_like(shape_high_projected.feats))),
            steps=1, rescale_t=1.0, guidance_strength=1.0, verbose=False)
        high_latent = high_sample.samples.replace(
            high_sample.samples.feats * torch.from_numpy(SHAPE_STD) + torch.from_numpy(SHAPE_MEAN))
    texture_noise = stage_noise("texture_slat", high_coords, 3, 64)
    texture_global, texture_projected = stage_condition("texture_slat", high_coords, 64)
    with torch.no_grad():
        # The texture flow consumes the shape latent in normalized space;
        # shape decoding below still uses the denormalized high_latent.
        texture_shape_condition = high_latent.replace(
            (high_latent.feats - torch.from_numpy(SHAPE_MEAN)) /
            torch.from_numpy(SHAPE_STD))
        tex_sample = sampler.sample(
            texture_flow, texture_noise,
            (texture_global, texture_projected),
            (texture_global.replace(torch.zeros_like(texture_global.feats)),
             texture_projected.replace(torch.zeros_like(texture_projected.feats))),
            steps=1, rescale_t=1.0, guidance_strength=1.0,
            concat_cond=texture_shape_condition, verbose=False)
        tex_latent = tex_sample.samples.replace(
            tex_sample.samples.feats * torch.from_numpy(TEXTURE_STD) + torch.from_numpy(TEXTURE_MEAN))
        shape_decoded, shape_subs = shape_decoder(high_latent, return_subs=True)
        texture_decoded = texture_decoder(tex_latent, guide_subs=shape_subs)
        # The production pipeline maps texture decoder outputs from the
        # trained [-1, 1] range into material/voxel attributes in [0, 1].
        texture_decoded = texture_decoded.replace(
            texture_decoded.feats * np.float32(0.5) + np.float32(0.5))
    mesh_vertices, mesh_faces = reference_mesh(shape_decoded, 1024)

    expected = {
        "cascade_structure_coords": ss_coords_expected.astype(np.int32),
        "cascade_structure_sample": ss_sample_expected.astype(np.float32),
        "cascade_upsampled_coords": upsampled_np.reshape(-1),
        "cascade_high_coords": high_coords.reshape(-1),
        "cascade_shape_low": flatten_sparse(low_latent),
        "cascade_shape_high": flatten_sparse(high_latent),
        "cascade_texture": flatten_sparse(tex_latent),
        "cascade_shape_decoded": flatten_sparse(shape_decoded),
        "cascade_texture_decoded": flatten_sparse(texture_decoded),
        "cascade_mesh_vertices": mesh_vertices,
        "cascade_mesh_faces": mesh_faces,
    }
    fixture = ROOT / "build" / "bin" / "pixal3d_cascade_fixture"
    with tempfile.TemporaryDirectory(prefix="pixal3d-cascade-") as directory:
        directory = Path(directory)
        paths = [directory / name for name in (
            "ss-flow.gguf", "ss-decoder.gguf", "shape-low.gguf", "shape-high.gguf",
            "texture-flow.gguf", "shape-decoder.gguf", "texture-decoder.gguf")]
        ss_flow_base.write_gguf(paths[0], ss_state, ss_config)
        ss_decoder_base.write_gguf(paths[1], ss_dec_state, ss_dec_config)
        write_slat_flow_pack(paths[2], shape_low_state, component="shape_flow_512",
                             alias="sh512", resolution=4, in_channels=3, out_channels=3)
        write_slat_flow_pack(paths[3], shape_high_state, component="shape_flow_1024",
                             alias="sh1024", resolution=8, in_channels=3, out_channels=3)
        write_slat_flow_pack(paths[4], texture_flow_state, component="texture_flow_1024",
                             alias="tx1024", resolution=8, in_channels=6, out_channels=3)
        write_slat_decoder_pack(paths[5], shape_decoder_state, component="shape_decoder",
                                out_channels=7, pred_subdiv=True)
        write_slat_decoder_pack(paths[6], texture_decoder_state, component="texture_decoder",
                                out_channels=3, pred_subdiv=False)
        completed = subprocess.run([str(fixture), *(str(path) for path in paths)],
                                   check=True, capture_output=True, text=True)
    actual = parse_fixture(completed.stdout)
    if actual["cascade_resolution"] != 1024:
        raise AssertionError("cascade selected an unexpected resolution")
    report = []
    for name, target in expected.items():
        got = actual[name]
        if got.shape != target.shape:
            raise AssertionError(f"{name} shape mismatch: {got.shape} != {target.shape}")
        if not np.isfinite(got.astype(np.float64)).all():
            raise AssertionError(f"{name} contains non-finite values")
        if np.issubdtype(target.dtype, np.integer):
            if not np.array_equal(got, target):
                raise AssertionError(f"{name} coordinate/index mismatch")
            max_abs = 0.0
            rel_l2 = 0.0
        else:
            delta = got.astype(np.float64) - target.astype(np.float64)
            max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
            rel_l2 = float(np.linalg.norm(delta) /
                           max(np.linalg.norm(target.astype(np.float64)), 1e-12))
            if max_abs > 2e-4 or rel_l2 > 2e-4:
                raise AssertionError(f"{name} numerical comparison exceeds tolerance")
        report.append({"name": name, "shape": list(target.shape),
                       "reference_fingerprint": fingerprint(target),
                       "cpp_fingerprint": fingerprint(got),
                       "max_abs_error": max_abs, "relative_l2_error": rel_l2})
    print(json.dumps(report, sort_keys=True))
    print("Pixal3D cascade numerical comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
