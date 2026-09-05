#!/usr/bin/env python3
"""Converter contract tests; run in the repository Docker image."""

import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pixal3d_converter", ROOT / "scripts" / "convert_pixal3d_to_gguf.py"
)
assert SPEC and SPEC.loader
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


class ConverterContractTest(unittest.TestCase):
    def test_auto_precision_policy_is_stable_mixed_deployment_mode(self):
        args = converter.build_parser().parse_args([])
        self.assertEqual(args.ftype, "auto")
        self.assertEqual(converter.AUTO_FTYPE_DESCRIPTION, "auto=F32 flow, F16 decoder")
        self.assertEqual(converter.default_ftype_for_bundle("shared"), 1)
        self.assertEqual(converter.default_ftype_for_bundle("base-flow"), 0)
        self.assertEqual(converter.default_ftype_for_bundle("mv-flow"), 0)
        self.assertEqual(converter.default_ftype_for_definition(converter.MODELS["ss-flow"]), 0)
        self.assertEqual(converter.default_ftype_for_definition(converter.MODELS["shape-dec"]), 1)
        for definition in converter.MODELS.values():
            expected = (converter.AUTO_FTYPE_DECODER
                        if definition.kind in ("ss-dec", "slat-dec")
                        else converter.AUTO_FTYPE_FLOW)
            self.assertEqual(converter.default_ftype_for_definition(definition), expected)

    def test_dino_component_is_explicit_f32_and_has_reference_contract(self):
        definition = converter.MODELS["dino"]
        self.assertEqual(definition.kind, "dino")
        self.assertEqual(converter.default_ftype_for_definition(definition), 0)
        self.assertEqual(converter.choose_dino_type("norm.weight", (8,), 0),
                         converter.GGML_TYPE_F32)
        with self.assertRaisesRegex(converter.ConversionError, "requires --ftype 0"):
            converter.choose_dino_type("linear.weight", (8, 8), 1)
        args = {
            "hidden_size": 1024, "intermediate_size": 4096,
            "num_hidden_layers": 24, "num_attention_heads": 16,
            "patch_size": 16, "num_channels": 3, "num_register_tokens": 4,
            "layer_norm_eps": 1e-5, "rope_theta": 100.0,
            "use_gated_mlp": False,
        }
        items = converter._dino_metadata(definition, args, "pixal3d.dino.")
        keys = []
        for item in items:
            size = struct.unpack_from("<Q", item, 0)[0]
            keys.append(item[8:8 + size].decode("utf-8"))
        self.assertEqual(len(keys), len(set(keys)))
        self.assertIn("pixal3d.dino.output_contract", keys)

    def test_naf_component_metadata_is_explicit_and_lossless(self):
        definition = converter.MODELS["naf"]
        self.assertEqual(definition.kind, "naf")
        self.assertEqual(converter.choose_naf_type("image_encoder.rope.periods", (16,), 0),
                         converter.GGML_TYPE_F32)
        with self.assertRaisesRegex(converter.ConversionError, "requires --ftype 0"):
            converter.choose_naf_type("image_encoder.encoder.0.weight", (128, 3, 1, 1), 1)
        args = {"dim": 256, "in_channels": 3, "heads_attn": 4,
                "heads_rope": 4, "kernel_size": 9, "img_layers": 2,
                "num_groups": 8, "rope_base": 100.0}
        items = converter._naf_metadata(definition, args, "pixal3d.naf.")
        keys = []
        for item in items:
            size = struct.unpack_from("<Q", item, 0)[0]
            keys.append(item[8:8 + size].decode("utf-8"))
        self.assertEqual(len(keys), len(set(keys)))
        self.assertIn("pixal3d.naf.padding", keys)

    def test_naf_uses_upstream_defaults_when_sidecar_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "naf.json"
            name, args = converter.load_model_config(config, "naf")
        self.assertEqual(name, "NAF")
        self.assertEqual(args, converter.NAF_DEFAULT_CONFIG)

    def test_slat_sparse_conv_layout_and_official_reader_roundtrip(self):
        kernel = torch.arange(2 * 3 * 3 * 3 * 4, dtype=torch.float32).reshape(2, 3, 3, 3, 4)
        linear = torch.tensor([[1.25, -2.5, 3.0], [4.0, 5.5, -6.0]], dtype=torch.float32)
        bias = torch.tensor([0.125, -0.25], dtype=torch.float32)
        definition = converter.ModelDefinition(
            "synthetic", "synthetic", "Synthetic", "slat-dec", "shape_decoder", "test"
        )

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "synthetic.safetensors"
            output = directory / "synthetic.gguf"
            save_file({"conv.weight": kernel, "linear.weight": linear, "bias": bias}, str(source))

            with converter.TensorStore(source) as store:
                specs = converter._tensor_specs(store, definition, 1, "shape_decoder.")
                by_name = {spec.name: spec for spec in specs}
                conv_spec = by_name["shape_decoder.conv.weight"]
                self.assertEqual(conv_spec.shape, (27, 2, 4))
                self.assertEqual(conv_spec.dims, (4, 2, 27))
                self.assertEqual(conv_spec.ggml_type, converter.GGML_TYPE_F16)
                self.assertEqual(by_name["shape_decoder.bias"].ggml_type,
                                 converter.GGML_TYPE_F32)
                metadata = converter._base_metadata("pixal3d", "synthetic", 1)
                converter._write_gguf(
                    output, metadata, specs, {definition.key: store}, force=False
                )

            converter.verify_gguf(output, specs)
            reader = GGUFReader(str(output))
            tensors = {tensor.name: tensor for tensor in reader.tensors}
            self.assertEqual(set(tensors), set(by_name))
            self.assertEqual(tuple(tensors["shape_decoder.conv.weight"].shape), (4, 2, 27))
            expected = kernel.permute(1, 2, 3, 0, 4).reshape(27, 2, 4).numpy().astype(np.float16)
            np.testing.assert_array_equal(
                tensors["shape_decoder.conv.weight"].data.reshape(27, 2, 4), expected
            )
            np.testing.assert_array_equal(
                tensors["shape_decoder.linear.weight"].data.reshape(linear.shape),
                linear.numpy().astype(np.float16),
            )
            np.testing.assert_array_equal(
                tensors["shape_decoder.bias"].data.reshape(bias.shape), bias.numpy()
            )

    def test_slat_sparse_conv_rejects_unknown_kernel_contract(self):
        bad = torch.zeros((2, 5, 5, 5, 4), dtype=torch.float16)
        with self.assertRaisesRegex(converter.ConversionError, "requires a 3x3x3 kernel"):
            converter._tensor_shape_and_value(bad, "slat-dec")

    def test_ss_decoder_conv3d_layout(self):
        value = torch.arange(2 * 4 * 3 * 3 * 3, dtype=torch.float32).reshape(2, 4, 3, 3, 3)
        shape, transformed = converter._tensor_shape_and_value(value, "ss-dec")
        self.assertEqual(shape, (8, 3, 3, 3))
        torch.testing.assert_close(transformed.reshape(value.shape), value)

    def test_slat_decoder_metadata_has_one_resolution_key(self):
        args = {
            "resolution": 256,
            "model_channels": [8, 4],
            "latent_channels": 3,
            "num_blocks": [1, 0],
            "block_type": ["SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d"],
            "up_block_type": ["SparseResBlockC2S3d"],
        }
        items = converter._slat_decoder_metadata(
            converter.MODELS["shape-dec"], args, "pixal3d.shape_decoder."
        )
        keys = []
        for item in items:
            size = struct.unpack_from("<Q", item, 0)[0]
            keys.append(item[8:8 + size].decode("utf-8"))
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(keys.count("pixal3d.shape_decoder.resolution"), 1)

    def test_file_type_metadata_values_match_pinned_ggml(self):
        self.assertEqual(converter.gguf_file_type(0), 0)
        self.assertEqual(converter.gguf_file_type(1), 1)
        self.assertEqual(converter.gguf_file_type(2), 24)

    def test_flow_tensor_names_fit_ggml_and_compact_stage_prefix(self):
        tensor = torch.zeros((2, 2), dtype=torch.float32)
        definition = converter.ModelDefinition(
            "synthetic-flow", "synthetic", "Synthetic", "synthetic",
            "shape_flow_1024", "base"
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "synthetic.safetensors"
            save_file({
                "blocks.29.cross_attn.cross_attn_block.q_rms_norm.gamma": tensor.clone(),
                "input_layer.weight": tensor.clone(),
                "input_layer.bias": torch.zeros(2),
                "out_layer.weight": tensor.clone(),
                "out_layer.bias": torch.zeros(2),
                "t_embedder.mlp.0.weight": tensor.clone(),
                "t_embedder.mlp.2.weight": tensor.clone(),
                "blocks.0.self_attn.to_qkv.weight": tensor.clone(),
            }, str(source))
            with converter.TensorStore(source) as store:
                specs = converter._tensor_specs(store, definition, 1, "shape_flow_1024.")
            names = [spec.name for spec in specs]
            self.assertIn(
                "sh1024.blocks.29.cross_attn.cross_attn_block.q_rms_norm.gamma",
                names,
            )
            self.assertTrue(all(len(name.encode("utf-8")) < converter.GGML_MAX_NAME
                                for name in names))


if __name__ == "__main__":
    unittest.main()
