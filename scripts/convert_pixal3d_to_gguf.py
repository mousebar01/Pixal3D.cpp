#!/usr/bin/env python3
"""Convert all Pixal3D checkpoints to architecture-aware GGUF v3 files.

The eleven Pixal3D source checkpoints are grouped into three deployment files;
the standalone DINOv3 vision checkpoint is available as a separate component:
``pixal3d-shared`` (three decoders), ``pixal3d-base-flow`` (four standard
flows), and ``pixal3d-mv-flow`` (four multi-view flows).  Tensor names are
prefixed by logical component inside each bundle and use the deterministic
``pixal3d.tensor_name_scheme=compact-v1`` aliases required by ggml's 64-byte
tensor-name field.

Examples:
  python scripts/convert_pixal3d_to_gguf.py --bundle all --dry-run
  python scripts/convert_pixal3d_to_gguf.py --bundle all
  python scripts/convert_pixal3d_to_gguf.py --bundle all --ftype 0
  python scripts/convert_pixal3d_to_gguf.py --component shape-dec
  python scripts/convert_pixal3d_to_gguf.py --component ss-flow
  python scripts/convert_pixal3d_to_gguf.py --component dino --model /path/dino.safetensors --config /path/config.json
  python scripts/convert_pixal3d_to_gguf.py --component naf --model /path/naf.safetensors

ftype: auto=the project default deployment mix (F32 flow, F16 decoder),
0=all F32, 1=mostly F16, 2=mostly BF16 (flow models only).  Use --ftype 0
for lossless F32 numerical-porting/reference runs; it does not change the
stable auto policy.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 30

# Keep the deployment default separate from the lossless/reference switch.
# ``auto`` is intentionally mixed: flow weights are stored as F32, while the
# released decoder weights remain F16.  ``--ftype 0`` is the explicit all-F32
# mode used for numerical porting and Python-reference parity.
AUTO_FTYPE_FLOW = 0
AUTO_FTYPE_DECODER = 1
AUTO_FTYPE_DESCRIPTION = "auto=F32 flow, F16 decoder"
GGML_MAX_DIMS = 4
GGML_MAX_NAME = 64  # ggml stores tensor names in char[GGML_MAX_NAME]
GGML_FTYPE_ALL_F32 = 0
GGML_FTYPE_MOSTLY_F16 = 1
GGML_FTYPE_MOSTLY_BF16 = 24
GGUF_VT_UINT32 = 4
GGUF_VT_FLOAT32 = 6
GGUF_VT_BOOL = 7
GGUF_VT_STRING = 8

PIXAL3D_ARCH = "pixal3d"
PIXAL3D_PREFIX = "pixal3d."
PIXAL3D_BUNDLE_FORMAT = 1
PIXAL3D_TENSOR_NAME_SCHEME = "compact-v1"
FLOW_COMPUTED_TENSORS = frozenset(("rope_phases",))

# ggml's tensor name field is 64 bytes including the terminating NUL.  The
# long stage prefixes are compacted while preserving the remainder of the
# PyTorch key verbatim.  This keeps names deterministic and easy to map in the
# C++ loader without changing decoder/source tensor names.
TENSOR_NAME_PREFIXES = (
    ("texture_flow_1024.", "tx1024."),
    ("shape_flow_1024.", "sh1024."),
    ("shape_flow_512.", "sh512."),
    ("ss_flow.", "ss."),
)


class ConversionError(RuntimeError):
    pass


class ModelDefinition(NamedTuple):
    key: str
    stem: str
    expected_class: str
    kind: str
    component: str
    variant: str


class TensorSpec(NamedTuple):
    name: str
    source_id: str
    source_name: str
    source_kind: str
    ggml_type: int
    dims: Tuple[int, ...]
    shape: Tuple[int, ...]
    nbytes: int


MODELS: Dict[str, ModelDefinition] = {
    # The DINOv3 checkpoint is kept as a separate component pack.  It lives
    # in the shared vision cache rather than weights/Pixal3D/ckpts, so callers
    # normally pass --model and --config explicitly.
    "dino": ModelDefinition("dino", "dinov3-vitl16-pretrain-lvd1689m",
        "DINOv3ViTModel", "dino", "dino", "vision"),
    "naf": ModelDefinition("naf", "naf", "NAF", "naf", "naf", "vision"),
    "ss-dec": ModelDefinition("ss-dec", "ss_dec_conv3d_16l8_fp16",
        "SparseStructureDecoder", "ss-dec", "ss_decoder", "shared"),
    "shape-dec": ModelDefinition("shape-dec", "shape_dec_next_dc_f16c32_fp16",
        "FlexiDualGridVaeDecoder", "slat-dec", "shape_decoder", "shared"),
    "tex-dec": ModelDefinition("tex-dec", "tex_dec_next_dc_f16c32_fp16",
        "SparseUnetVaeDecoder", "slat-dec", "texture_decoder", "shared"),
    "ss-flow": ModelDefinition("ss-flow", "ss_flow_img_dit_1_3B_64_bf16",
        "SparseStructureFlowModel", "ss-flow", "ss_flow", "base"),
    "shape-flow-512": ModelDefinition("shape-flow-512",
        "slat_flow_img2shape_dit_1_3B_512_bf16", "ElasticSLatFlowModel",
        "slat-flow", "shape_flow_512", "base"),
    "shape-flow-1024": ModelDefinition("shape-flow-1024",
        "slat_flow_img2shape_dit_1_3B_1024_bf16", "ElasticSLatFlowModel",
        "slat-flow", "shape_flow_1024", "base"),
    "tex-flow-1024": ModelDefinition("tex-flow-1024",
        "slat_flow_imgshape2tex_dit_1_3B_1024_bf16", "ElasticSLatFlowModel",
        "slat-flow", "texture_flow_1024", "base"),
    "ss-flow-mv": ModelDefinition("ss-flow-mv", "ss_flow_img_dit_1_3B_64_bf16_mv",
        "SparseStructureFlowModel", "ss-flow", "ss_flow", "mv"),
    "shape-flow-512-mv": ModelDefinition("shape-flow-512-mv",
        "slat_flow_img2shape_dit_1_3B_512_bf16_mv", "ElasticSLatFlowModel",
        "slat-flow", "shape_flow_512", "mv"),
    "shape-flow-1024-mv": ModelDefinition("shape-flow-1024-mv",
        "slat_flow_img2shape_dit_1_3B_1024_bf16_mv", "ElasticSLatFlowModel",
        "slat-flow", "shape_flow_1024", "mv"),
    "tex-flow-1024-mv": ModelDefinition("tex-flow-1024-mv",
        "slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv", "ElasticSLatFlowModel",
        "slat-flow", "texture_flow_1024", "mv"),
}

# The upstream valeoai/NAF torch.hub checkpoint is a bare state_dict and does
# not ship a JSON config.  Keep its constructor contract explicit so a caller
# can convert that canonical checkpoint without fabricating a sidecar file.
NAF_DEFAULT_CONFIG: Dict[str, Any] = {
    "model_type": "naf",
    "architectures": ["NAF"],
    "dim": 256,
    "in_channels": 3,
    "heads_attn": 4,
    "heads_rope": 4,
    "kernel_size": 9,
    "rope_base": 100.0,
    "img_layers": 2,
    "num_groups": 8,
}

BUNDLES: Dict[str, Tuple[str, ...]] = {
    "shared": ("ss-dec", "shape-dec", "tex-dec"),
    "base-flow": ("ss-flow", "shape-flow-512", "shape-flow-1024", "tex-flow-1024"),
    "mv-flow": ("ss-flow-mv", "shape-flow-512-mv", "shape-flow-1024-mv",
                "tex-flow-1024-mv"),
}
BUNDLE_OUTPUT_STEMS = {
    "shared": "pixal3d-shared",
    "base-flow": "pixal3d-base-flow",
    "mv-flow": "pixal3d-mv-flow",
}
EXPECTED_TENSOR_COUNTS = {
    "dino": 415,
    "ss-dec": 74,
    "shape-dec": 292,
    "tex-dec": 284,
}


class TensorStore:
    """Lazy safetensors reader used to avoid a second full checkpoint copy."""

    def __init__(self, path: Path):
        self.path = path
        self._context: Any = None
        self._handle: Any = None
        self._keys: Tuple[str, ...] = ()

    def __enter__(self) -> "TensorStore":
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ConversionError("missing dependency 'safetensors': {}".format(exc))
        try:
            self._context = safe_open(str(self.path), framework="pt", device="cpu")
            self._handle = self._context.__enter__()
            self._keys = tuple(sorted(self._handle.keys()))
            if not self._keys:
                raise ConversionError("empty checkpoint: {}".format(self.path))
            return self
        except ConversionError:
            self._close()
            raise
        except Exception as exc:
            self._close()
            raise ConversionError("cannot read {}: {}".format(self.path, exc))

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._close()

    def _close(self) -> None:
        if self._context is not None:
            try:
                self._context.__exit__(None, None, None)
            except Exception:
                pass
        self._context = self._handle = None
        self._keys = ()

    def keys(self) -> Tuple[str, ...]:
        if self._handle is None:
            raise ConversionError("tensor store is not open")
        return self._keys

    def get(self, name: str) -> Any:
        if self._handle is None:
            raise ConversionError("tensor store is not open")
        try:
            return self._handle.get_tensor(name)
        except Exception as exc:
            raise ConversionError("cannot read tensor '{}': {}".format(name, exc))


def _gguf_str(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _kv(key: str, value_type: int, payload: bytes) -> bytes:
    return _gguf_str(key) + struct.pack("<I", value_type) + payload


def kv_u32(key: str, value: int) -> bytes:
    return _kv(key, GGUF_VT_UINT32, struct.pack("<I", int(value)))


def kv_f32(key: str, value: float) -> bytes:
    return _kv(key, GGUF_VT_FLOAT32, struct.pack("<f", float(value)))


def kv_bool(key: str, value: bool) -> bytes:
    return _kv(key, GGUF_VT_BOOL, struct.pack("<?", bool(value)))


def kv_str(key: str, value: str) -> bytes:
    return _kv(key, GGUF_VT_STRING, _gguf_str(str(value)))


def align(value: int, alignment: int = GGUF_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def gguf_file_type(ftype: int) -> int:
    try:
        return {0: GGML_FTYPE_ALL_F32, 1: GGML_FTYPE_MOSTLY_F16,
                2: GGML_FTYPE_MOSTLY_BF16}[ftype]
    except KeyError:
        raise ConversionError("unsupported ftype {}".format(ftype))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_weights_dir() -> Path:
    return _project_root() / "weights" / "Pixal3D" / "ckpts"


def _default_output_dir() -> Path:
    return _project_root() / "build" / "weights"


def _normalise_model_path(path: Path) -> Path:
    return path if path.suffix == ".safetensors" else Path(str(path) + ".safetensors")


def resolve_checkpoint(weights_dir: Path, stem: str, model_arg: Optional[str] = None,
                       config_arg: Optional[str] = None,
                       allow_missing_config: bool = False) -> Tuple[Path, Path]:
    model = _normalise_model_path(Path(model_arg).expanduser()) if model_arg else \
        weights_dir / (stem + ".safetensors")
    part = Path(str(model) + ".part")
    if not model.exists():
        if part.exists():
            raise ConversionError("still downloading: {} ({} bytes)".format(part, part.stat().st_size))
        raise ConversionError("checkpoint not found: {}".format(model))
    if not model.is_file():
        raise ConversionError("checkpoint is not a file: {}".format(model))
    config = Path(config_arg).expanduser() if config_arg else \
        Path(str(model)[:-len(".safetensors")] + ".json")
    if not config.is_file() and not allow_missing_config:
        raise ConversionError("config not found: {}".format(config))
    return model, config


def load_model_config(path: Path, kind: str = "") -> Tuple[str, Dict[str, Any]]:
    if kind == "naf" and not path.is_file():
        return "NAF", dict(NAF_DEFAULT_CONFIG)
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ConversionError("cannot read config {}: {}".format(path, exc))
    if kind == "dino" and isinstance(config, dict) and \
            config.get("model_type") == "dinov3_vit":
        architectures = config.get("architectures", ("DINOv3ViTModel",))
        name = architectures[0] if isinstance(architectures, list) and architectures else \
            "DINOv3ViTModel"
        args = config
    elif kind == "naf" and isinstance(config, dict) and \
            config.get("model_type") == "naf":
        architectures = config.get("architectures", ("NAF",))
        name = architectures[0] if isinstance(architectures, list) and architectures else "NAF"
        args = config
    elif isinstance(config, dict) and isinstance(config.get("args"), dict):
        name, args = config.get("name", ""), config["args"]
    else:
        denoiser = config.get("models", {}).get("denoiser", {}) \
            if isinstance(config, dict) else {}
        name, args = denoiser.get("name", ""), denoiser.get("args", {})
    if not isinstance(name, str) or not name or not isinstance(args, dict):
        raise ConversionError("config lacks model name/args: {}".format(path))
    return name, args


def _require_args(args: Dict[str, Any], names: Iterable[str], key: str) -> None:
    missing = [name for name in names if name not in args]
    if missing:
        raise ConversionError("{} config missing: {}".format(key, ", ".join(missing)))


def _require_list(args: Dict[str, Any], name: str, key: str) -> Sequence[Any]:
    value = args.get(name)
    if not isinstance(value, (list, tuple)) or not value:
        raise ConversionError("{} '{}' must be a non-empty list".format(key, name))
    return value


def choose_flow_type(name: str, shape: Sequence[int], ftype: int) -> int:
    if ftype == 0:
        return GGML_TYPE_F32
    if ftype == 2:
        return GGML_TYPE_BF16
    sensitive = "gamma" in name or "modulation" in name or "norm" in name
    return GGML_TYPE_F16 if len(shape) >= 2 and not sensitive else GGML_TYPE_F32


def choose_decoder_type(shape: Sequence[int], ftype: int) -> int:
    if ftype == 0:
        return GGML_TYPE_F32
    return GGML_TYPE_F16 if len(shape) >= 2 else GGML_TYPE_F32


def choose_dino_type(name: str, shape: Sequence[int], ftype: int) -> int:
    # Native DINOv3 is deliberately introduced with a lossless F32 pack.  We
    # can add a mixed-precision policy after a reference comparison proves the
    # attention/LayerNorm path on the target backends.
    if ftype != 0:
        raise ConversionError("DINOv3 currently requires --ftype 0 (F32)")
    return GGML_TYPE_F32


def choose_naf_type(name: str, shape: Sequence[int], ftype: int) -> int:
    if ftype != 0:
        raise ConversionError("NAF currently requires --ftype 0 (F32)")
    return GGML_TYPE_F32


def _element_count(shape: Sequence[int]) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count


def _bytes_per_type(ggml_type: int) -> int:
    if ggml_type in (GGML_TYPE_F16, GGML_TYPE_BF16):
        return 2
    if ggml_type == GGML_TYPE_F32:
        return 4
    raise ConversionError("unsupported ggml type {}".format(ggml_type))


def _tensor_shape_and_value(tensor: Any, kind: str,
                            tensor_name: str = "") -> Tuple[Tuple[int, ...], Any]:
    """Apply the format-v1 model-specific storage transform."""
    try:
        if not tensor.is_floating_point():
            raise ConversionError("non-floating tensor dtype {}".format(tensor.dtype))
        value = tensor.detach().cpu().contiguous()
        shape = tuple(int(dim) for dim in value.shape)
        if kind == "ss-dec" and len(shape) == 5:
            # [OC, IC, kD, kH, kW] -> [OC*IC, kD, kH, kW].
            oc, ic, kd, kh, kw = shape
            shape = (oc * ic, kd, kh, kw)
            value = value.reshape(shape).contiguous()
        elif kind == "slat-dec" and len(shape) == 5:
            # flex_gemm [CO, kD, kH, kW, CI] -> [kernel_volume, CO, CI].
            # Putting the kernel offset first makes every [CO, CI] weight
            # matrix contiguous.  GGML sees [CI, CO, kernel_volume], so one
            # sparse-neighbor offset can be passed directly to ggml_mul_mat.
            co, kd, kh, kw, ci = shape
            if tensor_name and not tensor_name.endswith(
                    ("conv.weight", "conv1.weight", "conv2.weight")):
                raise ConversionError(
                    "unexpected five-dimensional SLat tensor '{}'".format(tensor_name))
            if (kd, kh, kw) != (3, 3, 3):
                raise ConversionError("SLat format v1 requires a 3x3x3 kernel, got {}x{}x{}"
                                      .format(kd, kh, kw))
            shape = (kd * kh * kw, co, ci)
            value = value.permute(1, 2, 3, 0, 4).reshape(shape).contiguous()
        return shape, value
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError("cannot inspect tensor: {}".format(exc))


def _validate_inventory(definition: ModelDefinition, store: TensorStore,
                        args: Dict[str, Any]) -> None:
    keys = set(store.keys())
    if definition.kind == "naf":
        # NAF is a small image-guided feature upsampler.  Its checkpoint has
        # no learned attention projections: only the two convolutional
        # encoder stacks and the persistent RoPE periods are serialized.
        dim = int(args.get("dim", 256))
        in_channels = int(args.get("in_channels", 3))
        heads_attn = int(args.get("heads_attn", 4))
        heads_rope = int(args.get("heads_rope", 4))
        kernel_size = int(args.get("kernel_size", 9))
        img_layers = int(args.get("img_layers", 2))
        num_groups = int(args.get("num_groups", 8))
        rope_base = float(args.get("rope_base", 100.0))
        if (dim <= 0 or dim % 2 != 0 or in_channels <= 0 or heads_attn <= 0 or
                heads_rope <= 0 or dim % heads_attn != 0 or dim % heads_rope != 0 or
                (dim // heads_rope) % 4 != 0 or kernel_size <= 0 or
                kernel_size % 2 == 0 or img_layers <= 0 or num_groups <= 0 or
                (dim // 2) % num_groups != 0 or rope_base <= 0.0):
            raise ConversionError("naf has invalid dimensions or numerical parameters")
        half = dim // 2
        required = {"image_encoder.encoder.0.weight", "image_encoder.encoder.0.bias",
                    "image_encoder.sem_encoder.0.weight", "image_encoder.sem_encoder.0.bias",
                    "image_encoder.rope.periods"}
        expected = {
            "image_encoder.encoder.0.weight": (half, in_channels, 1, 1),
            "image_encoder.encoder.0.bias": (half,),
            "image_encoder.sem_encoder.0.weight": (half, in_channels, 3, 3),
            "image_encoder.sem_encoder.0.bias": (half,),
            "image_encoder.rope.periods": (dim // heads_rope // 4,),
        }
        for stack in ("encoder", "sem_encoder"):
            block_kernel = 1 if stack == "encoder" else 3
            for layer in range(1, img_layers + 1):
                p = "image_encoder.{}.{}".format(stack, layer)
                required.update({p + ".norm1.weight", p + ".norm1.bias",
                                 p + ".conv1.weight", p + ".conv1.bias",
                                 p + ".norm2.weight", p + ".norm2.bias",
                                 p + ".conv2.weight", p + ".conv2.bias"})
                expected.update({
                    p + ".norm1.weight": (half,), p + ".norm1.bias": (half,),
                    p + ".conv1.weight": (half, half, block_kernel, block_kernel),
                    p + ".conv1.bias": (half,), p + ".norm2.weight": (half,),
                    p + ".norm2.bias": (half,),
                    p + ".conv2.weight": (half, half, block_kernel, block_kernel),
                    p + ".conv2.bias": (half,),
                })
        missing = sorted(required - keys)
        if missing:
            raise ConversionError("{} missing tensor(s): {}".format(
                definition.key, ", ".join(missing)))
        if keys != required:
            extra = sorted(keys - required)
            raise ConversionError("{} has unexpected tensor(s): {}".format(
                definition.key, ", ".join(extra)))
        for name, wanted in expected.items():
            got = tuple(int(dim_value) for dim_value in store.get(name).shape)
            if got != wanted:
                raise ConversionError("{} '{}' shape {} != config {}".format(
                    definition.key, name, got, wanted))
        return
    if definition.kind == "dino":
        _require_args(args, ("hidden_size", "intermediate_size", "num_hidden_layers",
                             "num_attention_heads", "patch_size", "num_channels",
                             "num_register_tokens", "layer_norm_eps", "rope_theta",
                             "use_gated_mlp"), definition.key)
        if args.get("use_gated_mlp", False):
            raise ConversionError("dino gated MLP is not supported by format v1")
        hidden = int(args["hidden_size"])
        intermediate = int(args["intermediate_size"])
        layers = int(args["num_hidden_layers"])
        heads = int(args["num_attention_heads"])
        patch = int(args["patch_size"])
        channels = int(args["num_channels"])
        registers = int(args["num_register_tokens"])
        if (hidden <= 0 or intermediate <= 0 or layers <= 0 or heads <= 0 or
                hidden % heads != 0 or patch <= 0 or channels <= 0 or
                registers < 0 or float(args["rope_theta"]) <= 0.0 or
                float(args["layer_norm_eps"]) <= 0.0):
            raise ConversionError("dino has invalid dimensions or numerical parameters")
        required = {
            "embeddings.cls_token", "embeddings.mask_token",
            "embeddings.patch_embeddings.bias", "embeddings.patch_embeddings.weight",
            "embeddings.register_tokens", "norm.bias", "norm.weight",
        }
        for index in range(layers):
            prefix = "layer.{}".format(index)
            layer_required = {
                prefix + ".attention.k_proj.weight",
                prefix + ".attention.o_proj.weight",
                prefix + ".attention.q_proj.weight",
                prefix + ".attention.v_proj.weight",
                prefix + ".layer_scale1.lambda1", prefix + ".layer_scale2.lambda1",
                prefix + ".mlp.down_proj.weight", prefix + ".mlp.up_proj.weight",
                prefix + ".norm1.bias", prefix + ".norm1.weight",
                prefix + ".norm2.bias", prefix + ".norm2.weight",
            }
            if args.get("query_bias", True):
                layer_required.add(prefix + ".attention.q_proj.bias")
            if args.get("value_bias", True):
                layer_required.add(prefix + ".attention.v_proj.bias")
            if args.get("proj_bias", True):
                layer_required.add(prefix + ".attention.o_proj.bias")
            if args.get("mlp_bias", True):
                layer_required.update({prefix + ".mlp.down_proj.bias",
                                      prefix + ".mlp.up_proj.bias"})
            if args.get("key_bias", False):
                layer_required.add(prefix + ".attention.k_proj.bias")
            required.update(layer_required)
        expected = {
            "embeddings.cls_token": (1, 1, hidden),
            "embeddings.mask_token": (1, 1, hidden),
            "embeddings.register_tokens": (1, registers, hidden),
            "embeddings.patch_embeddings.weight": (hidden, channels, patch, patch),
            "embeddings.patch_embeddings.bias": (hidden,),
            "norm.weight": (hidden,), "norm.bias": (hidden,),
        }
        for index in range(layers):
            prefix = "layer.{}".format(index)
            layer_expected = {
                prefix + ".attention.k_proj.weight": (hidden, hidden),
                prefix + ".attention.o_proj.weight": (hidden, hidden),
                prefix + ".attention.q_proj.weight": (hidden, hidden),
                prefix + ".attention.v_proj.weight": (hidden, hidden),
                prefix + ".layer_scale1.lambda1": (hidden,),
                prefix + ".layer_scale2.lambda1": (hidden,),
                prefix + ".mlp.down_proj.weight": (hidden, intermediate),
                prefix + ".mlp.up_proj.weight": (intermediate, hidden),
                prefix + ".norm1.weight": (hidden,), prefix + ".norm1.bias": (hidden,),
                prefix + ".norm2.weight": (hidden,), prefix + ".norm2.bias": (hidden,),
            }
            if args.get("query_bias", True): layer_expected[prefix + ".attention.q_proj.bias"] = (hidden,)
            if args.get("key_bias", False): layer_expected[prefix + ".attention.k_proj.bias"] = (hidden,)
            if args.get("value_bias", True): layer_expected[prefix + ".attention.v_proj.bias"] = (hidden,)
            if args.get("proj_bias", True): layer_expected[prefix + ".attention.o_proj.bias"] = (hidden,)
            if args.get("mlp_bias", True):
                layer_expected[prefix + ".mlp.down_proj.bias"] = (hidden,)
                layer_expected[prefix + ".mlp.up_proj.bias"] = (intermediate,)
            expected.update(layer_expected)
        missing = sorted(required - keys)
        if missing:
            raise ConversionError("{} missing tensor(s): {}".format(
                definition.key, ", ".join(missing)))
        if keys != required:
            extra = sorted(keys - required)
            raise ConversionError("{} has unexpected tensor(s): {}".format(
                definition.key, ", ".join(extra)))
        for name, wanted in expected.items():
            got = tuple(int(dim) for dim in store.get(name).shape)
            if got != wanted:
                raise ConversionError("{} '{}' shape {} != config {}".format(
                    definition.key, name, got, wanted))
        return
    if definition.kind in ("ss-flow", "slat-flow"):
        _require_args(args, ("in_channels", "out_channels", "model_channels",
                             "num_blocks"), definition.key)
        last = int(args["num_blocks"]) - 1
        required = {"input_layer.weight", "input_layer.bias", "out_layer.weight",
                    "out_layer.bias", "t_embedder.mlp.0.weight",
                    "t_embedder.mlp.2.weight", "blocks.0.self_attn.to_qkv.weight",
                    "blocks.{}.self_attn.to_qkv.weight".format(last)}
        if args.get("image_attn_mode") in ("proj", "gated_proj"):
            required.add("blocks.0.cross_attn.proj_linear.weight")
    elif definition.kind == "ss-dec":
        required = {"input_layer.weight", "input_layer.bias", "out_layer.2.weight",
                    "out_layer.2.bias", "middle_block.0.conv1.weight"}
    else:
        required = {"from_latent.weight", "from_latent.bias", "output_layer.weight",
                    "output_layer.bias", "blocks.0.0.conv.weight"}
    missing = sorted(required - keys)
    if missing:
        raise ConversionError("{} missing tensor(s): {}".format(definition.key, ", ".join(missing)))

    if definition.kind in ("ss-flow", "slat-flow"):
        expected = {
            "input_layer.weight": (int(args["model_channels"]), int(args["in_channels"])),
            "out_layer.weight": (int(args["out_channels"]), int(args["model_channels"])),
        }
    elif definition.kind == "slat-dec":
        channels = _require_list(args, "model_channels", definition.key)
        out_channels = 7 if definition.expected_class == "FlexiDualGridVaeDecoder" \
            else int(args["out_channels"])
        expected = {
            "from_latent.weight": (int(channels[0]), int(args["latent_channels"])),
            "output_layer.weight": (out_channels, int(channels[-1])),
        }
    else:
        expected = {}
    for name, wanted in expected.items():
        got = tuple(int(dim) for dim in store.get(name).shape)
        if got != wanted:
            raise ConversionError("{} '{}' shape {} != config {}".format(
                definition.key, name, got, wanted))


def _tensor_specs(store: TensorStore, definition: ModelDefinition, ftype: int,
                  output_prefix: str) -> List[TensorSpec]:
    specs: List[TensorSpec] = []
    source_5d_count = 0
    for source_name in store.keys():
        if definition.kind == "ss-flow" and source_name in FLOW_COMPUTED_TENSORS:
            continue
        source_tensor = store.get(source_name)
        if len(source_tensor.shape) == 5:
            source_5d_count += 1
        shape, _ = _tensor_shape_and_value(source_tensor, definition.kind, source_name)
        if not shape or any(dim <= 0 for dim in shape) or len(shape) > GGML_MAX_DIMS:
            raise ConversionError("unsupported transformed shape {} for '{}'".format(shape, source_name))
        if definition.kind in ("ss-flow", "slat-flow"):
            ggml_type = choose_flow_type(source_name, shape, ftype)
        elif definition.kind == "dino":
            ggml_type = choose_dino_type(source_name, shape, ftype)
        elif definition.kind == "naf":
            ggml_type = choose_naf_type(source_name, shape, ftype)
        else:
            ggml_type = choose_decoder_type(shape, ftype)
        output_name = output_prefix + source_name
        for long_prefix, compact_prefix in TENSOR_NAME_PREFIXES:
            if output_name.startswith(long_prefix):
                output_name = compact_prefix + output_name[len(long_prefix):]
                break
        if len(output_name.encode("utf-8")) >= GGML_MAX_NAME:
            raise ConversionError(
                "tensor name too long for ggml ({} bytes >= {}): '{}'"
                .format(len(output_name.encode("utf-8")), GGML_MAX_NAME, output_name))
        specs.append(TensorSpec(
            output_name, definition.key, source_name, definition.kind,
            ggml_type, tuple(reversed(shape)), shape,
            _element_count(shape) * _bytes_per_type(ggml_type)))
    if len({spec.name for spec in specs}) != len(specs):
        raise ConversionError("duplicate output tensor name")
    expected_count = 700 if definition.kind in ("ss-flow", "slat-flow") else \
        EXPECTED_TENSOR_COUNTS.get(definition.key)
    if expected_count is not None and len(specs) != expected_count:
        raise ConversionError("{} has {} serializable tensors, format v1 expects {}".format(
            definition.key, len(specs), expected_count))
    if definition.key in ("shape-dec", "tex-dec") and source_5d_count != 40:
        raise ConversionError("{} has {} sparse-conv weights, format v1 expects 40".format(
            definition.key, source_5d_count))
    return specs


def _tensor_bytes(tensor: Any, kind: str, spec: TensorSpec) -> bytes:
    shape, value = _tensor_shape_and_value(tensor, kind, spec.source_name)
    if shape != spec.shape:
        raise ConversionError("'{}' shape changed while writing".format(spec.name))
    try:
        import numpy as np
        arr = value.float().numpy()
        if spec.ggml_type == GGML_TYPE_F32:
            return arr.astype("<f4", copy=False).tobytes(order="C")
        if spec.ggml_type == GGML_TYPE_F16:
            return arr.astype("<f2", copy=False).tobytes(order="C")
        if spec.ggml_type == GGML_TYPE_BF16:
            bits = np.asarray(arr, dtype=np.float32).view(np.uint32)
            rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
            return rounded.astype("<u2", copy=False).tobytes(order="C")
    except ImportError as exc:
        raise ConversionError("NumPy is required: {}".format(exc))
    except Exception as exc:
        raise ConversionError("cannot serialize '{}': {}".format(spec.name, exc))
    raise ConversionError("unsupported ggml type {}".format(spec.ggml_type))


def _base_metadata(architecture: str, name: str, ftype: int) -> List[bytes]:
    return [kv_str("general.architecture", architecture), kv_str("general.name", name),
            kv_u32("general.file_type", gguf_file_type(ftype)),
            kv_u32("general.alignment", GGUF_ALIGNMENT)]


def _flow_metadata(definition: ModelDefinition, args: Dict[str, Any], prefix: str) -> List[bytes]:
    _require_args(args, ("resolution", "in_channels", "out_channels", "model_channels",
                         "cond_channels", "num_blocks", "num_heads", "mlp_ratio"), definition.key)
    rope = args.get("rope_freq", (1.0, 10000.0))
    if not isinstance(rope, (list, tuple)) or len(rope) != 2:
        raise ConversionError("{} rope_freq must contain two values".format(definition.key))
    result = [kv_str(prefix + "model_class", definition.expected_class),
        kv_str(prefix + "checkpoint", definition.stem), kv_str(prefix + "variant", definition.variant),
        kv_u32(prefix + "resolution", args["resolution"]),
        kv_u32(prefix + "in_channels", args["in_channels"]),
        kv_u32(prefix + "out_channels", args["out_channels"]),
        kv_u32(prefix + "model_channels", args["model_channels"]),
        kv_u32(prefix + "cond_channels", args["cond_channels"]),
        kv_u32(prefix + "num_blocks", args["num_blocks"]),
        kv_u32(prefix + "num_heads", args["num_heads"]),
        kv_f32(prefix + "mlp_ratio", args["mlp_ratio"]),
        kv_str(prefix + "pe_mode", args.get("pe_mode", "rope")),
        kv_bool(prefix + "share_mod", args.get("share_mod", False)),
        kv_bool(prefix + "qk_rms_norm", args.get("qk_rms_norm", False)),
        kv_bool(prefix + "qk_rms_norm_cross", args.get("qk_rms_norm_cross", False)),
        kv_f32(prefix + "rope_freq_min", rope[0]), kv_f32(prefix + "rope_freq_base", rope[1]),
        kv_str(prefix + "image_attn_mode", args.get("image_attn_mode", "cross"))]
    if args.get("proj_in_channels") is not None:
        result.append(kv_u32(prefix + "proj_in_channels", args["proj_in_channels"]))
    if args.get("vae_in_channels") is not None:
        result.append(kv_u32(prefix + "vae_in_channels", args["vae_in_channels"]))
    return result


def _ss_decoder_metadata(definition: ModelDefinition, args: Dict[str, Any],
                         prefix: str) -> List[bytes]:
    _require_args(args, ("out_channels", "latent_channels", "num_res_blocks",
                         "num_res_blocks_middle", "channels"), definition.key)
    channels = _require_list(args, "channels", definition.key)
    result = [kv_str(prefix + "model_class", definition.expected_class),
        kv_str(prefix + "checkpoint", definition.stem), kv_str(prefix + "variant", definition.variant),
        kv_u32(prefix + "out_channels", args["out_channels"]),
        kv_u32(prefix + "latent_channels", args["latent_channels"]),
        kv_u32(prefix + "num_res_blocks", args["num_res_blocks"]),
        kv_u32(prefix + "num_res_blocks_middle", args["num_res_blocks_middle"]),
        kv_u32(prefix + "n_levels", len(channels)),
        kv_str(prefix + "norm_type", args.get("norm_type", "layer")),
        kv_f32(prefix + "norm_eps", args.get("norm_eps", 1e-5)),
        kv_str(prefix + "conv_weight_source_layout", "out,in,kd,kh,kw"),
        kv_str(prefix + "conv_weight_gguf_layout", "out_mul_in,kd,kh,kw")]
    result.extend(kv_u32(prefix + "channels.{}".format(i), value)
                  for i, value in enumerate(channels))
    return result


def _slat_decoder_metadata(definition: ModelDefinition, args: Dict[str, Any],
                           prefix: str) -> List[bytes]:
    _require_args(args, ("model_channels", "latent_channels", "num_blocks", "block_type",
                         "up_block_type"), definition.key)
    channels = _require_list(args, "model_channels", definition.key)
    blocks = _require_list(args, "num_blocks", definition.key)
    block_types = _require_list(args, "block_type", definition.key)
    up_types = _require_list(args, "up_block_type", definition.key)
    if not (len(channels) == len(blocks) == len(block_types)) or len(up_types) != len(channels) - 1:
        raise ConversionError("{} decoder level lists have inconsistent lengths".format(definition.key))
    out_channels = 7 if definition.expected_class == "FlexiDualGridVaeDecoder" \
        else int(args["out_channels"])
    result = [kv_str(prefix + "model_class", definition.expected_class),
        kv_str(prefix + "checkpoint", definition.stem), kv_str(prefix + "variant", definition.variant),
        kv_u32(prefix + "out_channels", out_channels),
        kv_u32(prefix + "latent_channels", args["latent_channels"]),
        kv_f32(prefix + "norm_eps", args.get("norm_eps", 1e-6)),
        # The released SS decoder is fixed at 16^3 input resolution, but keep
        # the value explicit so synthetic/minimal fixtures and future decoder
        # variants do not need a hard-coded runtime assumption.
        kv_u32(prefix + "resolution", args.get("resolution", 16)),
        kv_u32(prefix + "n_levels", len(channels)),
        kv_bool(prefix + "pred_subdiv", args.get("pred_subdiv", True)),
        kv_bool(prefix + "use_fp16", args.get("use_fp16", False)),
        kv_str(prefix + "conv_weight_source_layout", "out,kd,kh,kw,in"),
        kv_str(prefix + "conv_weight_gguf_layout", "kernel_volume,out,in"),
        kv_u32(prefix + "conv_kernel_size", 3), kv_u32(prefix + "conv_kernel_volume", 27)]
    for i, value in enumerate(channels):
        result += [kv_u32(prefix + "model_channels.{}".format(i), value),
                   kv_u32(prefix + "num_blocks.{}".format(i), blocks[i]),
                   kv_str(prefix + "block_type.{}".format(i), block_types[i])]
    result.extend(kv_str(prefix + "up_block_type.{}".format(i), value)
                  for i, value in enumerate(up_types))
    return result


def _dino_metadata(definition: ModelDefinition, args: Dict[str, Any],
                   prefix: str) -> List[bytes]:
    _require_args(args, ("hidden_size", "intermediate_size", "num_hidden_layers",
                         "num_attention_heads", "patch_size", "num_channels",
                         "num_register_tokens", "layer_norm_eps", "rope_theta",
                         "use_gated_mlp"), definition.key)
    return [
        kv_str(prefix + "model_class", definition.expected_class),
        kv_str(prefix + "checkpoint", definition.stem),
        kv_str(prefix + "variant", definition.variant),
        kv_u32(prefix + "hidden_size", args["hidden_size"]),
        kv_u32(prefix + "intermediate_size", args["intermediate_size"]),
        kv_u32(prefix + "num_hidden_layers", args["num_hidden_layers"]),
        kv_u32(prefix + "num_attention_heads", args["num_attention_heads"]),
        kv_u32(prefix + "patch_size", args["patch_size"]),
        kv_u32(prefix + "num_channels", args["num_channels"]),
        kv_u32(prefix + "num_register_tokens", args["num_register_tokens"]),
        kv_f32(prefix + "layer_norm_eps", args["layer_norm_eps"]),
        kv_f32(prefix + "rope_theta", args["rope_theta"]),
        kv_bool(prefix + "use_gated_mlp", args["use_gated_mlp"]),
        kv_bool(prefix + "query_bias", args.get("query_bias", True)),
        kv_bool(prefix + "key_bias", args.get("key_bias", False)),
        kv_bool(prefix + "value_bias", args.get("value_bias", True)),
        kv_bool(prefix + "proj_bias", args.get("proj_bias", True)),
        kv_bool(prefix + "mlp_bias", args.get("mlp_bias", True)),
        kv_str(prefix + "patch_weight_source_layout", "out,in,kh,kw"),
        kv_str(prefix + "patch_weight_gguf_layout", "kw,kh,in,out"),
        kv_str(prefix + "output_contract", "affine_free_ln_cls_register_patch"),
    ]


def _naf_metadata(definition: ModelDefinition, args: Dict[str, Any],
                  prefix: str) -> List[bytes]:
    dim = int(args.get("dim", 256))
    heads_attn = int(args.get("heads_attn", 4))
    heads_rope = int(args.get("heads_rope", 4))
    kernel_size = int(args.get("kernel_size", 9))
    img_layers = int(args.get("img_layers", 2))
    num_groups = int(args.get("num_groups", 8))
    return [
        kv_str(prefix + "model_class", definition.expected_class),
        kv_str(prefix + "checkpoint", definition.stem),
        kv_str(prefix + "variant", definition.variant),
        kv_u32(prefix + "dim", dim), kv_u32(prefix + "in_channels", args.get("in_channels", 3)),
        kv_u32(prefix + "heads_attn", heads_attn), kv_u32(prefix + "heads_rope", heads_rope),
        kv_u32(prefix + "kernel_size", kernel_size), kv_u32(prefix + "img_layers", img_layers),
        kv_u32(prefix + "num_groups", num_groups),
        kv_f32(prefix + "rope_base", args.get("rope_base", 100.0)),
        kv_str(prefix + "padding", "reflect"),
        kv_str(prefix + "pooling", "adaptive_avg_pool2d"),
        kv_str(prefix + "attention", "neighborhood_cross_attention"),
    ]


def _model_metadata(definition: ModelDefinition, args: Dict[str, Any], prefix: str) -> List[bytes]:
    if definition.kind == "naf":
        return _naf_metadata(definition, args, prefix)
    if definition.kind == "dino":
        return _dino_metadata(definition, args, prefix)
    if definition.kind in ("ss-flow", "slat-flow"):
        return _flow_metadata(definition, args, prefix)
    if definition.kind == "ss-dec":
        return _ss_decoder_metadata(definition, args, prefix)
    return _slat_decoder_metadata(definition, args, prefix)


def _build_tensor_infos(specs: Sequence[TensorSpec]) -> Tuple[bytes, List[int]]:
    infos, offsets, offset = bytearray(), [], 0
    for spec in specs:
        if len(spec.name.encode("utf-8")) >= GGML_MAX_NAME:
            raise ConversionError(
                "tensor name too long for ggml ({} bytes >= {}): '{}'"
                .format(len(spec.name.encode("utf-8")), GGML_MAX_NAME, spec.name))
        offsets.append(offset)
        offset = align(offset + spec.nbytes)
    for spec, tensor_offset in zip(specs, offsets):
        infos += _gguf_str(spec.name) + struct.pack("<I", len(spec.dims))
        for dim in spec.dims:
            infos += struct.pack("<Q", int(dim))
        infos += struct.pack("<I", spec.ggml_type) + struct.pack("<Q", tensor_offset)
    return bytes(infos), offsets


def _serialized_layout(metadata: Sequence[bytes], specs: Sequence[TensorSpec]
                       ) -> Tuple[bytes, bytes, List[int], int, int]:
    if not specs:
        raise ConversionError("cannot write an empty GGUF")
    header = bytearray(GGUF_MAGIC) + struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<Q", len(specs)) + struct.pack("<Q", len(metadata))
    for item in metadata:
        header += item
    infos, offsets = _build_tensor_infos(specs)
    data_offset = align(len(header) + len(infos))
    size = data_offset + align(offsets[-1] + specs[-1].nbytes)
    return bytes(header), infos, offsets, data_offset, size


def _write_gguf(output: Path, metadata: Sequence[bytes], specs: Sequence[TensorSpec],
                stores: Dict[str, TensorStore], force: bool) -> None:
    if output.exists() and not force:
        raise ConversionError("output exists: {} (use --force)".format(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    header, infos, offsets, data_offset, expected_size = _serialized_layout(metadata, specs)
    temp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix="." + output.name + ".",
                suffix=".tmp", dir=str(output.parent), delete=False) as handle:
            temp_name = handle.name
            handle.write(header); handle.write(infos)
            handle.write(b"\x00" * (data_offset - handle.tell()))
            for spec, wanted_offset in zip(specs, offsets):
                if handle.tell() - data_offset != wanted_offset:
                    raise ConversionError("internal offset failure before '{}'".format(spec.name))
                raw = _tensor_bytes(stores[spec.source_id].get(spec.source_name),
                                    spec.source_kind, spec)
                if len(raw) != spec.nbytes:
                    raise ConversionError("'{}' wrote {} bytes, expected {}".format(
                        spec.name, len(raw), spec.nbytes))
                if handle.write(raw) != len(raw):
                    raise ConversionError("short write for '{}'".format(spec.name))
                handle.write(b"\x00" * (align(len(raw)) - len(raw)))
            if handle.tell() != expected_size:
                raise ConversionError("file size {} != plan {}".format(handle.tell(), expected_size))
            handle.flush(); os.fsync(handle.fileno())
        # NamedTemporaryFile honors a restrictive 0600 default.  Model files
        # are immutable deployment artifacts and should be readable by other
        # users/containers while remaining writable only by their owner.
        os.chmod(temp_name, 0o644)
        os.replace(temp_name, output)
        temp_name = None
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def _read_exact(handle: Any, count: int) -> bytes:
    data = handle.read(count)
    if len(data) != count:
        raise ConversionError("truncated GGUF")
    return data


def _u32(handle: Any) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _u64(handle: Any) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _string(handle: Any) -> str:
    return _read_exact(handle, _u64(handle)).decode("utf-8")


def _skip_value(handle: Any, value_type: int) -> None:
    if value_type in (GGUF_VT_UINT32, GGUF_VT_FLOAT32):
        _read_exact(handle, 4)
    elif value_type == GGUF_VT_BOOL:
        _read_exact(handle, 1)
    elif value_type == GGUF_VT_STRING:
        _read_exact(handle, _u64(handle))
    else:
        raise ConversionError("verifier cannot parse metadata type {}".format(value_type))


def verify_gguf(path: Path, specs: Sequence[TensorSpec]) -> None:
    """Independently validate the written header, table, offsets and size."""
    with path.open("rb") as handle:
        if _read_exact(handle, 4) != GGUF_MAGIC or _u32(handle) != GGUF_VERSION:
            raise ConversionError("verification failed: invalid GGUF header")
        tensor_count, metadata_count = _u64(handle), _u64(handle)
        if tensor_count != len(specs):
            raise ConversionError("verification failed: tensor count")
        keys = set()
        for _ in range(metadata_count):
            key = _string(handle)
            if key in keys:
                raise ConversionError("verification failed: duplicate KV '{}'".format(key))
            keys.add(key); _skip_value(handle, _u32(handle))
        required = {"general.architecture", "general.name", "general.file_type",
                    "general.alignment"}
        if not required.issubset(keys):
            raise ConversionError("verification failed: missing general metadata")
        offsets = []
        for spec in specs:
            name, n_dims = _string(handle), _u32(handle)
            dims = tuple(_u64(handle) for _ in range(n_dims))
            ggml_type, tensor_offset = _u32(handle), _u64(handle)
            if (name, dims, ggml_type) != (spec.name, spec.dims, spec.ggml_type):
                raise ConversionError("verification failed at tensor '{}'".format(name))
            if tensor_offset % GGUF_ALIGNMENT:
                raise ConversionError("verification failed: unaligned '{}'".format(name))
            offsets.append(tensor_offset)
        data_offset = align(handle.tell())
        for i, spec in enumerate(specs):
            wanted = 0 if i == 0 else align(offsets[i - 1] + specs[i - 1].nbytes)
            if offsets[i] != wanted:
                raise ConversionError("verification failed: bad offset for '{}'".format(spec.name))
        wanted_size = data_offset + align(offsets[-1] + specs[-1].nbytes)
        if path.stat().st_size != wanted_size:
            raise ConversionError("verification failed: file size")


def _prepare(definition: ModelDefinition, weights_dir: Path, stack: ExitStack, ftype: int,
             tensor_prefix: str, model_arg: Optional[str] = None,
             config_arg: Optional[str] = None) -> Tuple[TensorStore, Dict[str, Any], List[TensorSpec]]:
    if definition.kind in ("ss-dec", "slat-dec") and ftype == 2:
        raise ConversionError("{} is a decoder; ftype 2 is flow-only".format(definition.key))
    if definition.kind == "naf" and ftype != 0:
        raise ConversionError("NAF currently requires --ftype 0 (F32)")
    model_path, config_path = resolve_checkpoint(
        weights_dir, definition.stem, model_arg, config_arg,
        allow_missing_config=definition.kind == "naf" and config_arg is None)
    model_class, args = load_model_config(config_path, definition.kind)
    if model_class != definition.expected_class:
        raise ConversionError("{} declares '{}', expected '{}'".format(
            definition.key, model_class, definition.expected_class))
    store = stack.enter_context(TensorStore(model_path))
    _validate_inventory(definition, store, args)
    specs = _tensor_specs(store, definition, ftype, tensor_prefix)
    counts: Dict[int, int] = {}
    for spec in specs:
        counts[spec.ggml_type] = counts.get(spec.ggml_type, 0) + 1
    print("[{}] source : {}".format(definition.key, model_path))
    config_label = str(config_path) if config_path.is_file() else "<built-in NAF defaults>"
    print("[{}] config : {} ({})".format(definition.key, config_label, model_class))
    print("[{}] tensors: {} (f32={} f16={} bf16={})".format(definition.key, len(specs),
          counts.get(GGML_TYPE_F32, 0), counts.get(GGML_TYPE_F16, 0),
          counts.get(GGML_TYPE_BF16, 0)))
    return store, args, specs


def _report_size(label: str, byte_count: int) -> None:
    print("[{}] planned size: {} bytes ({:.2f} GiB)".format(
        label, byte_count, byte_count / 1024.0 ** 3))


def default_ftype_for_definition(definition: ModelDefinition) -> int:
    """Return the storage type selected by the stable mixed ``auto`` policy."""
    if definition.kind in ("ss-dec", "slat-dec"):
        return AUTO_FTYPE_DECODER
    # Flow, DINO and NAF all currently use the F32 path.  DINO/NAF also reject
    # reduced precision explicitly in their component-specific selectors.
    return AUTO_FTYPE_FLOW


def default_ftype_for_bundle(bundle: str) -> int:
    """Return the storage type selected by the stable mixed ``auto`` policy."""
    if bundle == "shared":
        return AUTO_FTYPE_DECODER
    if bundle in ("base-flow", "mv-flow"):
        return AUTO_FTYPE_FLOW
    raise ConversionError("unknown bundle '{}'".format(bundle))


def convert_bundle(bundle: str, weights_dir: Path, output: Path, ftype: int,
                   force: bool, dry_run: bool, verify: bool) -> Path:
    definitions = [MODELS[key] for key in BUNDLES[bundle]]
    if bundle == "shared" and ftype == 2:
        raise ConversionError("shared bundle supports only ftype 0 or 1")
    metadata = _base_metadata(PIXAL3D_ARCH, output.stem, ftype)
    metadata += [kv_u32(PIXAL3D_PREFIX + "bundle_format", PIXAL3D_BUNDLE_FORMAT),
                 kv_str(PIXAL3D_PREFIX + "tensor_name_scheme", PIXAL3D_TENSOR_NAME_SCHEME),
                 kv_str(PIXAL3D_PREFIX + "bundle_kind", bundle),
                 kv_u32(PIXAL3D_PREFIX + "component_count", len(definitions))]
    specs: List[TensorSpec] = []
    stores: Dict[str, TensorStore] = {}
    with ExitStack() as stack:
        for i, definition in enumerate(definitions):
            metadata.append(kv_str(PIXAL3D_PREFIX + "component.{}".format(i),
                                   definition.component))
            store, args, model_specs = _prepare(definition, weights_dir, stack, ftype,
                                                 definition.component + ".")
            stores[definition.key] = store
            metadata += _model_metadata(definition, args,
                                         PIXAL3D_PREFIX + definition.component + ".")
            specs += model_specs
        if len({spec.name for spec in specs}) != len(specs):
            raise ConversionError("duplicate tensor name in bundle")
        _, _, _, _, size = _serialized_layout(metadata, specs)
        print("[{}] output : {} (ftype={})".format(bundle, output, ftype))
        _report_size(bundle, size)
        if dry_run:
            print("[{}] dry-run: no GGUF written".format(bundle))
            return output
        _write_gguf(output, metadata, specs, stores, force)
    if verify:
        verify_gguf(output, specs)
        print("[{}] verified header, metadata, tensor table, offsets and size".format(bundle))
    print("[{}] wrote {} ({} bytes)".format(bundle, output, output.stat().st_size))
    return output


def convert_component(definition: ModelDefinition, weights_dir: Path, output: Path,
                      ftype: int, force: bool, dry_run: bool, verify: bool,
                      model_arg: Optional[str] = None,
                      config_arg: Optional[str] = None) -> Path:
    source = _normalise_model_path(Path(model_arg).expanduser()) if model_arg else \
        weights_dir / (definition.stem + ".safetensors")
    if output.resolve() == source.resolve():
        raise ConversionError("output must not overwrite source checkpoint")
    architecture, prefix = PIXAL3D_ARCH, PIXAL3D_PREFIX + definition.component + "."
    tensor_prefix = definition.component + "."
    with ExitStack() as stack:
        store, args, specs = _prepare(definition, weights_dir, stack, ftype,
                                      tensor_prefix, model_arg, config_arg)
        metadata = _base_metadata(architecture, definition.stem, ftype)
        metadata += [kv_u32(PIXAL3D_PREFIX + "bundle_format", PIXAL3D_BUNDLE_FORMAT),
                     kv_str(PIXAL3D_PREFIX + "tensor_name_scheme", PIXAL3D_TENSOR_NAME_SCHEME),
                     kv_str(PIXAL3D_PREFIX + "bundle_kind", "component"),
                     kv_u32(PIXAL3D_PREFIX + "component_count", 1),
                     kv_str(PIXAL3D_PREFIX + "component.0", definition.component)]
        metadata += _model_metadata(definition, args, prefix)
        _, _, _, _, size = _serialized_layout(metadata, specs)
        print("[{}] output : {} (ftype={})".format(definition.key, output, ftype))
        _report_size(definition.key, size)
        if dry_run:
            print("[{}] dry-run: no GGUF written".format(definition.key))
            return output
        _write_gguf(output, metadata, specs, {definition.key: store}, force)
    if verify:
        verify_gguf(output, specs)
        print("[{}] verified header, metadata, tensor table, offsets and size".format(definition.key))
    print("[{}] wrote {} ({} bytes)".format(definition.key, output, output.stat().st_size))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--bundle", choices=("shared", "base-flow", "mv-flow", "all"),
                           help="deployment bundle (default: all three)")
    selection.add_argument("--component", choices=tuple(MODELS.keys()),
                           help="one namespaced component for development")
    parser.add_argument("--weights-dir", default=str(_default_weights_dir()))
    parser.add_argument("--model", help="explicit checkpoint (single component)")
    parser.add_argument("--config", help="explicit config (single component)")
    parser.add_argument("--output", help="explicit output (single selection)")
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    parser.add_argument(
        "--ftype", choices=("auto", "0", "1", "2"), default="auto",
        help="GGUF storage type: {} (stable deployment default), ".format(
             AUTO_FTYPE_DESCRIPTION) +
             "0=all F32 (reference parity), 1=mostly F16, "
             "2=mostly BF16 (flow only)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    weights_dir, output_dir = Path(args.weights_dir).expanduser(), Path(args.output_dir).expanduser()
    requested_ftype = None if args.ftype == "auto" else int(args.ftype)
    bundle = args.bundle
    if bundle is None and args.component is None:
        bundle = "all"
    single = args.component is not None or bundle in ("shared", "base-flow", "mv-flow")
    if (args.model or args.config) and not args.component:
        parser.error("--model/--config require one --component")
    if args.output and not single:
        parser.error("--output requires a single selection")
    try:
        if bundle:
            names = tuple(BUNDLES.keys()) if bundle == "all" else (bundle,)
            for name in names:
                ftype = requested_ftype if requested_ftype is not None else \
                    default_ftype_for_bundle(name)
                suffix = {0: "f32", 1: "f16", 2: "bf16"}[ftype]
                output = Path(args.output).expanduser() if args.output else output_dir / \
                    (BUNDLE_OUTPUT_STEMS[name] + "-" + suffix + ".gguf")
                convert_bundle(name, weights_dir, output, ftype, args.force,
                               args.dry_run, not args.no_verify)
        else:
            definition = MODELS[args.component]
            ftype = requested_ftype if requested_ftype is not None else \
                default_ftype_for_definition(definition)
            suffix = {0: "f32", 1: "f16", 2: "bf16"}[ftype]
            output = Path(args.output).expanduser() if args.output else output_dir / \
                (definition.stem + "-" + suffix + ".gguf")
            convert_component(definition, weights_dir, output, ftype, args.force,
                              args.dry_run, not args.no_verify, args.model, args.config)
    except ConversionError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
