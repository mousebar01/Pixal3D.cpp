# Pixal3D.cpp

This is the first C++/ggml migration scaffold for Pixal3D. The original
Python implementation and the `trellis2cpp` port are kept under [`ref/`](ref/)
as reference sources; they are not compiled or linked by the production build.

## Current scope

- `pixal3d_core` is the outer API boundary for future Pixal3D stages.
- The top-level `ggml` submodule is the sole runtime dependency of the
  production C++ library. The projects under `ref/` are reference-only.
- The `pixal3d` CLI can inspect Pixal3D GGUF bundles. `inspect-pack` validates
  architecture, bundle metadata, tensor names/shapes, alignment, and payload
  file bounds.
- Native F32 DINOv3 ViT-L/16 and NAF component loaders are available behind
  explicit GGUF contracts; both have deterministic Python reference oracles.
- CUDA builds select GPU execution for DINOv3, SLat flow, SS decoder, and
  sparse SLat decoder
  convolution/coordinate-upsample, and NAF convolution stacks, with validated
  CPU fallbacks.
- CTest covers the CLI smoke tests, projection core, flow-conditioning
  primitives, sparse decoder/transformer fixtures, and Dual Grid extraction.
- Docker provides a reproducible C++ build/runtime image. The legacy Python
  environment has a separate optional Dockerfile so its large dependencies do
  not affect normal C++ builds.

## Native build

The production build only needs the top-level ggml submodule:

```sh
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
(cd build && ctest --output-on-failure)
```

`PIXAL3D_GGML_NATIVE=ON` can be enabled when the image will only run on the
same CPU family used for the build.

Run the scaffold CLI:

```sh
./build/bin/pixal3d --help
./build/bin/pixal3d --version
./build/bin/pixal3d inspect-pack build/weights/pixal3d-shared-f16.gguf
# Metadata-only SS-flow inspection (does not allocate the 22 GiB weights).
./build/bin/pixal3d inspect-ss-flow build/weights/pixal3d-base-flow-f32.gguf
# Metadata-only SS decoder inspection (does not allocate decoder payloads).
./build/bin/pixal3d inspect-ss-decoder build/weights/pixal3d-shared-f16.gguf
# Metadata-only SLat flow inspection (does not allocate the 21 GiB flow pack).
./build/bin/pixal3d inspect-slat-flow \
  build/weights/pixal3d-base-flow-f32.gguf shape_flow_512
```

## Convert downloaded weights

The repository-level converter reads all eleven local Pixal3D checkpoints from
`weights/Pixal3D/ckpts`; it also supports the standalone DINOv3 vision
component via `--component dino`. Validate model classes, required tensors,
config-dependent boundary shapes and storage layouts before writing anything:

```sh
python3 scripts/convert_pixal3d_to_gguf.py --bundle all --dry-run
```

The production layout groups the checkpoints into three files rather than
eleven: `pixal3d-shared-f16.gguf` contains all three decoders,
`pixal3d-base-flow-f32.gguf` contains the four standard flow models, and
`pixal3d-mv-flow-f32.gguf` contains the four multi-view flow models. Standard
inference opens `shared + base-flow`; multi-view inference opens
`shared + mv-flow`. The default `auto` mode is exactly
`auto=F32 flow, F16 decoder`: decoder matrix tensors use F16 where eligible,
required sensitive/scalar tensors remain F32, and every flow model uses F32.

Important: `auto` is a mixed deployment policy, not an all-F32 mode. In this
repository it is fixed to **F32 for every flow bundle and the F16 decoder
policy** (eligible decoder matrices are F16; sensitive/scalar tensors remain
F32). Use `--ftype 0` explicitly when the purpose is numerical porting or
Python-reference parity; never infer that choice from the word `auto`.

The flow filenames contain `_bf16`, but their downloaded safetensors payloads
are stored primarily as F32; the checkpoint config selects BF16 for the model
runtime. The decoder checkpoints are already FP16 (the sparse-structure
decoder is mixed FP16/F32), so the stable `auto` deployment mode keeps their
original storage precision. This `auto` choice is deliberate and must not be
read as the numerical-porting precision: use `--ftype 0` when comparing a
new C++ implementation against the Python reference. That produces an
all-F32 GGUF (including promoted decoder tensors), but cannot recover precision
lost in the source FP16 tensors.

When using the existing Docker Python image, run as the host user so the
generated GGUF files remain writable by the checkout owner:

```sh
docker run --rm --user "$(id -u):$(id -g)" \
  --entrypoint /opt/conda/bin/python \
  -v "$PWD:/workspace/Pixal3D.cpp" \
  -w /workspace/Pixal3D.cpp \
  local/comfyui-h3:0.34.0-cu126 \
  scripts/convert_pixal3d_to_gguf.py --bundle all
```

Convert one namespaced component for development:

```sh
python3 scripts/convert_pixal3d_to_gguf.py --component shape-dec
```

`auto` always selects the F16 decoder policy and F32 for flow components;
this is the project default deployment policy and is intentionally stable. Use `--ftype 0` for the
all-F32 numerical development/reference pack, or `--ftype 1`/`2` to opt into
the explicitly named reduced-precision policies. The script refuses incomplete
`.safetensors.part` files, reads tensors lazily
with `safetensors.safe_open`, and uses an atomic temporary output. It then
independently parses the resulting GGUF header, metadata, tensor table,
alignment and final size. Flow bundles support `--ftype 0`, `1`, and `2`;
decoder bundles support `0` and `1`.

GGUF tensor names use the fixed `pixal3d.tensor_name_scheme=compact-v1`
contract. The long flow prefixes are mapped as `ss_flow→ss`,
`shape_flow_512→sh512`, `shape_flow_1024→sh1024`, and
`texture_flow_1024→tx1024`; all names remain below ggml's 64-byte limit.
The source tensor key is still retained by the converter and the mapping is
deterministic for the C++ loader.

SLat decoder sparse-convolution weights originate in flex_gemm layout
`[out,kD,kH,kW,in]`. GGUF bundle format v1 validates the released 3x3x3
kernels and stores them as `[kernel_volume,out,in]`, exposed to ggml as
`[in,out,kernel_volume]`. Each of the 27 neighbor matrices is therefore
contiguous and is consumed directly by the standalone CPU sparse-convolution
fallback or the ggml GPU gather+matmul path.

The downloaded pipeline contains seven logical model blocks: sparse-structure
flow, sparse-structure decoder, shape SLat flow at 512 and 1024, shape SLat
decoder, texture SLat flow at 1024, and texture SLat decoder. The multi-view
pipeline adds four separately trained `_mv` flow checkpoints while reusing the
three decoders. The converter covers all eleven physical checkpoints. The
C++ `load_pack_info()` boundary now reads all three packs without allocating
their payloads and validates the compact naming scheme, components, tensor
descriptors, alignment, and file bounds. Projection, sparse SLat operators,
the complete SLat flow stack, and inference-time Flexible Dual Grid extraction
now have standalone CPU/F32 implementations and deterministic Python
comparisons. The external-condition `run_pixal3d_cascade_f32()` boundary now
composes all seven blocks, including SLat denormalization, coordinate
quantization, texture shape concatenation, decoder subdivisions, and mesh
extraction. Native DINOv3 ViT-L/16 F32 loading and inference are now available
as a separate ggml component; the standalone NAF F32 component and its
released-weight converter are also available. A bounded real-weight
image-to-mesh run is covered by the CUDA smoke path; an unconstrained quality
comparison remains a separate integration step.

## Pixal3D projection core

The first Pixal3D-specific C++ module is now available as `pixal3d/projection.h`.
It implements the CPU reference contract for `ProjGrid` and `ProjGridMV`:
Blender `-Z` camera projection, `camera_angle_x`/`distance`/`mesh_scale`,
`align_corners=false` bilinear sampling with border padding, and elementwise
multi-view feature averaging. It also exposes the relative-camera transform
contract `F * inverse(C0) * Ci` used by `ProjGridMV`. It accepts externally
produced HxWxC DINO/NAF features, so DINOv3 itself can be ported independently
from the 3D graph.

The numerical smoke test is built and run by CTest:

```sh
(cd build && ctest --output-on-failure -R pixal3d_projection)
```

For a direct Python/C++ numerical comparison, build the deterministic fixture
and run the comparison inside the Python reference image:

```sh
cmake --build build --target pixal3d_projection_fixture
docker run --rm --user "$(id -u):$(id -g)" \
  --entrypoint /opt/conda/bin/python \
  -v "$PWD:/workspace/Pixal3D.cpp" \
  -w /workspace/Pixal3D.cpp \
  local/comfyui-h3:0.34.0-cu126 scripts/compare_projection.py
```

The script executes the projection definitions from the original Python
reference, then compares shapes, finite values, fingerprints, maximum
absolute error, relative L2 error, and validity-mask agreement. The current
float32 fixture uses `2e-5` maximum absolute and `2e-6` relative-L2
tolerances; it is a projection-core check, not full model inference parity.

The projection layer is intentionally separate from model execution. The GGUF
boundary now has both `load_pack_info()` for zero-payload validation and
`Pixal3DPackReader` for on-demand tensor reads; opening a 43 GiB bundle does
not allocate its payload. A reader test writes a tiny GGUF, round-trips one
tensor, and checks missing/closed-reader failures. The SS-flow loader now binds
these descriptors to ggml backend tensors and exposes a metadata-only
`inspect-ss-flow` CLI path.

`projected_grid_to_sparse_f32()` is the explicit bridge from a projected dense
feature grid to the sparse coordinates used by the shape/texture flow. It
preserves the reference's sampled values (including border samples) and point
order; it does not silently discard cells marked invalid by the diagnostic
mask. This lets an external DINO/NAF encoder feed the C++ graph without
coupling the camera projection code to a particular vision runtime.

The first dense-flow conditioning primitives are now in
[`include/pixal3d/flow.h`](include/pixal3d/flow.h): the Python
`TimestepEmbedder` frequency/MLP path, float32 `LayerNorm32` rows, and AdaLN
`shift/scale` modulation. They use the converter's PyTorch `[out,in]` linear
weight contract. The deterministic fixture can be checked against the
original reference implementation inside the Python image:

```sh
cmake --build build --target pixal3d_flow_fixture
docker run --rm --user "$(id -u):$(id -g)" \
  --entrypoint /opt/conda/bin/python \
  -v "$PWD:/workspace/Pixal3D.cpp" \
  -w /workspace/Pixal3D.cpp \
  local/comfyui-h3:0.34.0-cu126 scripts/compare_flow.py
```

The comparison checks shapes, finite values, SHA-256 fingerprints, maximum
absolute error, and relative L2 error. Current float32 tolerances are
`3e-5` absolute and `3e-6` relative L2; the fixture passes with a maximum
error below `5e-7`.

## Flow-Euler sampler

[`include/pixal3d/flow_sampler.h`](include/pixal3d/flow_sampler.h) provides a
model-independent CPU/F32 flow-matching sampler for the sparse stages. It
implements the reference `t=1→0` Euler schedule, `rescale_t`, positive and
negative CFG callbacks, guidance intervals, and Pixal3D's sparse CFG-rescale
semantics. Coordinates are checked on every model callback and remain fixed
throughout integration; optional trajectories expose both `x_t` and `x_0`
predictions for diagnostics.

The fixed callback fixture compares the complete loop, including the warped
timestep schedule and per-batch rescaling, with
`FlowEulerGuidanceIntervalSampler`:

```sh
cmake --build build --target pixal3d_flow_sampler_fixture
docker run --rm --user "$(id -u):$(id -g)" \
  --entrypoint /opt/conda/bin/python \
  -v "$PWD:/workspace/Pixal3D.cpp" \
  -w /workspace/Pixal3D.cpp \
  local/comfyui-h3:0.34.0-cu126 scripts/compare_flow_sampler.py
```

The comparison passes with maximum absolute error `3.10e-6` and relative L2
error `2.09e-6`. `SLatFlowModel::sample()` now binds this sampler to the loaded
SLat flow forward path; the tiny GGUF comparison exercises positive/zero CFG
branches through that adapter as well. Texture flow's `concat_cond` path is
supported: the shape latent is concatenated before the input projection and is
kept for both positive and negative CFG branches, matching the Python
pipeline. A separate tiny fixture compares both forward and sampling paths.
`run_slat_stage_f32()` additionally applies the configured SLat latent
denormalization and is checked by the same Python comparison. It accepts an
optional concatenated sparse condition for the texture model. The sampler
still does not choose pipeline-specific noise shapes or image-conditioning
tensors.

The SS-flow graph is exposed as
[`include/pixal3d/ss_flow.h`](include/pixal3d/ss_flow.h). It follows the
validated trellis2cpp graph layout but uses the `ss.*` compact-v1 names from
the grouped Pixal3D bundle. Its tiny end-to-end fixture uses one block and a
projection-attention condition, then compares the complete output against
`SparseStructureFlowModel` with the same weights, x, t, global condition, and
projected condition:

```sh
cmake --build build --target pixal3d_ss_flow_fixture
docker run --rm --user "$(id -u):$(id -g)" \
  --entrypoint /opt/conda/bin/python \
  -v "$PWD:/workspace/Pixal3D.cpp" \
  -w /workspace/Pixal3D.cpp \
  local/comfyui-h3:0.34.0-cu126 scripts/compare_ss_flow.py
```

The fixture currently passes with relative L2 error below `1e-5`; it exercises
the forward graph and the model-bound four-step Flow-Euler sampler, including
zero negative global/projection conditions and CFG rescaling. A real 30-block
production checkpoint fixture is still required before claiming full model
equivalence.

CUDA builds select the ggml GPU backend for SS-flow and use tiled
FlashAttention for its full-grid self/cross attention; unsupported backend
operations fall back to the CPU graph. The recommended development path uses
the F32 flow pack, while the decoder can remain in its released F16 storage
format. `SSFlowModel::backend_name()` and the `run-image` loader log expose the
selected device so a run cannot silently be mistaken for CPU execution.

`SSFlowModel::sample()` accepts a complete regular-grid `SparseTensorF32` as
noise and returns the same coordinate-preserving sparse representation used by
the SLat sampler. `ss_occupancy_to_coords_f32()` and
`SSDecoderModel::decode_coords()` reproduce the next pipeline boundary:
threshold occupancy logits and optionally max-pool by an integer resolution
ratio to obtain `[batch,x,y,z]` active coordinates. The pooling/threshold
contract has a standalone CTest fixture.

The dense sparse-structure decoder is exposed as
[`include/pixal3d/ss_decoder.h`](include/pixal3d/ss_decoder.h). Its graph is
adapted directly from the corresponding trellis2cpp stage: Conv3d weights use
the compact-v1 `[out_mul_in,kD,kH,kW]` storage transform, normalization is
per-voxel over channels, and each upsample is the same three-axis pixel
shuffle. The tiny fixture uses a 2³ latent grid so all operations can be
checked quickly:

```sh
cmake --build build --target pixal3d_ss_decoder_fixture
docker run --rm --user "$(id -u):$(id -g)" \
  --entrypoint /opt/conda/bin/python \
  -v "$PWD:/workspace/Pixal3D.cpp" \
  -w /workspace/Pixal3D.cpp \
  local/comfyui-h3:0.34.0-cu126 scripts/compare_ss_decoder.py
```

The tiny F32 comparison currently passes below `1e-6` maximum absolute error
and checks the exact active-coordinate list produced by `decode_coords()`.
The production check reads `pixal3d-shared-f16.gguf` and compares the real
16³→64³ decoder against the same checkpoint values in an F32 PyTorch oracle;
its current relative L2 error is about `4.3e-5` (the F16 source weights are
not silently promoted in the GGUF pack).

In a CUDA build, `SSDecoderModel` probes the actual `IM2COL_3D` and `MUL_MAT`
backend operations and selects the GPU when both are available.  The CUDA
graph uses an F32 im2col activation matrix with the serialized F16 kernels,
which avoids F16 intermediate overflow in the deep decoder while preserving
the pack's storage type.  The CPU backend keeps ggml's native Conv3d path as
the reference fallback.  `backend_name()` reports the selected device; a
production F16 pack currently compares to CPU with max absolute error about
`2.8e-2` and relative L2 `3.3e-5` on the RTX 4090D test host.

```sh
docker run --rm --user "$(id -u):$(id -g)" \
  --entrypoint /opt/conda/bin/python \
  -v "$PWD:/workspace/Pixal3D.cpp" \
  -w /workspace/Pixal3D.cpp \
  local/comfyui-h3:0.34.0-cu126 scripts/compare_ss_decoder_production.py
```

The first composed pipeline boundary is exposed by
[`include/pixal3d/pipeline.h`](include/pixal3d/pipeline.h):
`run_sparse_structure_stage_f32()` binds SS-flow sampling to SS-decoder
occupancy extraction while keeping image features as an explicit
`Pixal3DImageConditionF32` input. This makes the stage usable with the native
DINO/NAF components and bridge, or with precomputed F32 conditions. A two-pack tiny
fixture compares the complete flow→decoder composition, including the
row-major/channel-major bridge and 2× occupancy max-pool:

```sh
cmake --build build --target pixal3d_ss_stage1_fixture
docker run --rm --user "$(id -u):$(id -g)" \
  --entrypoint /opt/conda/bin/python \
  -v "$PWD:/workspace/Pixal3D.cpp" \
  -w /workspace/Pixal3D.cpp \
  local/comfyui-h3:0.34.0-cu126 scripts/compare_ss_stage1.py
```

The composed fixture passes with the same `2e-5` maximum absolute sample
error and exact active-coordinate agreement.

## SLat sparse foundation

[`include/pixal3d/sparse.h`](include/pixal3d/sparse.h) now provides the first
CPU fallback primitives needed by the Shape/Texture SLat stages: a validated
`[batch,x,y,z]` active-coordinate tensor, `[out,in]` sparse linear, and the
3x3x3 submanifold convolution used by the reference `flex_gemm` backend. The
convolution consumes the converter's `[kernel,out,in]` layout and preserves
active point order; average downsampling follows the reference's sorted
`torch.unique` grouping and ceil-shaped grid. Batch-isolated multi-head
scaled-dot-product attention and coordinate-driven 3D RoPE are also available
as CPU fallbacks, along with the per-head RMS normalization used by Q/K.
A deterministic fixture mirrors the reference
submanifold accumulation (the optional `flex_gemm` wheel is not present in
every legacy image):

```sh
cmake --build build --target pixal3d_sparse_fixture
docker run --rm --user "$(id -u):$(id -g)" \
  --entrypoint /opt/conda/bin/python \
  -v "$PWD:/workspace/Pixal3D.cpp" \
  -w /workspace/Pixal3D.cpp \
  local/comfyui-h3:0.34.0-cu126 scripts/compare_sparse.py
```

The sparse fixture passes at float32 precision. These standalone sparse
attention, spatial-resampling, RoPE, and submanifold-convolution functions
remain CPU reference primitives; production SLat flow uses their equivalent
ggml CUDA graph operators when a CUDA backend is available.

## SLat Transformer block

[`include/pixal3d/sparse_transformer.h`](include/pixal3d/sparse_transformer.h)
now contains one CPU/F32 `ModulatedSparseTransformerCrossBlock`. Its operation
order follows the Pixal3D reference: shared AdaLN modulation, sparse
self-attention with optional 3D RoPE and per-head Q/K RMS normalization,
global variable-length cross-attention, projection-attention fusion, and the
gated GELU MLP. The implementation uses the standalone sparse primitives in
`sparse.h`; it does not link to or copy code from either reference checkout.
The SS-flow and SS-decoder graph organization was cross-checked against
`ref/trellis2cpp`, while this SLat block is checked against the original
Pixal3D Python module because trellis2cpp does not contain this block.

Build the deterministic block fixture and compare all 120 output values with
the Python oracle in the legacy image:

```sh
cmake --build build --target pixal3d_sparse_transformer_fixture
docker run --rm --user "$(id -u):$(id -g)" \
  --entrypoint /opt/conda/bin/python \
  -v "$PWD:/workspace/Pixal3D.cpp" \
  -w /workspace/Pixal3D.cpp \
  local/comfyui-h3:0.34.0-cu126 scripts/compare_sparse_transformer.py
```

The current F32 comparison passes with maximum absolute error below `4e-7`
and relative L2 error below `1e-7`. This validates one block and its tensor
layouts. The complete SLat flow stack is covered in the next section;
production 30-block execution still needs a full end-to-end run with image
features and its memory budget accounted for.

## SLat flow stack

[`include/pixal3d/slat_flow.h`](include/pixal3d/slat_flow.h) and
[`src/slat_flow.cpp`](src/slat_flow.cpp) implement the complete
`ElasticSLatFlowModel` graph used by the three base-flow components. The CPU
path follows the reference order: input sparse linear, timestep embedding and
shared AdaLN modulation, repeated sparse self/cross/projection transformer
blocks, final non-affine layer normalization, and output sparse linear.
Weights are loaded from compact-v1 aliases `sh512`, `sh1024`, and `tx1024`;
F32, F16, and BF16 payloads are converted to host F32 at the loader boundary.

The metadata-only CLI path validates the real 30-block packs without reading
their multi-gigabyte payloads:

```sh
./build/bin/pixal3d inspect-slat-flow \
  build/weights/pixal3d-base-flow-f32.gguf shape_flow_512
./build/bin/pixal3d inspect-slat-flow \
  build/weights/pixal3d-base-flow-f32.gguf shape_flow_1024
./build/bin/pixal3d inspect-slat-flow \
  build/weights/pixal3d-base-flow-f32.gguf texture_flow_1024
```

A one-block projection fixture compares the complete C++ graph directly with
`SLatFlowModel`, and a second pass writes the same state into a tiny GGUF and
loads it through `SLatFlowModel`:

```sh
cmake --build build --target pixal3d_slat_flow_fixture \
  pixal3d_slat_flow_loader_fixture
docker run --rm --user "$(id -u):$(id -g)" \
  --entrypoint /opt/conda/bin/python \
  -v "$PWD:/workspace/Pixal3D.cpp" \
  -w /workspace/Pixal3D.cpp \
  local/comfyui-h3:0.34.0-cu126 scripts/compare_slat_flow.py
```

The current F32 direct and loader comparisons pass with maximum absolute
error `3.73e-7` and relative L2 error `3.27e-7`. This validates the reference
CPU graph and tensor layouts. A previous CUDA writeback path transposed every
SLat `[point, channel]` row once more after `ggml_backend_tensor_get`; that
bug produced a plausible-looking but semantically scrambled latent and was
the cause of the earlier completely incorrect meshes. The writeback now keeps
the backend's token-major order. A compact CUDA-vs-CPU check after the fix is
below `4e-7` absolute error; a production shape-flow pack (five active points)
is finite with relative L2 difference about `1.3e-3` from CUDA arithmetic.

For production batch-1 inference, a CUDA build uploads one SLat component on
demand and releases it after its sampler stage so the three 5-GiB-class flow
components do not all occupy VRAM at once. CUDA builds disable TF32 by default
(`PIXAL3D_CUDA_TF32=OFF`) so F32 matrix products keep the precision expected
by the Python oracle. Set `-DPIXAL3D_CUDA_TF32=ON` only for a speed-oriented,
numerically approximate run. If a CUDA allocation or operation is unavailable,
the call falls back to the validated CPU/F32 path. Set
`PIXAL3D_SLAT_BACKEND=cpu` to force that fallback for a diagnostic comparison,
or `PIXAL3D_SLAT_VERBOSE=1` to log the reason for a per-forward fallback.

A real-image GPU smoke run with deliberately diagnostic settings
(`--vision-resolution 64 --steps 2 --max-structure-points 5
--occupancy-threshold -100`) completes all three SLat stages and writes a
finite OBJ. It is an execution/finite-value gate, not a quality benchmark;
the production-default unconstrained image run still needs to be rerun after
this fix before judging visual fidelity.

### Long-sequence CUDA attention

The released flow checkpoints are BF16 even though the development GGUF packs
are stored as F32. The CUDA graph therefore keeps graph tensors in F32. The
SS grid uses the generic ggml SDPA graph for its 4096-token sequence, matching
the reference implementation in `ref/trellis2cpp`; short SLat grids also use
that generic path, while longer SLat grids use the streaming CUDA attention
kernel. The bundled ggml backend has a bounds-checked
BF16 path for explicitly selected BF16 deployments, but the default Pixal3D
F32 build does not silently cast flow activations to BF16. `PIXAL3D_SLAT_VERBOSE=1`
remains available to make any genuine GPU-to-CPU fallback visible.

Texture flow's concatenated shape-latent condition is checked separately:

```sh
cmake --build build --target pixal3d_slat_flow_concat_fixture
docker run --rm --user "$(id -u):$(id -g)" \
  --entrypoint /opt/conda/bin/python \
  -v "$PWD:/workspace/Pixal3D.cpp" \
  -w /workspace/Pixal3D.cpp \
  local/comfyui-h3:0.34.0-cu126 scripts/compare_slat_flow_concat.py
```

It compares the texture-style `x || shape_slat` input contract with maximum
absolute error below `8e-7` for both forward and three-step sampling.

## SLat sparse decoder blocks

[`include/pixal3d/slat_decoder.h`](include/pixal3d/slat_decoder.h) now
contains the CPU/F32 reference decoder building blocks used by the Shape and
Texture SLat VAEs: `SparseConvNeXtBlock3d`, `SparseResBlockC2S3d`,
packed-channel to spatial rearrangement, subdivision prediction, guide
subdivision, and the final channel-wise normalization/output projection. The
implementation keeps the reference's child ordering (`x` bit, then `y`, then
`z`) and compact-v1 `[kernel_volume,out,in]` convolution layout.

When built with CUDA, `SLatDecoderModel` uploads each decoder component on
demand and evaluates its sparse 3x3x3 convolutions on the selected ggml GPU
backend using device-side gather plus matrix multiplication. Coordinate
bookkeeping, sparse rearrangement, normalization, and dense projections remain
on the host for now; any unsupported GPU operation falls back to the validated
CPU/F32 implementation. Shape decoder coordinate upsampling uses the same GPU
convolution path. Set `PIXAL3D_SLAT_DECODER_BACKEND=cpu` to force the reference
path and `PIXAL3D_SLAT_VERBOSE=1` to log GPU fallback reasons.

The two-level fixture compares coordinates, subdivision logits, and output
features against `SparseUnetVaeDecoder` using a small Python fallback for the
documented submanifold-convolution contract:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --target pixal3d_slat_decoder_fixture
docker run --rm --user "$(id -u):$(id -g)" \
  --entrypoint /opt/conda/bin/python \
  -v "$PWD:/workspace/Pixal3D.cpp" \
  -w /workspace/Pixal3D.cpp \
  local/comfyui-h3:0.34.0-cu126 scripts/compare_slat_decoder.py
```

The fixture has exact coordinate/subdivision agreement and a maximum feature
error of `1.7e-5` (the scalar CPU reduction and vectorized PyTorch reduction
have different accumulation order). The same comparator also writes a tiny
grouped GGUF component, loads it through `SLatDecoderModel`, and reruns the
decoder; this validates the metadata, compact tensor names, F32 payload path,
and the production F16-to-F32 conversion path. BF16 decoding is implemented
at the same loader boundary but has not yet received a dedicated fixture. The
same comparison invokes `SLatDecoderModel::upsample_coords()` and checks the
coordinate-only prefix against `SparseUnetVaeDecoder.upsample(1)`. The
coordinate fingerprints are exact for direct weights and both GGUF storage
paths.

The cascade helpers in [`include/pixal3d/pipeline.h`](include/pixal3d/pipeline.h)
now expose the remaining deterministic wiring around a sampled SLat:
`run_slat_stage_f32()` applies the reference `latent * std + mean`
denormalization, while `quantize_slat_coords_f32()` implements the
decoder-coordinate mapping, lexicographic deduplication, and 1536→1024
token-budget fallback used by `Pixal3DImageTo3DPipeline`. They take noise and
image conditions as explicit inputs; random generation and the DINO/NAF image
encoder are intentionally not hidden in the C++ library. The small
`pixal3d_pipeline_utils` CTest covers the rounding, deduplication, and hard
resolution floor.

The complete external-condition cascade is exposed by
`run_pixal3d_cascade_f32()`. It follows the reference order
SS-flow/decoder → low-resolution shape SLat → decoder coordinate upsample →
high-resolution shape SLat → texture SLat with concatenated shape latent →
shape/texture decode → Flexible Dual Grid mesh. Noise and image features are
callbacks keyed by stage and exact sparse coordinates, so a native DINOv3/NAF
encoder can be added without changing the model graph. A tiny all-stage
fixture compares coordinates, F32 tensors, decoded outputs, and mesh indices:

```sh
cmake --build build --target pixal3d_cascade_fixture
docker run --rm --user "$(id -u):$(id -g)" \
  --entrypoint /opt/conda/bin/python \
  -v "$PWD:/workspace/Pixal3D.cpp" \
  -w /workspace/Pixal3D.cpp \
  local/comfyui-h3:0.34.0-cu126 scripts/compare_pixal3d_cascade.py
```

The deterministic cascade comparison passes with exact coordinate and mesh
index agreement; scalar F32 paths stay below `3e-5` maximum absolute error.
It validates the seven-model wiring without allocating production checkpoints.

## Flexible Dual Grid mesh extraction

[`include/pixal3d/dual_grid.h`](include/pixal3d/dual_grid.h) and
[`src/dual_grid.cpp`](src/dual_grid.cpp) provide the inference-time CPU path
corresponding to O-Voxel's `flexible_dual_grid_to_mesh(..., train=False)`.
It reconstructs complete edge quads with the reference neighbor ordering,
maps dual vertices into the decoder AABB, applies the positive split-weight
diagonal rule, and returns packed F32 vertices plus int32 triangle indices.
`SLatDecoderModel::decode_shape_mesh()` connects this path to the seven-channel
FlexiDualGrid decoder output (sigmoid dual vertices, intersection flags, and
softplus split weights).

The standalone comparator checks weighted and geometric diagonal selection,
coordinate mapping, and the seven-channel wrapper against a Python oracle:

```sh
cmake --build build --target pixal3d_dual_grid_fixture
docker run --rm --user "$(id -u):$(id -g)" \
  --entrypoint /opt/conda/bin/python \
  -v "$PWD:/workspace/Pixal3D.cpp" \
  -w /workspace/Pixal3D.cpp \
  local/comfyui-h3:0.34.0-cu126 scripts/compare_dual_grid.py
```

The extraction comparator passes with exact face-index agreement and maximum
vertex error below `1.2e-7`. The memory-bounded `run-image` path has also been
exercised with real packed weights on an RTX 4090D; GPU/CPU mesh topology is
not required to be bitwise identical near occupancy thresholds.

## Docker build

The default image is CPU-capable and does not install the large Python
dependency set:

```sh
docker build -f docker/Dockerfile --target runtime -t pixal3d:scaffold .
docker run --rm pixal3d:scaffold --version
docker compose -f docker/compose.yaml run --rm dev
# If the Compose plugin is not installed, use the legacy binary instead:
# docker-compose -f docker/compose.yaml run --rm dev
```

For a CUDA-enabled ggml build, use the CUDA developer image and pair it with a
CUDA runtime image:

```sh
docker build -f docker/Dockerfile --target runtime -t pixal3d:cuda \
  --build-arg DEV_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  --build-arg RUNTIME_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
  --build-arg PIXAL3D_ENABLE_CUDA=ON .
```

The CUDA image build compiles the backend but skips CTest because the Docker
build stage has no NVIDIA driver library. Run the compiled tests in the
developer stage with GPU access when that gate is required:

```sh
docker build -f docker/Dockerfile --target cpp-build -t pixal3d:cuda-build \
  --build-arg DEV_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04 \
  --build-arg PIXAL3D_ENABLE_CUDA=ON .
docker run --rm --gpus all pixal3d:cuda-build \
  sh -lc 'cd /workspace/build && ctest --output-on-failure'
```

Run the CUDA runtime with a GPU and enough shared memory for the attention
workspaces:

```sh
docker run --rm --gpus all --shm-size=24g -e OMP_NUM_THREADS=32 \
  -v "$PWD:/workspace/Pixal3D.cpp" -w /workspace/Pixal3D.cpp \
  pixal3d:cuda run-image \
  build/weights/pixal3d-shared-f16.gguf \
  build/weights/pixal3d-base-flow-f32.gguf \
  build/weights/dinov3-vitl16-pretrain-lvd1689m-f32.gguf \
  weights/NAF/naf_release-f32.gguf input.ppm output.obj \
  --max-model-gib 32 --seed 42
```

The legacy reference environment is opt-in:

```sh
docker build -f docker/Dockerfile.python -t pixal3d:python \
  --build-arg INSTALL_PIXAL3D_DEPS=ON .
```

The Python image is intended for reproducing the original reference pipeline;
the outer C++ scaffold does not depend on it.

## DINO condition input

The trellis2cpp-compatible `DINOCOND` reader is exposed as
[`include/pixal3d/dino.h`](include/pixal3d/dino.h). It accepts the canonical
little-endian F32 `[1,tokens,channels]` export (and the equivalent unbatched
`[tokens,channels]` form), validates finiteness and shape, and presents it as a
`VarLenTensorF32` suitable for `SSFlowModel::sample()` or
`run_sparse_structure_stage_f32()`. The native
[`include/pixal3d/dino_vit.h`](include/pixal3d/dino_vit.h) component loads the
standalone DINOv3 ViT-L/16 F32 GGUF and emits the same affine-free final
LayerNorm contract (global CLS/register rows plus an HxWxC patch map). The
converter currently accepts F32 only; mixed F16/BF16 will be enabled after a
backend-specific precision comparison.

Convert the released vision checkpoint without copying it into the project:

```sh
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -v "$HOME/.polykit/models/trellis2/Vision:/vision:ro" \
  -w /workspace local/comfyui-h3:0.34.0-cu126 \
  python scripts/convert_pixal3d_to_gguf.py --component dino \
    --model /vision/dinov3-vitl16-pretrain-lvd1689m.safetensors \
    --config /vision/config.json \
    --output build/weights/dinov3-vitl16-pretrain-lvd1689m-f32.gguf \
    --ftype 0
```

Inspect the converted tensor table and metadata with:

```sh
./build/bin/pixal3d inspect-dino-vit \
  build/weights/dinov3-vitl16-pretrain-lvd1689m-f32.gguf
```

The deterministic tiny-model oracle covers all 415 tensor slots:

```sh
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  local/comfyui-h3:0.34.0-cu126 python tests/compare_dino_vit.py
```

The converted production checkpoint is also checked at a 32x32 dynamic input
(short sequence, all 1.2 GiB weights):

```sh
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -v "$HOME/.polykit/models/trellis2/Vision:/vision:ro" \
  -w /workspace local/comfyui-h3:0.34.0-cu126 \
  python tests/compare_dino_vit_real.py --vision-dir /vision
```

## NAF feature upsampler

[`include/pixal3d/naf.h`](include/pixal3d/naf.h) is a native F32 implementation
of the NAF module vendored by ComfyUI's Pixal3D/Trellis pipeline. It preserves
the two reflect-padded convolution stacks, learned 2D RoPE, adaptive pooling,
and neighborhood cross-attention. The GGUF converter records the NAF
hyperparameters and rejects lossy F16/BF16 conversion until a separate
precision comparison is available.

In a CUDA build, NAF's two reflect-padded convolution stacks use ggml GPU
reflection-padding and 2-D convolution. Adaptive pooling, learned 2-D RoPE,
and neighborhood cross-attention remain on the validated CPU path for now, so
the native image boundary still benefits from CPU headroom at 512/1024 pixels.
Set `PIXAL3D_NAF_BACKEND=cpu` to force the complete CPU reference path, or set
`PIXAL3D_SLAT_VERBOSE=1` to log a GPU-convolution fallback.

Convert a NAF checkpoint (when present) and inspect its metadata with:

```sh
python scripts/convert_pixal3d_to_gguf.py --component naf \
  --model /path/to/naf.safetensors --config /path/to/naf.json \
  --output build/weights/naf-f32.gguf --ftype 0
./build/bin/pixal3d inspect-naf build/weights/naf-f32.gguf
```

For the canonical upstream `valeoai/NAF` torch.hub state dict, `--config` may
be omitted; the converter applies the released 256-channel/4-head/9x9/2-block
defaults. A sidecar JSON is still required when using a custom NAF variant.

The official release is a PyTorch `.pth` state dict. The adapter converts it to
F32 safetensors and then invokes the same verified GGUF converter:

```sh
scripts/download_naf_weights.sh
python scripts/convert_naf_checkpoint.py \
  --input weights/NAF/naf_release.pth \
  --safetensors weights/NAF/naf_release-f32.safetensors \
  --output weights/NAF/naf_release-f32.gguf
```

Validate the released weights against ComfyUI's exact NAF implementation:

```sh
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  local/comfyui-h3:0.34.0-cu126 python tests/compare_naf_real.py
```

The tiny oracle imports the exact `comfy/image_encoders/naf.py` reference and
compares all NAF operations against the native implementation:

```sh
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  local/comfyui-h3:0.34.0-cu126 python tests/compare_naf.py
```

[`include/pixal3d/vision_condition.h`](include/pixal3d/vision_condition.h)
bridges native DINO/NAF outputs into the existing cascade boundary. It keeps
the low-resolution DINO map and optional NAF map in HWC order, assigns the
original image resolution to projection metadata, and projects only the exact
sparse coordinates requested by the sampler.
`encode_pixal3d_condition_stage_f32()` is the native raw-image entry point: it
accepts a square CHW F32 image in `[0,1]`, applies the DINO ImageNet
normalization, runs DINO and optional NAF, and returns one validated stage.
`make_pixal3d_multiview_condition_stage_f32()` assembles per-view outputs for
the relative-camera `ProjGridMV` average path. The tiny end-to-end bridge
oracle is:

```sh
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  local/comfyui-h3:0.34.0-cu126 python tests/compare_vision_condition.py
```

`run_pixal3d_from_condition_stages()` is the in-memory cascade entry point for
these assembled stages; the P3DCOND file path remains available for large or
external encoders. `save_pixal3d_condition_bundle()` writes the same validated
format atomically, so native encoder output can be handed to the existing file
based tools without a Python-side repackaging step.

Inspect a condition file with:

```sh
./build/bin/pixal3d inspect-dinodata /path/to/image.dinodata
```

For the projection-attention Pixal3D cascade, an external encoder can export
the four stage conditions as NumPy arrays and package them with
[`scripts/export_pixal3d_condition_bundle.py`](scripts/export_pixal3d_condition_bundle.py).
The resulting `P3DCOND` file stores global DINO tokens plus HWC DINO/optional
NAF feature maps; it does not store dense 3D grids. The C++ loader validates
all dimensions and finite F32 payloads, and `inspect-condition` reports the
stage map contracts:

```sh
python3 scripts/export_pixal3d_condition_bundle.py conditions.json conditions.p3dcond
./build/bin/pixal3d inspect-condition conditions.p3dcond
```

The manifest accepts `ss`, `shape_512`, `shape_1024`, and `tex_1024` stages.
For each stage, `global` is `[tokens,1024]`, `dino_map` is `[H,W,1024]`, and
the optional `naf_map` is `[H,W,1024]`; map objects may specify `layout: CHW`
for exporters that write channel-first arrays. The C++ projection bridge
reprojects maps at the actual 64³ or 96³ cascade grid and gathers only active
sparse coordinates.

For a native image boundary, `pixal3d encode-condition-stage` decodes PNG/JPEG
(when CMake finds libpng/libjpeg) or PNM, resizes with a separable Lanczos-3
filter and half-pixel centers compatible with the reference (without its final
uint8 quantization), runs the converted DINOv3 model and
optional NAF model, and writes a one-stage `P3DCOND` bundle. The stage defaults
match the released pipeline: `ss` 512px without NAF, `shape_512` 512/512,
`shape_1024` 1024/512, and `tex_1024` 1024/1024. Use `-` as the NAF path for
the `ss` stage:

```sh
./build/bin/pixal3d encode-condition-stage \
  build/weights/dinov3-vitl16-pretrain-lvd1689m-f32.gguf - input.png ss ss.p3dcond
./build/bin/pixal3d encode-condition-stage \
  build/weights/dinov3-vitl16-pretrain-lvd1689m-f32.gguf \
  build/weights/naf-f32.gguf input.png shape_512 shape_512.p3dcond
```

The command is intentionally stage-level: production inference still needs a
four-stage bundle (or a caller can use the in-memory vision-condition API to
encode all stages without intermediate files).

For a single-process path, `run-image` performs the same four-stage encoding
in memory and immediately enters the native cascade:

```sh
./build/bin/pixal3d run-image \
  build/weights/pixal3d-shared-f16.gguf \
  build/weights/pixal3d-base-flow-f32.gguf \
  build/weights/dinov3-vitl16-pretrain-lvd1689m-f32.gguf \
  weights/NAF/naf_release-f32.gguf input.png output.obj \
  --max-model-gib 32 --seed 42
```

`run-image` uses the same explicit front-view camera defaults as `run-cascade`;
`--fov`, `--distance`, `--mesh-scale`, `--resolution`, `--max-tokens`, and
`--steps` override them. The native executable deliberately does not hide a
Python matting or MoGe dependency. For an RGB image, run the matching
preprocess helper first (it follows the reference 1024px resize, alpha mask,
1.1x square crop, and black compositing):

```sh
python3 scripts/preprocess_pixal3d_image.py input.png input-preprocessed.png
./build/bin/pixal3d run-image \
  build/weights/pixal3d-shared-f16.gguf \
  build/weights/pixal3d-base-flow-f32.gguf \
  build/weights/dinov3-vitl16-pretrain-lvd1689m-f32.gguf \
  weights/NAF/naf_release-f32.gguf input-preprocessed.png output.obj \
  --max-model-gib 32 --seed 42
```

The helper uses rembg's `birefnet-general` model by default; use
`--model u2net` only when BiRefNet is unavailable. Camera estimation is still
explicit: native runs use the supplied/manual front-view defaults, while the
reference Python pipeline additionally estimates FOV and distance with
MoGe-2. Pass the same `--fov`/`--distance` values when reproducing a reference
camera. `--max-model-gib` is checked against the metadata-only F32 resident
estimate before DINO/NAF image encoding starts, so an undersized host exits
without allocating the vision models.

For a bounded GPU smoke test, `--vision-resolution N` overrides all four image
condition inputs to the same multiple-of-16 size (the default production
inputs remain 512/512/1024/1024). This is a throughput/debug option only and
does not represent the released image-quality path. The reusable test command
is [`tests/run_image_smoke.py`](tests/run_image_smoke.py); it uses the
experimental F16 flow pack to fit a 24-GiB development GPU and deliberately
uses a diagnostic occupancy threshold (`-100`) plus a one-point structure cap.
These flags only exercise the downstream mesh writer quickly; production keeps
the trained threshold (`0`) and disables the structure-point cap.

For the multi-view reference pipeline, use
[`scripts/export_pixal3d_multiview_condition_bundle.py`](scripts/export_pixal3d_multiview_condition_bundle.py).
Each `views` entry carries one global/map set and an absolute row-major c2w
matrix per camera. The resulting `P3DMVCON` file preserves the view set; the
C++ bridge computes `F @ inverse(C0) @ Ci` and applies the reference
elementwise average fusion to global tokens and projected rows. All stages
must contain the same number of views. The current ggml denoisers expose the
reference `average` fusion mode. Native callers can write an already assembled
bundle with `save_pixal3d_multiview_condition_bundle()`; it uses the same
atomic byte layout as the Python exporter:

```sh
python3 scripts/export_pixal3d_multiview_condition_bundle.py \
  multiview_conditions.json multiview_conditions.p3dmvcon
./build/bin/pixal3d inspect-mv-condition multiview_conditions.p3dmvcon
```

The projection/fusion contract has a standalone Python-oracle comparison:

```sh
cmake --build build --target pixal3d_multiview_condition_fixture
docker run --rm --user "$(id -u):$(id -g)" \
  --entrypoint /opt/conda/bin/python \
  -v "$PWD:/workspace/Pixal3D.cpp" \
  -w /workspace/Pixal3D.cpp \
  local/comfyui-h3:0.34.0-cu126 scripts/compare_condition_multiview.py
```

It checks both global-token averaging and sparse projected DINO+NAF rows
against the reference `compute_relative_calc_mat()` plus `ProjGridMV` path.

With a complete four-stage bundle, the production-side C++ driver can run the
seven GGUF components and write the decoded geometry as OBJ. It performs a
metadata-only F32 resident-memory estimate before loading; use the optional
limit to fail before allocating if the host budget is too small:

```sh
./build/bin/pixal3d estimate-model \
  build/weights/pixal3d-shared-f16.gguf \
  build/weights/pixal3d-base-flow-f32.gguf
./build/bin/pixal3d run-cascade \
  build/weights/pixal3d-shared-f16.gguf \
  build/weights/pixal3d-base-flow-f32.gguf \
  conditions.p3dcond output.obj \
  --resolution 1536 --max-tokens 49152 --seed 42 \
  --max-model-gib 32
```

`run-cascade` uses the pipeline defaults from `pipeline.json` (12 Euler steps,
front-view camera defaults) and accepts `--fov`, `--distance`, and
`--mesh-scale` for camera overrides. `--occupancy-threshold` is available for
diagnostic runs; production inference should keep the trained default (0).
`--max-structure-points N` is another optional development smoke-test cap
applied after SS occupancy extraction; it is disabled by default and should
not be used for final geometry.
The output OBJ contains geometry only;
decoded texture voxel attributes remain in the C++ result for the forthcoming
material/voxel sidecar writer. The native DINOv3/NAF encoder and camera
estimation are intentionally not hidden in this command.

The multi-view command uses the converted `pixal3d-mv-flow` pack and the same
sampler options:

```sh
./build/bin/pixal3d run-cascade-mv \
  build/weights/pixal3d-shared-f16.gguf \
  build/weights/pixal3d-mv-flow-f32.gguf \
  multiview_conditions.p3dmvcon output.obj --max-model-gib 32
```

Callers that already hold native per-view DINO/NAF stages can use
`run_pixal3d_from_multiview_condition_stages()` and avoid writing a temporary
`P3DMVCON` file; it applies the same relative-camera average fusion.

## Weights

Pixal3D model files are kept outside the source tree under
`weights/Pixal3D/` (the directory is ignored by Git). The downloader pins the
official Hugging Face snapshot revision in `.download-metadata.json` and keeps
partial files as `*.part` so interrupted transfers can be resumed safely.

Start the full download in the background (the current snapshot is about
42.9 GiB):

```sh
cd /home/sy/Pixal3D.cpp
chmod +x scripts/download_pixal3d_weights.sh
nohup env PIXAL3D_DOWNLOAD_WORKERS=8 \
  ./scripts/download_pixal3d_weights.sh \
  > /tmp/pixal3d-download.log 2>&1 &
echo $! > /tmp/pixal3d-download.pid
```

The command resumes the existing `.part` files and verifies completed files
with their expected size/SHA-256. Monitor it with:

```sh
tail -f /tmp/pixal3d-download.log
ps -p "$(cat /tmp/pixal3d-download.pid)" -o pid,etime,cmd
du -sh weights/Pixal3D
```

If the process exits because of a transient network error, run the same
`nohup` command again. Lower `PIXAL3D_DOWNLOAD_WORKERS` (for example, to `2`)
if the network or disk cannot sustain eight parallel streams.

For unreliable proxies, use the persistent retry monitor instead. It downloads
in bounded HTTP Range chunks and keeps retrying until the snapshot is complete:

```sh
setsid nohup env PIXAL3D_DOWNLOAD_WORKERS=4 \
  PIXAL3D_DOWNLOAD_CHUNK_BYTES=67108864 \
  bash scripts/monitor_pixal3d_weights.sh \
  </dev/null > /tmp/pixal3d-download.log 2>&1 &
```

Set `PIXAL3D_PROXY=http://host:port` when a specific proxy must be used. The
downloader synchronizes uppercase/lowercase proxy variables so a stale
lowercase value does not silently override the selected proxy.

The separate NAF release is about 2.66 MB and is pinned by
[`scripts/download_naf_weights.sh`](scripts/download_naf_weights.sh); it is
stored under `weights/NAF/` and verified against its fixed SHA-256 before use.
