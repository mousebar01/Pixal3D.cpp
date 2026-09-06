# Pixal3D.cpp

Pixal3D.cpp is a C++17/ggml migration scaffold for Pixal3D image-to-3D. The
outer `src/` and `include/` implementation is authoritative; `ref/Pixal3D/` and
`ref/trellis2cpp/` are reference-only. GGUF loaders, CPU/F32 paths, native
DINOv3/NAF components, and a bounded real-weight smoke path are available, but
unrestricted Python parity and production image quality are not claimed.

## Requirements

- CMake 3.16 or newer
- A C++17 compiler
- The top-level `ggml` submodule (the default configuration)
- Python 3 only for conversion, preprocessing, and reference-comparison tools
- PNG/JPEG development libraries are optional; PNM (`P2`, `P3`, `P5`, `P6`)
  input is always available
- OpenMP is optional

Initialize the submodule and make a portable CPU build:

```sh
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
(cd build && ctest --output-on-failure)
./build/bin/pixal3d --help
./build/bin/pixal3d --version
```

Options: `PIXAL3D_ENABLE_CUDA=ON` enables CUDA; `PIXAL3D_CUDA_TF32=ON` allows
approximate TF32; `PIXAL3D_GGML_NATIVE=ON` reduces CPU portability;
`PIXAL3D_BUILD_CLI=OFF` omits the CLI/tests; and `PIXAL3D_USE_BUNDLED_GGML=OFF`
uses an installed ggml package. TF32 is OFF by default. Do not reuse a Docker
build directory at a different path: configure CPU and CUDA builds separately.

## CLI

`pixal3d --help` is the executable source of truth. Supported command groups:

```text
inspect-pack, estimate-model, run-cascade, run-image, run-cascade-mv
inspect-dinodata, inspect-dino-vit, inspect-naf, inspect-condition
inspect-mv-condition, inspect-ss-flow, inspect-ss-decoder
inspect-slat-decoder, inspect-slat-flow
encode-condition-stage, encode-condition-bundle
```

Positional arguments are the corresponding GGUF, condition, image, and OBJ
paths; SLat inspection also takes its component name.

The SLat components are `shape_decoder|texture_decoder` and
`shape_flow_512|shape_flow_1024|texture_flow_1024`. Typical runs:

```sh
./build/bin/pixal3d inspect-pack build/weights/pixal3d-shared-f16.gguf
./build/bin/pixal3d estimate-model build/weights/pixal3d-shared-f16.gguf build/weights/pixal3d-base-flow-f32.gguf
./build/bin/pixal3d run-cascade build/weights/pixal3d-shared-f16.gguf build/weights/pixal3d-base-flow-f32.gguf conditions.p3dcond output.obj --resolution 1024 --seed 42 --max-model-gib 32
./build/bin/pixal3d run-cascade-mv build/weights/pixal3d-shared-f16.gguf build/weights/pixal3d-mv-flow-f32.gguf views.p3dmvcon output.obj --resolution 1024 --max-model-gib 32
```

Run options are `--seed`, `--resolution` (>=1024, divisible by 16),
`--max-tokens`, `--steps` (>0), `--occupancy-threshold`,
`--max-structure-points` (>0), `--fov` (0..pi radians), `--distance` (>0),
`--mesh-scale` (>0), and `--max-model-gib`. The parser accepts any resolution
that passes the numeric constraint, but that is not a model-support guarantee;
see [Resolution support](#resolution-support). Defaults are seed `42`, the
legacy experimental resolution `1536`, 12 Euler steps, threshold `0`, and a
front-view camera. Use `--resolution 1024` for the primary supported target.
The token cap may lower a requested value in 128-voxel steps, never below 1024.
The point cap and changed threshold are diagnostic controls. `run-image`
additionally accepts `--vision-resolution N` (>=16, divisible by 16) as a small
smoke-test override.

## Resolution support

The cascade resolution is not the same thing as the resolution of every model
stage. Pixal3D.cpp uses the following support matrix:

| Resolution | Role | Status |
|---|---|---|
| `512³` | Coarse/internal structure and low-resolution shape stage | Required internal cascade stage; not a standalone final-output guarantee |
| `1024³` | Standard high-resolution shape/texture cascade target | Primary supported end-to-end target for the current C++ pipeline |
| `1536³` | Test-time high-resolution cascade extension | Experimental and unverified; do not treat as formally supported |

The original TRELLIS paper does not define a `1536³` output target. The
`1536³` path is a TRELLIS.2-style test-time cascade extension, not an
independently trained model stage. It remains outside the standard validated
matrix until the complete CUDA execution, numerical checks, and Python-reference
parity gates pass.

Other values accepted by `--resolution` are parser-compatible experiments, not
supported resolutions. A multiple of 16 only satisfies argument validation; it
does not prove that the model, decoder, memory use, or output quality has been
validated at that size.

`encode-condition-stage` accepts `--resolution` and `--naf-resolution`; stages
are `ss`, `shape_512`, `shape_1024`, and `tex_1024`. `-` is valid as the NAF
path only for `ss`. `encode-condition-bundle` requires NAF and accepts the four
stage resolution flags plus four `--*-naf-resolution` flags. A stage file is
only one stage; a production cascade needs all four stages or the in-memory API.

## Model packs and conversion

Keep checkpoints, GGUF files, conditions, latent dumps, and OBJ outputs outside
Git. Pixal3D weights are stored under the ignored directory `weights/Pixal3D/`.
The NAF release is stored under `weights/NAF/`.

Download the pinned assets in the foreground:

```sh
./scripts/download_pixal3d_weights.sh
./scripts/download_naf_weights.sh
```

The Pixal3D downloader resumes `.part` files and verifies completed files by
size and available SHA-256 metadata. The NAF downloader requires the GitHub CLI
(`gh`) and verifies its pinned release. The full Pixal3D snapshot is roughly
43 GiB; do not copy it into a Docker image.

The converter groups eleven checkpoints into three packs:

| output | contents | use |
|---|---|---|
| `pixal3d-shared-*.gguf` | three decoders | both pipelines |
| `pixal3d-base-flow-*.gguf` | four single-view flows | cascade/image |
| `pixal3d-mv-flow-*.gguf` | four multi-view flows | multi-view cascade |

Convert all deployment packs or one namespaced component:

```sh
python3 scripts/convert_pixal3d_to_gguf.py --bundle all --dry-run
python3 scripts/convert_pixal3d_to_gguf.py --bundle all
python3 scripts/convert_pixal3d_to_gguf.py --component shape-dec
```

For a single component, `--model`, `--config`, `--output`, `--weights-dir`,
and `--output-dir` override the defaults. `--force`, `--dry-run`, and
`--no-verify` are available. Supported component selectors include `dino`,
`naf`, `ss-dec`, `shape-dec`, `tex-dec`, the three base flow stages, and the
four `_mv` flow stages.

### Precision policy

`--ftype auto` is the stable deployment policy, exactly:

```text
auto=F32 flow, F16 decoder
```

Every flow bundle is stored as F32. The shared decoder bundle stores eligible
matrix tensors as F16 while sensitive and scalar tensors remain F32. `auto` is
not an all-F32 mode.

Use `--ftype 0` explicitly for numerical porting and Python-reference parity.
It writes all-F32 GGUF storage, but cannot restore precision lost in an FP16
source checkpoint. `--ftype 1` is mostly F16. `--ftype 2` is mostly BF16 and
is flow-only; shared decoder bundles accept only `0` and `1`. DINOv3 and NAF
currently require `--ftype 0`. A `_bf16` filename suffix does not prove the
payload dtype; inspect generated metadata when it matters.

## Conditions and image input

`P3DCOND` stores the four stages (`ss`, `shape_512`, `shape_1024`, `tex_1024`)
with global F32 tokens, DINO maps, and optional NAF maps:

```sh
python3 scripts/export_pixal3d_condition_bundle.py manifest.json conditions.p3dcond
./build/bin/pixal3d inspect-condition conditions.p3dcond
```

`P3DMVCON` stores the same data per view plus row-major 4x4 camera-to-world
matrices; all stages must have the same view count:

```sh
python3 scripts/export_pixal3d_multiview_condition_bundle.py views.json views.p3dmvcon
./build/bin/pixal3d inspect-mv-condition views.p3dmvcon
```

The native bridge applies the reference relative-camera transform and average
fusion, projecting only the sparse coordinates requested by the sampler.

`run-image` reads PNG/JPEG when those libraries were found at configure time;
otherwise use PNM. Native image inference does not perform background matting
or MoGe camera estimation. For photographs, the optional helper can prepare an
explicit input using rembg:

```sh
python3 scripts/preprocess_pixal3d_image.py input.png prepared.png
```

The native path uses manual front-camera defaults or the camera options above;
`run-image` requires DINO and NAF GGUFs.

## Backend selection

The shared backend policy accepts `auto`, `cpu`, `gpu`, and `gpu:<index>`:

```sh
PIXAL3D_BACKEND=cpu ./build/bin/pixal3d run-cascade ...
PIXAL3D_DINO_BACKEND=gpu:0 ...
```

Stage-specific variables are `PIXAL3D_DINO_BACKEND`,
`PIXAL3D_SLAT_BACKEND`, `PIXAL3D_SLAT_DECODER_BACKEND`,
`PIXAL3D_SS_FLOW_BACKEND`, `PIXAL3D_SS_DECODER_BACKEND`, and
`PIXAL3D_NAF_BACKEND`; each overrides `PIXAL3D_BACKEND`. `auto` lets
`ggml_backend_sched` place supported graph nodes across GPU/ACCEL and CPU,
with fallback reasons logged when a preferred GPU path is unavailable. `cpu`
forces CPU-only execution. `gpu:<index>` fixes the primary GPU: core matmul,
attention, gather, im2col, and convolution nodes cannot silently downgrade,
while explicit CPU-resident work remains on the host. `PIXAL3D_SLAT_VERBOSE=1`
logs SLat fallback reasons. Set `PIXAL3D_BACKEND_TRACE=1` to print final scheduler
node placement, split/copy counts, per-backend compute buffers, and device memory
snapshots (including persistent weight allocation). NAF GPU acceleration covers only the
reflection/convolution stacks; GroupNorm, pooling, learned 2-D RoPE, and
neighborhood attention remain CPU-resident. SLat decoder sparse coordinate,
subdivision, packing, and host reference operations likewise remain CPU-side;
its GPU path is partial, not a claim of whole-stage GPU execution.

## Docker and CUDA

The small C++ Dockerfile provides CPU-capable `dev`, `cpp-build`, and `runtime`
stages:

```sh
docker build -f docker/Dockerfile --target runtime -t pixal3d:scaffold .
docker run --rm pixal3d:scaffold --version
docker compose -f docker/compose.yaml run --rm dev
```

Build CUDA with matching NVIDIA developer/runtime images and a separate build
directory:

```sh
docker build -f docker/Dockerfile --target runtime -t pixal3d:cuda \
  --build-arg DEV_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04 \
  --build-arg RUNTIME_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04 \
  --build-arg PIXAL3D_ENABLE_CUDA=ON .
```

The CUDA image compiles the backend but skips CTest without NVIDIA driver
libraries. Run GPU tests and inference with `--gpus all`; attention may need
`--shm-size=24g`. `local/comfyui-h3:0.34.0-cu126` is an external oracle image.
`docker/Dockerfile.python` is opt-in and installs large dependencies only with
`INSTALL_PIXAL3D_DEPS=ON`.

CUDA is not required for CPU runs. Repeated CUDA compilation usually means a
CUDA cache was reused after source changes or a container build tree was moved;
keep CPU, CUDA, and container build directories separate.

## Tests and validation

CTest is authoritative for registered tests. The default testing build
registers 18 tests covering CLI startup, projection, flow primitives and
sampler, pack reading, DINO, backend management, image/condition helpers,
inference utilities, occupancy coordinates, Dual Grid, and the registered SLat
fixtures. Run them with:

```sh
(cd build && ctest --output-on-failure)
```

Many deterministic executables are build targets but not CTest tests, including
NAF, sparse, SS, SLat loader/concat, and cascade fixtures. Their Python oracle
comparators under `scripts/` and `tests/` are manual tools requiring declared
fixtures and often the external Python image. Building a `*_fixture` target does
not run it under CTest.

The bounded real-image smoke path uses deliberately small vision inputs, two
steps, an occupancy threshold of `-100`, a one-point structure cap, and an
experimental F16 flow pack. It checks finite OBJ geometry only. Full 30-block
production parity, unrestricted image quality, and bitwise-identical GPU/CPU
mesh topology near occupancy thresholds are not claimed.

## Current limitations

- Wavefront OBJ output contains geometry only. Texture voxel attributes remain
  in memory; material and texture sidecar output is not implemented.
- Native image inference has no hidden rembg/matting, MoGe camera estimation,
  or Python preprocessing step.
- GPU execution is partial and capability-driven. `auto` uses scheduler-managed
  CPU/GPU cooperation; forced GPU reports failures for required core nodes.
  CPU-resident preprocessing, sampling, sparse topology, and mesh extraction
  are intentionally not GPU-required.
- All current model execution paths use the shared backend scheduler; any
  remaining backend-specific limitations are capability or stage-boundary
  constraints, not a reason to duplicate backend selection logic.
- DINO, NAF, and model stages should not be treated as fully supported merely
  because a checkpoint can be serialized; metadata, layout, loader, forward,
  and reference checks remain the support gates.

## Repository layout

```text
src/, include/       outer Pixal3D C++ library and CLI
scripts/             download, conversion, preprocessing, and oracle helpers
tests/               C++ tests, fixtures, and Python comparisons
ref/Pixal3D/         original Python reference
ref/trellis2cpp/     bundled ggml stage-1 reference
docker/              reproducible C++ and optional Python images
weights/             ignored local model data
```

Keep generated weights, GGUF packs, condition files, latent dumps, meshes, and
build directories out of version control. This README is a support and status
guide, not a claim that every serialized component is production-ready.
