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

Positional arguments are the corresponding GGUF, condition, and image paths;
SLat inspection also takes its component name.  The output path for
`run-cascade`, `run-image`, and `run-cascade-mv` is optional: when omitted,
the command writes `output.glb` in the current directory.  GLB is the
recommended textured asset format; an explicit `.obj` path selects the
geometry-only compatibility writer.

The SLat components are `shape_decoder|texture_decoder` and
`shape_flow_512|shape_flow_1024|texture_flow_1024`. Typical runs:

```sh
./build/bin/pixal3d inspect-pack build/weights/pixal3d-shared-f16.gguf
./build/bin/pixal3d estimate-model build/weights/pixal3d-shared-f16.gguf build/weights/pixal3d-base-flow-f32.gguf
# Recommended: omit the output path to create ./output.glb with PBR textures.
./build/bin/pixal3d run-cascade build/weights/pixal3d-shared-f16.gguf build/weights/pixal3d-base-flow-f32.gguf conditions.p3dcond --resolution 1024 --texture-size 1024 --seed 42 --max-model-gib 32
# The same textured output with an explicit path.
./build/bin/pixal3d run-cascade build/weights/pixal3d-shared-f16.gguf build/weights/pixal3d-base-flow-f32.gguf conditions.p3dcond output.glb --resolution 1024 --texture-size 1024 --seed 42 --max-model-gib 32
# Compatibility output: explicit OBJ is geometry-only.
./build/bin/pixal3d run-cascade build/weights/pixal3d-shared-f16.gguf build/weights/pixal3d-base-flow-f32.gguf conditions.p3dcond output.obj --resolution 1024 --seed 42 --max-model-gib 32
./build/bin/pixal3d run-cascade-mv build/weights/pixal3d-shared-f16.gguf build/weights/pixal3d-mv-flow-f32.gguf views.p3dmvcon output.glb --resolution 1024 --texture-size 1024 --max-model-gib 32
```

Run options are `--seed`, `--resolution 1024`, `--max-tokens`, `--steps` (>0),
`--occupancy-threshold`, `--max-structure-points` (>0), `--fov` (0..pi radians),
`--distance` (>0), `--mesh-scale` (>0), `--max-model-gib`, and
`--texture-size` (1..4096 for `.glb` output). The native exporter bakes a
chart-unwrapped atlas through the mesh postprocess chain in
`src/mesh_postprocess.cpp` (adapted from pwilkin/trellis.cpp, MIT): weld hairline cracks,
unify face winding, drop floating fragments, Taubin-smooth the voxel
stair-step noise, then a CuMesh-port QEM decimation to the reference 1,000,000
face target (the reference `to_glb` decimation target), hole filling, and
xatlas UV unwrap with a trilinear bake of the decoded texture volume - the
same parameterization backend the reference wraps. A BVH over the
pre-decimation surface snaps texels that fall between voxels back onto the
surface, matching the reference cuBVH correction. The raw dual grid mesh is
inconsistently wound and sliver heavy, which stalls plain quadric
simplification and degenerates charting; the postprocess chain exists to
digest it. GLB texture output requires a build with libpng; PNM input remains
available without PNG/JPEG support, and explicit `.obj` output remains
geometry-only. The default final cascade resolution is `1024`,
with seed `42`, 12 Euler steps, threshold `0`, and a front-view camera.
`max_num_tokens` is a hard guard at this resolution; it does not silently
select another final resolution. The point cap and changed threshold are
diagnostic controls. `run-image` additionally accepts `--vision-resolution N`
(>=16, divisible by 16) as a small smoke-test override.

### Per-image camera estimation (run-image)

The projection conditioning is sensitive to the camera that maps the image
into the conditioning volume: images shot with a narrow lens combined with the
wide default front camera (`--fov 0.8576 --distance 2.0`, the TRELLIS render
convention) can place the subject outside the conditioning volume, and the
generative stages then truncate it (for example a figure without its head).

`run-image` therefore estimates the camera per image by default, matching the
reference `inference.py` wild path: the official MoGe-2 ONNX export
(`Ruicheng/moge-2-vitl-normal-onnx`) runs through onnxruntime, the pinhole
focal is recovered from the predicted point map with the reference
`recover_focal_shift` fit (64x64 nearest downsample plus a one-parameter
Levenberg-Marquardt solve), and the distance follows the reference
`distance_from_fov` rule, which keeps the subject inside the conditioning
volume. The estimate is printed as `camera_estimated:` and echoed in the run
summary as `camera:`. Explicit `--fov`/`--distance` still win over the
estimation and skip it.

When onnxruntime or the model file is unavailable the run falls back to the
fixed front camera with a loud `pixal3d: warning:` naming the reason
(estimate ~2 degrees from the reference path, well inside the pipeline's
tolerance, but not bit-identical). `--moge-onnx <path>` selects the model
file; the default location is `weights/MoGe/moge-2-vitl-normal.onnx`.

Camera controls:

- default: estimate when possible, warn and fall back otherwise.
- `--camera auto`: require estimation; fail loudly when onnxruntime or the
  model is missing. Cannot be combined with explicit `--fov`/`--distance`.
- `--camera default`: always use the fixed front camera (also silences the
  fallback warning).

Download the pinned model with `./scripts/download_moge_weights.sh` (pinned
size and SHA-256, `.part` resume, atomic rename).

Building the estimation needs onnxruntime: point `PIXAL3D_ONNXRUNTIME_ROOT`
at a prebuilt onnxruntime tree (containing `include/onnxruntime_cxx_api.h`
and `lib/libonnxruntime.so`) when configuring CMake. Without it the build
succeeds, the default run warns and falls back, and `--camera auto` fails
with an explicit error.

### Mesh coordinate frames

The decoded `DualGridMeshF32` vertices stay in the canonical decoder AABB
frame, with axes corresponding to the sparse `[x, y, z]` grid.  The explicit
OBJ compatibility writer keeps the shape-only export frame
`(x, y, z) -> (-x, -z, -y)`.  The textured GLB writer follows the main Python
`to_glb()` path, whose internal axis swap followed by Pixal3D's outer rotation
produces the final GLB frame `(x, y, z) -> (-x, +y, -z)`.  Both transforms have
positive determinant, so face indices and winding are preserved.

For the official textured GLB front view, place the camera on the GLB `-Z`
side, look toward `+Z`, and use `+Y` as image-up.  For the explicit legacy OBJ
frame, the equivalent reference-facing view is on the `-Y` side, looking toward
`+Y`, with `-Z` as image-up.  Preview tools must not apply another axis swap.

These export conversions only change coordinate frames; they do not fill
missing geometry or replace the Python CuMesh remesh/decimation postprocess.

A headless preview of the textured GLB is available with
`blender -b -P scripts/render_glb_preview.py -- <model.glb> <out_prefix>`.
Its orbit azimuths follow the multiview dataset naming (azim000 is the front
view), and `--transforms <path/to/transforms.json>` reproduces the capture
viewpoints and field of view exactly.  The capture matrices predate the
GLB's H flip, so the script applies the matching 180-degree turn before
placing the cameras.


## Resolution support

The cascade resolution is not the same thing as the resolution of every model
stage. Pixal3D.cpp uses the following support matrix:

| Resolution | Role | Status |
|---|---|---|
| `512³` | Coarse/internal structure and low-resolution shape stage | Required internal cascade stage; not a standalone final-output guarantee |
| `1024³` | Standard high-resolution shape/texture cascade target | Only supported final output resolution for the current C++ pipeline |

The `512³` stage is an internal coarse shape stage. The outer C++ pipeline
intentionally exposes only the validated `1024³` final target; upstream
reference implementations may expose additional test-time cascade modes, but
they are not part of this C++ configuration contract.

`encode-condition-stage` accepts `--resolution` and `--naf-resolution`; stages
are `ss`, `shape_512`, `shape_1024`, and `tex_1024`. `-` is valid as the NAF
path only for `ss`. `encode-condition-bundle` requires NAF and accepts the four
stage resolution flags plus four `--*-naf-resolution` flags. A stage file is
only one stage; a production cascade needs all four stages or the in-memory API.

## Model packs and conversion

Keep checkpoints, GGUF files, conditions, latent dumps, OBJ/GLB outputs, and
texture images outside Git. Pixal3D weights are stored under the ignored
directory `weights/Pixal3D/`.
The NAF release is stored under `weights/NAF/`.

Download the pinned assets in the foreground:

```sh
./scripts/download_pixal3d_weights.sh
./scripts/download_naf_weights.sh
./scripts/download_moge_weights.sh
```

The Pixal3D downloader resumes `.part` files and verifies completed files by
size and available SHA-256 metadata. The NAF downloader requires the GitHub CLI
(`gh`) and verifies its pinned release. The MoGe downloader fetches the pinned
`Ruicheng/moge-2-vitl-normal-onnx` asset used by `run-image` camera estimation
(~1.3 GiB) and verifies size and SHA-256 before an atomic rename. The full
Pixal3D snapshot is roughly 43 GiB; do not copy it into a Docker image.

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
otherwise use PNM. RGBA PNG inputs now follow the reference foreground path:
longest-side cap, alpha foreground crop with 10% square margin, and black
background compositing. RGB/JPEG inputs still do not run rembg, and native
image inference does not perform MoGe camera estimation. For photographs, the
optional helper can prepare an explicit input using rembg:

```sh
python3 scripts/preprocess_pixal3d_image.py input.png prepared.png
```

The native path uses manual front-camera defaults or the camera options above;
`run-image` requires DINO and NAF GGUFs. Set
`PIXAL3D_CASCADE_SPATIAL_TRACE=1` to print x/y/z occupancy buckets after SS,
coordinate upsampling, high-resolution quantization, and shape decoding.

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

## Release packages

Release archives contain the CLI and documentation only. Model weights, GGUF
packs, condition files, latent dumps, meshes, and build directories are never
included. The local packaging helper writes ignored artifacts under `dist/`:

```sh
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTING=ON \
  -DPIXAL3D_ENABLE_CUDA=OFF \
  -DPIXAL3D_GGML_NATIVE=OFF
cmake --build build-release --parallel
(cd build-release && ctest --output-on-failure)
./scripts/package_release.sh --build-dir build-release --variant cpu
(cd dist && sha256sum -c SHA256SUMS)
```

The resulting `pixal3d-<version>-linux-<arch>-cpu.tar.gz` is a CLI-only
runtime archive. It contains `bin/pixal3d`, `README.md`, runtime dependency
notes, build metadata, and third-party license text. It does not contain the
CMake SDK library/headers or model files. The generated `RELEASE-MANIFEST.json`
records the asset name, size, SHA-256, platform, backend, source commit, ggml
submodule commit, workflow run, and validation status.

CPU ggml graph execution reuses a manager-owned threadpool for the lifetime of
one backend manager. Stage code still selects its requested thread count through
the shared backend manager; this does not change scheduler placement or numerical
behavior. Builds with ggml OpenMP use OpenMP's worker team for each parallel
region, so a persistent pool should not be interpreted as a permanent set of
OS threads in that configuration.

The outer project is released under the MIT License in `LICENSE`. `NOTICE`
contains the upstream Pixal3D attribution, third-party component summary, and
responsible-use guidance. README is explanatory documentation; it does not
replace those files. Both files are included in the CMake install tree and in
formal binary archives, alongside `ggml-LICENSE`. The ggml license is not a
license for the outer Pixal3D.cpp project.

Model weights are not included in the binary archive. They are downloaded
separately and remain subject to the upstream Pixal3D model repository's
license, notice, and access metadata.

Following the convention used by whisper.cpp and llama.cpp, keep these items in
Git:

```text
source, headers, CMake files
.github/workflows/*
scripts/package_release.sh
scripts/validate_release_candidate.sh
README.md and release documentation
```

Keep these items out of Git. `dist/` is ignored and the generated files belong
in Actions artifacts or the GitHub Release instead:

```text
build*/
dist/
*.tar.gz, *.zip, *.gguf, *.safetensors
weights/, meshes, latent dumps, Blender files
```

Download a formal release from the repository's [GitHub Releases](/../../releases)
page and verify it in the directory containing the downloaded assets:

```sh
sha256sum -c SHA256SUMS
```

The release manifest is generated from the same candidate as the archive; it is
not hand-maintained in the source repository. Formal publication requires both
`LICENSE` and `NOTICE` to be present in the candidate archive.

Releases use a two-stage, manually gated workflow modeled after whisper.cpp.
`.github/workflows/release.yml` is a candidate build only: it runs the portable
CPU tests and uploads an Actions artifact, but never creates a tag or modifies a
GitHub Release. The candidate can optionally include a separate CUDA
`compile-only` artifact. That artifact is not a claim of GPU runtime support:
forced `gpu:<index>` execution, scheduler placement, memory usage, and CPU/GPU
numerical parity still require a matching NVIDIA runner or local CUDA host.

After reviewing a successful candidate, use `.github/workflows/make-release.yml`
with the exact candidate run ID. It downloads and rechecks the candidate
archive, verifies the CLI version, checksum, source commit, and release tag,
and rejects an existing tag or Release. A dry run performs all checks without
publishing; only `dry_run=false` creates the tag at the tested commit and
uploads the immutable CPU assets:

```sh
# 1. Build and test a candidate; this creates no GitHub Release.
gh workflow run release.yml --ref master \
  -f build_cuda_compile=false

# 2. Inspect the successful candidate run and record its run ID.
gh run list --workflow release.yml --limit 5

# 3. Validate that exact candidate without publishing.
gh workflow run make-release.yml --ref master \
  -f candidate_run_id=123456789 \
  -f release_tag=v0.3.0 \
  -f dry_run=true

# 4. After review, publish the same candidate run.
gh workflow run make-release.yml --ref master \
  -f candidate_run_id=123456789 \
  -f release_tag=v0.3.0 \
  -f dry_run=false
```

The final workflow does not rebuild from a moving branch and does not overwrite
existing tags or Release assets. CUDA packages require a compatible NVIDIA
driver/CUDA runtime and are not bundled with model weights; the compile-only
CUDA candidate is not included in the formal CPU Release.

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
registers 19 tests covering CLI startup, projection, flow primitives and
sampler, pack reading, DINO, backend management, image/condition helpers,
inference utilities, occupancy coordinates, Dual Grid, bounded mesh hole
filling, and the registered SLat fixtures. Run them with:

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

The deterministic SLat decoder parity tool uses a three-level decoder fixture
and compares every subdivision level, strict `logit > 0` mask, coordinate set,
and final features against the Python module graph:

```sh
python3 scripts/compare_slat_decoder.py
```

The fixture is also registered as `pixal3d_slat_decoder_fixture` with
`unit;parity;cpu` labels. For a real cascade, set
`PIXAL3D_SLAT_DECODER_VERBOSE=1` to log shape-high coordinate fingerprints,
per-level decoder point/active counts, and final decoded coordinate
fingerprints. This is diagnostic only and does not change decoder precision or
backend placement.

## Current limitations

- OBJ output remains geometry-only. GLB output now includes an approximate
  single PBR material, base-color texture, and metallic-roughness texture
  baked from the six-channel texture voxel output. Its native spherical UV
  unwrap is deterministic but is not bit-exact with Python CuMesh unwrap,
  remeshing, or decimation.
- Cascade mesh output applies a CPU bounded boundary-loop fan fill matching the
  observable CuMesh `fill_holes(0.03)` contract. A separate
  `repair_non_manifold_edges_f32()` helper can split vertices for export-stage
  diagnostics, but it is not enabled by default: on the 1024/12-step diagnostic
  mesh it removed 218,118 over-shared edges while increasing the mesh from 10
  to 57,343 connected components. It is not the Python CuMesh implementation
  and does not include the separate Python remesh/decimation postprocess.
- Native image inference now applies the alpha-aware RGBA crop and black matte
  used by the reference. It still has no rembg fallback for RGB inputs and no
  MoGe camera estimation.
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
