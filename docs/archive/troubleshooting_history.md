# Pixal3D.cpp Troubleshooting History

> **历史归档：** 本文件保留用于追溯，不是当前实现状态、性能基线或支持矩阵。


- **Recorded:** 2026-09-11
- **Scope:** findings and mitigations from the Pixal3D C++ migration and local
  end-to-end validation session

This document is a session-level index. It records both confirmed root causes
and important false leads so that a future investigation does not repeat the
same work. The detailed gray-texture analysis is in
[`texture_gray_diagnosis.md`](texture_gray_diagnosis.md).

## Findings at a glance

| issue | status | decision |
| --- | --- | --- |
| Textured output looked gray for `armor_knight.png` | diagnosed | No decoder, channel-order, UV-bake, or GLB-export change; use a saturated-color fixture for color checks |
| Texture projection or preprocessing suspected of causing gray output | ruled out for this fixture | Preserve the current coordinate order and preprocessing path; require a reference mismatch before changing either |
| Python/C++ final texture voxel counts differ | unresolved | Track as a separate F32/BF16 and sparse-decoder parity task |
| Host build was slow and reported old/incompatible environment versions | mitigated | Build, test, and run inference in Docker with separate build directories |
| Parallel worktree changes and generated build/run state risked accidental rollback or contamination | process risk | Use path-scoped Git operations and keep generated state local and isolated |

## 1. Gray textured output was a misleading symptom

### Symptom

The textured GLB preview generated from the fixed image appeared nearly gray.
The initial suspects were the decoder channel layout, the output conversion,
UV baking, and GLB base-color export.

### Evidence and diagnosis

The fixed fixture is:

```text
weights/Pixal3D/test_inputs/armor_knight.png
SHA-256: d4e48ed407101749dc7bfe0bcafcdadd27278fbe563038a9c49ad9e5ff58bc19
```

It is mostly a silver-gray metal knight, dark-gray chain mail, brown leather,
and black background. It is not a high-saturation color-chart image.

The C++ `texture_decoded` values were already low-chroma before UV baking:

```text
mean RGB       = (0.28026, 0.25714, 0.24046)
mean |R-G|     = 0.02305
mean |G-B|     = 0.01681
mean |R-B|     = 0.03975
```

The Python reference, using the same image, seed `42`, `fov=0.558434`,
`distance=1.743943`, and `mesh_scale=1.0`, showed the same behavior:

```text
mean RGB       = (0.26124, 0.24864, 0.24482)
mean |R-G|     = 0.01325
mean |G-B|     = 0.00634
mean |R-B|     = 0.01693
```

The C++/Python Texture projection comparison was also close:

```text
correlation   ~= 0.9999886
relative L2   ~= 0.00481
```

Swapping coordinate axes increased the error to approximately `0.35`–`0.46`.
The preprocessed images differed by at most `3/255`, with a mean difference of
approximately `0.149/255`.

### Resolution and prevention rule

This was not evidence of a C++-only grayscale conversion bug. The low
saturation is shared by the reference and C++ paths and is consistent with the
fixture's mostly gray material.

- Do not change the decoder, channel order, `value * 0.5f + 0.5f` conversion,
  UV bake, or GLB exporter solely to make this fixture look more colorful.
- Keep `armor_knight.png` as an end-to-end regression fixture for pipeline
  stability and output structure.
- Use a separate synthetic or photographed high-saturation fixture with clear
  red, green, and blue regions for channel-order and color-fidelity tests.
- On a new gray-output report, inspect `texture_decoded` before UV baking and
  compare with the Python reference before changing model math.

**Status:** resolved as a false lead for this fixture; no code fix was required.

## 2. Projection and preprocessing were ruled out as the gray-output cause

### Problem considered

A mismatch in camera projection, XYZ coordinate order, or image preprocessing
could have changed the texture conditions and made the output appear gray.

### Resolution

The projection correlation, relative L2 error, axis-permutation experiment, and
preprocessing pixel differences did not support that hypothesis. Preserve the
current coordinate order and preprocessing behavior. Treat an axis or
preprocessing change as a code change only when a reference comparison
identifies a reproducible mismatch and the proposed fix reduces it.

This finding is part of the gray-output incident, but it is recorded separately
because it is a reusable diagnostic checkpoint.

## 3. Python/C++ texture voxel counts differ independently of the gray output

### Symptom

The final texture voxel counts did not match:

```text
Python = 4,644,079
C++    = 4,127,527
```

### Current status

This discrepancy is not solved in this session. There is no evidence that it
caused the low-saturation appearance. Do not use the gray preview as proof of
its cause and do not fold this investigation into an unvalidated color fix.

### Required next step

Track it as a separate numerical-parity task. Compare lossless F32 outputs
first, then inspect sparse decoder coordinate sets, BF16/F32 conversion, tensor
layouts, and active-mask/subdivision behavior. Record the exact fixture, seed,
backend, fingerprints, relative L2 error, and discrete coordinate agreement
before testing F16 or quantized storage.

## 4. Host build and version errors are an environment problem

### Symptom

Host-side compilation was slow, and the local development environment reported
old or incompatible tool versions during the migration work.

### Resolution and operating rule

Use the repository Docker setup for compilation, tests, and inference instead of
falling back to a host build for the normal workflow:

- use `docker/Dockerfile` and the appropriate `dev`, `cpp-build`, or `runtime`
  stage;
- use the CUDA `12.6.3` development/runtime images for CUDA validation;
- keep CPU, CUDA, and container build directories separate;
- mount model weights from the ignored `weights/` directory rather than baking
  them into an image;
- treat a successful image build as an environment check, not as numerical or
  CPU/GPU parity evidence.

The recorded Docker CUDA build completed successfully, and the current CUDA
build has 23 registered tests (including the sparse-linear fixture). Future
handoffs must still include the exact image, build command, backend policy, and
test result.

A separate host-side invocation of `build-cuda-docker/bin/pixal3d` failed before
startup because `libonnxruntime.so.1` was not on the host loader path. The
working invocation mounts `/home/sy/.local/opt/onnxruntime-1.23.2` into the
container and sets `LD_LIBRARY_PATH=/opt/onnxruntime/lib:/workspace/build-cuda-docker/bin`;
keep using the Docker invocation rather than changing source or baking a host
mount-specific path into the project.

## 5. Worktree changes and generated state must not be rolled back broadly

### Problem

The workspace contained changes from a separate performance-optimization
process as well as build and diagnostic outputs. A broad rollback or cleanup
could erase work that was not owned by the current task.

### Resolution and operating rule

Before reverting or cleaning anything:

1. inspect `git status --short` and identify the owner and purpose of each
   changed path;
2. use path-scoped `git add`, `git restore`, and cleanup commands;
3. never use a blanket reset or clean operation when another process has active
   work in the same workspace;
4. commit only the files belonging to the current task;
5. leave model weights, logs, meshes, containers, and build directories outside
   the source change and outside version control.

For this session, the gray-diagnosis documentation was committed separately;
other worktree changes were intentionally left untouched.

## 6. Local diagnostic artifacts are evidence, not project sources

The gray-output investigation produced temporary evidence such as:

```text
/tmp/pixal3d-gray-debug-0910b/cascade/texture_decoded.bin
/tmp/pixal3d-condition-compare/cpp_tex_projection.bin
/tmp/pixal3d-condition-compare/reference_texture_attrs.npy
/tmp/python_full_20260911_retry.log
```

These files are useful for the current investigation but must not be copied
into `docs/`, committed, or used as a substitute for a reproducible fixture.
Keep generated outputs in ignored `build*`, `weights/`, or temporary directories
and document their role and cleanup/retention decision in the handoff.


## 7. Current real image-to-3D validation

### Fixture and command

The current worktree was exercised with the fixed image used in the gray-output
investigation:

```text
weights/Pixal3D/test_inputs/armor_knight.png
SHA-256: d4e48ed407101749dc7bfe0bcafcdadd27278fbe563038a9c49ad9e5ff58bc19
```

The diagnostic validation used the current CUDA Docker image and current
`build-cuda-docker/bin/pixal3d` binary. It kept `resolution=1024`, the
production `steps=12`, occupancy threshold `0`, and no structure-point cap.
It explicitly used `texture-size=64` only to check the end-to-end path quickly;
the decoded geometry and texture channels still ran through the normal
four-stage cascade, but this atlas size is not a visual quality reference.

The command was:

```sh
docker run --rm --gpus all \
  -v /home/sy/Pixal3D.cpp:/workspace \
  -v /home/sy/.local/opt/onnxruntime-1.23.2:/opt/onnxruntime:ro \
  -w /workspace \
  -e LD_LIBRARY_PATH=/opt/onnxruntime/lib:/workspace/build-cuda-docker/bin \
  pixal3d:cuda-host-opt \
  ./build-cuda-docker/bin/pixal3d run-image \
    weights/Pixal3D/gguf/pixal3d-shared-f16.gguf \
    weights/Pixal3D/gguf/pixal3d-base-flow-f32.gguf \
    weights/Pixal3D/gguf/dinov3-vitl16-pretrain-lvd1689m-f32.gguf \
    weights/NAF/naf_release-f32.gguf \
    weights/Pixal3D/test_inputs/armor_knight.png \
    build-cuda-docker/armor_knight.current-quality.glb \
    --moge-onnx weights/MoGe/moge-2-vitl-normal.onnx \
    --resolution 1024 --steps 12 --texture-size 64 --max-model-gib 32
```

### Result and checks

The run completed with exit code `0` in `2951` seconds. All image and cascade
stages completed, including DINO, NAF, sparse-structure flow/decoder, shape
512/1024 flow, texture flow, shape decoder, and texture decoder. The MoGe
camera estimate was `fov=0.558434` radians and `distance=1.743943`.

The ignored output `build-cuda-docker/armor_knight.current-quality.glb` was
validated independently:

- GLB 2.0 header and declared file length match (`79,526,520` bytes);
- one mesh, one PBR material, two embedded PNG textures;
- `2,116,014` finite positions/normals/UVs and `982,948` triangles;
- every index is in range, normals are unit length within `2e-7`, and UVs stay
  in `[0, 1]`;
- both embedded PNGs decode with valid CRCs at `64x64`;
- the base-color image has `1,439` distinct RGB values, so it is not a constant
  gray image;
- Blender 5.2.1 imported the GLB and rendered all four azimuth previews
  successfully.

This confirms the current code can perform a real image-to-3D run and emit a
structurally valid, renderable textured asset. The 64x64 result is a workflow
smoke test only. A separate upstream-aligned `texture-size=4096` run also
completed successfully; its output was structurally valid and rendered in
Blender, but it took about 3069 seconds on the local RTX 4090 D because the
current C++ mesh postprocess is CPU-heavy. The earlier bounded smoke output
(`steps=2`, threshold `-100`, and a `4096`-point cap) is intentionally not a
quality reference; its fragmented preview is an expected consequence of those
diagnostic settings and must not be used to diagnose the model or texture path.

**Status:** current production geometry path passed. The default atlas is kept
at `512` as a practical quality/speed compromise; `256` is available for the
fastest textured iteration, `1024` for more detail, and `4096` remains the
explicit upstream-aligned quality setting. Full Python numerical parity remains
a separate, explicitly unclaimed task.

## 8. Upstream inference parameters and the C++ compatibility boundary

The official `TencentARC/Pixal3D` repository exposes two different contracts:

- the `paper` branch is the original Direct3D-S2-based implementation used for
  the paper results; its CLI uses a dense stage (`50` steps), sparse `512`
  (`30` steps), sparse `1024` (`15` steps), fixed camera FOV `0.2`,
  `mesh_scale=0.9`, and dense/MC thresholds `0.1`/`0.2`;
- the default branch is the later TRELLIS.2-based release used by the current
  `ref/Pixal3D` checkout. Its image inference uses 12 steps for all three
  samplers, `max_num_tokens=49152`, `decimation_target=1000000`, and
  `texture_size=4096`. The released condition inputs are `512/512/1024/1024`
  pixels with NAF targets `0/512/512/1024` for `ss/shape_512/shape_1024/tex_1024`.
  Standard inference defaults to `1536`; the low-VRAM path uses `1024`.

The C++ model packs and four-stage cascade are ported from the latter contract,
not from the paper branch. Applying the paper-branch step schedule or fixed
camera to this binary would mix incompatible model/checkpoint contracts rather
than reproduce the paper. The C++ pipeline currently supports the upstream
low-VRAM-equivalent `1024` final cascade only; adding `1536` would require the
corresponding model stages and a separate compatibility task.

The C++ sampler, guidance, normalization, token limit, decimation target, and
condition resolutions already match the TRELLIS.2-based contract. The remaining parameter difference is intentional: the upstream inference
script uses `4096`, but the C++ public exporter and CLI commands default to
`512` because the current mesh postprocess and texture bake are CPU-heavy.
`256` remains available for the fastest textured iteration; `1024` and `4096`
remain available explicitly for higher-detail and upstream-aligned quality
validation. A local full run with `texture-size=4096` took about `3069` seconds; the phase
log is cumulative, with approximately `851` seconds spent in QEM decimation
and `186` seconds in UV bake/rasterization/inpaint. The 4096 atlas itself is
not the source of the entire wall time.

A fixed decoded cascade dump was exported at `texture-size=256` and `512` and
rendered with Blender. Both sizes retained the major material regions without
the broad metallic-looking collapse seen in the 64 preview; 512 preserved a
little more local surface detail while remaining visually close to 256. The
same mesh was produced in both exports (`Fo=981800`), and the mean baked
base-color values stayed nearly the same. Therefore, the brighter 4096 preview
is not evidence that higher resolution changes the whole material to white; it
exposes more local variation and PBR highlights. The low-resolution result is a
different sampling/averaging trade-off, not a more correct material.

**Prevention rule:** use `512` for normal development, `256` for the fastest
textured iteration, `1024` when more texture detail is needed, and explicit
`4096` for upstream-aligned quality runs. Never use `texture-size=64` as a
quality claim; it is only a fast workflow smoke-test setting.

## 9. GPU-first SLat decoder rollout and C2S boundary

### Scope

On September 11, 2026, the level-local GPU-first decoder path was committed as
`d355be9` (`perf: keep SLat decoder levels GPU-first`). It uses the existing
GGML scheduler and CUDA operations for the fixed-topology level graph, while
keeping coordinates and topology-changing metadata on the host.

The current working-tree patch extends that path to the channel-to-spatial
(C2S) transition. The C2S feature arithmetic (LayerNorm, SiLU, both sparse
convolutions, channel regrouping, skip-channel repeat, and residual add) is
represented by one scheduler graph. CPU code still builds the child-selection
index and output coordinates. To make the topology decision observable and
conservative, `to_subdiv` is currently evaluated with the CPU
`sparse_linear()` reference, but it consumes the hidden features produced by
the preceding GPU graph; this is not equivalent to running the whole prefix on
CPU.

### Regressions found and fixed during implementation

1. The first C2S attempt passed `hidden` as both the const input and output of
the new helper. The helper cleared `output` at entry, which also cleared the
input object through the alias and produced `input_channels=0`. The call sites
now write to a temporary `SparseTensorF32` and move it back into `hidden` only
after the graph succeeds. This aliasing rule is part of the GPU-helper
contract.
2. The original C2S graph used `ggml_concat(ctx, input, zero_column, 1)` to
append the sentinel row. On CUDA, that maps the row count directly to
`grid.y`; a real input with more than 65,535 rows failed with
`CUDA error: invalid configuration argument`. The graph now uses a repeated
zero tensor followed by `ggml_set()`, which preserves the sentinel semantics
without the `grid.y` limit.

### Real-input evidence

The real input was
`/tmp/pixal3d-old-pre-f37-0910/shape_slat_high.bin`, decoded with
`pixal3d-shared-f16.gguf` in the CUDA Docker image (`CUDA 12.6.3`, ggml
`0.9.9`, RTX 4090 D). The synthetic CUDA fixture still passes:

```text
SLat GPU-first fixture: PASS
subdivision_max_abs=9.54932e-06
subdivision_max_rel=0.00611262
max_abs=3.11807e-05
max_rel=0.0157362
mean_abs=7.29475e-06
```

The real probe completed successfully in both GPU-first modes:

| Run | Decoder-reported time | Probe wall time | Result directory |
| --- | ---: | ---: | --- |
| GPU-first, GPU `to_subdiv` | 45.2109 s | 83.40 s | `/tmp/pixal3d-gpu-first-real-0911` |
| GPU-first, CPU `to_subdiv` | 43.0445 s | 81.36 s | `/tmp/pixal3d-gpu-first-real-cpu-topology-0911` |
| Forced CPU reference | not emitted | 1822.58 s | `/tmp/pixal3d-cpu-reference-0911` |
| Previous hybrid GPU path | 596.873 s | not recorded in the same probe | `/tmp/pixal3d-slat-full-cache-IfQAGY` |

The CPU probe's wall time includes dump writing and mesh extraction, so it is
not a decoder-only timing. The GPU rows use the decoder timing emitted by the
model plus the complete dump-probe wall time; they are retained as evidence of
the scale of the performance difference, not as a like-for-like microbenchmark.

The active-child topology is not exact on the real input. The observed point
counts are:

| Path | level 0 | level 1 | level 2 | level 3 | final output |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous hybrid GPU path | 9,857 | 46,537 | 206,980 | 909,164 | 4,005,638 |
| GPU-first, GPU `to_subdiv` | 9,857 | 46,537 | 206,980 | 909,168 | 4,005,677 |
| GPU-first, CPU `to_subdiv` | 9,857 | 46,537 | 206,980 | 909,164 | 4,005,584 |
| Forced CPU reference | 9,857 | 46,537 | 206,981 | 909,172 | 4,005,637 |

For the current GPU-first/CPU-`to_subdiv` run compared with the forced CPU
reference, coordinate-aligned feature diagnostics were:

| Tensor | Common coordinates | CPU-only | GPU-only | Sign differences | Mean absolute error | Max absolute error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `subdiv_0` | 9,857 | 0 | 0 | 0 | 0.00845 | 0.10417 |
| `subdiv_1` | 46,537 | 0 | 0 | 1 | 0.00935 | 0.07845 |
| `subdiv_2` | 206,980 | 1 | 0 | 88 | 0.00531 | 13.42216 |
| `subdiv_3` | 909,124 | 48 | 40 | 1,624 | 0.00515 | 14.62862 |
| `output` | 4,004,739 | 898 | 845 | 17,763 | 0.00876 | 80.01348 |

Relative error is not used as a gate in this table: it is dominated by values
near zero and by rows whose downstream topology has already diverged. The
`subdiv_1` sign difference is enough to alter one child coordinate at the next
level; after that point, row-wise comparison without coordinate alignment is
misleading.

### Root cause and interpretation

The CPU reference accumulates each output in the portable loop order
`out -> in`. The GPU path uses ggml CUDA `mul_mat` and sparse gather/matrix
products, whose reduction and operation ordering are different. Both paths
use F32 tensors and the same logical weights, but they need not produce the
same last bits (or even the same few ulps around zero). The subdivision decision
is discontinuous because a child is selected when its logit is `> 0`, so a small
arithmetic difference can change the sparse coordinate set and amplify through
all later levels.

The forced CPU result also differs from the previous hybrid GPU result at
levels 2--4 (for example, 206,981 versus 206,980 points at level 2). That is
consistent with the same CPU/GPU reduction-order sensitivity and means that the
old hybrid output cannot be treated as an exact CPU oracle for topology.

Using CPU `to_subdiv` reduces one source of topology drift, but it cannot restore
exact CPU topology after the hidden feature tensor has already been computed by
the GPU. Therefore the current real-model result demonstrates a large
performance improvement and finite outputs, but **does not establish exact
active-child or coordinate parity**.

### Current status and next step

- The `ggml_concat` large-row CUDA failure is fixed and covered by the real
  smoke run plus the synthetic C2S fixture.
- The current GPU-first path has `cpu_fallback_ops=0` inside the feature graph,
  and no custom CUDA kernel, weight relayout, persistent activation tensor,
  validation redesign, or neighbor-map cache extension was introduced.
- The synthetic fixture verifies exact coordinates for its deterministic small
  pack. It does not prove real-weight topology parity.
- Do not describe the real GPU-first decoder as exact-parity or merge a relaxed
  real-model tolerance solely to make the point counts pass. The next change
  should isolate the topology decision and define an explicit CPU/GPU parity
  policy before formal decoder rollout. If exact topology is required, the
  options are a CPU/reference topology path with a documented performance cost,
  or a numerically matched GPU implementation; neither is completed by this
  patch.

**Status:** performance validation succeeded; real-weight topology parity is
unresolved. The current working-tree C2S extension remains an investigation
patch and has not been committed.

## 10. Decoder appears slower while GPU utilization is lower

### Symptom

A full image-to-3D run could appear to have a slower decoder even though the
CUDA graph was selected, while a live GPU monitor showed a lower or more
intermittent GPU utilization peak. This was a symptom of the pre-`d355be9`
hybrid decoder path, not evidence that the texture atlas setting made the
decoder slower. The numbers below are retained as historical baseline data.

### Evidence

The comparable full runs used the fixed
`weights/Pixal3D/test_inputs/armor_knight.png` input and the CUDA Docker
runtime with an RTX 4090 D:

| Run | Shape decoder | Texture decoder | Sum |
| --- | ---: | ---: | ---: |
| `/tmp/pixal3d-armor-knight-current-quality.log` | 581.635 s | 583.526 s | 1165.161 s |
| `/tmp/pixal3d-armor-knight-upstream-aligned-4096.log` | 579.086 s | 588.874 s | 1167.960 s |
| `/tmp/pixal3d-final-0910/run.log` | 602.989 s | 607.868 s | 1210.857 s |

The latest 4096 run therefore differs from the immediately preceding full run
by only `+2.799 s` for the two decoders, while it is about `42.897 s` faster
than the older comparable run. The output sizes and cascade point counts are
unchanged in the two latest logs. `texture_size` is passed to `uv_bake()` only
after both decoders have completed, so changing the atlas from 64/256/512/1024
to 4096 cannot change decoder arithmetic.

The full decoder profile in
`/tmp/pixal3d-slat-full-cache-IfQAGY/full.log` gives the more important
breakdown:

```text
shape decoder wall time:       596.873 s
GPU sparse-conv compute:         4.29051 s
sparse_linear total:           432.953 s
SparseTensorF32::valid():       74.3051 s
sparse_layer_norm:              24.8565 s
sparse_channel_to_spatial:      15.2814 s
finish_sparse:                  18.3754 s
output_validation:              11.8979 s
```

The measured GPU graph compute occupies only about `0.72%` of that decoder
wall time. The dominant work is still CPU-side sparse linear, validation, and
sparse layout handling. The detailed linear profile also reports
`omp_max_threads=32` and `omp_actual_threads=32`; the separate
`cpu_threads=4` backend log is the ggml CPU backend threadpool setting and does
not mean that the OpenMP sparse-linear hot loop is limited to four threads.

The neighbor-map cache is not the cause of a slowdown. In the same profiling
setup, cache enabled versus disabled measured:

```text
cache enabled:  596.873 s, neighbor_map_builds=5,  cache_hits=35
cache disabled: 635.113 s, neighbor_map_builds=40, cache_hits=0
```

### Root cause

At that point, the SLat decoder was only partially GPU-resident. Sparse
convolution graph nodes ran on CUDA, but the surrounding feature operations
returned to host-side C++ vectors. Each decoder block therefore alternated
between CPU work and short GPU bursts. A GPU monitor could show low average
utilization or low peaks while the end-to-end decoder remained slow; the
observed wall time was real, but it was CPU-bound rather than GPU-compute-bound.
The later GPU-first rollout is documented in section 9, with its separate
real-weight topology-parity limitation.

The normal backend `memory=` log reports free VRAM/total VRAM, not GPU
utilization. It must not be interpreted as a compute-usage or throughput
measurement. The normal image-run logs also do not record a reliable
utilization peak; point-in-time `nvidia-smi` samples can miss short sparse
kernels.

### Prevention and next diagnostic

For decoder comparisons:

- compare the explicit `shape_decoder` and `texture_decoder` stage timings,
  not only the total run time, because QEM and UV bake are cumulative export
  costs;
- disable `PIXAL3D_SLAT_DECODER_PROFILE`, `PIXAL3D_SLAT_DECODER_TRACE`, and
  `PIXAL3D_VALIDATE_SPARSE_MAP_CACHE` for production timing;
- use Nsight or a continuous `nvidia-smi dmon` sample if GPU utilization is
  needed; do not infer it from the backend free-memory snapshot;
- prioritize the already confirmed CPU `sparse_linear` path before judging
  CUDA kernel peak utilization.

**Status:** decoder regression not reproduced. The lower GPU utilization is
consistent with the known CPU-dominated partial GPU decoder path.
