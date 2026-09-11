# Pixal3D.cpp Troubleshooting History

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
