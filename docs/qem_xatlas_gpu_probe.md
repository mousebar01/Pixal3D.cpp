# QEM and xatlas backend boundary

- **Updated:** 2026-09-12
- **Status:** CUDA QEM is experimental; xatlas remains CPU-only

## Effective backend

`decimate_qem()` attempts the custom CUDA QEM implementation only in a CUDA
build. It preserves the CPU implementation as an explicit fallback. Each
production call reports one of these states:

```text
pixal3d: qem backend=gpu ...
pixal3d: qem backend=cpu fallback=true ... reason=...
pixal3d: qem backend=none ... reason=already_at_target
```

The generic export phase label `export qem decimate` is not sufficient to tell
which path ran. In particular, the historical `898.323 s` phase timing cannot
be used as a CUDA-QEM benchmark because its backend was not recorded.

## Measurements

The deterministic torus fixture uses the same CPU and GPU entry points:

| Input | CPU | CUDA QEM | Result |
| --- | ---: | ---: | --- |
| 4,096 V / 8,192 F → 2,048 F | 0.27 s | 0.22 s | finite, valid, watertight fixture passed |
| 250,000 V / 500,000 F → 100,000 F | 15.8 s | about 0.33 s | prior probe; non-bit-exact output |

The GPU result may choose different equal-cost collapses. The fixture checks
finite values, valid indices, face budget, topology counters and a shape
envelope; it does not prove parity for real non-manifold decoder meshes.

A real-mesh replay showed the experimental CUDA call completing in about 1.7 s,
but the output still contained many open/non-manifold edges. That result is not
a production quality gate and must not be compared with a different historical
CPU input.

## xatlas and UV bake

The vendored xatlas implementation has no CUDA/HIP/Vulkan execution path.
Charting, packing, atlas rasterization and inpaint are host-side work. A CUDA
build of the model backend does not change that. A profiled replay measured
roughly 77.8 s for packing and 101.0 s for the complete 512-pixel UV bake.

## Production decision

Keep QEM GPU-first in a CUDA build only as an experimental path with explicit
fallback diagnostics. Before treating it as a validated replacement, run CPU
and GPU on the same preprocessed real mesh and compare:

- output face/vertex counts and valid indices;
- open and non-manifold edge counts;
- finite values and bounds;
- UV coverage, atlas dimensions and texture sampling;
- a visual render or equivalent golden output.

The custom QEM probe also uses an independent CUDA runtime device check. It
still needs to consume the shared backend policy/device before it can be
considered fully integrated with multi-backend execution.

## Reproduction

Build the fixture in the CUDA development container and run it in a matching
runtime container. The binary under `build-cuda-docker/` was built against the
container's newer glibc/libstdc++; running it directly on this host fails before
CUDA initialization with missing `GLIBCXX_3.4.29`/`GLIBC_2.34` symbols. That is a
runtime-image mismatch, not evidence that the QEM kernel is unavailable.

```sh
cmake --build build-cuda-docker --parallel 4 \
  --target pixal3d_core pixal3d_mesh_qem_cuda_test

docker run --rm --gpus all \
  -v /home/sy/Pixal3D.cpp:/workspace \
  -w /workspace \
  pixal3d:cuda-host-opt \
  ./build-cuda-docker/bin/pixal3d_mesh_qem_cuda_test
```

If the CUDA image is named differently, use the image that was used for the
build; do not copy its binary into an older host runtime. The fixture is
intentionally separate from the full image-to-3D pipeline so that QEM
performance and correctness can be tested without loading model weights.
