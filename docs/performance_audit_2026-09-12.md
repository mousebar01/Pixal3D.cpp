# Performance audit

- **Updated:** 2026-09-12
- **Scope:** projection, inference runtime setup, CLI process boundaries and
  mesh export
- **Status:** one measured hot loop fixed; remaining costs are recorded for
  targeted follow-up

## Measured results

### Projection

A deterministic microbenchmark with 50,000 points and 1,024 channels changed
from `1.90023 s` to `0.32508 s` after caching camera/scale terms and moving
bilinear coordinates and weights out of the channel loop. Dense and sparse
projection results have a regression test. This is a local microbenchmark, not
an end-to-end speedup claim.

### Mesh export

Historical real-image logs contain:

```text
export qem decimate: 898.323 s
export uv bake:      1037.96 s
```

The QEM line was an old generic phase label and did not record its effective
backend. It is not evidence that CUDA QEM took 898.323 seconds; it may reflect
CPU QEM or a failed GPU attempt followed by CPU fallback. The current wrapper
records `qem backend=gpu`, `qem backend=cpu fallback=true reason=...`, or
`qem backend=none reason=already_at_target`.

The current CUDA QEM path is experimental. On a regular torus fixture it ran in
about `0.22 s` versus `0.27 s` for CPU at 8,192 input faces. A larger prior
probe measured roughly `0.33 s` GPU versus `15.8 s` CPU. These are QEM-only
measurements and do not certify decoder-mesh topology or visual parity.

xatlas charting, packing and UV baking remain CPU work. A profiled replay
measured approximately `77.8 s` in chart packing and `101.0 s` for the complete
512-pixel UV bake. CUDA model execution does not move these phases to the GPU.

## Confirmed follow-up costs

- DINO and NAF still rebuild graph/scheduler state on repeated calls instead of
  reusing a runtime keyed by stage, shape and backend policy.
- Each `run-image` CLI invocation reloads its model components. The local demo
  starts one child process per job and therefore does not provide model reuse.
- The dense multi-view projection helper repeatedly allocates a temporary
  buffer. It is not the dominant path for the current sparse image condition.
- QEM GPU output is not bit-exact with CPU and has not passed same-input parity
  on highly non-manifold decoder meshes. Keep the CPU fallback and compare
  topology/visual output before changing the production policy.

## Profiler boundary

Profiling is opt-in through `PIXAL3D_PROFILE=off|summary|trace`; the default is
off. Disabled profiling does not read the clock, synchronize the GPU, or copy
activations merely to produce diagnostics. Required error/fallback and backend
policy logs are separate from optional timing output.

## Validation

```sh
cmake --build build-backend-test --parallel 4
(cd build-backend-test && ctest --output-on-failure)
cmake --build build-cuda-docker --parallel 4 \
  --target pixal3d_core pixal3d_mesh_qem_cuda_test
./build-cuda-docker/bin/pixal3d_mesh_qem_cuda_test

git diff --check
```

The CPU suite passed `24/24`; the CUDA QEM fixture passed. Full real-weight
inference/export benchmarking was not repeated in this cleanup.

## Measurement rules

1. Record input size, build type, policy and effective backend with every timing.
2. Separate exclusive phase time from nested/cumulative stage time.
3. Treat QEM, xatlas and UV export as separate phases; do not infer GPU use from
   a CUDA-enabled build alone.
4. Keep weights, meshes and raw logs outside `docs/` and version control.
