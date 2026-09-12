# Multi-backend integration

- **Updated:** 2026-09-12
- **Scope:** outer CMake options, ggml backend registry, device inventory and
  execution policy
- **Status:** CPU baseline is validated; additional backend runtime parity is
  not certified

## Current contract

- CPU is always built and is the final fallback.
- Backend discovery comes only from the ggml registry
  (`ggml_backend_dev_count()` / `ggml_backend_dev_get()`); the outer project
  does not maintain a second CUDA/Vulkan device list.
- The policy accepted by the outer API is `auto`, `cpu`, `gpu` or
  `gpu:<index>`. GPU indices are assigned to registry devices of type GPU or
  IGPU; CPU and ACCEL do not consume a GPU index.
- `auto` may fall back after logging the reason. A forced `gpu:<index>` must
  fail if the selected device or a required operation is unavailable.
- Production graph execution uses `ggml_backend_sched`; new model code must not
  add direct `ggml_backend_graph_compute()` calls.
- Device listing reports availability and memory information. It does not prove
  that a stage supports every operation or that a model has passed parity.

## Build/runtime status

| Backend | Current evidence | Not yet proven |
| --- | --- | --- |
| CPU | Portable Release build, 24/24 CTest, CLI smoke tests | — |
| CUDA | CUDA build and QEM CUDA fixture run in the local NVIDIA container | Full model-stage placement, transfer accounting and CPU/CUDA parity |
| Vulkan | Outer option is exposed | Loader/ICD, graph execution and parity |
| Metal | Outer option is exposed on Apple builds | Apple hardware runtime and parity |
| HIP/SYCL | Outer options are exposed | Matching device/toolchain runtime and parity |

An enabled CMake option means that the backend can be requested from the build;
it is not a support claim. A stage is supported only after its metadata, tensor
layout, loader, forward pass and reference comparison are complete.

## CLI check

```sh
./build/bin/pixal3d --list-devices
```

The output is the same registry inventory used by backend selection. Typical
CPU-only output contains one `type=CPU` entry. The command does not initialize
model weights and should not allocate persistent GPU model memory.

For an explicit policy, use the environment variable consumed by the selected
stage, or `PIXAL3D_BACKEND` as the common fallback:

```sh
PIXAL3D_BACKEND=cpu  ./build/bin/pixal3d run-cascade ...
PIXAL3D_BACKEND=auto  ./build/bin/pixal3d run-cascade ...
PIXAL3D_BACKEND=gpu:0 ./build/bin/pixal3d run-cascade ...
```

## Validation completed on 2026-09-12

CPU baseline:

```sh
cmake --build build-backend-test --parallel 4
(cd build-backend-test && ctest --output-on-failure)
./build-backend-test/bin/pixal3d --help
./build-backend-test/bin/pixal3d --version
./build-backend-test/bin/pixal3d --list-devices
```

Result: build succeeded, 24/24 CTest tests passed, and all three CLI smoke
commands succeeded.

CUDA postprocess check:

```sh
cmake --build build-cuda-docker --parallel 4 \
  --target pixal3d_core pixal3d_mesh_qem_cuda_test
./build-cuda-docker/bin/pixal3d_mesh_qem_cuda_test
```

Result: CUDA QEM fixture passed. This validates the isolated QEM kernel path,
not the complete image-to-3D pipeline.

## Open work

1. Add stage-specific capability and placement fixtures for every eligible
   operation, including activation/residual/normalization paths.
2. Count H2D/D2H transfers and bytes at stage boundaries and reject an
   unapproved CPU fallback under forced GPU policy.
3. Pass the shared backend policy/device into custom postprocess code; the
   experimental QEM wrapper currently has its own CUDA runtime probe.
4. Validate each enabled backend on matching hardware. A compile-only build or
   software Vulkan pass is not a vendor-device parity result.
