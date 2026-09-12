# QEM / xatlas GPU probe

- **Recorded:** 2026-09-11
- **Status:** GPU QEM runs as an experimental CUDA path; xatlas remains CPU-only.
- **Scope:** Docker CUDA probe and real-mesh replay only. No decoder, sparse op,
  neighbor-map cache, `ref/`, or model weights were changed.

## Environment and commands

The probe ran in the `pixal3d-dev:cuda` image with CUDA 12.6.85 on an NVIDIA
GeForce RTX 4090 D. The CUDA build used:

```sh
cmake -S . -B build-cuda-docker -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPIXAL3D_ENABLE_CUDA=ON \
  -DPIXAL3D_CUDA_ARCHITECTURES=89 \
  -DPIXAL3D_GGML_NATIVE=OFF \
  -DBUILD_TESTING=ON
```

The original outer project had a CUDA ggml model path, but `src/mesh_qem_decimate.cpp`
was still CPU QEM and the vendored `third_party/xatlas/xatlas.cpp` had no CUDA or
HIP implementation. The new QEM probe is based on the CUDA implementation in
`ref/trellis.cpp/src/decimate_qem.cu`; the reference directory itself was not
modified.

## CUDA QEM results

The independent fixture compares the existing serial CPU implementation with the
CUDA implementation. GPU edge ownership uses atomic minima, so the collapse order
is not guaranteed to match the serial CPU traversal exactly.

| fixture | CPU | GPU | speedup | topology / parity result |
| --- | ---: | ---: | ---: | --- |
| torus, 4,096 V / 8,192 F, target 2,048 F | 0.277 s | 0.236 s | 1.18x | CPU 2,032 F, GPU 1,982 F; both finite, no open/non-manifold edges; envelope within 1% |
| torus, 250,000 V / 500,000 F, target 100,000 F | 15.75–15.80 s | 0.331–0.333 s | 47.4–47.6x | three repeats: GPU 92,630–92,782 F; face delta 0.146–0.310%; open edges 0; non-manifold edges 0–1 |

These results prove that the CUDA QEM kernels can execute on the available GPU.
They do **not** prove bit-exact CPU parity or production-quality parity for every
input mesh. The first version of the kernel only propagated ownership per face;
repeated runs then produced materially different face counts because two edges
sharing a high-valence vertex could collapse concurrently. A local per-vertex
ownership guard was added after that observation. The repeated numbers in the
table are from the guarded version.

### Correctness status

The CUDA path is currently a non-bit-exact prototype. The small torus fixture
checks finite values, valid indices, face budget, open-edge count, and a 1%
shape envelope. It does not establish equivalence for non-manifold decoder
meshes. The production wrapper keeps the CPU implementation as an explicit
fallback; a same-input CPU replay and visual comparison are still required
before treating the GPU output as a fully validated replacement.

### Real mesh replay

The replay used the existing local artifact:

```text
/tmp/pixal3d-gpu-first-real-0911/mesh.obj
```

Its raw size was 4,005,677 vertices and 9,081,864 triangles. The production
pre-QEM cleanup was run first:

```text
weld + clean + drop_small_components + Taubin smooth: 55.399 s
pre-QEM mesh: V=3,925,466 F=9,004,618
pre-QEM topology: open edges=520,060, non-manifold edges=1,309,620
```

The CUDA QEM call then took 1.696 s and returned:

```text
V=216,630 F=969,084
open edges=49,330
non-manifold edges=502,898
finite=1
```

This input is not the same decode artifact as the older CPU quality log, so it
must not be used as a direct CPU-vs-GPU quality comparison. It does show an
important limitation: the current GPU path is not yet a sufficient production
correctness gate for highly non-manifold input. A CPU replay on the same
preprocessed mesh and a visual/structural comparison are still required before
making the GPU path the unconditional production default.

The current branch keeps `decimate_qem_cpu()` as the explicit fallback and
`decimate_qem_gpu()` as an independently testable entry point. The CUDA wrapper
currently tries GPU first in CUDA builds, but this should remain treated as an
experimental path until the same-input real-mesh parity gate is complete.

## xatlas result

A source scan of the vendored xatlas implementation found no CUDA, HIP, or
Vulkan backend. Building the outer project with CUDA therefore does not move
xatlas to the GPU. The CUDA build's observed GPU utilization stayed at 0% while
xatlas was running; the CPU process was active instead.

For profiling, xatlas was compiled with its existing `XA_PROFILE=1` support and
verbose printing was enabled only in a temporary replay runner. The replay used
the CUDA-QEM output above and a 512-pixel bake atlas.

| phase | time |
| --- | ---: |
| outer `uv_bake` total | 143.853 s |
| `AddMesh` total (real) | 41.22 s |
| `ComputeCharts` total (real) | 3.34 s |
| `ComputeCharts` total (thread/cumulative) | 98.64 s |
| chart-group compute | 96.39 s |
| create chart-group mesh | 34.15 s |
| create chart mesh + parameterize | 41.45 s |
| `PackCharts` total | 40.46 s |
| pack `Find location` | 36.31 s |
| build output meshes | 0.98 s |

The replay added **551,914** cluster meshes to xatlas and xatlas produced
**609,968** charts. Most added meshes were one or two triangles. This is the
main actionable signal for the next xatlas optimization: reduce or batch the
large number of tiny cluster mesh declarations without changing the baked
coordinate contract. It is not evidence that a CUDA flag is missing.

The resulting bake was structurally complete:

```text
xatlas atlas: 2,276 x 2,276
final atlas: 512 x 512
output: V=2,174,059 F=969,084
uncharted faces kept with point-collapsed UVs: 198
```

## Timing instrumentation

`src/mesh_postprocess.cpp` already reports exclusive and cumulative `uv_bake`
phases. `src/texture_export.cpp` now reports the same distinction for:

```text
weld+clean+smooth
qem decimate
uv bake
encode_png
build_glb
write_glb
```

This prevents nested/cumulative export timers from making QEM or xatlas appear
to cost more than they actually do.

## Conclusions and next safe steps

1. **QEM:** GPU execution is real and can be dramatically faster on regular
   meshes. The CUDA path is not yet a drop-in numerical/topological replacement;
   keep CPU fallback and complete same-input CPU/GPU parity on the real mesh.
2. **xatlas:** no existing GPU path is available to enable. Do not claim xatlas
   is GPU-accelerated merely because the process was built with CUDA.
3. **Low-risk xatlas work:** prototype batching/combining disconnected tiny
   cluster declarations while preserving cluster isolation, then compare UV
   coverage, chart count, atlas utilization, output face count, and a golden
   render. Do not change chart semantics and batching in the same patch.
4. **Backend policy:** before making the experimental QEM wrapper a permanent
   production default, pass the selected GPU device/policy explicitly instead
   of using an independent `cudaGetDeviceCount()` probe that implicitly uses
   device 0.
5. **Replay:** keep using the local raw mesh replay to avoid repeating the full
   image-to-3D inference run. Logs, raw meshes, weights, and build directories
   remain outside version control.
