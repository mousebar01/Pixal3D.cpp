# xatlas packing optimization

- **Date:** 2026-09-12
- **Status:** implemented in the outer `src/mesh_postprocess.cpp`; not a CUDA
  xatlas implementation
- **Scope:** xatlas submission and packing only. QEM, decoder execution,
  validation, neighbor-map cache, and `ref/` are unchanged.

## Symptom and reproduction

The real QEM replay showed that UV baking was still a large postprocess cost.
The measurements below use one fixed local replay mesh kept outside the
repository:

```text
vertices: 225,509
triangles: 997,620
preclusters: 570,442
```

The previous per-precluster path submitted 570,442 separate xatlas meshes.
For the stock xatlas pack options (`rotateCharts=true`), the final replay was:

```text
xatlas AddMesh: 42.553 s
xatlas ComputeCharts wall phase: 3.802 s
xatlas PackCharts: 90.569 s
atlas reindex + rasterize + inpaint: 3.922 s
complete uv_bake: 153.234 s
```

The largest cost is still CPU xatlas chart packing. CuMesh does not provide a
CUDA implementation of xatlas parameterization or packing; its GPU clustering
is therefore not a direct fix for this 90-second phase. The existing outer
precluster is already a CPU implementation of the same broad mutual-argmin
chart-merging family, so moving only that step to CUDA would not remove the
main hotspot.

## Implemented changes

### 1. Material-isolated MeshDecl batching

For large precluster sets (`>= 4096` clusters), `uv_bake()` now builds one
xatlas `MeshDecl` instead of one declaration per precluster. Each precluster
gets a distinct `faceMaterialData` value. xatlas explicitly treats a material
change as a chart-group boundary, so preclusters remain isolated for chart
growth while the per-`AddMesh` allocation/progress/task overhead is removed.

The batch uses a compact original-vertex-to-batch-vertex map. Output positions
are still restored through `l2orig`; no mesh coordinate is translated or
modified.

This is an xatlas API use, not a claim that xatlas is GPU accelerated.

### 2. Disable packing rotation search by default

`rotateCharts` is now false by default. xatlas's stock behavior can be restored
for A/B comparison with:

```sh
PIXAL3D_XATLAS_ROTATE_CHARTS=1
```

The option is intentionally limited to rotation search. `bilinear`,
`blockAlign`, `bruteForce`, `rotateChartsToAxis`, `resolution`, and
`maxChartSize` retain their existing defaults unless explicitly overridden by
the diagnostic environment variables.

### 3. Explicit rollback and diagnostics

```sh
# Restore the old per-precluster submission path.
PIXAL3D_XATLAS_BATCH_MATERIAL=0

# Restore stock xatlas rotation search.
PIXAL3D_XATLAS_ROTATE_CHARTS=1

# Print selected options and resulting atlas metadata.
PIXAL3D_XATLAS_LOG_OPTIONS=1
```

Small meshes below the 4096-cluster threshold retain the old per-cluster path
unless `PIXAL3D_XATLAS_BATCH_MATERIAL=1` is explicitly set.

The earlier translation-batching and tiny-cluster bypass probes are not part of
the implementation. Translation changed the atlas density/layout, while the
bypass path could overflow its fallback strip and cause texel sharing.

## Performance evidence

Same-input full replay after the change:

| mode | AddMesh | PackCharts | complete `uv_bake` | xatlas atlas | charts | output V/F |
|---|---:|---:|---:|---|---:|---|
| old path: material batch off, rotation on | 42.553 s | 90.569 s | 153.234 s | 2358x2360 | 654,133 | 2,295,849 / 997,620 |
| material batch on, rotation on | 0.580 s | 90.159 s | 112.547 s | 2360x2359 | 654,302 | 2,296,149 / 997,620 |
| old path, rotation off | 42.319 s | 73.652 s | 136.285 s | 2359x2358 | 654,133 | 2,295,849 / 997,620 |
| **default after change** | **0.503 s** | **77.808 s** | **101.035 s** | **2359x2358** | **654,302** | **2,296,149 / 997,620** |

The default replay is about 52.2 seconds faster than the old path, a reduction
of roughly 34%. The exact wall time varies with CPU scheduling, but the
large-mesh direction was reproduced in both the full replay and a 100k-face
fixture:

```text
100k old path (batch=0, rotate=1): 16.080 s
100k default fast path:             10.275 s
```

The full replay's texture density remains effectively unchanged:

```text
old tpu: 505.449219
new tpu: 505.449158
```

The atlas dimensions differ by only one or two texels because chart placement
order and the rotation policy are different; final GLB UVs are still normalized
by the resulting atlas dimensions.

## Correctness checks

A deterministic 100k-face real-mesh fixture ran the old path and all three
candidate variants. For the default fast path compared with the old path:

```text
mapped source faces: 100,000 / 100,000
UV NaN/Inf:          0
UV out of [0,1]:     0
output faces:        100,000 / 100,000
output T:            512 / 512
base MAE:            4.789580 byte values
base max error:      116.515244 byte values
metal/rough MAE:     0.858103 byte values
metal/rough max:     9.631104 byte values
```

The comparison bakes a deterministic position-derived PBR field and samples
both output textures at the same barycentric points of every source face. It
is a texture-sampling/structural check, not a claim of bit-exact UV equality:
packing is allowed to choose a different valid layout, so isolated texels can
show a larger error even though the mean error is small. No source face was
lost in the comparison; duplicate geometric faces were matched by occurrence.

The full replay also preserved the triangle count and the same xatlas density
within float rounding. The material batch adds no coordinate offset and uses
the existing `l2orig` restoration path.

## Why this is the low-risk CuMesh-informed choice

CuMesh's useful lesson is to separate highly parallel chart/topology work from
the mature CPU xatlas stages. In this repository:

- precluster is already a CuMesh-like CPU mutual-argmin pass;
- GPUizing it has an upper-bound benefit of about 11 seconds on this replay;
- xatlas `ComputeCharts` and especially `PackCharts` remain CPU-only;
- unconditionally merging meshes without a boundary marker is unsafe because
  xatlas can form cross-cluster colocals;
- material boundaries are a documented xatlas mechanism for preventing chart
  growth across those boundaries, and the real replay retained the chart count
  and density at the measured level.

Therefore the current optimization batches only through the public material
boundary mechanism. It does not modify vendored `third_party/xatlas`, does not
introduce a custom CUDA kernel, and does not call the path GPU-accelerated.

## Validation

The outer CUDA build compiled the affected xatlas/export code, and the
deterministic texture-export test passed. The same-input replay used for the
table above preserved source-face coverage, finite UVs and the measured texture
density. The optimization remains a CPU xatlas change; it does not add a CUDA
xatlas backend.

Raw meshes, weights, replay runners and logs remain outside the repository.
