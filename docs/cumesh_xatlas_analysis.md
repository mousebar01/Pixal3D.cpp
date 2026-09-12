# CuMesh atlas / xatlas 源码调查

- **记录日期：** 2026-09-11
- **状态：** 调查完成，尚未把 CuMesh clustering 接入正式流水线
- **范围：** 官方 CuMesh `main` 源码、当前外层 `precluster -> xatlas` 调用链，以及临时 xatlas batching fixture
- **工作区约束：** 本轮没有修改 `ref/`、QEM、decoder、neighbor-map 或正式 xatlas 路径；已有未提交改动保持不变

## 1. 调查对象

官方仓库：[`JeffreyXiang/CuMesh`](https://github.com/JeffreyXiang/CuMesh)

本轮通过 GitHub API 读取了这些文件：

- `src/atlas.cu`
- `src/connectivity.cu`
- `src/geometry.cu`
- `src/cumesh.cu` / `src/cumesh.h`
- `cumesh/cumesh.py`
- `cumesh/xatlas.py`
- `third_party/xatlas/xatlas.cpp` / `xatlas.h`

CuMesh README 的表述与源码一致：它是 **GPU mesh clustering + CPU xatlas**，不是 CUDA 版完整 xatlas。其 `third_party/xatlas` 仍然是 CPU C++ 实现。

## 2. CuMesh 的真实调用链

```text
Python CuMesh.uv_unwrap()
  -> remove_degenerate_faces()                 GPU
  -> CuMesh.compute_charts()                   GPU clustering / topology
  -> read_atlas_charts()                       GPU chart labels / maps
  -> chart tensors .cpu()                      D2H
  -> Atlas.add_mesh(chart_i)                   CPU xatlas, 每个 chart 一次
  -> Atlas.compute_charts()                    CPU xatlas parameterization
  -> Atlas.pack_charts()                      CPU xatlas packing
  -> Atlas.get_mesh(i)                         CPU 输出
```

对应源码位置：

- `cumesh/cumesh.py:408-480`：`uv_unwrap()` 在 GPU clustering 后把 chart 数据转到 CPU，并逐个 `xatlas.add_mesh()`。
- `cumesh/xatlas.py:11-53`：`Atlas.add_mesh()` 明确要求 CPU contiguous F32 / INT32 tensors。
- `cumesh/xatlas.py:55-142`：chart computation 和 packing 都通过 CPU xatlas wrapper。

因此，**单纯把 CuMesh 的 Python/CUDA extension 引入当前工程，不会让 xatlas 的 parameterization 或 packing 上 GPU。**

## 3. CuMesh GPU clustering 做了什么

### 3.1 几何量

`src/geometry.cu` 为每个 face 启动一个 CUDA thread，计算：

- face area
- normalized face normal

这部分适合直接并行，且可以在同一 GPU buffer 上供后续 chart 计算使用。

### 3.2 拓扑 / adjacency

`src/connectivity.cu` 的主路线是：

```text
face indices
  -> 每个 face 展开 3 个无向 uint64 edge key
  -> CUB radix sort
  -> run-length encode 得到 unique edges / edge multiplicity
  -> CSR 形式的 edge -> face adjacency
  -> 只选择 multiplicity == 2 的 manifold face pairs
```

关键实现：

- `get_edges()`：`expand_edges_kernel` + CUB `DeviceRadixSort::SortKeys` + `DeviceRunLengthEncode::Encode`
- `get_edge_face_adjacency()`：构造 `edge2face` 和 `face2edge`
- `get_manifold_face_adjacency()`：CUB `DeviceSelect::If` 过滤 manifold edges

这比当前 `src/mesh_postprocess.cpp` 中每轮使用 CPU `unordered_map` 构造 face/edge 关系更适合移到 CUDA。

### 3.3 chart collapse

`src/atlas.cu` 的 `compute_charts()` 逻辑是：

1. 每个 face 初始化为一个 chart。
2. 从 manifold face adjacency 生成 chart-pair key，并通过 CUB sort/reduce 汇总共享边长。
3. 用 segmented reduction 计算每个 chart 的：
   - 面积
   - 法线轴
   - normal cone half-angle
4. 对每条相邻 chart edge 计算：

```text
cost = merged_cone_half_angle
     + area_penalty_weight * merged_area
     + perimeter_area_ratio_weight * merged_perimeter^2 / merged_area
```

5. 每个 chart 选成本最小的 incident edge。
6. 只有同时成为两端 argmin 的 edge 才 collapse（mutual argmin）。
7. 压缩 chart labels，重复到局部不再能合并。

对应的 CUDA kernel 包括：

- `init_chart_adj_kernel`
- `compute_chart_adjacency_cost_kernel`
- `propagate_cost_kernel`
- `collapse_edges_kernel`

### 3.4 refinement 和 chart mesh 输出

CuMesh 还有两个容易被误认为“只是小优化”的阶段：

- `refine_charts_kernel`：每个 face 根据 chart normal similarity + shared edge length 重新选 chart。
- `reassign_chart_ids()`：refine 后用 GPU DSU 把重新变成不连通的 chart 拆开。

最后 `construct_chart_mesh()` 使用：

```text
sort chart ids
  -> run-length encode / exclusive scan
  -> pack (chart_id, original_vertex_id) 到 uint64
  -> sort/unique
  -> 输出 chart vertex map / chart faces / offsets
```

这部分可以作为当前 `build_cluster_meshes()` 的 CUDA 参考，但它不是当前主瓶颈。

## 4. 与当前实现的逐项对照

当前 `src/mesh_postprocess.cpp:613-753` 已经是一个 **CPU 版的 bottom-up mutual-argmin chart merging**：

- face normal / area：已有
- manifold face-pair adjacency：已有
- chart pair shared length：已有
- cone half-angle：已有
- area / perimeter cost：已有
- mutual argmin collapse：已有
- 每轮压缩 chart label：已有

所以当前并不是“完全没有 CuMesh 思路”，而是把同一类算法放在 CPU 上执行。真实 replay 中：

```text
precluster: 10.647 s
```

这说明 CuMesh 风格 GPU clustering 有价值，但它最多首先解决这约 10.6 秒，不能解释或消除 xatlas 的几十秒 AddMesh 和八十多秒 PackCharts。

### 4.1 不能直接照搬的语义差异

当前实现和 CuMesh 并非数学等价，最重要的差异是 perimeter：

- CuMesh 的 `get_chart_connectivity()` 基于 `manifold_face_adj`，`chart_perims` 只累计跨 chart 的 manifold shared edges。
- 当前实现先扫描所有 face edges，再减去 same-chart manifold interior edges，因此会把 open edges / non-manifold edges 也纳入 chart perimeter。

当前真实 QEM 输出包含：

```text
open edges:       49,159
non-manifold edges: 503,949
```

如果直接替换成 CuMesh 的 perimeter 定义，chart merge 结果可能改变，进而改变 UV seam、纹理覆盖和最终渲染。这个差异不能被当作纯性能改动。

另外：

- CuMesh 默认 `refine_iterations=100`、`global_iterations=3`；当前外层 precluster 没有启用 refine。
- CuMesh 在 collapse/refine 过程中多次 `cudaStreamSynchronize()`，并重新申请 CUB / 临时 buffer；它是有效的 GPU 实现，但不是“无同步、无分配”的理想模板。
- 当前实现的 pair 容器先按 chart id 排序，CuMesh 的 argmin tie-break 还显式使用 edge id；相同 cost 时结果也可能不同。

因此，**不能把 `atlas.cu` 直接复制进当前正式路径，然后只看耗时判断成功。**

## 5. 当前真正的热点

最新真实 replay（RTX 4090 D / Docker CUDA build，QEM 后网格约 216k V / 970k F）为：

```text
component triage:       0.284 s
precluster:            10.647 s
build cluster meshes:   0.556 s
xatlas AddMesh:        40.665 s
xatlas ComputeCharts:   3.602 s
xatlas PackCharts:     84.637 s
其他 rasterize/inpaint:约 2.8 s
uv_bake total:        144.199 s
cluster 数量:          552,787
```

因此排序是：

```text
1. PackCharts：由 chart 数量和每个 chart 的 packing 工作驱动
2. AddMesh：大量小 MeshDecl / colocals / task / allocation 管理开销
3. precluster：可以 GPU 化，但不是第一大头
```

`third_party/xatlas/xatlas.cpp` 也直接验证了这一点：

- `pack::Atlas::addCharts()` 对每个 chart 建一个 task；
- `pack::Atlas::packCharts()` 的工作集是全局 chart 数，而不是 AddMesh 调用次数；
- 所以把多个 chart 放进同一个 `MeshDecl`，如果 chart 数不变，PackCharts 主成本不会消失。

## 6. 临时 batching fixture 的实测

为了验证“把多个 chart 合成一个 xatlas MeshDecl”是否能解决问题，使用当前 vendored xatlas 做了临时 fixture，未写入仓库。

### 6.1 空间分离、无坐标重叠的 10,000 个三角形

```text
                         separate       one combined MeshDecl
AddMesh + Join              0.696 s             0.008 s
ComputeCharts               0.153 s             0.525 s
PackCharts                  3.293 s             3.290 s
charts                      10,000             10,000
```

结论：batching 确实可以几乎消除大量小 `AddMesh` 的管理开销，但不能降低 `PackCharts`，因为 chart 数没有变化。

### 6.2 具有大量 coincident positions 的合并

当 1,000 个三角形都占用完全相同的三个坐标、但每个三角形仍使用独立 vertex index 时：

```text
separate total: 1.114 s
combined total: 22.440 s
```

10,000 个完全重叠三角形的 combined probe 在 60 秒以上没有结束，已中止。

原因是 xatlas 在 `AddMeshJoin()` 阶段对一个 `Mesh` 全局建立 colocal vertex 关系；不同 chart 原本分属不同 `AddMesh` 时不会互相参与，合并后会被放进同一个 colocal 域，可能令边界/parameterization 代价爆炸。`faceMaterialData` 只能影响 face grouping，不能作为禁用跨 cluster colocals 的可靠开关。

因此，**不能把当前 552,787 个 cluster 无条件拼成一个大 MeshDecl。**

## 7. 可行优化方向排序

### A. 先做安全 batching probe（推荐下一步）

不是直接把所有 cluster 合并，而是先按以下条件分批：

1. 同一 batch 内不同 cluster 不共享原始 vertex id；
2. 同一 batch 内不同 cluster 的 position 不在 xatlas epsilon 内重合；
3. 对高重叠坐标 / 高 valence cluster 设置 batch 隔离；
4. 给 batch 数量和单 batch vertex/face 数设置上限。

然后只比较：

```text
AddMesh + Join
ComputeCharts
PackCharts
chart count
output face count
atlas width/height/utilization
```

这条路径有希望把 40.665 秒的 AddMesh 管理成本降下来，但即便成功，也不能期待 84.637 秒 PackCharts 自动消失。

### B. 安全地减少进入 xatlas 的 tiny charts

当前代码只对 tiny **connected components** 做 strip fallback；但真实大组件内部仍有大量 1～2 triangle chart。可以单独识别这些 chart，使用与 tiny component 类似的 planar projection + shelf strip，绕过 xatlas。

这是更可能影响 PackCharts 的方向，但需要验证：

- texel density 是否与大 chart 对齐；
- strip 是否溢出；
- seam / bake coverage 是否改变；
- 真实渲染是否出现材质块或采样断层。

这不是 CuMesh 直接提供的功能，必须作为独立的 UV 语义变更来做。

### C. GPU 化当前 precluster

可以借鉴 CuMesh 的：

```text
face normal/area
edge sort + unique
CSR edge/face adjacency
chart-pair sort/reduce
segmented cone reduction
```

但必须先保留当前 perimeter 语义，或者明确接受语义变化并增加同输入 CPU fixture。正确的低风险形式是：

```text
CPU current precluster labels
vs.
GPU candidate labels
-> cluster count / histogram / adjacency boundary
-> same mesh fed to xatlas
-> UV and render comparison
```

收益上限首先约为当前 10.6 秒，不应把它当作 xatlas 主优化。

### D. 直接复制 CuMesh refine / global iterations（不推荐）

这会改变 chart labels，并且当前项目对非流形 mesh 的处理语义与 CuMesh 不同。它不是纯性能 patch，应暂缓。

### E. 修改 xatlas 内部 packing（高风险）

当前 PackCharts 的主要工作是 chart placement / rasterization。要真正减少 84 秒，需要：

- 减少 chart 数；或
- 改变 packer 的搜索/placement 算法；或
- 为 tiny chart 做外部专用 packing。

这会触及 vendored `third_party/xatlas`，除非有独立 correctness fixture，否则不建议直接动。

## 8. 结论

**有优化空间，但 CuMesh 能直接带来的主要是 precluster 加速，不是当前最大热点。**

当前最合理的判断：

```text
CuMesh GPU clustering：值得做独立 prototype，预计优化约 10.6 s 级别
无条件合并所有 cluster：不可行，coincident vertex 会触发 xatlas colocal 风险
安全 batching：可能明显降低 AddMesh，但对 PackCharts 基本无效
真正影响总耗时：减少进入 xatlas 的 chart 数，或为 tiny chart 建立独立 UV fallback
```

下一步建议顺序：

```text
1. 统计真实 552,787 clusters 的共享 vertex / coincident position 冲突图
2. 离线模拟 greedy safe batching，估算可降到多少 batch
3. 只在临时 runner 中跑 batching，不接正式 decoder
4. 如果 AddMesh 确实明显下降，再做独立 patch
5. 之后再评估 tiny chart bypass；最后才考虑 CuMesh GPU precluster
```

## 9. 可复现材料

临时文件（均不属于仓库）：

```text
/tmp/cumesh-src/
/tmp/xatlas_batch_probe.cpp
/tmp/xatlas_batch_probe_grid.cpp
/tmp/xatlas_batch_probe_overlap.cpp
/tmp/xatlas_batch_probe
/tmp/pixal3d-xatlas-current-0911/run.log
```

当前工作区既有改动保持原样，未提交、未 push、未修改 `ref/`。
