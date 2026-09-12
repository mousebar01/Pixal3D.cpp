# CPU 小算子线程管理审计

- **日期：** 2026-09-12
- **范围：** 外层 `src/` 的 CPU/F32 小算子、ggml CPU backend、MoGe
  ONNX Runtime，以及 mesh/export 中的独立 CPU 线程入口。
- **状态：** 第一阶段已实现进程级 CPU thread budget；底层仍由 ggml、OpenMP、
  ONNX Runtime 和 xatlas 各自管理执行，不声称它们共享一个物理 worker pool。

## 结论先行

当前不需要把所有 CPU 小算子强行改造成一个统一线程池。需要统一的是：

1. 进程级有效线程预算及其优先级；
2. 预算向各运行时的显式传播；
3. 小任务的串行阈值、并行层级和诊断字段；
4. GPU stage 中 host-only 工作的边界说明。

这与 whisper.cpp 的做法一致：入口有显式线程参数，随后分别传给 mel、模型图和
backend；跨任务并发度与单任务线程数分开，库内部仍保留自己的执行机制。为此，
Pixal3D.cpp 采用统一配置入口，而不是修改 vendored ggml 或建立一个跨库大而全的
线程池。

## 问题与证据

### 原来的线程预算分裂

审计时发现三套互不共享的线程机制：

1. ggml CPU backend 的 `ggml_threadpool`，只服务 ggml 图执行；
2. 外层 C++ 小算子的 OpenMP `parallel for`；
3. MoGe 的 ONNX Runtime IntraOp 线程池；xatlas 还会在图表计算调用期间创建
   一次性 `std::thread`。

历史真实 decoder profile 中出现过：

```text
omp_max_threads=32
omp_actual_threads=32
cpu_threads=4
```

这里的 `cpu_threads=4` 只代表 ggml CPU backend 的配置，不代表外层
`sparse_linear` 只用了 4 个线程。此前 `src/moge_camera.cpp` 还固定调用
`SetIntraOpNumThreads(4)`，所以同一进程的不同阶段可能采用不同预算。

本机审计环境为 Intel Core i9-14900K、32 个逻辑 CPU、24 个物理核心，构建启用
`GGML_USE_OPENMP` 和外层 OpenMP，未设置 `OMP_NUM_THREADS`。一次临时
`sparse_linear` micro-probe（100,000 points、16 输入通道、32 输出通道）显示：

| `OMP_NUM_THREADS` | total | compute |
| ---: | ---: | ---: |
| 1 | 94.94 ms | 57.33 ms |
| 4 | 50.28 ms | 14.43 ms |
| 8 | 48.47 ms | 11.90 ms |

该数据只用于说明预算和并行层级的影响，不是端到端性能承诺。

## whisper.cpp 调研结论

已对官方 `ggml-org/whisper.cpp` 的固定提交 `1da4dc82fa7996d4edda05890dca65aeceaafd6d`
进行只读源码审计，重点是 `src/whisper.cpp`、`include/whisper.h` 和
`examples/cli/cli.cpp`：

- `whisper_full_params` 有显式 `n_threads`，默认值约为
  `min(4, hardware_concurrency())`；
- CLI 的 `-t/--threads` 会传给 mel、encoder、decoder 和 ggml backend；
- ggml 图通过 `ggml_backend_sched` 执行，compute 前设置 backend 线程数；
- mel 和 `whisper_full_parallel()` 在调用期间创建临时 `std::thread`；
- `n_processors` 是独立的跨任务并发度，潜在并发近似为
  `n_threads × n_processors`；
- VAD 仍有独立 `n_threads`，没有跨所有 CPU 子系统共享的全局 worker pool。

因此，whisper.cpp 支持“统一预算、显式传播、避免 oversubscription”的方向，
但不支持“所有模块必须共享同一物理线程池”的方向。

## 第一阶段实现

### 统一策略和优先级

新增 `include/pixal3d/cpu_threads.h` 与 `src/cpu_threads.cpp`，有效线程数的优先级为：

```text
CLI/process override
    > PIXAL3D_CPU_THREADS
    > OMP_NUM_THREADS 的第一个数值
    > min(4, hardware_concurrency())
```

`PIXAL3D_CPU_THREADS` 必须是正整数；`OMP_NUM_THREADS=3,2` 兼容 OpenMP 的
列表写法并取第一个数。无效的项目变量会记录 warning 后继续回退，不会静默地
把错误值当成有效线程数。

CLI 的 `--cpu-threads N` 已接入：

- `run-cascade`
- `run-cascade-mv`
- `run-image`
- `encode-condition-stage`
- `encode-condition-bundle`

例如：

```sh
PIXAL3D_CPU_THREADS=8 ./build/bin/pixal3d run-cascade ...
./build/bin/pixal3d run-image ... --cpu-threads 8
```

### 各运行时的传播

- **ggml：** `BackendManager` 初始化时按统一预算创建并绑定 CPU
  `ggml_threadpool`；`BackendScheduler::compute()` 前再次应用当前预算。
  manager-owned pool 只服务 ggml graph。
- **外层 OpenMP：** 当前 27 个主要并行区显式使用
  `num_threads(cpu_thread_count())`，并通过 `cpu_should_parallelize()` 对小工作量
  走串行路径。覆盖 `sparse.cpp`、`sparse_transformer.cpp`、`naf.cpp`、
  `slat_flow.cpp` 和 `slat_decoder.cpp`，以及对应的 CUDA benchmark helper。
- **MoGe/ONNX Runtime：** `SetIntraOpNumThreads()` 使用同一预算；图设置为
  `ORT_SEQUENTIAL`，`SetInterOpNumThreads(1)`，避免在单模型调用中再引入第二层
  inter-op 线程预算。每次创建 MoGe session 时记录 `cpu_threads`、来源和
  `inter_op_threads=1`。
- **小算子：** 没有为了“统一”而给所有串行操作增加 fork/join。元素量小、分配
  或同步成本占主导的工作继续串行，后续只有在 microbench 证明收益时才单独扩大
  并行范围。
- **NAF：** 每像素 softmax workspace 仍位于并行循环内部，保持 thread-local；
  没有改成跨调用共享 workspace，也没有改变参考实现的累加顺序。

这套实现统一的是预算和诊断，不是物理 worker pool。OpenMP、ggml 和 ORT 的
barrier、线程生命周期及线程局部状态仍由各自运行时负责；xatlas 也不接受 ggml
worker 注入。

## GPU/host 边界

线程预算不能掩盖 backend placement。当前 mesh export 的边界是：

- **QEM：** CUDA 构建会优先尝试可选的 CUDA QEM 路径；设备、分配或 kernel
  失败时记录原因并显式回退到 CPU QEM；非 CUDA 构建直接使用 CPU QEM。CUDA
  路径目前是实验性实现，独立 fixture 已验证有限输入，但同一真实 decoder
  mesh 的拓扑、质量和 CPU/GPU parity 尚未完成，因此不能称为生产级 GPU 支持。
- **xatlas/UV bake：** 当前 vendored xatlas 没有 CUDA/HIP/Vulkan 执行路径，
  charting、packing、atlas rasterization 和 inpaint 仍是 host-side CPU 工作。

QEM 不能再被笼统写成 CPU-only；准确描述是“CUDA build 下实验性 GPU-first
QEM + CPU fallback”。xatlas 则仍然是 CPU-only。详细证据见
`docs/qem_xatlas_gpu_probe.md` 和 `docs/performance_audit_2026-09-12.md`。

## 不采用全局物理线程池的原因

- pinned ggml 没有面向外层任务提交的通用 pool API；
- OpenMP、ggml、ORT 各自管理 barrier、线程生命周期和线程局部状态；
- xatlas 是独立库，不能安全假设它接受 ggml worker；
- 修改 `ref/trellis2cpp` 或 vendored ggml 会扩大变更面，并引入重入、亲和性、
  异常和并发模型问题；
- OpenMP runtime 通常会复用 team，统一预算已经能解决当前主要 oversubscription
  风险。

如果未来需要跨图片吞吐，应新增独立的 job-level `n_processors`/队列配置，
并明确其与单任务 `cpu_threads` 的乘法关系；不能隐式地把两个维度都调大。

## 验证状态

本轮代码验证命令：

```sh
cmake -S . -B build-backend-test \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DPIXAL3D_ONNXRUNTIME_ROOT=/home/sy/.local/opt/onnxruntime-1.23.2
cmake --build build-backend-test --parallel 4 \
  --target pixal3d_core pixal3d pixal3d_cpu_threads_test \
           pixal3d_backend_manager_test pixal3d_sparse_fixture
./build-backend-test/bin/pixal3d_cpu_threads_test
./build-backend-test/bin/pixal3d_backend_manager_test
PIXAL3D_CPU_THREADS=1 ./build-backend-test/bin/pixal3d_sparse_fixture
PIXAL3D_CPU_THREADS=8 ./build-backend-test/bin/pixal3d_sparse_fixture
./build-backend-test/bin/pixal3d --help
./build-backend-test/bin/pixal3d --version
(cd build-backend-test && ctest --output-on-failure)
```

验证结果：配置成功；受影响目标全部编译成功；`pixal3d_cpu_threads_test`、
`pixal3d_backend_manager_test` 和 `PIXAL3D_CPU_THREADS=1/8` 的 sparse fixture
均通过；CLI help/version/list-devices smoke check 通过；CTest **25/25 通过**，
总耗时约 13 秒。注意本机 CMake 使用 CTest 3.16，`ctest --test-dir ...` 不会
切换测试目录，必须使用上面的 subshell 写法。

构建产物、权重、mesh 和完整日志留在 ignored 路径，不写入文档。

## 后续工作

- 为 MoGe 增加无需加载模型的线程配置单测或 session-options probe；
- 对强制 `PIXAL3D_CPU_THREADS=1/8` 的 sparse/NAF 结果做固定 fixture parity；
- 记录真实权重下的线程曲线，区分单任务延迟与多任务吞吐；
- 继续检查 GPU stage 的 eligible operations、activation residency 和 H2D/D2H
  计数，避免部分 GPU 路径反复跨设备传输。
