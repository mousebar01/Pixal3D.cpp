# Pixal3D.cpp engineering notes

这里的文档只记录已经检查过的工程事实、当前实现边界和可复现的验证结果。
它们不是模型权重、运行日志或平台支持承诺的替代品。

## 当前文档

| 文档 | 用途 | 当前状态 |
| --- | --- | --- |
| [`multi_backend_integration.md`](multi_backend_integration.md) | 外层 CMake、ggml registry、设备策略和 CLI inventory | 已实现，跨设备运行时 parity 仍未完成 |
| [`performance_audit_2026-09-12.md`](performance_audit_2026-09-12.md) | 已测性能热点、未解决的长尾和测量边界 | 审计记录 |
| [`cpu_threading_audit_2026-09-12.md`](cpu_threading_audit_2026-09-12.md) | CPU 小算子、ggml/OpenMP/ORT 线程预算和统一策略 | 第一阶段已实现，底层仍非共享物理线程池 |
| [`cuda_build_time_investigation_2026-09-12.md`](cuda_build_time_investigation_2026-09-12.md) | CUDA/ggml 编译耗时、架构矩阵、缓存和重复构建调查 | 调查完成，默认构建策略未改变 |
| [`profiler_opt_in_audit.md`](profiler_opt_in_audit.md) | profiler 默认关闭和条件采集规则 | 已实现 |
| [`qem_xatlas_gpu_probe.md`](qem_xatlas_gpu_probe.md) | QEM CUDA 探针与 xatlas CPU 边界 | 实验性路径 |
| [`xatlas_pack_optimization.md`](xatlas_pack_optimization.md) | xatlas 提交/打包的低风险优化 | 已实现，仍为 CPU xatlas |

## 历史归档

`archive/` 保存过去的调查和故障记录。归档内容可能包含旧构建目录、临时
日志路径、外部调研链接或当时尚未验证的方案；它们用于追溯原因，**不能当作
当前实现状态或支持矩阵**。

- `archive/troubleshooting_history.md`：迁移期间的 session 级排障记录
- `archive/texture_gray_diagnosis.md`：固定图片的灰度外观诊断
- `archive/cumesh_xatlas_analysis.md`：CuMesh/xatlas 源码调研
- `archive/linux_multi_backend_validation_strategy.md`：Linux-only 的跨后端验证策略

## 文档规则

1. 将“已测量”“从代码确认”“推测/待验证”分开写。
2. 性能数字必须带输入规模、backend、构建方式和测试命令；阶段名本身不等于实际执行设备。
3. “支持”必须同时有 metadata、tensor layout、loader、forward 和 parity 证据；仅编译通过不算支持。
4. 权重、mesh、完整日志和 build 目录留在 ignored 路径，不写入当前文档。
5. 历史记录若不再指导当前实现，移到 `archive/`，不要在顶层重复维护。
