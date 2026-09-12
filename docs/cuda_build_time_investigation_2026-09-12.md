# CUDA 编译耗时调查

- **Updated:** 2026-09-12
- **Scope:** bundled ggml CUDA 编译、CMake 架构选择、Ninja 并行度和重复构建
- **Status:** 调查完成；已加入可选的本机架构、ccache 和 Docker 并行度配置，portable 默认行为保持不变

## 结论先行

这次 CUDA 冷编译耗时很长，**有一部分是正常的**，但当前构建过程还有明显的可避免开销：

1. bundled ggml 当前生成了 130 个 CUDA object；其中多个量化矩阵乘和 FlashAttention 模板实例本身就很重。
2. `GGML_NATIVE=OFF` 且没有设置 `CMAKE_CUDA_ARCHITECTURES` 时，CUDA 12.6 的 ggml 默认使用 7 组架构：
   `50-virtual;61-virtual;70-virtual;75-virtual;80-virtual;86-real;89-real`。
3. `PIXAL3D_CUDA_ARCHITECTURES=89` **只约束外层 `pixal3d_core` 的自定义 QEM CUDA 文件**，不会约束 ggml。当前 `compile_commands.json` 也证明了两者的差异。
4. 调查时顶层 CMake 强制 `GGML_CCACHE=OFF`。机器虽然安装了 ccache，但当时的 CUDA 命令直接调用 `nvcc`，没有经过 ccache；现在已改为显式 opt-in。后续 sm_89 验证构建确认了 `ccache nvcc` 规则生效。
5. 调查期间曾有两个完整 CUDA 构建并行运行，而且 Docker 构建目录记录的是 `/workspace`，当前源码路径是 `/home/sy/Pixal3D.cpp`。这会放大耗时，也会阻断构建缓存复用。

因此：**首次冷编译达到十几到几十分钟是可以解释的；但重复启动多个全量构建、跨容器/宿主复用错误 build 目录，以及没有利用架构裁剪和缓存，不应作为正常基线。**

## 本地环境和构建证据

### 构建环境

历史上的 `build-cuda-docker` 是在 CUDA 开发容器路径 `/workspace` 下配置的，配置文件显示：

```text
CMake:                 3.28.3
CUDA compiler:         12.6.85
Generator:             Ninja
Build type:            Release
GGML_NATIVE:           OFF
GGML_CUDA:             ON
GGML_CUDA_FA:          ON
GGML_CUDA_FA_ALL_QUANTS: OFF
GGML_CCACHE:           OFF
PIXAL3D_CUDA_ARCHITECTURES: 89
CMAKE_HOME_DIRECTORY:  /workspace
```

当前宿主机 shell 的 `/usr/bin/nvcc` 是 CUDA 10.1，而上述 build 目录的 compiler-id 明确是 CUDA 12.6。因此不要在宿主机直接复用这个 Docker build 目录；这不仅是路径问题，也是 CUDA toolchain 不同的问题。

本机硬件检查结果为 32 个逻辑 CPU、NVIDIA GeForce RTX 4090 D（compute capability 8.9）。本次构建可见的 GPU 目标和构建时的 CUDA 工具链并不意味着其他 GPU 也经过了运行时 parity 验证。

本次隔离验证使用同一 CUDA 12.6 开发镜像、独立的
`build-cuda-sm89` 目录和 `--parallel 8`。配置耗时约 7.9 秒，完整构建耗时
916 秒（15 分 16 秒）。编译阶段没有挂载 GPU；之后使用 `--gpus all` 运行
时，ggml registry 正确发现 RTX 4090 D，说明 CUDA 工具链编译和 CUDA 运行时
验证是两个独立步骤。

### 目标数量和架构矩阵

`build-cuda-docker/build.ninja` 中有：

```text
CUDA object build rules: 131
  ggml CUDA objects:     130
  Pixal3D custom QEM:      1
```

130 个 ggml CUDA 编译命令都包含下面 7 个 `--generate-code` 参数：

```text
compute_50 -> PTX
compute_61 -> PTX
compute_70 -> PTX
compute_75 -> PTX
compute_80 -> PTX
compute_86 -> sm_86
compute_89 -> sm_89
```

这不是严格意义上的 910 个独立 Ninja job，因为多个架构是在同一次 `nvcc` 调用中生成的；但它说明每个 CUDA 源文件都要处理多组 device code。以 130 个 ggml object 粗略计算，就是约 910 个“源文件 × 架构”组合需要经过 CUDA 编译流程。

同时，QEM 文件的命令只含：

```text
compute_89 -> compute_89,sm_89
```

所以 `PIXAL3D_CUDA_ARCHITECTURES=89` 没有减少 ggml 的 7 架构编译量。这是目前最容易误解的配置点。

### Ninja 日志的耗时范围

以下数字来自被多次尝试写入的 Ninja 日志，只用于判断数量级，**不是干净的 A/B benchmark**：

| 构建目录 | 有记录的 ggml CUDA object | 日志时间跨度 | 单个最慢 object |
| --- | ---: | ---: | ---: |
| `build-cuda-xa-profile` | 63 | 约 21.8 分钟 | 约 663 秒 |
| `build-cuda-verify` | 111 | 约 22.3 分钟 | 约 685 秒 |
| `build-cuda-docker` | 131 | 约 51.3 分钟 | 约 788 秒 |

`build-cuda-docker` 中最慢的对象包括：

```text
fattn-tile-instance-dkq256-dv256.cu.o  788.057 s
mmq-instance-q2_k.cu.o                  751.735 s
mmvq.cu.o                               676.080 s
mmq-instance-q6_k.cu.o                  574.769 s
```

这些是 ggml 的矩阵乘/注意力模板实例，不是 QEM。相比之下，日志中 Pixal3D 的 `decimate_qem.cu.o` 约 37 秒；仅改动 C++ 文件后的增量编译中，`mesh_qem_decimate.cpp.o` 约 3.9 秒，随后静态库和测试链接均小于 0.3 秒。

日志还存在重复对象路径和重叠编译区间，和调查期间同时启动多个完整构建的现象一致。因此 51 分钟不能直接当作“单次、单进程、固定并行度”的准确基线；它更像是重复/并发构建被记录在一起后的上界。

### sm_89 + ccache 验证结果

新配置的实际 CMake/Ninja 结果如下：

```text
PIXAL3D_GGML_CUDA_ARCHITECTURES=89
CMAKE_CUDA_ARCHITECTURES=89
PIXAL3D_CUDA_ARCHITECTURES=89
GGML_CCACHE=ON
ggml CUDA compile commands=130
non-sm_89 generate-code commands=0
```

`compile_commands.json` 不包含 Ninja 的 `RULE_LAUNCH_COMPILE` 前缀，因此不能
用它判断是否经过 ccache；在 `build.ninja` 中实际看到的命令是：

```text
ccache /usr/local/cuda/bin/nvcc ... --generate-code=arch=compute_89,code=[compute_89,sm_89]
```

为两个对象各执行一次 miss、再各执行一次 hit 的小型缓存验证得到：

```text
Cacheable calls: 4 / 4
Hits: 2 / 4
Misses: 2 / 4
```

这不是完整构建的缓存命中率，只证明 CUDA 和 C++ 编译规则都进入了 ccache；
要获得重复构建收益，必须把 ccache 目录挂载到稳定的宿主机路径，不能让每次
临时 Docker 容器退出时一起丢弃缓存。

这次 sm_89 冷构建仍用了 916 秒，说明架构裁剪主要减少每个 nvcc 调用中的
device-code 生成量，并不会减少 130 个 ggml CUDA 源文件或其模板实例本身。
因此这个结果不能宣称“sm_89 一定比旧的多架构日志快多少”；旧日志包含并发
构建和不同路径，无法作为干净的 A/B 基线。但它已经验证了新选项没有被 QEM
选项误用，并且只生成了 sm_89 的 ggml CUDA code。

随后在同一容器路径中使用 `--gpus all` 执行 CTest，结果为 `28/28` 通过，
其中 4 个 CUDA 标签测试实际走了 RTX 4090 D。单独的 QEM fixture 也通过，
但这只覆盖 QEM/小型 ggml 图，不代表完整 Pixal3D 推理已经完成 GPU parity。

## 社区资料和可比结论

### llama.cpp 社区讨论

[Speedup build cu files](https://github.com/ggml-org/llama.cpp/discussions/11582) 直接讨论了 llama.cpp 的 CUDA 编译很慢，参与者把主要瓶颈归因于大量 `.cu` 文件和 `nvcc`，并列出选择性编译、拆分大型 CUDA 文件、复用预编译 object、提高并行度等方向。讨论中还提到低端机器冷编译达到 30 分钟到 1 小时并不罕见。它是社区经验，不是严格 benchmark，但和本项目“ggml CUDA 文件数量多、模板实例重”的现象吻合。

### llama.cpp 官方构建说明

llama.cpp 的 CUDA 构建文档明确说明：

- 非 native 构建会覆盖更多 GPU，生成更大的 binary，编译时间也会明显增加；
- 可以用 `CMAKE_CUDA_ARCHITECTURES` 显式限制目标架构；
- `-j`/Ninja 可提高并行度；
- ccache 适合加速重复编译。

参考：[llama.cpp build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)。

### CMake 社区关于多架构

[CMake Discourse #3777](https://discourse.cmake.org/t/create-separate-make-command-for-each-of-cmake-cuda-architectures/3777) 说明了一个关键细节：当一个 target 设置多个 CUDA architecture 时，生成的 `nvcc` 命令会列出所有架构，CUDA 编译器会为每个架构生成代码；该讨论还建议用 Ninja pool 控制 CUDA 编译并行度，避免外层 `-j` 和 CUDA 内部架构并行相互过度订阅。

### ggml/llama.cpp 的模板实例优化

[llama.cpp PR #21768](https://github.com/ggml-org/llama.cpp/pull/21768) 通过跳过不会使用的 FlashAttention 模板特化，把一台 64-core EPYC 上的完整构建（无 ccache）从 330 秒降到 300 秒。这个结果不能直接套用到 Pixal3D，但它证明模板实例选择确实会影响冷编译时间；也说明不同 CPU、并行度、架构矩阵和源码版本之间的绝对耗时差异会很大。

### NVIDIA nvcc 选项

当前 CUDA 12.6 的本地 `nvcc --help` 可见：

- `--threads (-t)`：为多架构编译创建内部并行线程；
- `--split-compile`：并行化部分编译器优化；
- `--time (-time)`：输出各编译阶段耗时。

NVIDIA 文档：[nvcc compiler driver](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)。这些选项不能无条件叠加：外层 Ninja `-j`、nvcc `-t` 和机器内存需要一起测量，否则可能过度订阅 CPU 或内存。

### ccache 的注意点

[ccache 关于 NVCC 的讨论](https://github.com/ccache/ccache/discussions/1300)指出，使用 CUDA 时应缓存 `nvcc` 调用，而不是只把 host compiler（例如 gcc）替换成 ccache；同时 CUDA 编译链中的不同阶段对缓存收益并不完全相同。当前项目的 pinned ggml CMake 已有 `GGML_CCACHE` 支持；历史构建曾被外层 CMake 强制关闭，现在通过 `PIXAL3D_GGML_CCACHE=ON` 显式启用。

## “正常”与“不正常”的边界

### 可以认为正常的部分

- 第一次启用 CUDA，需要从零编译 ggml 的大量 `.cu` 文件；
- 每个源文件都要经过 7 组 CUDA 架构代码生成；
- `mmq`、`mmvq`、`mmvf` 和 FlashAttention 模板实例的单文件耗时远高于普通 C++；
- `GGML_CUDA_FA_ALL_QUANTS=OFF` 已经是当前 pinned ggml 的默认值，因此这次不能把“所有 FlashAttention 量化特化都打开”作为主要原因；
- Docker 内部的 CUDA 12.6 和宿主机 CUDA 10.1 不同，重新配置或重新编译也属于预期成本。

### 不应作为正常基线的部分

- 同时启动两个完整 CUDA 构建；
- 在 `/workspace` 配置的 Docker build 目录上直接使用 `/home/sy/Pixal3D.cpp` 宿主路径；
- 每次小改动都删除 build 目录或重新配置到新路径；
- 已安装 ccache 却在顶层强制 `GGML_CCACHE=OFF`；
- 用 `PIXAL3D_CUDA_ARCHITECTURES=89` 误以为已经把 ggml 限制到 sm_89；
- 在没有验证模型所需算子和 parity 的情况下，随意删掉 ggml CUDA 源文件或关闭 attention/matmul kernel。

## 推荐的验证顺序

暂不改变 portable 默认构建，先建立两个明确的配置：

### 1. 可发布/可移植构建

保留当前 `GGML_NATIVE=OFF` 和 ggml 默认架构矩阵，单独使用固定的容器 build 目录。不要让宿主机和容器共享同一 build 目录。

### 2. 本机 RTX 4090 D 开发构建

在 CUDA 开发容器中使用 CMake 3.18+，显式给 ggml 设置架构，例如：

```sh
cmake -S . -B build-cuda-sm89 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPIXAL3D_ENABLE_CUDA=ON \
  -DPIXAL3D_GGML_CUDA_ARCHITECTURES=89 \
  -DPIXAL3D_CUDA_ARCHITECTURES=89 \
  -DPIXAL3D_GGML_CCACHE=ON \
  -DPIXAL3D_GGML_NATIVE=OFF \
  -DBUILD_TESTING=ON
cmake --build build-cuda-sm89 --parallel 8
```

这里两个变量含义不同：

- `PIXAL3D_GGML_CUDA_ARCHITECTURES=89`：由外层项目传给 bundled ggml，限制 ggml CUDA 架构；
- `CMAKE_CUDA_ARCHITECTURES=89`：仍可由调用方直接设置，适用于明确控制整个 CMake CUDA target 初始化的场景；
- `PIXAL3D_CUDA_ARCHITECTURES=89`：限制 Pixal3D 自定义 QEM target。

如果采用 `89-real` 而不是 `89`，需要把它作为本机专用构建决定，并确认是否仍需要 PTX 前向兼容；不要把它作为可移植默认值。

### 3. 再测缓存和并行度

在固定路径、单个构建进程下依次比较：

```text
-j 4
-j 8
-j 12
```

然后单独测试启用 ccache 后的第二次增量构建。不要一开始就把 `-j` 调到 32，也不要同时启用较大的 nvcc `-t`；先观察 CPU、内存、swap 和单个编译阶段耗时。

## 已实施的构建配置变更

本次调查后的低风险构建改动位于外层项目，不修改 pinned `ggml`：

- 新增 `PIXAL3D_GGML_CUDA_ARCHITECTURES`。非空时，它会在 bundled ggml 配置前设置 `CMAKE_CUDA_ARCHITECTURES`；空值保持 ggml 的 portable 默认矩阵。
- 保留 `PIXAL3D_CUDA_ARCHITECTURES` 只控制外层自定义 QEM target，避免两个语义混淆。
- 新增 `PIXAL3D_GGML_CCACHE`，由外层显式传给 pinned ggml 的 `GGML_CCACHE`。默认仍为 OFF；开启后由 ggml 检测 ccache/sccache。
- Docker dev/cpp-build 增加 ccache 包、架构参数和可选 `PIXAL3D_BUILD_PARALLEL`，便于使用固定的单构建并行度。

本次仍然没有：

- 删除 CUDA kernel 或关闭 `GGML_CUDA_FA`；
- 默认把 portable 构建切换为 sm_89；
- 默认开启 ccache；
- 把本机 sm_89 构建当成发布构建；
- 宣称 CUDA 全 pipeline 或 QEM 已达到生产级 parity。

本次已经完成：sm_89 配置与架构覆盖检查、固定 `--parallel 8` 的完整冷构建、
ccache miss/hit 小验证、CUDA QEM fixture 和带 GPU 的 CTest。仍需后续做干净的
portable 多架构与 sm_89 A/B 测量、不同并行度的系统比较，以及完整模型算子
覆盖和运行时 parity；这些不应由一次 QEM fixture 代替。

## 复现和检查命令

```sh
# 检查实际 CUDA 架构和 compiler
sed -n '1,90p' build-cuda-docker/CMakeFiles/3.28.3/CMakeCUDACompiler.cmake
grep -n 'CMAKE_CUDA_ARCHITECTURES' build-cuda-docker/CMakeFiles/CMakeConfigureLog.yaml

# 检查实际编译命令是否经过 ccache、以及 ggml 架构参数
python3 - <<'PY'
import json
from pathlib import Path
commands = json.loads(Path('build-cuda-docker/compile_commands.json').read_text())
for item in commands:
    if 'ggml/src/ggml-cuda/' in item.get('file', '') and item['file'].endswith('.cu'):
        print(item['command'])
        break
PY

# 查看 Ninja 的单目标实际命令和增量状态
ninja -C build-cuda-docker -t commands ggml/src/ggml-cuda/libggml-cuda.a | sed -n '1,5p'
ninja -C build-cuda-docker -n pixal3d
```
