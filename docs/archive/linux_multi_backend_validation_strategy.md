# Multi-backend validation from a Linux-only development environment

> **历史归档：** 本文件保留用于追溯，不是当前实现状态、性能基线或支持矩阵。


Research date: 2026-09-12. This is a proposed validation strategy, not a
certification of any additional backend. No CI jobs were dispatched and no
cloud resources or runner registrations were created.

## Local evidence and misleading symptom

A CPU-only build must not be confused with a CPU-only host. Read-only command:

```sh
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

Result: NVIDIA GeForce RTX 4090 D, driver 560.35.03. `vulkaninfo` was not found
on PATH. Thus CUDA hardware is present; Vulkan availability and required
extensions still need a loader/ICD/toolchain probe. Do not treat this inventory
as a successful graph run. Current project workflows inspected were Linux
release workflows, including an optional explicitly CUDA-compile-only job.

## Upstream practice / primary sources

- llama.cpp's self-hosted workflow separates Linux NVIDIA Vulkan, AMD ROCm,
  macOS ARM64 Metal, and additional hardware jobs:
  https://github.com/ggml-org/llama.cpp/blob/master/.github/workflows/build-self-hosted.yml
- Its 2025 CI migration discusses GPU hardware runners and software Vulkan
  testing on ordinary CI machines:
  https://github.com/ggml-org/llama.cpp/pull/16116
- Upstream build instructions document Linux Vulkan, HIP and SYCL toolchains:
  https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- GitHub currently documents GPU hardware acceleration on macOS M2 XLarge
  runners; do not generalize GPU availability from a generic macOS label:
  https://docs.github.com/en/actions/reference/runners/larger-runners
- Standard macOS/Windows jobs can cover native build and CPU tests. Larger
  runners have account/plan and billing requirements:
  https://docs.github.com/en/actions/reference/runners/github-hosted-runners
- Self-hosted runners require careful trust boundaries for untrusted PRs:
  https://docs.github.com/en/actions/reference/security/secure-use

These upstream master URLs are research references, not instructions to upgrade
the project's pinned ggml or copy newer compiler flags without verification.

## Recommended staged plan

1. Required Linux portable CPU Debug/Release and sanitizer checks, independent
   of model downloads and GPU availability.
2. Separate Linux CUDA and Vulkan builds on the existing NVIDIA host. Start
   with registry/capability inspection, tiny deterministic F32 operation and
   block fixtures, then forced-placement/transfer checks and model parity.
   Confirm Vulkan driver extensions before choosing optimized paths.
3. Remote native macOS build/CPU checks. For Metal runtime checks use an
   explicitly GPU-capable hosted Mac or an isolated self-hosted Apple Silicon
   Mac. Check device visibility first; missing hardware is not a passing GPU
   test. Use tiny fixtures before provisioning enough RAM for full models.
4. HIP and SYCL remain experimental. Linux toolchain compile checks are useful,
   but AMD/Intel target runtime claims need matching device/driver validation.
5. Software Vulkan is an optional shader/graph smoke tier, never a hardware
   placement, performance, or vendor-driver parity gate. Keep its label separate
   from forced-GPU tests; do not weaken GPU policy to accommodate a CPU device.

Use distinct results: configured, compile-verified, fixture-parity-verified,
and stage-validated (including placement/residency). Do not collapse these into
one 'supported' flag. A Vulkan pass on NVIDIA does not certify AMD or Intel.

Containerize SDK dependencies where useful, but do not treat a container image
or cross-compiled binary as proof of target-device execution. Start hardware
jobs manually on trusted commits; do not attach the everyday development
machine as an unrestricted public-PR runner.

## Next diagnostic

Probe Vulkan SDK/compiler, loader and NVIDIA ICD plus `vulkaninfo --summary`;
then build a separate Vulkan configuration against pinned ggml and run the same
small all-F32 fixture used for CPU/CUDA. No installation or GPU build was done
in this research turn. Only this documentation file was added.
