#pragma once

#include <cstddef>
#include <string>

namespace pixal3d {

// Process-level CPU execution policy.  This controls the budget passed to
// ggml, OpenMP regions, and ONNX Runtime.  It deliberately does not expose or
// replace the worker implementation owned by any of those runtimes.
struct CpuThreadPolicy {
    int threads = 1;
    std::string source;
    std::string warning;
};

// Parse one positive decimal thread count.  This is used for both the CLI
// option and the PIXAL3D_CPU_THREADS environment variable.
bool parse_cpu_thread_count(const std::string & value,
                            int & threads,
                            std::string * error = nullptr);

// Resolve the effective process-level budget.  Precedence is:
// CLI/process override, PIXAL3D_CPU_THREADS, the first value in
// OMP_NUM_THREADS, and the portable project default.
CpuThreadPolicy cpu_thread_policy_from_environment();
int default_cpu_thread_count() noexcept;
int cpu_thread_count() noexcept;

// Set or clear the explicit CLI/process override.  This is configuration
// state, not model state, and must be set before starting a pipeline stage.
bool set_cpu_thread_override(int threads, std::string * error = nullptr);
void clear_cpu_thread_override() noexcept;

// Avoid launching a full team for small operations.  The default requires at
// least min_items_per_thread scalar work items per effective thread.
bool cpu_should_parallelize(
    std::size_t work_items,
    std::size_t min_items_per_thread = 256) noexcept;

} // namespace pixal3d
