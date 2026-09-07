#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <string>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;

namespace pixal3d {

enum class BackendPolicyKind {
    auto_select,
    cpu,
    gpu,
};

struct BackendPolicy {
    BackendPolicyKind kind = BackendPolicyKind::auto_select;
    int gpu_index = 0;

    static bool parse(const std::string & value,
                      BackendPolicy & output,
                      std::string * error = nullptr);
    static BackendPolicy from_environment(const char * stage_variable = nullptr,
                                          std::string * source = nullptr,
                                          std::string * error = nullptr);

    std::string name() const;
};

struct BackendDeviceInfo {
    std::size_t registry_index = 0;
    int gpu_index = -1;
    enum ggml_backend_dev_type type = GGML_BACKEND_DEVICE_TYPE_CPU;
    std::string name;
    std::string description;
    std::size_t memory_free = 0;
    std::size_t memory_total = 0;
};

// Shared backend ownership and policy for all Pixal3D stages.  The backend
// list is ordered with the selected GPU/IGPU first (when present), ACCEL
// devices next, and CPU last so it is directly usable by ggml_backend_sched.
class BackendManager {
public:
    BackendManager() = default;
    ~BackendManager();

    BackendManager(const BackendManager &) = delete;
    BackendManager & operator=(const BackendManager &) = delete;

    bool initialize(const BackendPolicy & policy,
                    std::string * error = nullptr);
    bool initialize_from_environment(const char * stage_variable = nullptr,
                                     std::string * error = nullptr);
    void close() noexcept;

    bool initialized() const noexcept { return initialized_; }
    const BackendPolicy & policy() const noexcept { return policy_; }
    const std::vector<ggml_backend_t> & backends() const noexcept { return backends_; }
    const std::vector<BackendDeviceInfo> & devices() const noexcept { return devices_; }

    ggml_backend_t primary() const noexcept;
    const std::string & primary_name() const noexcept { return primary_name_; }
    bool requires_primary_backend() const noexcept {
        return policy_.kind == BackendPolicyKind::gpu;
    }
    bool primary_supports_op(const ggml_tensor * op,
                             std::string * error = nullptr) const;

    // Returns the first backend that advertises support for an operation.
    // The returned handle is owned by this manager and remains valid until
    // close() or destruction.
    ggml_backend_t backend_for_op(const ggml_tensor * op) const noexcept;
    bool supports_op(const ggml_tensor * op,
                     std::string * error = nullptr) const;
    // Emit a runtime memory snapshot for one configured backend.  The
    // snapshot includes the caller-provided buffer size, if any.
    void log_buffer(const char * phase,
                    ggml_backend_t backend,
                    std::size_t buffer_bytes = 0) const noexcept;
    void set_n_threads(int n_threads) const noexcept;

private:
    BackendPolicy policy_;
    std::vector<ggml_backend_t> backends_;
    std::vector<BackendDeviceInfo> devices_;
    std::string primary_name_;
    bool initialized_ = false;
};

// RAII wrapper around ggml's multi-backend graph scheduler.  The manager must
// outlive this object because it owns the backend handles passed to ggml.
class BackendScheduler {
public:
    BackendScheduler() = default;
    BackendScheduler(const BackendManager & manager,
                     std::size_t graph_size,
                     bool parallel = false,
                     bool op_offload = true,
                     std::string * error = nullptr,
                     const char * stage = nullptr);
    ~BackendScheduler();

    BackendScheduler(const BackendScheduler &) = delete;
    BackendScheduler & operator=(const BackendScheduler &) = delete;

    BackendScheduler(BackendScheduler && other) noexcept;
    BackendScheduler & operator=(BackendScheduler && other) noexcept;

    bool valid() const noexcept { return scheduler_ != nullptr; }
    // Mark one operation as GPU-required when the selected policy is forced
    // GPU.  In auto mode the scheduler remains free to place it on any
    // configured backend that advertises support.
    bool require_primary(ggml_tensor * node, std::string * error = nullptr);
    bool require_primary_graph(ggml_cgraph * graph, std::string * error = nullptr);
    bool allocate_graph(ggml_cgraph * graph, std::string * error = nullptr);
    bool preflight(ggml_cgraph * graph, std::string * error = nullptr) const;
    bool validate_placement(ggml_cgraph * graph,
                            std::string * error = nullptr) const;
    ggml_status compute(ggml_cgraph * graph,
                        std::string * error = nullptr);
    void set_eval_callback(ggml_backend_sched_eval_callback callback,
                           void * user_data) noexcept;
    void reset() noexcept;
    void synchronize() noexcept;

    std::size_t buffer_size(ggml_backend_t backend) const noexcept;
    ggml_backend_t tensor_backend(ggml_tensor * node) const noexcept;
    // Print final scheduler placement and allocation statistics when tracing
    // is enabled through PIXAL3D_BACKEND_TRACE=1.
    void log_trace(ggml_cgraph * graph,
                   const char * phase = nullptr) const noexcept;

private:
    ggml_backend_sched_t scheduler_ = nullptr;
    const BackendManager * manager_ = nullptr;
    std::string stage_;
    std::vector<ggml_tensor *> required_nodes_;
};

const char * backend_device_type_name(enum ggml_backend_dev_type type) noexcept;

// Return a monotonic wall-clock timestamp in milliseconds.  The helper is
// intended for optional backend performance diagnostics and has no effect on
// graph placement or numerical behavior.
double backend_time_now_ms() noexcept;

// Emit one timing record when PIXAL3D_BACKEND_TRACE is enabled.
void backend_log_timing(const char * stage,
                        const char * phase,
                        double elapsed_ms) noexcept;

} // namespace pixal3d
