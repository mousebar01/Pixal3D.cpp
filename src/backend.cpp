#include "pixal3d/backend.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <chrono>
#include <utility>

namespace pixal3d {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

std::string lower_ascii(std::string value) {
    for (char & ch : value) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return value;
}

const char * env_value(const char * name) {
    if (!name || !*name) return nullptr;
    const char * value = std::getenv(name);
    return value && *value ? value : nullptr;
}

bool is_gpu_type(enum ggml_backend_dev_type type) {
    return type == GGML_BACKEND_DEVICE_TYPE_GPU ||
           type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

bool is_gpu_core_op(enum ggml_op op) {
    switch (op) {
        case GGML_OP_GET_ROWS:
        case GGML_OP_MUL_MAT:
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_3D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_FLASH_ATTN_EXT:
            return true;
        default:
            return false;
    }
}

std::string device_label(ggml_backend_dev_t device) {
    const char * description = ggml_backend_dev_description(device);
    if (description && *description) return description;
    const char * name = ggml_backend_dev_name(device);
    return name && *name ? name : "unknown";
}

BackendDeviceInfo describe_device(ggml_backend_dev_t device,
                                  std::size_t registry_index,
                                  int gpu_index) {
    BackendDeviceInfo info;
    info.registry_index = registry_index;
    info.gpu_index = gpu_index;
    info.type = ggml_backend_dev_type(device);
    const char * name = ggml_backend_dev_name(device);
    info.name = name && *name ? name : "unknown";
    info.description = device_label(device);
    ggml_backend_dev_memory(device, &info.memory_free, &info.memory_total);
    return info;
}

std::string bytes_string(std::size_t bytes) {
    if (bytes == 0) return "unknown";
    const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
    return std::to_string(mib) + " MiB";
}

bool trace_enabled() {
    const char * value = std::getenv("PIXAL3D_BACKEND_TRACE");
    return value && *value && std::strcmp(value, "0") != 0;
}

} // namespace

const char * backend_device_type_name(enum ggml_backend_dev_type type) noexcept {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU: return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU: return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
    }
    return "UNKNOWN";
}

double backend_time_now_ms() noexcept {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
        clock::now().time_since_epoch()).count();
}

void backend_log_timing(const char * stage,
                        const char * phase,
                        double elapsed_ms) noexcept {
    if (!trace_enabled()) return;
    std::cerr << "pixal3d: backend timing stage="
              << (stage && *stage ? stage : "unknown")
              << " phase=" << (phase && *phase ? phase : "unknown")
              << " elapsed_ms=" << elapsed_ms << std::endl;
}

bool BackendPolicy::parse(const std::string & value,
                          BackendPolicy & output,
                          std::string * error) {
    if (error) error->clear();
    const std::string normalized = lower_ascii(value);
    if (normalized == "auto" || normalized.empty()) {
        output = BackendPolicy{};
        return true;
    }
    if (normalized == "cpu") {
        output.kind = BackendPolicyKind::cpu;
        output.gpu_index = 0;
        return true;
    }
    if (normalized == "gpu") {
        output.kind = BackendPolicyKind::gpu;
        output.gpu_index = 0;
        return true;
    }
    constexpr const char prefix[] = "gpu:";
    if (normalized.compare(0, sizeof(prefix) - 1, prefix) == 0) {
        const std::string index_text = normalized.substr(sizeof(prefix) - 1);
        if (index_text.empty()) {
            set_error(error, "backend policy gpu:<index> requires an index");
            return false;
        }
        errno = 0;
        char * end = nullptr;
        const long parsed = std::strtol(index_text.c_str(), &end, 10);
        if (errno == ERANGE || end == index_text.c_str() || *end != '\0' ||
            parsed < 0 || parsed > std::numeric_limits<int>::max()) {
            set_error(error, "invalid GPU backend index: " + index_text);
            return false;
        }
        output.kind = BackendPolicyKind::gpu;
        output.gpu_index = static_cast<int>(parsed);
        return true;
    }

    set_error(error, "unknown backend policy: " + value +
                     " (expected auto, cpu, gpu, or gpu:<index>)");
    return false;
}

BackendPolicy BackendPolicy::from_environment(const char * stage_variable,
                                              std::string * source,
                                              std::string * error) {
    BackendPolicy result;
    if (source) source->clear();
    if (error) error->clear();

    const char * selected_name = nullptr;
    const char * selected_value = nullptr;
    if (stage_variable) {
        selected_value = env_value(stage_variable);
        if (selected_value) selected_name = stage_variable;
    }
    if (!selected_value) {
        selected_value = env_value("PIXAL3D_BACKEND");
        if (selected_value) selected_name = "PIXAL3D_BACKEND";
    }
    if (!selected_value) {
        if (source) *source = "default";
        return result;
    }

    if (!parse(selected_value, result, error)) return BackendPolicy{};
    if (source) *source = selected_name;
    return result;
}

std::string BackendPolicy::name() const {
    switch (kind) {
        case BackendPolicyKind::auto_select: return "auto";
        case BackendPolicyKind::cpu: return "cpu";
        case BackendPolicyKind::gpu: return "gpu:" + std::to_string(gpu_index);
    }
    return "unknown";
}

BackendManager::~BackendManager() {
    close();
}

void BackendManager::close() noexcept {
    for (ggml_backend_t backend : backends_) {
        if (backend && ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_threadpool(backend, nullptr);
        }
    }
    if (cpu_threadpool_) {
        ggml_threadpool_free(cpu_threadpool_);
        cpu_threadpool_ = nullptr;
    }
    cpu_threadpool_n_threads_ = 0;
    for (auto it = backends_.rbegin(); it != backends_.rend(); ++it) {
        if (*it) ggml_backend_free(*it);
    }
    backends_.clear();
    devices_.clear();
    primary_name_.clear();
    initialized_ = false;
}

bool BackendManager::initialize_from_environment(const char * stage_variable,
                                                 std::string * error) {
    std::string source;
    std::string parse_error;
    BackendPolicy policy = BackendPolicy::from_environment(stage_variable,
                                                            &source, &parse_error);
    if (!parse_error.empty()) {
        set_error(error, parse_error);
        return false;
    }
    if (!source.empty()) {
        std::cerr << "pixal3d: backend policy " << policy.name()
                  << " (source=" << source << ")" << std::endl;
    }
    return initialize(policy, error);
}

bool BackendManager::initialize(const BackendPolicy & policy,
                                std::string * error) {
    if (error) error->clear();
    close();
    policy_ = policy;

    struct DeviceEntry {
        ggml_backend_dev_t device = nullptr;
        BackendDeviceInfo info;
    };
    const std::size_t device_count = ggml_backend_dev_count();
    const std::size_t no_entry = std::numeric_limits<std::size_t>::max();
    std::vector<DeviceEntry> inventory;
    inventory.reserve(device_count);
    std::size_t selected_gpu_entry = no_entry;
    std::size_t cpu_entry = no_entry;
    int gpu_count = 0;
    for (std::size_t index = 0; index < device_count; ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
        const int ordinal = is_gpu_type(type) ? gpu_count++ : -1;
        inventory.push_back({device, describe_device(device, index, ordinal)});
        const DeviceEntry & entry = inventory.back();
        std::cerr << "pixal3d: backend device " << index << ": "
                  << entry.info.description << " ("
                  << backend_device_type_name(type) << ", memory="
                  << bytes_string(entry.info.memory_free) << "/"
                  << bytes_string(entry.info.memory_total) << ")" << std::endl;
        if (is_gpu_type(type) &&
            ((policy.kind == BackendPolicyKind::gpu && ordinal == policy.gpu_index) ||
             (policy.kind == BackendPolicyKind::auto_select &&
              selected_gpu_entry == no_entry))) {
            selected_gpu_entry = index;
        }
        if (type == GGML_BACKEND_DEVICE_TYPE_CPU && cpu_entry == no_entry) {
            cpu_entry = index;
        }
    }

    if (policy.kind == BackendPolicyKind::gpu && selected_gpu_entry == no_entry) {
        set_error(error, "requested GPU backend " + std::to_string(policy.gpu_index) +
                         " was not found");
        return false;
    }

    const auto register_backend = [this](const DeviceEntry & entry,
                                         ggml_backend_t backend,
                                         bool make_primary) {
        backends_.push_back(backend);
        devices_.push_back(entry.info);
        if (make_primary || primary_name_.empty()) {
            primary_name_ = entry.info.description;
        }
    };

    if (policy.kind != BackendPolicyKind::cpu && selected_gpu_entry != no_entry) {
        const DeviceEntry & entry = inventory[selected_gpu_entry];
        ggml_backend_t backend = ggml_backend_dev_init(entry.device, nullptr);
        if (!backend) {
            if (policy.kind == BackendPolicyKind::gpu) {
                set_error(error, "failed to initialize requested GPU backend " +
                                 entry.info.description);
                return false;
            }
            std::cerr << "pixal3d: GPU backend " << entry.info.description
                      << " failed to initialize; falling back" << std::endl;
        } else {
            register_backend(entry, backend, true);
            std::cerr << "pixal3d: using " << primary_name_ << " backend"
                      << " (policy=" << policy.name() << ")" << std::endl;
        }
    }

    if (policy.kind == BackendPolicyKind::auto_select) {
        for (const DeviceEntry & entry : inventory) {
            if (entry.info.type != GGML_BACKEND_DEVICE_TYPE_ACCEL) continue;
            ggml_backend_t backend = ggml_backend_dev_init(entry.device, nullptr);
            if (!backend) {
                std::cerr << "pixal3d: ACCEL backend " << entry.info.description
                          << " failed to initialize; skipping" << std::endl;
                continue;
            }
            register_backend(entry, backend, false);
            std::cerr << "pixal3d: using " << entry.info.description
                      << " backend (ACCEL)" << std::endl;
        }
    }

    if (cpu_entry == no_entry) {
        set_error(error, "ggml CPU backend is not registered");
        close();
        return false;
    }
    const DeviceEntry & cpu = inventory[cpu_entry];
    ggml_backend_t cpu_backend = ggml_backend_dev_init(cpu.device, nullptr);
    if (!cpu_backend) {
        set_error(error, "failed to initialize ggml CPU backend");
        close();
        return false;
    }
    register_backend(cpu, cpu_backend, false);
    configure_cpu_threadpool(GGML_DEFAULT_N_THREADS);
    initialized_ = true;
    std::cerr << "pixal3d: backend manager ready (policy=" << policy.name()
              << ", backends=" << backends_.size()
              << ", cpu_threads=" << cpu_threadpool_n_threads_ << ")" << std::endl;
    return true;
}

ggml_backend_t BackendManager::primary() const noexcept {
    return backends_.empty() ? nullptr : backends_.front();
}

bool BackendManager::primary_supports_op(const ggml_tensor * op,
                                         std::string * error) const {
    if (!initialized_ || !primary()) {
        set_error(error, "backend manager is not initialized");
        return false;
    }
    if (!op) {
        set_error(error, "cannot probe a null ggml operation");
        return false;
    }
    if (ggml_backend_supports_op(primary(), op)) return true;
    const char * backend = ggml_backend_name(primary());
    const char * operation = ggml_op_desc(op);
    const char * tensor_name = ggml_get_name(op);
    set_error(error, "backend " + std::string(backend ? backend : "unknown") +
                     " does not support operation " +
                     std::string(operation ? operation : "unknown") +
                     " for tensor " +
                     std::string(tensor_name && *tensor_name ? tensor_name : "<unnamed>"));
    return false;
}

ggml_backend_t BackendManager::backend_for_op(const ggml_tensor * op) const noexcept {
    if (!op) return nullptr;
    for (ggml_backend_t backend : backends_) {
        if (backend && ggml_backend_supports_op(backend, op)) return backend;
    }
    return nullptr;
}

bool BackendManager::supports_op(const ggml_tensor * op,
                                 std::string * error) const {
    if (!initialized_) {
        set_error(error, "backend manager is not initialized");
        return false;
    }
    if (!op) {
        set_error(error, "cannot probe a null ggml operation");
        return false;
    }
    if (backend_for_op(op)) return true;
    set_error(error, "no configured backend supports ggml operation " +
                     std::to_string(static_cast<int>(op->op)));
    return false;
}

void BackendManager::log_buffer(const char * phase,
                                ggml_backend_t backend,
                                std::size_t buffer_bytes) const noexcept {
    if (!trace_enabled() || !backend) return;
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (device) ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
    const char * name = ggml_backend_name(backend);
    std::cerr << "pixal3d: backend memory phase="
              << (phase && *phase ? phase : "snapshot")
              << " backend=" << (name && *name ? name : "unknown")
              << " buffer=" << bytes_string(buffer_bytes)
              << " free=" << bytes_string(free_bytes)
              << " total=" << bytes_string(total_bytes)
              << std::endl;
}

bool BackendManager::configure_cpu_threadpool(int n_threads) const noexcept {
    if (n_threads <= 0) return false;

    if (cpu_threadpool_ && cpu_threadpool_n_threads_ == n_threads) {
        ggml_threadpool_resume(cpu_threadpool_);
        return true;
    }

    struct ggml_threadpool_params params = ggml_threadpool_params_default(n_threads);
    ggml_threadpool * replacement = ggml_threadpool_new(&params);
    if (!replacement) {
        std::cerr << "pixal3d: failed to create CPU ggml threadpool"
                  << " (threads=" << n_threads
                  << "); using the backend's disposable pool" << std::endl;
        return false;
    }

    ggml_threadpool * previous = cpu_threadpool_;
    cpu_threadpool_ = replacement;
    cpu_threadpool_n_threads_ = n_threads;
    for (ggml_backend_t backend : backends_) {
        if (backend && ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_threadpool(backend, cpu_threadpool_);
        }
    }
    if (previous) ggml_threadpool_free(previous);

    if (trace_enabled()) {
        std::cerr << "pixal3d: CPU ggml threadpool configured"
                  << " threads=" << n_threads << std::endl;
    }
    return true;
}

void BackendManager::set_n_threads(int n_threads) const noexcept {
    if (n_threads <= 0) return;
    for (ggml_backend_t backend : backends_) {
        if (!backend) continue;
        ggml_backend_dev_t device = ggml_backend_get_device(backend);
        ggml_backend_reg_t registry = device ? ggml_backend_dev_backend_reg(device) : nullptr;
        if (!registry) continue;
        auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(registry, "ggml_backend_set_n_threads"));
        if (set_threads) set_threads(backend, n_threads);
    }
    configure_cpu_threadpool(n_threads);
}

BackendScheduler::BackendScheduler(const BackendManager & manager,
                                   std::size_t graph_size,
                                   bool parallel,
                                   bool op_offload,
                                   std::string * error,
                                   const char * stage)
    : manager_(&manager), stage_(stage ? stage : "") {
    if (!manager.initialized() || manager.backends().empty()) {
        set_error(error, "cannot create scheduler from an uninitialized backend manager");
        return;
    }
    if (graph_size == 0) {
        set_error(error, "backend scheduler graph size must be non-zero");
        return;
    }
    // ggml_backend_sched_new takes a mutable pointer for historical API
    // reasons but does not mutate the backend array.
    auto & backends = const_cast<std::vector<ggml_backend_t> &>(manager.backends());
    scheduler_ = ggml_backend_sched_new(backends.data(), nullptr,
                                        static_cast<int>(backends.size()),
                                        graph_size, parallel, op_offload);
    if (!scheduler_) set_error(error, "failed to initialize ggml backend scheduler");
}

BackendScheduler::~BackendScheduler() {
    if (scheduler_) ggml_backend_sched_free(scheduler_);
}

BackendScheduler::BackendScheduler(BackendScheduler && other) noexcept
    : scheduler_(other.scheduler_), manager_(other.manager_), stage_(std::move(other.stage_)),
      required_nodes_(std::move(other.required_nodes_)) {
    other.scheduler_ = nullptr;
    other.manager_ = nullptr;
    other.required_nodes_.clear();
}

BackendScheduler & BackendScheduler::operator=(BackendScheduler && other) noexcept {
    if (this == &other) return *this;
    if (scheduler_) ggml_backend_sched_free(scheduler_);
    scheduler_ = other.scheduler_;
    manager_ = other.manager_;
    stage_ = std::move(other.stage_);
    required_nodes_ = std::move(other.required_nodes_);
    other.scheduler_ = nullptr;
    other.manager_ = nullptr;
    other.required_nodes_.clear();
    return *this;
}

bool BackendScheduler::require_primary(ggml_tensor * node, std::string * error) {
    if (!scheduler_ || !manager_ || !manager_->initialized()) {
        set_error(error, "backend scheduler is not initialized");
        return false;
    }
    if (!node) {
        set_error(error, "cannot require a null tensor on the primary backend");
        return false;
    }
    if (std::find(required_nodes_.begin(), required_nodes_.end(), node) ==
        required_nodes_.end()) {
        required_nodes_.push_back(node);
    }
    return true;
}

bool BackendScheduler::require_primary_graph(ggml_cgraph * graph, std::string * error) {
    if (!graph) {
        set_error(error, "cannot require a null graph on the primary backend");
        return false;
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (!node || node->op == GGML_OP_NONE || !is_gpu_core_op(node->op)) continue;
        if (!require_primary(node, error)) return false;
    }
    return true;
}

bool BackendScheduler::preflight(ggml_cgraph * graph, std::string * error) const {
    if (!manager_ || !manager_->initialized()) {
        set_error(error, "backend scheduler has no initialized manager");
        return false;
    }
    if (!graph) {
        set_error(error, "cannot preflight a null ggml graph");
        return false;
    }
    if (!manager_->requires_primary_backend()) return true;
    for (ggml_tensor * node : required_nodes_) {
        std::string node_error;
        if (!manager_->primary_supports_op(node, &node_error)) {
            const std::string prefix = stage_.empty() ? "backend graph" : stage_;
            set_error(error, prefix + ": required node: " + node_error);
            return false;
        }
    }
    return true;
}

bool BackendScheduler::validate_placement(ggml_cgraph * graph,
                                           std::string * error) const {
    if (!manager_ || !manager_->initialized() || !scheduler_) {
        set_error(error, "backend scheduler is not initialized");
        return false;
    }
    if (!graph) {
        set_error(error, "cannot validate placement for a null ggml graph");
        return false;
    }
    if (!manager_->requires_primary_backend()) return true;
    const ggml_backend_t primary = manager_->primary();
    if (!primary) {
        set_error(error, "backend scheduler has no primary backend");
        return false;
    }
    for (ggml_tensor * node : required_nodes_) {
        if (!node) continue;
        ggml_backend_t placed = ggml_backend_sched_get_tensor_backend(scheduler_, node);
        if (placed != primary) {
            const char * operation = ggml_op_desc(node);
            const char * tensor_name = ggml_get_name(node);
            const char * placed_name = placed ? ggml_backend_name(placed) : "unassigned";
            const std::string stage = stage_.empty() ? "backend graph" : stage_;
            set_error(error, stage + ": required tensor " +
                             std::string(tensor_name && *tensor_name ? tensor_name : "<unnamed>") +
                             " operation " +
                             std::string(operation ? operation : "unknown") +
                             " was placed on " + (placed_name ? placed_name : "unknown") +
                             ", expected " +
                             (ggml_backend_name(primary) ? ggml_backend_name(primary) : "primary"));
            return false;
        }
    }
    return true;
}

bool BackendScheduler::allocate_graph(ggml_cgraph * graph, std::string * error) {
    if (!scheduler_) {
        set_error(error, "backend scheduler is not initialized");
        return false;
    }
    if (!graph) {
        set_error(error, "cannot allocate a null ggml graph");
        return false;
    }
    if (!preflight(graph, error)) return false;
    for (ggml_tensor * node : required_nodes_) {
        ggml_backend_sched_set_tensor_backend(scheduler_, node, manager_->primary());
    }
    const double allocation_start = backend_time_now_ms();
    if (!ggml_backend_sched_alloc_graph(scheduler_, graph)) {
        backend_log_timing(stage_.c_str(), "scheduler_allocate",
                           backend_time_now_ms() - allocation_start);
        set_error(error, "failed to allocate ggml graph on configured backends");
        return false;
    }
    backend_log_timing(stage_.c_str(), "scheduler_allocate",
                       backend_time_now_ms() - allocation_start);
    if (!validate_placement(graph, error)) return false;
    log_trace(graph, "allocated");
    return true;
}

ggml_status BackendScheduler::compute(ggml_cgraph * graph, std::string * error) {
    if (!scheduler_) {
        set_error(error, "backend scheduler is not initialized");
        return GGML_STATUS_FAILED;
    }
    if (!graph) {
        set_error(error, "cannot compute a null ggml graph");
        return GGML_STATUS_FAILED;
    }
    if (!preflight(graph, error) || !validate_placement(graph, error)) {
        return GGML_STATUS_FAILED;
    }
    const double compute_start = backend_time_now_ms();
    const ggml_status status = ggml_backend_sched_graph_compute(scheduler_, graph);
    backend_log_timing(stage_.c_str(), "scheduler_compute",
                       backend_time_now_ms() - compute_start);
    if (status != GGML_STATUS_SUCCESS) {
        set_error(error, "ggml backend scheduler graph compute failed");
    }
    log_trace(graph, status == GGML_STATUS_SUCCESS ? "computed" : "compute_failed");
    return status;
}

void BackendScheduler::set_eval_callback(ggml_backend_sched_eval_callback callback,
                                          void * user_data) noexcept {
    if (scheduler_) {
        ggml_backend_sched_set_eval_callback(scheduler_, callback, user_data);
    }
}

void BackendScheduler::reset() noexcept {
    if (scheduler_) ggml_backend_sched_reset(scheduler_);
}

void BackendScheduler::synchronize() noexcept {
    if (scheduler_) ggml_backend_sched_synchronize(scheduler_);
}

std::size_t BackendScheduler::buffer_size(ggml_backend_t backend) const noexcept {
    return scheduler_ && backend ? ggml_backend_sched_get_buffer_size(scheduler_, backend) : 0;
}

ggml_backend_t BackendScheduler::tensor_backend(ggml_tensor * node) const noexcept {
    return scheduler_ && node ? ggml_backend_sched_get_tensor_backend(scheduler_, node) : nullptr;
}

void BackendScheduler::log_trace(ggml_cgraph * graph,
                                  const char * phase) const noexcept {
    if (!trace_enabled() || !scheduler_ || !manager_ || !graph) return;
    const std::string prefix = stage_.empty() ? "backend" : stage_;
    std::cerr << "pixal3d: scheduler stage=" << prefix
              << " phase=" << (phase && *phase ? phase : "snapshot")
              << " nodes=" << ggml_graph_n_nodes(graph)
              << " splits=" << ggml_backend_sched_get_n_splits(scheduler_)
              << " copies=" << ggml_backend_sched_get_n_copies(scheduler_)
              << std::endl;
    for (std::size_t index = 0; index < manager_->backends().size(); ++index) {
        ggml_backend_t backend = manager_->backends()[index];
        const char * name = backend ? ggml_backend_name(backend) : nullptr;
        const std::size_t bytes = buffer_size(backend);
        std::cerr << "pixal3d: scheduler backend="
                  << (name && *name ? name : "unknown")
                  << " buffer=" << bytes_string(bytes) << std::endl;
        manager_->log_buffer(phase, backend, bytes);
    }
    for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
        ggml_tensor * node = ggml_graph_node(graph, index);
        if (!node || node->op == GGML_OP_NONE) continue;
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(scheduler_, node);
        const char * backend_name = backend ? ggml_backend_name(backend) : nullptr;
        const char * tensor_name = ggml_get_name(node);
        std::cerr << "pixal3d: scheduler node=" << index
                  << " op=" << ggml_op_name(node->op)
                  << " name=" << (tensor_name && *tensor_name ? tensor_name : "<unnamed>")
                  << " backend=" << (backend_name && *backend_name ? backend_name : "unassigned")
                  << std::endl;
    }
}

} // namespace pixal3d
