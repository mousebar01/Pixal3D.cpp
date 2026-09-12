#include "pixal3d/cpu_threads.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <thread>

namespace pixal3d {
namespace {

std::atomic<int> g_cpu_thread_override{0};

const char * skip_space(const char * value) noexcept {
    if (!value) return nullptr;
    while (*value && std::isspace(static_cast<unsigned char>(*value))) ++value;
    return value;
}

bool parse_env_count(const char * value, bool allow_openmp_list, int & threads) noexcept {
    const char * begin = skip_space(value);
    if (!begin || !*begin) return false;

    errno = 0;
    char * end = nullptr;
    const long parsed = std::strtol(begin, &end, 10);
    if (errno == ERANGE || end == begin || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max()) return false;

    while (*end && std::isspace(static_cast<unsigned char>(*end))) ++end;
    if (*end != '\0' && !(allow_openmp_list && *end == ',')) return false;
    threads = static_cast<int>(parsed);
    return true;
}

int resolve_cpu_thread_count() noexcept {
    const int override = g_cpu_thread_override.load(std::memory_order_relaxed);
    if (override > 0) return override;

    int threads = 0;
    if (parse_env_count(std::getenv("PIXAL3D_CPU_THREADS"), false, threads)) {
        return threads;
    }
    if (parse_env_count(std::getenv("OMP_NUM_THREADS"), true, threads)) {
        return threads;
    }
    return default_cpu_thread_count();
}

} // namespace

bool parse_cpu_thread_count(const std::string & value,
                            int & threads,
                            std::string * error) {
    if (error) error->clear();
    const char * begin = skip_space(value.c_str());
    if (!begin || !*begin) {
        if (error) *error = "CPU thread count must be a positive integer";
        return false;
    }

    errno = 0;
    char * end = nullptr;
    const long parsed = std::strtol(begin, &end, 10);
    while (end && *end && std::isspace(static_cast<unsigned char>(*end))) ++end;
    if (errno == ERANGE || end == begin || !end || *end != '\0' ||
        parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        if (error) *error = "CPU thread count must be a positive integer: " + value;
        return false;
    }
    threads = static_cast<int>(parsed);
    return true;
}

CpuThreadPolicy cpu_thread_policy_from_environment() {
    CpuThreadPolicy policy;
    const int override = g_cpu_thread_override.load(std::memory_order_relaxed);
    if (override > 0) {
        policy.threads = override;
        policy.source = "override";
        return policy;
    }

    int threads = 0;
    const char * configured = std::getenv("PIXAL3D_CPU_THREADS");
    if (configured && *configured) {
        if (parse_env_count(configured, false, threads)) {
            policy.threads = threads;
            policy.source = "PIXAL3D_CPU_THREADS";
            return policy;
        }
        policy.warning = "invalid PIXAL3D_CPU_THREADS='" + std::string(configured) +
                         "'; falling back to OMP_NUM_THREADS or the project default";
    }

    const char * openmp = std::getenv("OMP_NUM_THREADS");
    if (parse_env_count(openmp, true, threads)) {
        policy.threads = threads;
        policy.source = "OMP_NUM_THREADS";
        return policy;
    }

    policy.threads = default_cpu_thread_count();
    policy.source = "default";
    return policy;
}

int default_cpu_thread_count() noexcept {
    const unsigned int hardware = std::thread::hardware_concurrency();
    if (hardware == 0) return 1;
    return std::min(4u, hardware);
}

int cpu_thread_count() noexcept {
    return resolve_cpu_thread_count();
}

bool set_cpu_thread_override(int threads, std::string * error) {
    if (error) error->clear();
    if (threads <= 0) {
        if (error) *error = "CPU thread count must be positive";
        return false;
    }
    g_cpu_thread_override.store(threads, std::memory_order_relaxed);
    return true;
}

void clear_cpu_thread_override() noexcept {
    g_cpu_thread_override.store(0, std::memory_order_relaxed);
}

bool cpu_should_parallelize(std::size_t work_items,
                            std::size_t min_items_per_thread) noexcept {
    const int threads = cpu_thread_count();
    if (threads <= 1 || work_items == 0 || min_items_per_thread == 0) return threads > 1 && work_items > 0;
    return work_items / static_cast<std::size_t>(threads) >= min_items_per_thread;
}

} // namespace pixal3d
