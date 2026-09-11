#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pixal3d {
namespace detail {

struct SparseValidationProfileRecord {
    std::string label;
    std::size_t points = 0;
    int channels = 0;
    double elapsed_ms = 0.0;
    double coord_hashmap_ms = 0.0;
    double feature_scan_ms = 0.0;
    bool success = false;
};

struct SparseLinearProfileRecord {
    std::string label;
    std::size_t points = 0;
    int input_channels = 0;
    int output_channels = 0;
    double total_ms = 0.0;
    double input_validation_ms = 0.0;
    double copy_shape_ms = 0.0;
    double output_allocation_ms = 0.0;
    double actual_compute_ms = 0.0;
    double bias_ms = 0.0;
    double output_validation_ms = 0.0;
    double other_ms = 0.0;
    int omp_max_threads = 1;
    int omp_actual_threads = 1;
    bool profile_split_bias = false;
    bool success = false;
};

// This is intentionally an internal, thread-local diagnostic surface.  The
// sparse kernels remain usable without a profiler, and a profiling scope only
// adds work when a caller explicitly enables it.
struct SparseProfileStats {
    std::uint64_t valid_calls = 0;
    double valid_ms = 0.0;
    double valid_coord_hashmap_ms = 0.0;
    double valid_feature_scan_ms = 0.0;
    std::vector<SparseValidationProfileRecord> validation_records;

    std::uint64_t copy_shape_calls = 0;
    std::uint64_t copy_shape_bytes = 0;
    double copy_shape_ms = 0.0;

    std::uint64_t sparse_linear_calls = 0;
    double sparse_linear_ms = 0.0;
    double sparse_linear_input_validation_ms = 0.0;
    double sparse_linear_copy_shape_ms = 0.0;
    double sparse_linear_output_allocation_ms = 0.0;
    double sparse_linear_actual_compute_ms = 0.0;
    double sparse_linear_bias_ms = 0.0;
    double sparse_linear_output_validation_ms = 0.0;
    double sparse_linear_other_ms = 0.0;
    std::vector<SparseLinearProfileRecord> sparse_linear_records;

    std::uint64_t sparse_layer_norm_calls = 0;
    double sparse_layer_norm_ms = 0.0;
    std::uint64_t sparse_channel_to_spatial_calls = 0;
    double sparse_channel_to_spatial_ms = 0.0;
    std::uint64_t sparse_submanifold_conv3d_calls = 0;
    double sparse_submanifold_conv3d_ms = 0.0;

    std::uint64_t add_residual_calls = 0;
    double add_residual_ms = 0.0;
    std::uint64_t silu_calls = 0;
    double silu_ms = 0.0;
    std::uint64_t repeat_channels_calls = 0;
    double repeat_channels_ms = 0.0;
};

inline thread_local SparseProfileStats * active_sparse_profile = nullptr;
inline thread_local const char * active_sparse_linear_label = nullptr;
inline thread_local const char * active_sparse_validation_label = nullptr;

class SparseProfileScope {
public:
    explicit SparseProfileScope(SparseProfileStats * stats) noexcept
        : previous_(active_sparse_profile) {
        active_sparse_profile = stats;
    }

    ~SparseProfileScope() {
        active_sparse_profile = previous_;
    }

    SparseProfileScope(const SparseProfileScope &) = delete;
    SparseProfileScope & operator=(const SparseProfileScope &) = delete;

private:
    SparseProfileStats * previous_ = nullptr;
};

class SparseLinearProfileLabelScope {
public:
    explicit SparseLinearProfileLabelScope(const char * label) noexcept
        : previous_(active_sparse_linear_label) {
        active_sparse_linear_label = label;
    }

    ~SparseLinearProfileLabelScope() {
        active_sparse_linear_label = previous_;
    }

    SparseLinearProfileLabelScope(const SparseLinearProfileLabelScope &) = delete;
    SparseLinearProfileLabelScope & operator=(const SparseLinearProfileLabelScope &) = delete;

private:
    const char * previous_ = nullptr;
};

class SparseValidationProfileLabelScope {
public:
    explicit SparseValidationProfileLabelScope(const char * label) noexcept
        : previous_(active_sparse_validation_label) {
        active_sparse_validation_label = label;
    }

    ~SparseValidationProfileLabelScope() {
        active_sparse_validation_label = previous_;
    }

    SparseValidationProfileLabelScope(const SparseValidationProfileLabelScope &) = delete;
    SparseValidationProfileLabelScope & operator=(const SparseValidationProfileLabelScope &) = delete;

private:
    const char * previous_ = nullptr;
};

class SparseProfileTimer {
public:
    SparseProfileTimer() noexcept = default;

    SparseProfileTimer(std::uint64_t * calls, double * elapsed_ms) noexcept
        : calls_(calls), elapsed_ms_(elapsed_ms), start_(clock::now()) {}

    SparseProfileTimer(const SparseProfileTimer &) = delete;
    SparseProfileTimer & operator=(const SparseProfileTimer &) = delete;

    SparseProfileTimer(SparseProfileTimer && other) noexcept
        : calls_(other.calls_), elapsed_ms_(other.elapsed_ms_), start_(other.start_) {
        other.calls_ = nullptr;
        other.elapsed_ms_ = nullptr;
    }

    SparseProfileTimer & operator=(SparseProfileTimer && other) noexcept {
        if (this != &other) {
            stop();
            calls_ = other.calls_;
            elapsed_ms_ = other.elapsed_ms_;
            start_ = other.start_;
            other.calls_ = nullptr;
            other.elapsed_ms_ = nullptr;
        }
        return *this;
    }

    ~SparseProfileTimer() {
        stop();
    }

    void stop() noexcept {
        if (!calls_ || !elapsed_ms_) return;
        ++(*calls_);
        *elapsed_ms_ += std::chrono::duration<double, std::milli>(
            clock::now() - start_).count();
        calls_ = nullptr;
        elapsed_ms_ = nullptr;
    }

private:
    using clock = std::chrono::steady_clock;
    std::uint64_t * calls_ = nullptr;
    double * elapsed_ms_ = nullptr;
    clock::time_point start_{};
};

inline SparseProfileTimer make_sparse_profile_timer(
    std::uint64_t SparseProfileStats::* calls,
    double SparseProfileStats::* elapsed_ms) noexcept {
    SparseProfileStats * stats = active_sparse_profile;
    if (!stats) return SparseProfileTimer{};
    return SparseProfileTimer(&(stats->*calls), &(stats->*elapsed_ms));
}

inline SparseProfileStats * sparse_profile_current() noexcept {
    return active_sparse_profile;
}

inline const char * sparse_linear_profile_label() noexcept {
    return active_sparse_linear_label;
}

inline const char * sparse_validation_profile_label() noexcept {
    return active_sparse_validation_label;
}

} // namespace detail
} // namespace pixal3d
