#include "pixal3d/sparse.h"
#include "pixal3d/cpu_threads.h"

#include "sparse_profile.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>

#if defined(_OPENMP)
    #include <omp.h>
#endif

namespace pixal3d {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

struct Coord {
    std::int32_t batch = 0;
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::int32_t z = 0;

    bool operator==(const Coord & other) const noexcept {
        return batch == other.batch && x == other.x && y == other.y && z == other.z;
    }
};

struct CoordHash {
    std::size_t operator()(const Coord & value) const noexcept {
        std::size_t result = static_cast<std::size_t>(value.batch);
        result = result * 1000003u + static_cast<std::size_t>(value.x);
        result = result * 1000003u + static_cast<std::size_t>(value.y);
        result = result * 1000003u + static_cast<std::size_t>(value.z);
        return result;
    }
};

bool checked_product(std::size_t a, std::size_t b, std::size_t & result) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) return false;
    result = a * b;
    return true;
}

bool check_output_channels(int channels, std::string * error) {
    if (channels <= 0) {
        set_error(error, "sparse output channel count must be positive");
        return false;
    }
    return true;
}

bool check_weights(const SparseTensorF32 & input, int out_channels,
                   const float * weight, const float * bias,
                   std::string * error) {
    if (!input.valid(error) || !check_output_channels(out_channels, error)) return false;
    if (!weight) {
        set_error(error, "sparse weight pointer is null");
        return false;
    }
    if (!bias) {
        set_error(error, "sparse bias pointer is null");
        return false;
    }
    return true;
}

double sparse_profile_elapsed_ms(
    const std::chrono::steady_clock::time_point begin) noexcept {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
}

class SparseValidationProfileRecordScope {
public:
    SparseValidationProfileRecordScope(const SparseTensorF32 & input,
                                       detail::SparseProfileStats * stats)
        : input_(input), stats_(stats), begin_(stats ? clock::now() : clock::time_point{}) {
        if (stats_) {
            const char * label = detail::sparse_validation_profile_label();
            label_ = label && *label ? label : "unlabelled";
        }
    }

    ~SparseValidationProfileRecordScope() {
        if (!stats_) return;
        detail::SparseValidationProfileRecord record;
        record.label = label_;
        record.points = input_.points();
        record.channels = input_.channels;
        record.elapsed_ms = sparse_profile_elapsed_ms(begin_);
        record.coord_hashmap_ms = coord_hashmap_ms_;
        record.feature_scan_ms = feature_scan_ms_;
        record.success = success_;
        stats_->validation_records.push_back(std::move(record));
    }

    void set_coord_hashmap_ms(double elapsed_ms) noexcept {
        coord_hashmap_ms_ = elapsed_ms;
    }

    void set_feature_scan_ms(double elapsed_ms) noexcept {
        feature_scan_ms_ = elapsed_ms;
    }

    void mark_success() noexcept { success_ = true; }

    SparseValidationProfileRecordScope(const SparseValidationProfileRecordScope &) = delete;
    SparseValidationProfileRecordScope & operator=(
        const SparseValidationProfileRecordScope &) = delete;

private:
    using clock = std::chrono::steady_clock;

    const SparseTensorF32 & input_;
    detail::SparseProfileStats * stats_ = nullptr;
    clock::time_point begin_{};
    std::string label_;
    double coord_hashmap_ms_ = 0.0;
    double feature_scan_ms_ = 0.0;
    bool success_ = false;
};

class SparseLinearProfileRecordScope {
public:
    SparseLinearProfileRecordScope(const SparseTensorF32 & input,
                                   int output_channels,
                                   detail::SparseProfileStats * stats)
        : stats_(stats), begin_(stats ? clock::now() : clock::time_point{}) {
        record_.points = input.points();
        record_.input_channels = input.channels;
        record_.output_channels = output_channels;
        if (stats_) {
            const char * label = detail::sparse_linear_profile_label();
            record_.label = label && *label ? label : "unlabelled";
#if defined(_OPENMP)
            record_.omp_max_threads = omp_get_max_threads();
#endif
        }
    }

    ~SparseLinearProfileRecordScope() {
        if (!stats_) return;
        record_.total_ms = sparse_profile_elapsed_ms(begin_);
        const double accounted = record_.input_validation_ms + record_.copy_shape_ms +
                                 record_.output_allocation_ms + record_.actual_compute_ms +
                                 record_.bias_ms + record_.output_validation_ms;
        record_.other_ms = std::max(0.0, record_.total_ms - accounted);
        stats_->sparse_linear_input_validation_ms += record_.input_validation_ms;
        stats_->sparse_linear_copy_shape_ms += record_.copy_shape_ms;
        stats_->sparse_linear_output_allocation_ms += record_.output_allocation_ms;
        stats_->sparse_linear_actual_compute_ms += record_.actual_compute_ms;
        stats_->sparse_linear_bias_ms += record_.bias_ms;
        stats_->sparse_linear_output_validation_ms += record_.output_validation_ms;
        stats_->sparse_linear_other_ms += record_.other_ms;
        stats_->sparse_linear_records.push_back(std::move(record_));
    }

    detail::SparseLinearProfileRecord & record() noexcept { return record_; }

    void mark_success() noexcept { record_.success = true; }

    SparseLinearProfileRecordScope(const SparseLinearProfileRecordScope &) = delete;
    SparseLinearProfileRecordScope & operator=(const SparseLinearProfileRecordScope &) = delete;

private:
    using clock = std::chrono::steady_clock;

    detail::SparseProfileStats * stats_ = nullptr;
    clock::time_point begin_{};
    detail::SparseLinearProfileRecord record_;
};

bool validate_sparse_tensor(const SparseTensorF32 & input,
                            const char * operation,
                            std::string * error) {
    detail::SparseProfileStats * stats = detail::sparse_profile_current();
    if (!stats) return input.valid(error);

    std::string label = operation && *operation ? operation : "unlabelled";
    if (const char * linear_label = detail::sparse_linear_profile_label();
        linear_label && *linear_label) {
        label += "[";
        label += linear_label;
        label += "]";
    }
    detail::SparseValidationProfileLabelScope label_scope(label.c_str());
    return input.valid(error);
}

void copy_shape(const SparseTensorF32 & input, int channels, SparseTensorF32 & output) {
    auto timer = detail::make_sparse_profile_timer(
        &detail::SparseProfileStats::copy_shape_calls,
        &detail::SparseProfileStats::copy_shape_ms);
    if (detail::SparseProfileStats * stats = detail::sparse_profile_current()) {
        stats->copy_shape_bytes += static_cast<std::uint64_t>(
            input.coords.size() * sizeof(std::int32_t));
    }
    output.batch_size = input.batch_size;
    output.channels = channels;
    output.spatial_x = input.spatial_x;
    output.spatial_y = input.spatial_y;
    output.spatial_z = input.spatial_z;
    output.coords = input.coords;
}

} // namespace

bool VarLenTensorF32::valid(std::string * error) const {
    if (batch_size <= 0 || channels <= 0 || offsets.size() !=
        static_cast<std::size_t>(batch_size + 1) || offsets.front() != 0) {
        set_error(error, "invalid variable-length tensor shape");
        return false;
    }
    for (std::size_t index = 1; index < offsets.size(); ++index) {
        if (offsets[index] < offsets[index - 1]) {
            set_error(error, "variable-length tensor offsets are not monotonic");
            return false;
        }
    }
    std::size_t expected = 0;
    if (!checked_product(offsets.back(), static_cast<std::size_t>(channels), expected) ||
        expected != feats.size()) {
        set_error(error, "variable-length tensor feature count does not match offsets");
        return false;
    }
    for (float value : feats) {
        if (!std::isfinite(value)) {
            set_error(error, "variable-length tensor contains a non-finite value");
            return false;
        }
    }
    return true;
}

bool SparseTensorF32::valid(std::string * error) const {
    auto timer = detail::make_sparse_profile_timer(
        &detail::SparseProfileStats::valid_calls,
        &detail::SparseProfileStats::valid_ms);
    detail::SparseProfileStats * stats = detail::sparse_profile_current();
    SparseValidationProfileRecordScope profile_record(*this, stats);
    if (batch_size <= 0 || channels <= 0 || spatial_x <= 0 ||
        spatial_y <= 0 || spatial_z <= 0) {
        set_error(error, "invalid sparse tensor shape");
        return false;
    }
    if (coords.size() % 4 != 0) {
        set_error(error, "sparse coordinates must have four columns");
        return false;
    }
    std::size_t expected = 0;
    if (!checked_product(points(), static_cast<std::size_t>(channels), expected) ||
        feats.size() != expected) {
        set_error(error, "sparse feature count does not match coordinates");
        return false;
    }

    if (stats) {
        // In profiling mode use two coarse timers rather than one clock read
        // per point.  The normal path below keeps the original validation
        // order and has no profiling branches in its hot loop.
        const auto coord_begin = std::chrono::steady_clock::now();
        std::unordered_map<Coord, std::size_t, CoordHash> seen;
        seen.reserve(points());
        for (std::size_t point = 0; point < points(); ++point) {
            const Coord coord{coords[point * 4 + 0], coords[point * 4 + 1],
                              coords[point * 4 + 2], coords[point * 4 + 3]};
            if (coord.batch < 0 || coord.batch >= batch_size || coord.x < 0 ||
                coord.x >= spatial_x || coord.y < 0 || coord.y >= spatial_y ||
                coord.z < 0 || coord.z >= spatial_z) {
                set_error(error, "sparse coordinate is outside its shape");
                return false;
            }
            if (!seen.emplace(coord, point).second) {
                set_error(error, "sparse coordinates contain a duplicate point");
                return false;
            }
            if (point > 0 && coords[(point - 1) * 4] > coord.batch) {
                set_error(error, "sparse coordinates must be contiguous by batch");
                return false;
            }
        }
        const double coord_elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - coord_begin).count();
        stats->valid_coord_hashmap_ms += coord_elapsed;
        profile_record.set_coord_hashmap_ms(coord_elapsed);
        const auto feature_begin = std::chrono::steady_clock::now();
        for (float value : feats) {
            if (!std::isfinite(value)) {
                set_error(error, "sparse features contain a non-finite value");
                return false;
            }
        }
        const double feature_elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - feature_begin).count();
        stats->valid_feature_scan_ms += feature_elapsed;
        profile_record.set_feature_scan_ms(feature_elapsed);
        profile_record.mark_success();
        return true;
    }

    std::unordered_map<Coord, std::size_t, CoordHash> seen;
    seen.reserve(points());
    for (std::size_t point = 0; point < points(); ++point) {
        const Coord coord{coords[point * 4 + 0], coords[point * 4 + 1],
                          coords[point * 4 + 2], coords[point * 4 + 3]};
        if (coord.batch < 0 || coord.batch >= batch_size || coord.x < 0 ||
            coord.x >= spatial_x || coord.y < 0 || coord.y >= spatial_y ||
            coord.z < 0 || coord.z >= spatial_z) {
            set_error(error, "sparse coordinate is outside its shape");
            return false;
        }
        if (!seen.emplace(coord, point).second) {
            set_error(error, "sparse coordinates contain a duplicate point");
            return false;
        }
        if (point > 0 && coords[(point - 1) * 4] > coord.batch) {
            set_error(error, "sparse coordinates must be contiguous by batch");
            return false;
        }
        for (int channel = 0; channel < channels; ++channel) {
            if (!std::isfinite(feats[point * static_cast<std::size_t>(channels) +
                                      static_cast<std::size_t>(channel)])) {
                set_error(error, "sparse features contain a non-finite value");
                return false;
            }
        }
    }
    return true;
}

bool sparse_linear(const SparseTensorF32 & input,
                  const float * weight,
                  const float * bias,
                  int out_channels,
                  SparseTensorF32 & output,
                  std::string * error) {
    auto timer = detail::make_sparse_profile_timer(
        &detail::SparseProfileStats::sparse_linear_calls,
        &detail::SparseProfileStats::sparse_linear_ms);

    detail::SparseProfileStats * stats = detail::sparse_profile_current();
    SparseLinearProfileRecordScope profile_record(input, out_channels, stats);

    const auto input_validation_begin = stats
        ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool input_valid = validate_sparse_tensor(input, "sparse_linear.input", error);
    if (stats) {
        profile_record.record().input_validation_ms = sparse_profile_elapsed_ms(
            input_validation_begin);
    }
    if (!input_valid) return false;
    if (!check_output_channels(out_channels, error)) return false;
    if (!weight) {
        set_error(error, "sparse weight pointer is null");
        return false;
    }
    if (!bias) {
        set_error(error, "sparse bias pointer is null");
        return false;
    }

    const auto copy_shape_begin = stats
        ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    copy_shape(input, out_channels, output);
    if (stats) {
        profile_record.record().copy_shape_ms = sparse_profile_elapsed_ms(copy_shape_begin);
    }

    const auto output_allocation_begin = stats
        ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::size_t count = 0;
    if (!checked_product(input.points(), static_cast<std::size_t>(out_channels), count)) {
        set_error(error, "sparse linear output size overflows size_t");
        return false;
    }
    output.feats.assign(count, 0.0f);
    if (stats) {
        profile_record.record().output_allocation_ms = sparse_profile_elapsed_ms(
            output_allocation_begin);

        // Profiling-only decomposition: keep the release path's fused
        // bias-plus-dot-product loop below, but split the same arithmetic into
        // two passes here so bias initialization and multiply/add work can be
        // measured independently.  The operation order is unchanged for each
        // output value: bias, then input channels in ascending order.
        int actual_threads = 1;
        const auto bias_begin = std::chrono::steady_clock::now();
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) num_threads(cpu_thread_count()) if(cpu_should_parallelize(output.feats.size()))
#endif
        for (std::size_t point = 0; point < input.points(); ++point) {
#if defined(_OPENMP)
            if (point == 0) actual_threads = omp_get_num_threads();
#endif
            float * destination = output.feats.data() +
                point * static_cast<std::size_t>(out_channels);
            for (int out = 0; out < out_channels; ++out) {
                destination[out] = bias[out];
            }
        }
        profile_record.record().bias_ms = sparse_profile_elapsed_ms(bias_begin);

        const auto compute_begin = std::chrono::steady_clock::now();
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) num_threads(cpu_thread_count()) if(cpu_should_parallelize(output.feats.size()))
#endif
        for (std::size_t point = 0; point < input.points(); ++point) {
            const float * source = input.feats.data() +
                point * static_cast<std::size_t>(input.channels);
            float * destination = output.feats.data() +
                point * static_cast<std::size_t>(out_channels);
            for (int out = 0; out < out_channels; ++out) {
                float value = destination[out];
                for (int in = 0; in < input.channels; ++in) {
                    value += weight[static_cast<std::size_t>(out) * input.channels + in] *
                             source[in];
                }
                destination[out] = value;
            }
        }
        profile_record.record().actual_compute_ms = sparse_profile_elapsed_ms(compute_begin);
        profile_record.record().omp_actual_threads = actual_threads;
        profile_record.record().profile_split_bias = true;
    } else {
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static) num_threads(cpu_thread_count()) if(cpu_should_parallelize(output.feats.size()))
#endif
        for (std::size_t point = 0; point < input.points(); ++point) {
            const float * source = input.feats.data() +
                point * static_cast<std::size_t>(input.channels);
            float * destination = output.feats.data() +
                point * static_cast<std::size_t>(out_channels);
            for (int out = 0; out < out_channels; ++out) {
                float value = bias[out];
                for (int in = 0; in < input.channels; ++in) {
                    value += weight[static_cast<std::size_t>(out) * input.channels + in] *
                             source[in];
                }
                destination[out] = value;
            }
        }
    }
    profile_record.mark_success();
    return true;
}

bool varlen_linear(const VarLenTensorF32 & input,
                   const float * weight,
                   const float * bias,
                   int out_channels,
                   VarLenTensorF32 & output,
                   std::string * error) {
    if (!input.valid(error) || !check_output_channels(out_channels, error)) return false;
    if (!weight || !bias) {
        set_error(error, "variable-length linear weight or bias pointer is null");
        return false;
    }
    output.batch_size = input.batch_size;
    output.channels = out_channels;
    output.offsets = input.offsets;
    std::size_t count = 0;
    if (!checked_product(input.tokens(), static_cast<std::size_t>(out_channels), count)) {
        set_error(error, "variable-length linear output size overflows size_t");
        return false;
    }
    output.feats.assign(count, 0.0f);
 #if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(cpu_thread_count()) if(cpu_should_parallelize(output.feats.size()))
 #endif
    for (std::size_t token = 0; token < input.tokens(); ++token) {
        const float * source = input.feats.data() + token * static_cast<std::size_t>(input.channels);
        float * destination = output.feats.data() + token * static_cast<std::size_t>(out_channels);
        for (int out = 0; out < out_channels; ++out) {
            float value = bias[out];
            for (int in = 0; in < input.channels; ++in) {
                value += weight[static_cast<std::size_t>(out) * input.channels + in] * source[in];
            }
            destination[out] = value;
        }
    }
    return true;
}

bool sparse_layer_norm(const SparseTensorF32 & input,
                       float epsilon,
                       const float * gamma,
                       const float * beta,
                       SparseTensorF32 & output,
                       std::string * error) {
    auto timer = detail::make_sparse_profile_timer(
        &detail::SparseProfileStats::sparse_layer_norm_calls,
        &detail::SparseProfileStats::sparse_layer_norm_ms);
    if (!validate_sparse_tensor(input, "sparse_layer_norm.input", error)) return false;
    if (!(epsilon > 0.0f) || !std::isfinite(epsilon)) {
        set_error(error, "sparse layer norm epsilon must be finite and positive");
        return false;
    }
    if ((gamma == nullptr) != (beta == nullptr)) {
        set_error(error, "sparse layer norm affine parameters must be both null or non-null");
        return false;
    }
    copy_shape(input, input.channels, output);
    output.feats.resize(input.feats.size());
    const std::size_t channels = static_cast<std::size_t>(input.channels);
 #if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(cpu_thread_count()) if(cpu_should_parallelize(output.feats.size()))
 #endif
    for (std::size_t point = 0; point < input.points(); ++point) {
        const float * source = input.feats.data() + point * channels;
        float * destination = output.feats.data() + point * channels;
        float mean = 0.0f;
        for (std::size_t channel = 0; channel < channels; ++channel) mean += source[channel];
        mean /= static_cast<float>(channels);
        float variance = 0.0f;
        for (std::size_t channel = 0; channel < channels; ++channel) {
            const float delta = source[channel] - mean;
            variance += delta * delta;
        }
        variance /= static_cast<float>(channels);
        const float inverse = 1.0f / std::sqrt(variance + epsilon);
        for (std::size_t channel = 0; channel < channels; ++channel) {
            float value = (source[channel] - mean) * inverse;
            if (gamma) value = value * gamma[channel] + beta[channel];
            destination[channel] = value;
        }
    }
    return true;
}

bool sparse_downsample_mean(const SparseTensorF32 & input,
                            int factor,
                            SparseTensorF32 & output,
                            std::string * error) {
    if (!validate_sparse_tensor(input, "sparse_downsample_mean.input", error)) return false;
    if (factor <= 0) {
        set_error(error, "sparse downsample factor must be positive");
        return false;
    }
    struct Group {
        Coord coord;
        std::vector<float> sum;
        std::size_t count = 0;
    };
    std::unordered_map<Coord, std::size_t, CoordHash> lookup;
    std::vector<Group> groups;
    lookup.reserve(input.points());
    for (std::size_t point = 0; point < input.points(); ++point) {
        const Coord source{input.coords[point * 4 + 0], input.coords[point * 4 + 1],
                           input.coords[point * 4 + 2], input.coords[point * 4 + 3]};
        const Coord target{source.batch, source.x / factor, source.y / factor,
                           source.z / factor};
        auto found = lookup.find(target);
        std::size_t group_index = 0;
        if (found == lookup.end()) {
            group_index = groups.size();
            lookup.emplace(target, group_index);
            groups.push_back(Group{target, std::vector<float>(static_cast<std::size_t>(input.channels), 0.0f), 0});
        } else {
            group_index = found->second;
        }
        Group & group = groups[group_index];
        const float * source_features = input.feats.data() +
            point * static_cast<std::size_t>(input.channels);
        for (int channel = 0; channel < input.channels; ++channel) {
            group.sum[static_cast<std::size_t>(channel)] += source_features[channel];
        }
        ++group.count;
    }
    std::sort(groups.begin(), groups.end(), [](const Group & left, const Group & right) {
        if (left.coord.batch != right.coord.batch) return left.coord.batch < right.coord.batch;
        if (left.coord.x != right.coord.x) return left.coord.x < right.coord.x;
        if (left.coord.y != right.coord.y) return left.coord.y < right.coord.y;
        return left.coord.z < right.coord.z;
    });
    output.batch_size = input.batch_size;
    output.channels = input.channels;
    output.spatial_x = (input.spatial_x + factor - 1) / factor;
    output.spatial_y = (input.spatial_y + factor - 1) / factor;
    output.spatial_z = (input.spatial_z + factor - 1) / factor;
    output.coords.resize(groups.size() * 4);
    output.feats.resize(groups.size() * static_cast<std::size_t>(input.channels));
    for (std::size_t index = 0; index < groups.size(); ++index) {
        const Group & group = groups[index];
        output.coords[index * 4 + 0] = group.coord.batch;
        output.coords[index * 4 + 1] = group.coord.x;
        output.coords[index * 4 + 2] = group.coord.y;
        output.coords[index * 4 + 3] = group.coord.z;
        for (int channel = 0; channel < input.channels; ++channel) {
            output.feats[index * static_cast<std::size_t>(input.channels) +
                         static_cast<std::size_t>(channel)] =
                group.sum[static_cast<std::size_t>(channel)] /
                static_cast<float>(group.count);
        }
    }
    return true;
}

bool sparse_channel_to_spatial(const SparseTensorF32 & input,
                               int factor,
                               const SparseTensorF32 * subdivision,
                               SparseTensorF32 & output,
                               std::string * error) {
    auto timer = detail::make_sparse_profile_timer(
        &detail::SparseProfileStats::sparse_channel_to_spatial_calls,
        &detail::SparseProfileStats::sparse_channel_to_spatial_ms);
    if (!validate_sparse_tensor(input, "sparse_channel_to_spatial.input", error)) return false;
    if (factor <= 0) {
        set_error(error, "sparse channel-to-spatial factor must be positive");
        return false;
    }
    std::size_t child_count = 1;
    for (int axis = 0; axis < 3; ++axis) {
        if (child_count > std::numeric_limits<std::size_t>::max() /
                              static_cast<std::size_t>(factor)) {
            set_error(error, "sparse channel-to-spatial factor overflows size_t");
            return false;
        }
        child_count *= static_cast<std::size_t>(factor);
    }
    if (input.channels <= 0 || input.channels % static_cast<int>(child_count) != 0) {
        set_error(error, "sparse channel-to-spatial channels are not divisible by factor^3");
        return false;
    }
    if (subdivision) {
        if (!validate_sparse_tensor(*subdivision, "sparse_channel_to_spatial.subdivision", error) ||
            subdivision->batch_size != input.batch_size ||
            subdivision->channels != static_cast<int>(child_count) ||
            subdivision->coords != input.coords) {
            set_error(error, "sparse subdivision shape or coordinates do not match input");
            return false;
        }
    }
    const long long factor_ll = static_cast<long long>(factor);
    const long long shape_x = static_cast<long long>(input.spatial_x) * factor_ll;
    const long long shape_y = static_cast<long long>(input.spatial_y) * factor_ll;
    const long long shape_z = static_cast<long long>(input.spatial_z) * factor_ll;
    if (shape_x > std::numeric_limits<int>::max() ||
        shape_y > std::numeric_limits<int>::max() ||
        shape_z > std::numeric_limits<int>::max()) {
        set_error(error, "sparse channel-to-spatial output shape overflows int");
        return false;
    }
    const int output_channels = input.channels / static_cast<int>(child_count);
    std::size_t selected = 0;
    if (subdivision) {
        for (std::size_t point = 0; point < input.points(); ++point) {
            for (std::size_t child = 0; child < child_count; ++child) {
                if (subdivision->feats[point * child_count + child] > 0.0f) ++selected;
            }
        }
    } else {
        if (input.points() > std::numeric_limits<std::size_t>::max() / child_count) {
            set_error(error, "sparse channel-to-spatial output point count overflows size_t");
            return false;
        }
        selected = input.points() * child_count;
    }
    output.batch_size = input.batch_size;
    output.channels = output_channels;
    output.spatial_x = static_cast<int>(shape_x);
    output.spatial_y = static_cast<int>(shape_y);
    output.spatial_z = static_cast<int>(shape_z);
    output.coords.resize(selected * 4);
    output.feats.resize(selected * static_cast<std::size_t>(output_channels));
    std::size_t destination_point = 0;
    for (std::size_t point = 0; point < input.points(); ++point) {
        const int batch = input.coords[point * 4 + 0];
        const int x = input.coords[point * 4 + 1];
        const int y = input.coords[point * 4 + 2];
        const int z = input.coords[point * 4 + 3];
        const float * source = input.feats.data() + point * input.channels;
        for (std::size_t child = 0; child < child_count; ++child) {
            if (subdivision && subdivision->feats[point * child_count + child] <= 0.0f) {
                continue;
            }
            const int child_x = static_cast<int>(child % static_cast<std::size_t>(factor));
            const int child_y = static_cast<int>((child / static_cast<std::size_t>(factor)) %
                                                  static_cast<std::size_t>(factor));
            const int child_z = static_cast<int>(child / (static_cast<std::size_t>(factor) *
                                                           static_cast<std::size_t>(factor)));
            const std::size_t coord_offset = destination_point * 4;
            output.coords[coord_offset + 0] = batch;
            output.coords[coord_offset + 1] = x * factor + child_x;
            output.coords[coord_offset + 2] = y * factor + child_y;
            output.coords[coord_offset + 3] = z * factor + child_z;
            const float * child_source = source + child * static_cast<std::size_t>(output_channels);
            float * destination = output.feats.data() +
                destination_point * static_cast<std::size_t>(output_channels);
            for (int channel = 0; channel < output_channels; ++channel) {
                destination[channel] = child_source[channel];
            }
            ++destination_point;
        }
    }
    return validate_sparse_tensor(output, "sparse_channel_to_spatial.output", error);
}

bool sparse_scaled_dot_product_attention(const SparseTensorF32 & query,
                                         const SparseTensorF32 & key,
                                         const SparseTensorF32 & value,
                                         int num_heads,
                                         int head_dim,
                                         float scale,
                                         SparseTensorF32 & output,
                                         std::string * error) {
    if (!query.valid(error) || !key.valid(error) || !value.valid(error)) return false;
    if (num_heads <= 0 || head_dim <= 0 || query.channels != num_heads * head_dim ||
        key.channels != num_heads * head_dim || value.channels <= 0 ||
        value.channels % num_heads != 0) {
        set_error(error, "sparse attention channel/head dimensions are invalid");
        return false;
    }
    if (query.batch_size != key.batch_size || query.batch_size != value.batch_size) {
        set_error(error, "sparse attention batch sizes do not match");
        return false;
    }
    if (!(scale > 0.0f) || !std::isfinite(scale)) {
        set_error(error, "sparse attention scale must be finite and positive");
        return false;
    }
    copy_shape(query, value.channels, output);
    output.feats.assign(query.points() * static_cast<std::size_t>(value.channels), 0.0f);
    const int value_head_dim = value.channels / num_heads;

    auto batch_range = [](const SparseTensorF32 & tensor, int batch) {
        std::size_t begin = 0;
        while (begin < tensor.points() && tensor.coords[begin * 4] < batch) ++begin;
        std::size_t end = begin;
        while (end < tensor.points() && tensor.coords[end * 4] == batch) ++end;
        return std::pair<std::size_t, std::size_t>(begin, end);
    };
    for (int batch = 0; batch < query.batch_size; ++batch) {
        const auto q_range = batch_range(query, batch);
        const auto k_range = batch_range(key, batch);
        const auto v_range = batch_range(value, batch);
        if (k_range.first == k_range.second ||
            v_range.second - v_range.first != k_range.second - k_range.first) {
            set_error(error, "sparse attention has no keys or mismatched key/value rows");
            return false;
        }
        const std::size_t key_count = k_range.second - k_range.first;
        for (std::size_t q_index = q_range.first; q_index < q_range.second; ++q_index) {
            std::vector<float> scores(key_count);
            const float * q_row = query.feats.data() + q_index * query.channels;
            float * out_row = output.feats.data() + q_index * value.channels;
            for (int head = 0; head < num_heads; ++head) {
                const float * q_head = q_row + static_cast<std::size_t>(head * head_dim);
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::size_t offset = 0; offset < key_count; ++offset) {
                    const float * k_row = key.feats.data() +
                        (k_range.first + offset) * static_cast<std::size_t>(key.channels);
                    const float * k_head = k_row + static_cast<std::size_t>(head * head_dim);
                    float score = 0.0f;
                    for (int dim = 0; dim < head_dim; ++dim) score += q_head[dim] * k_head[dim];
                    scores[offset] = score * scale;
                    maximum = std::max(maximum, scores[offset]);
                }
                float denominator = 0.0f;
                for (float & score : scores) {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                float * out_head = out_row + static_cast<std::size_t>(head * value_head_dim);
                for (std::size_t offset = 0; offset < key_count; ++offset) {
                    const float probability = scores[offset] / denominator;
                    const float * v_row = value.feats.data() +
                        (v_range.first + offset) * static_cast<std::size_t>(value.channels);
                    const float * v_head = v_row + static_cast<std::size_t>(head * value_head_dim);
                    for (int dim = 0; dim < value_head_dim; ++dim) {
                        out_head[dim] += probability * v_head[dim];
                    }
                }
            }
        }
    }
    return true;
}

bool sparse_cross_attention(const SparseTensorF32 & query,
                            const VarLenTensorF32 & key,
                            const VarLenTensorF32 & value,
                            int num_heads,
                            int head_dim,
                            float scale,
                            SparseTensorF32 & output,
                            std::string * error) {
    if (!query.valid(error) || !key.valid(error) || !value.valid(error)) return false;
    if (key.batch_size != value.batch_size || key.offsets != value.offsets ||
        num_heads <= 0 || head_dim <= 0 || query.channels != num_heads * head_dim ||
        key.channels != num_heads * head_dim || value.channels <= 0 ||
        value.channels % num_heads != 0) {
        set_error(error, "sparse cross-attention dimensions or offsets are invalid");
        return false;
    }
    if (query.batch_size != key.batch_size) {
        set_error(error, "sparse cross-attention batch sizes do not match");
        return false;
    }
    if (!(scale > 0.0f) || !std::isfinite(scale)) {
        set_error(error, "sparse cross-attention scale must be finite and positive");
        return false;
    }
    copy_shape(query, value.channels, output);
    output.feats.assign(query.points() * static_cast<std::size_t>(value.channels), 0.0f);
    const int value_head_dim = value.channels / num_heads;
    std::size_t query_begin = 0;
    for (int batch = 0; batch < query.batch_size; ++batch) {
        std::size_t query_end = query_begin;
        while (query_end < query.points() && query.coords[query_end * 4] == batch) ++query_end;
        const std::size_t key_begin = key.offsets[static_cast<std::size_t>(batch)];
        const std::size_t key_end = key.offsets[static_cast<std::size_t>(batch + 1)];
        if (key_begin == key_end) {
            set_error(error, "sparse cross-attention has an empty key sequence");
            return false;
        }
        const std::size_t key_count = key_end - key_begin;
        for (std::size_t q_index = query_begin; q_index < query_end; ++q_index) {
            std::vector<float> scores(key_count);
            const float * q_row = query.feats.data() + q_index * query.channels;
            float * out_row = output.feats.data() + q_index * value.channels;
            for (int head = 0; head < num_heads; ++head) {
                const float * q_head = q_row + static_cast<std::size_t>(head * head_dim);
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::size_t offset = 0; offset < key_count; ++offset) {
                    const float * k_row = key.feats.data() +
                        (key_begin + offset) * static_cast<std::size_t>(key.channels);
                    const float * k_head = k_row + static_cast<std::size_t>(head * head_dim);
                    float score = 0.0f;
                    for (int dim = 0; dim < head_dim; ++dim) score += q_head[dim] * k_head[dim];
                    scores[offset] = score * scale;
                    maximum = std::max(maximum, scores[offset]);
                }
                float denominator = 0.0f;
                for (float & score : scores) {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                float * out_head = out_row + static_cast<std::size_t>(head * value_head_dim);
                for (std::size_t offset = 0; offset < key_count; ++offset) {
                    const float probability = scores[offset] / denominator;
                    const float * v_row = value.feats.data() +
                        (key_begin + offset) * static_cast<std::size_t>(value.channels);
                    const float * v_head = v_row + static_cast<std::size_t>(head * value_head_dim);
                    for (int dim = 0; dim < value_head_dim; ++dim) {
                        out_head[dim] += probability * v_head[dim];
                    }
                }
            }
        }
        query_begin = query_end;
    }
    return true;
}

bool sparse_rotary_position_embedding(const SparseTensorF32 & input,
                                      int num_heads,
                                      int head_dim,
                                      float frequency_min,
                                      float frequency_base,
                                      SparseTensorF32 & output,
                                      std::string * error) {
    if (!input.valid(error)) return false;
    if (num_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0 ||
        input.channels != num_heads * head_dim) {
        set_error(error, "sparse RoPE head dimensions are invalid");
        return false;
    }
    if (!(frequency_min > 0.0f) || !std::isfinite(frequency_min) ||
        !(frequency_base > 0.0f) || !std::isfinite(frequency_base)) {
        set_error(error, "sparse RoPE frequencies must be finite and positive");
        return false;
    }
    copy_shape(input, input.channels, output);
    output.feats.resize(input.feats.size());
    const int pair_count = head_dim / 2;
    const int frequency_dim = pair_count / 3;
    std::vector<float> frequencies(static_cast<std::size_t>(std::max(frequency_dim, 0)));
    for (int index = 0; index < frequency_dim; ++index) {
        frequencies[static_cast<std::size_t>(index)] = frequency_min /
            std::pow(frequency_base, static_cast<float>(index) /
                     static_cast<float>(frequency_dim));
    }
 #if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(cpu_thread_count()) if(cpu_should_parallelize(output.feats.size()))
 #endif
    for (std::size_t point = 0; point < input.points(); ++point) {
        const int coordinates[3] = {
            input.coords[point * 4 + 1], input.coords[point * 4 + 2],
            input.coords[point * 4 + 3]};
        const float * source = input.feats.data() + point * input.channels;
        float * destination = output.feats.data() + point * input.channels;
        for (int head = 0; head < num_heads; ++head) {
            const std::size_t head_offset = static_cast<std::size_t>(head * head_dim);
            for (int pair = 0; pair < pair_count; ++pair) {
                float phase = 0.0f;
                if (pair < 3 * frequency_dim) {
                    phase = static_cast<float>(coordinates[pair / frequency_dim]) *
                            frequencies[static_cast<std::size_t>(pair % frequency_dim)];
                }
                const float cosine = std::cos(phase);
                const float sine = std::sin(phase);
                const std::size_t offset = head_offset + static_cast<std::size_t>(pair * 2);
                const float even = source[offset];
                const float odd = source[offset + 1];
                destination[offset] = even * cosine - odd * sine;
                destination[offset + 1] = even * sine + odd * cosine;
            }
        }
    }
    return true;
}

bool sparse_multihead_rms_norm(const SparseTensorF32 & input,
                               int num_heads,
                               int head_dim,
                               const float * gamma,
                               SparseTensorF32 & output,
                               std::string * error) {
    if (!input.valid(error)) return false;
    if (num_heads <= 0 || head_dim <= 0 || input.channels != num_heads * head_dim) {
        set_error(error, "sparse RMS norm head dimensions are invalid");
        return false;
    }
    if (!gamma) {
        set_error(error, "sparse RMS norm gamma pointer is null");
        return false;
    }
    copy_shape(input, input.channels, output);
    output.feats.resize(input.feats.size());
    const float multiplier = std::sqrt(static_cast<float>(head_dim));
 #if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(cpu_thread_count()) if(cpu_should_parallelize(output.feats.size()))
 #endif
    for (std::size_t point = 0; point < input.points(); ++point) {
        const float * source = input.feats.data() + point * input.channels;
        float * destination = output.feats.data() + point * input.channels;
        for (int head = 0; head < num_heads; ++head) {
            const std::size_t offset = static_cast<std::size_t>(head * head_dim);
            float squared = 0.0f;
            for (int dim = 0; dim < head_dim; ++dim) squared += source[offset + dim] * source[offset + dim];
            const float inverse = 1.0f / std::max(std::sqrt(squared), 1e-12f);
            for (int dim = 0; dim < head_dim; ++dim) {
                destination[offset + dim] = source[offset + dim] * inverse * multiplier * gamma[offset + dim];
            }
        }
    }
    return true;
}

bool varlen_multihead_rms_norm(const VarLenTensorF32 & input,
                               int num_heads,
                               int head_dim,
                               const float * gamma,
                               VarLenTensorF32 & output,
                               std::string * error) {
    if (!input.valid(error)) return false;
    if (num_heads <= 0 || head_dim <= 0 || input.channels != num_heads * head_dim) {
        set_error(error, "variable-length RMS norm head dimensions are invalid");
        return false;
    }
    if (!gamma) {
        set_error(error, "variable-length RMS norm gamma pointer is null");
        return false;
    }
    output.batch_size = input.batch_size;
    output.channels = input.channels;
    output.offsets = input.offsets;
    output.feats.resize(input.feats.size());
    const float multiplier = std::sqrt(static_cast<float>(head_dim));
 #if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(cpu_thread_count()) if(cpu_should_parallelize(output.feats.size()))
 #endif
    for (std::size_t token = 0; token < input.tokens(); ++token) {
        const float * source = input.feats.data() + token * input.channels;
        float * destination = output.feats.data() + token * input.channels;
        for (int head = 0; head < num_heads; ++head) {
            const std::size_t offset = static_cast<std::size_t>(head * head_dim);
            float squared = 0.0f;
            for (int dim = 0; dim < head_dim; ++dim) {
                squared += source[offset + static_cast<std::size_t>(dim)] *
                           source[offset + static_cast<std::size_t>(dim)];
            }
            const float inverse = 1.0f / std::max(std::sqrt(squared), 1e-12f);
            for (int dim = 0; dim < head_dim; ++dim) {
                const std::size_t index = offset + static_cast<std::size_t>(dim);
                destination[index] = source[index] * inverse * multiplier * gamma[index];
            }
        }
    }
    return true;
}

bool sparse_submanifold_conv3d(const SparseTensorF32 & input,
                               const float * weight,
                               const float * bias,
                               int out_channels,
                               SparseTensorF32 & output,
                               std::string * error) {
    auto timer = detail::make_sparse_profile_timer(
        &detail::SparseProfileStats::sparse_submanifold_conv3d_calls,
        &detail::SparseProfileStats::sparse_submanifold_conv3d_ms);
    if (!check_weights(input, out_channels, weight, bias, error)) return false;
    copy_shape(input, out_channels, output);
    std::size_t count = 0;
    if (!checked_product(input.points(), static_cast<std::size_t>(out_channels), count)) {
        set_error(error, "sparse convolution output size overflows size_t");
        return false;
    }
    output.feats.assign(count, 0.0f);

    std::unordered_map<Coord, std::size_t, CoordHash> indices;
    indices.reserve(input.points());
    for (std::size_t point = 0; point < input.points(); ++point) {
        indices.emplace(Coord{input.coords[point * 4 + 0], input.coords[point * 4 + 1],
                              input.coords[point * 4 + 2], input.coords[point * 4 + 3]}, point);
    }
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(cpu_thread_count()) if(cpu_should_parallelize(output.feats.size()))
#endif
    for (std::size_t point = 0; point < input.points(); ++point) {
        const Coord center{input.coords[point * 4 + 0], input.coords[point * 4 + 1],
                           input.coords[point * 4 + 2], input.coords[point * 4 + 3]};
        float * destination = output.feats.data() + point * static_cast<std::size_t>(out_channels);
        for (int kd = 0; kd < 3; ++kd) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    const Coord neighbor{center.batch, center.x + kd - 1,
                                         center.y + kh - 1, center.z + kw - 1};
                    const auto found = indices.find(neighbor);
                    if (found == indices.end()) continue;
                    const float * source = input.feats.data() +
                        found->second * static_cast<std::size_t>(input.channels);
                    const float * kernel = weight +
                        (static_cast<std::size_t>(kd) * 9 +
                         static_cast<std::size_t>(kh) * 3 + kw) *
                        static_cast<std::size_t>(out_channels) * input.channels;
                    for (int out = 0; out < out_channels; ++out) {
                        const float * row = kernel + static_cast<std::size_t>(out) * input.channels;
                        float value = 0.0f;
                        for (int in = 0; in < input.channels; ++in) value += row[in] * source[in];
                        destination[out] += value;
                    }
                }
            }
        }
        for (int out = 0; out < out_channels; ++out) destination[out] += bias[out];
    }
    return true;
}

} // namespace pixal3d
