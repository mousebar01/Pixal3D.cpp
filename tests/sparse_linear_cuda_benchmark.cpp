#include "pixal3d/backend.h"
#include "pixal3d/sparse.h"
#include "pixal3d/cpu_threads.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(_OPENMP)
    #include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Shape {
    const char * name = nullptr;
    std::size_t points = 0;
    int input_channels = 0;
    int output_channels = 0;
    int batch_size = 1;
    int spatial = 1024;
    bool with_bias = true;
};

const Shape kHotShapes[] = {
    {"n46537_c2048_to_c512", 46537, 2048, 512},
    {"n46537_c512_to_c2048", 46537, 512, 2048},
    {"n206980_c1024_to_c256", 206980, 1024, 256},
    {"n206980_c256_to_c1024", 206980, 256, 1024},
    {"n909164_c512_to_c128", 909164, 512, 128},
    {"n909164_c128_to_c512", 909164, 128, 512},
    {"n9857_c4096_to_c1024", 9857, 4096, 1024},
    {"n9857_c1024_to_c4096", 9857, 1024, 4096},
    {"n4005638_c64_to_c7", 4005638, 64, 7},
};

const Shape kFixtureShapes[] = {
    {"fixture_rect_bias_multibatch", 11, 3, 5, 2, 8, true},
    {"fixture_square_no_bias_multibatch", 13, 4, 4, 2, 8, false},
    {"fixture_rect_bias", 17, 5, 2, 1, 8, true},
};

struct Options {
    bool fixture = false;
    std::size_t warmup = 2;
    std::size_t iterations = 3;
    std::vector<std::string> selected_shapes;
};

struct PhaseStats {
    std::size_t count = 0;
    double min_ms = 0.0;
    double median_ms = 0.0;
};

struct IterationTiming {
    double input_upload_ms = 0.0;
    double weight_upload_ms = 0.0;
    double bias_upload_ms = 0.0;
    double gpu_compute_ms = 0.0;
    double output_download_ms = 0.0;
    double total_ms = 0.0;
};

struct ErrorStats {
    bool coords_exact = false;
    bool shape_exact = false;
    std::size_t cpu_nan = 0;
    std::size_t gpu_nan = 0;
    std::size_t cpu_inf = 0;
    std::size_t gpu_inf = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    double mean_abs = 0.0;
    std::size_t worst_index = 0;
    std::size_t worst_point = 0;
    int worst_channel = 0;
    float cpu_value = 0.0f;
    float gpu_value = 0.0f;
    double worst_abs = 0.0;
    double worst_rel = 0.0;
    std::size_t worst_relative_index = 0;
    std::size_t worst_relative_point = 0;
    int worst_relative_channel = 0;
    float worst_relative_cpu_value = 0.0f;
    float worst_relative_gpu_value = 0.0f;
    double worst_relative_abs = 0.0;
    double worst_relative = 0.0;
};

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool parse_size(const char * text, std::size_t & value) {
    if (!text || !*text) return false;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0) return false;
    value = static_cast<std::size_t>(parsed);
    return static_cast<unsigned long long>(value) == parsed;
}

std::string size_string(std::size_t value) {
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

PhaseStats summarize(const std::vector<double> & samples) {
    PhaseStats result;
    if (samples.empty()) return result;
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    result.count = sorted.size();
    result.min_ms = sorted.front();
    if (sorted.size() % 2 == 1) {
        result.median_ms = sorted[sorted.size() / 2];
    } else {
        result.median_ms = 0.5 * (sorted[sorted.size() / 2 - 1] +
                                  sorted[sorted.size() / 2]);
    }
    return result;
}

void print_phase(const char * prefix, const char * phase,
                 const std::vector<double> & samples) {
    const PhaseStats stats = summarize(samples);
    if (stats.count == 0) return;
    std::cout << prefix << " phase=" << phase
              << " count=" << stats.count
              << " median_ms=" << stats.median_ms
              << " min_ms=" << stats.min_ms << '\n';
}

std::size_t checked_count(std::size_t points, int channels, std::string * error) {
    if (channels <= 0 || points > std::numeric_limits<std::size_t>::max() /
                             static_cast<std::size_t>(channels)) {
        set_error(error, "sparse linear benchmark tensor size overflows size_t");
        return 0;
    }
    return points * static_cast<std::size_t>(channels);
}

std::vector<std::int32_t> make_coords(const Shape & shape) {
    const std::size_t points_per_batch =
        (shape.points + static_cast<std::size_t>(shape.batch_size) - 1) /
        static_cast<std::size_t>(shape.batch_size);
    const std::size_t side = static_cast<std::size_t>(shape.spatial);
    const std::size_t side_squared = side * side;
    std::vector<std::int32_t> coords(shape.points * 4);
    for (std::size_t point = 0; point < shape.points; ++point) {
        const std::size_t batch = point / points_per_batch;
        const std::size_t local = point % points_per_batch;
        const std::size_t z = local / side_squared;
        const std::size_t rem = local % side_squared;
        const std::size_t y = rem / side;
        const std::size_t x = rem % side;
        coords[point * 4 + 0] = static_cast<std::int32_t>(batch);
        coords[point * 4 + 1] = static_cast<std::int32_t>(x);
        coords[point * 4 + 2] = static_cast<std::int32_t>(y);
        coords[point * 4 + 3] = static_cast<std::int32_t>(z);
    }
    return coords;
}

float deterministic_value(std::size_t index, std::uint64_t salt, float scale) {
    const std::uint64_t mixed = index * 6364136223846793005ULL + salt;
    const int bucket = static_cast<int>((mixed >> 32) % 251ULL) - 125;
    return static_cast<float>(bucket) * scale;
}

void fill_values(std::vector<float> & values, std::uint64_t salt, float scale) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = deterministic_value(index, salt, scale);
    }
}

void fill_fixture_values(std::vector<float> & input,
                         std::vector<float> & weight,
                         std::vector<float> & bias) {
    // Use exactly representable binary fractions for the small fixtures.  This
    // makes the layout/reduction check sensitive to transposes and indexing
    // bugs without making the fixture fail merely because CUDA and CPU use
    // different reduction orders for arbitrary decimal inputs.
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = 0.125f * static_cast<float>((index % 13) + 1);
    }
    for (std::size_t index = 0; index < weight.size(); ++index) {
        weight[index] = 0.0625f * static_cast<float>((index % 11) + 1);
    }
    for (std::size_t index = 0; index < bias.size(); ++index) {
        bias[index] = 0.25f * static_cast<float>(index + 1);
    }
}

pixal3d::SparseTensorF32 make_input(const Shape & shape,
                                    std::vector<float> & input_values) {
    pixal3d::SparseTensorF32 input;
    input.batch_size = shape.batch_size;
    input.channels = shape.input_channels;
    input.spatial_x = shape.spatial;
    input.spatial_y = shape.spatial;
    input.spatial_z = shape.spatial;
    input.coords = make_coords(shape);
    input_values.resize(shape.points * static_cast<std::size_t>(shape.input_channels));
    fill_values(input_values, 0x13579bdf2468ace0ULL, 0.00075f);
    input.feats = input_values;
    return input;
}

void cpu_reference_linear(const std::vector<float> & input,
                         const std::vector<float> & weight,
                         const std::vector<float> & bias,
                         bool with_bias,
                         std::size_t points,
                         int input_channels,
                         int output_channels,
                         std::vector<float> & output) {
    // This is the production sparse_linear() arithmetic contract copied into
    // the benchmark without validation or allocation timing:
    // input/feature storage is row-major [point, channel], while weight is
    // row-major [out_channel, in_channel].  The inner reduction order matches
    // src/sparse.cpp exactly.
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(pixal3d::cpu_thread_count()) if(pixal3d::cpu_should_parallelize(output.size()))
#endif
    for (std::size_t point = 0; point < points; ++point) {
        const float * source = input.data() + point * static_cast<std::size_t>(input_channels);
        float * destination = output.data() + point * static_cast<std::size_t>(output_channels);
        for (int out = 0; out < output_channels; ++out) {
            float value = with_bias ? bias[static_cast<std::size_t>(out)] : 0.0f;
            for (int in = 0; in < input_channels; ++in) {
                value += weight[static_cast<std::size_t>(out) *
                                static_cast<std::size_t>(input_channels) +
                                static_cast<std::size_t>(in)] * source[in];
            }
            destination[out] = value;
        }
    }
}

std::size_t count_nan(const std::vector<float> & values) {
    std::size_t result = 0;
    for (float value : values) {
        if (std::isnan(value)) ++result;
    }
    return result;
}

std::size_t count_inf(const std::vector<float> & values) {
    std::size_t result = 0;
    for (float value : values) {
        if (std::isinf(value)) ++result;
    }
    return result;
}

ErrorStats compare_results(const pixal3d::SparseTensorF32 & input,
                           const pixal3d::SparseTensorF32 & output,
                           const std::vector<float> & cpu,
                           const std::vector<float> & gpu) {
    ErrorStats result;
    result.coords_exact = input.coords == output.coords;
    result.shape_exact = input.batch_size == output.batch_size &&
                         output.channels == static_cast<int>(
                             cpu.size() / std::max<std::size_t>(input.points(), 1)) &&
                         input.spatial_x == output.spatial_x &&
                         input.spatial_y == output.spatial_y &&
                         input.spatial_z == output.spatial_z;
    result.cpu_nan = count_nan(cpu);
    result.gpu_nan = count_nan(gpu);
    result.cpu_inf = count_inf(cpu);
    result.gpu_inf = count_inf(gpu);
    if (cpu.size() != gpu.size() || cpu.empty()) return result;

    double sum_abs = 0.0;
    bool have_worst_abs = false;
    bool have_worst_relative = false;
    for (std::size_t index = 0; index < cpu.size(); ++index) {
        const float cpu_value = cpu[index];
        const float gpu_value = gpu[index];
        if (!std::isfinite(cpu_value) || !std::isfinite(gpu_value)) continue;
        const double absolute = std::fabs(static_cast<double>(gpu_value) -
                                          static_cast<double>(cpu_value));
        // Keep the requested raw relative metric.  Values near zero can make
        // max_rel very large even when max_abs is small, so report both and
        // identify the separate worst-relative element below.
        const double relative = absolute / std::max(std::fabs(static_cast<double>(cpu_value)),
                                                     1e-12);
        sum_abs += absolute;
        if (absolute > result.max_abs) {
            result.max_abs = absolute;
        }
        if (!have_worst_relative || relative > result.max_rel) {
            have_worst_relative = true;
            result.max_rel = relative;
            result.worst_relative_index = index;
            result.worst_relative_cpu_value = cpu_value;
            result.worst_relative_gpu_value = gpu_value;
            result.worst_relative_abs = absolute;
            result.worst_relative = relative;
        }
        if (!have_worst_abs || absolute > result.worst_abs) {
            have_worst_abs = true;
            result.worst_abs = absolute;
            result.worst_rel = relative;
            result.worst_index = index;
            result.cpu_value = cpu_value;
            result.gpu_value = gpu_value;
        }
    }
    result.mean_abs = sum_abs / static_cast<double>(cpu.size());
    if (output.channels > 0) {
        // The comparison buffer is [point, output_channel], so decode both
        // error locations with Cout rather than the input Cin.
        const std::size_t output_channels = static_cast<std::size_t>(output.channels);
        result.worst_point = result.worst_index / output_channels;
        result.worst_channel = static_cast<int>(result.worst_index % output_channels);
        result.worst_relative_point = result.worst_relative_index / output_channels;
        result.worst_relative_channel = static_cast<int>(
            result.worst_relative_index % output_channels);
    }
    return result;
}

class GpuLinearGraph {
public:
    GpuLinearGraph() = default;
    ~GpuLinearGraph() { close(); }

    GpuLinearGraph(const GpuLinearGraph &) = delete;
    GpuLinearGraph & operator=(const GpuLinearGraph &) = delete;

    bool initialize(const Shape & shape, std::string * error) {
        close();
        const auto backend_begin = Clock::now();
        if (!manager_.initialize({pixal3d::BackendPolicyKind::gpu, 0}, error)) {
            return false;
        }
        backend_init_ms_ = elapsed_ms(backend_begin, Clock::now());

        const auto graph_begin = Clock::now();
        ggml_init_params params{};
        params.mem_size = ggml_tensor_overhead() * 8 +
                          ggml_graph_overhead_custom(16, false) + 4096;
        params.no_alloc = true;
        context_ = ggml_init(params);
        if (!context_) {
            set_error(error, "failed to create sparse linear benchmark ggml context");
            close();
            return false;
        }

        // CPU -> ggml layout mapping is intentionally explicit:
        //
        //   CPU input  X[N, Cin]  is stored point-major as [point][channel].
        //   CPU weight W[Cout, Cin] is stored output-major as [out][in].
        //
        // ggml uses ne[0] as the contiguous/column dimension.  Therefore the
        // same flat CPU buffers are represented as:
        //
        //   ggml input  ne = [Cin, N]
        //   ggml weight ne = [Cin, Cout]
        //   ggml output ne = [Cout, N]
        //
        // ggml_mul_mat(weight, input) computes the expected [Cout, N]
        // result, whose flat storage is again point-major [point][out].  No
        // transpose, weight-file relayout, or coordinate tensor is involved.
        weight_ = ggml_new_tensor_2d(context_, GGML_TYPE_F32,
                                     shape.input_channels, shape.output_channels);
        input_ = ggml_new_tensor_2d(context_, GGML_TYPE_F32,
                                    shape.input_channels,
                                    static_cast<int64_t>(shape.points));
        if (shape.with_bias) {
            bias_ = ggml_new_tensor_1d(context_, GGML_TYPE_F32, shape.output_channels);
        }
        product_ = weight_ && input_ ? ggml_mul_mat(context_, weight_, input_) : nullptr;
        if (product_) ggml_mul_mat_set_prec(product_, GGML_PREC_F32);
        result_ = product_;
        if (result_ && bias_) result_ = ggml_add(context_, result_, bias_);
        graph_ = result_ ? ggml_new_graph_custom(context_, 16, false) : nullptr;
        if (!weight_ || !input_ || (shape.with_bias && !bias_) || !product_ || !result_ || !graph_) {
            set_error(error, "failed to build sparse linear benchmark ggml graph");
            close();
            return false;
        }
        ggml_set_input(weight_);
        ggml_set_input(input_);
        if (bias_) ggml_set_input(bias_);
        ggml_set_output(result_);
        ggml_build_forward_expand(graph_, result_);
        graph_build_ms_ = elapsed_ms(graph_begin, Clock::now());

        const auto scheduler_begin = Clock::now();
        scheduler_ = pixal3d::BackendScheduler(manager_, 16, false, true,
                                               error, "sparse_linear_cuda_benchmark");
        // Pin every graph tensor to the forced primary GPU.  Marking only the
        // operation nodes is insufficient for this benchmark: the scheduler
        // may otherwise leave an input/weight tensor on CPU and silently add
        // cross-backend copies, which would invalidate the resident/non-
        // resident transfer measurements.
        if (!scheduler_.valid() ||
            !scheduler_.require_primary(weight_, error) ||
            !scheduler_.require_primary(input_, error) ||
            (bias_ && !scheduler_.require_primary(bias_, error)) ||
            !scheduler_.require_primary(product_, error) ||
            !scheduler_.require_primary(result_, error) ||
            !scheduler_.allocate_graph(graph_, error)) {
            close();
            return false;
        }
        scheduler_allocate_ms_ = elapsed_ms(scheduler_begin, Clock::now());
        if (scheduler_.tensor_backend(product_) != manager_.primary() ||
            scheduler_.tensor_backend(result_) != manager_.primary() ||
            scheduler_.tensor_backend(input_) != manager_.primary() ||
            scheduler_.tensor_backend(weight_) != manager_.primary() ||
            (bias_ && scheduler_.tensor_backend(bias_) != manager_.primary())) {
            set_error(error, "sparse linear benchmark tensors were not placed on the primary GPU");
            close();
            return false;
        }
        return true;
    }

    void close() noexcept {
        scheduler_ = pixal3d::BackendScheduler{};
        if (context_) ggml_free(context_);
        context_ = nullptr;
        graph_ = nullptr;
        weight_ = nullptr;
        input_ = nullptr;
        bias_ = nullptr;
        product_ = nullptr;
        result_ = nullptr;
        manager_.close();
        backend_init_ms_ = 0.0;
        graph_build_ms_ = 0.0;
        scheduler_allocate_ms_ = 0.0;
    }

    bool upload_input(const std::vector<float> & values, double & elapsed,
                      std::string * error) {
        return upload(input_, values.data(), values.size() * sizeof(float), elapsed, error);
    }

    bool upload_weight(const std::vector<float> & values, double & elapsed,
                      std::string * error) {
        return upload(weight_, values.data(), values.size() * sizeof(float), elapsed, error);
    }

    bool upload_bias(const std::vector<float> & values, double & elapsed,
                    std::string * error) {
        if (!bias_) {
            elapsed = 0.0;
            return true;
        }
        return upload(bias_, values.data(), values.size() * sizeof(float), elapsed, error);
    }

    bool compute(double & elapsed, std::string * error) {
        const auto begin = Clock::now();
        const ggml_status status = scheduler_.compute(graph_, error);
        scheduler_.synchronize();
        elapsed = elapsed_ms(begin, Clock::now());
        return status == GGML_STATUS_SUCCESS;
    }

    bool download(std::vector<float> & values, double & elapsed,
                  std::string * error) {
        if (!result_) {
            set_error(error, "sparse linear benchmark has no GPU result tensor");
            return false;
        }
        const std::size_t expected_bytes = ggml_nbytes(result_);
        const std::size_t bytes = values.size() * sizeof(float);
        if (values.empty() || bytes != expected_bytes) {
            set_error(error, "sparse linear benchmark output buffer size does not match GPU result");
            return false;
        }
        const auto begin = Clock::now();
        ggml_backend_tensor_get(result_, values.data(), 0, bytes);
        scheduler_.synchronize();
        elapsed = elapsed_ms(begin, Clock::now());
        // Keep finite-value diagnosis in compare_results().  The benchmark
        // must report NaN/Inf counts instead of failing before correctness
        // decomposition can explain them.
        return true;
    }

    double backend_init_ms() const noexcept { return backend_init_ms_; }
    double graph_build_ms() const noexcept { return graph_build_ms_; }
    double scheduler_allocate_ms() const noexcept { return scheduler_allocate_ms_; }
    const pixal3d::BackendManager & manager() const noexcept { return manager_; }
    const ggml_tensor * weight() const noexcept { return weight_; }
    const ggml_tensor * input() const noexcept { return input_; }
    const ggml_tensor * bias() const noexcept { return bias_; }
    const ggml_tensor * product() const noexcept { return product_; }
    const ggml_tensor * result() const noexcept { return result_; }

private:
    bool upload(ggml_tensor * tensor, const void * data, std::size_t bytes,
                double & elapsed, std::string * error) {
        if (!tensor || !data || bytes == 0) {
            set_error(error, "invalid sparse linear benchmark upload");
            return false;
        }
        const auto begin = Clock::now();
        ggml_backend_tensor_set(tensor, data, 0, bytes);
        scheduler_.synchronize();
        elapsed = elapsed_ms(begin, Clock::now());
        return true;
    }

    pixal3d::BackendManager manager_;
    pixal3d::BackendScheduler scheduler_;
    ggml_context * context_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * weight_ = nullptr;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * bias_ = nullptr;
    ggml_tensor * product_ = nullptr;
    ggml_tensor * result_ = nullptr;
    double backend_init_ms_ = 0.0;
    double graph_build_ms_ = 0.0;
    double scheduler_allocate_ms_ = 0.0;
};

bool run_compute_only(GpuLinearGraph & graph,
                      const std::vector<float> & input,
                      const std::vector<float> & weight,
                      const std::vector<float> & bias,
                      std::size_t warmup,
                      std::size_t iterations,
                      std::vector<double> & samples,
                      std::string * error) {
    double ignored = 0.0;
    if (!graph.upload_weight(weight, ignored, error) ||
        !graph.upload_bias(bias, ignored, error) ||
        !graph.upload_input(input, ignored, error)) {
        return false;
    }
    for (std::size_t index = 0; index < warmup; ++index) {
        if (!graph.compute(ignored, error)) return false;
    }
    samples.clear();
    samples.reserve(iterations);
    for (std::size_t index = 0; index < iterations; ++index) {
        double elapsed = 0.0;
        if (!graph.compute(elapsed, error)) return false;
        samples.push_back(elapsed);
    }
    return true;
}

bool run_end_to_end_iteration(GpuLinearGraph & graph,
                              const std::vector<float> & input,
                              const std::vector<float> & weight,
                              const std::vector<float> & bias,
                              bool upload_static,
                              std::vector<float> * output,
                              IterationTiming & timing,
                              std::string * error) {
    const auto total_begin = Clock::now();
    if (!graph.upload_input(input, timing.input_upload_ms, error)) return false;
    if (upload_static &&
        (!graph.upload_weight(weight, timing.weight_upload_ms, error) ||
         !graph.upload_bias(bias, timing.bias_upload_ms, error))) {
        return false;
    }
    if (!graph.compute(timing.gpu_compute_ms, error)) return false;
    if (!output) {
        set_error(error, "sparse linear benchmark requires a complete output download buffer");
        return false;
    }
    // End-to-end timing intentionally includes the complete [N, Cout] D2H
    // transfer.  Callers provide a reusable full-sized scratch vector so host
    // allocation is not mixed into the measured transfer.
    if (!graph.download(*output, timing.output_download_ms, error)) return false;
    timing.total_ms = elapsed_ms(total_begin, Clock::now());
    return true;
}

bool run_resident_end_to_end(GpuLinearGraph & graph,
                             const std::vector<float> & input,
                             const std::vector<float> & weight,
                             const std::vector<float> & bias,
                             std::size_t output_count,
                             std::size_t warmup,
                             std::size_t iterations,
                             double & setup_weight_ms,
                             double & setup_bias_ms,
                             std::vector<IterationTiming> & samples,
                             std::string * error) {
    if (!graph.upload_weight(weight, setup_weight_ms, error) ||
        !graph.upload_bias(bias, setup_bias_ms, error)) {
        return false;
    }
    std::vector<float> output_scratch(output_count, 0.0f);
    IterationTiming warmup_timing;
    for (std::size_t index = 0; index < warmup; ++index) {
        if (!run_end_to_end_iteration(graph, input, weight, bias, false,
                                       &output_scratch, warmup_timing, error)) {
            return false;
        }
    }
    samples.clear();
    samples.reserve(iterations);
    for (std::size_t index = 0; index < iterations; ++index) {
        IterationTiming timing;
        if (!run_end_to_end_iteration(graph, input, weight, bias, false,
                                       &output_scratch, timing, error)) {
            return false;
        }
        samples.push_back(timing);
    }
    return true;
}

bool run_nonresident_end_to_end(GpuLinearGraph & graph,
                                const std::vector<float> & input,
                                const std::vector<float> & weight,
                                const std::vector<float> & bias,
                                std::size_t output_count,
                                std::size_t warmup,
                                std::size_t iterations,
                                std::vector<IterationTiming> & samples,
                                std::string * error) {
    std::vector<float> output_scratch(output_count, 0.0f);
    IterationTiming warmup_timing;
    for (std::size_t index = 0; index < warmup; ++index) {
        if (!run_end_to_end_iteration(graph, input, weight, bias, true,
                                       &output_scratch, warmup_timing, error)) {
            return false;
        }
    }
    samples.clear();
    samples.reserve(iterations);
    for (std::size_t index = 0; index < iterations; ++index) {
        IterationTiming timing;
        if (!run_end_to_end_iteration(graph, input, weight, bias, true,
                                       &output_scratch, timing, error)) {
            return false;
        }
        samples.push_back(timing);
    }
    return true;
}

std::vector<double> phase_values(const std::vector<IterationTiming> & samples,
                                 double IterationTiming::* field) {
    std::vector<double> values;
    values.reserve(samples.size());
    for (const IterationTiming & sample : samples) values.push_back(sample.*field);
    return values;
}

void print_layout(const Shape & shape, const GpuLinearGraph & graph) {
    std::cout << "layout cpu_input_ne=[" << shape.points << ','
              << shape.input_channels << "]"
              << " cpu_weight_ne=[" << shape.output_channels << ','
              << shape.input_channels << "]"
              << " cpu_output_ne=[" << shape.points << ','
              << shape.output_channels << "]\n";
    std::cout << "layout ggml_input_ne=[" << graph.input()->ne[0] << ','
              << graph.input()->ne[1] << "]"
              << " ggml_weight_ne=[" << graph.weight()->ne[0] << ','
              << graph.weight()->ne[1] << "]"
              << " ggml_product_ne=[" << graph.product()->ne[0] << ','
              << graph.product()->ne[1] << "]"
              << " ggml_result_ne=[" << graph.result()->ne[0] << ','
              << graph.result()->ne[1] << "]\n";
}

bool run_case(const Shape & shape, const Options & options, bool fixture,
              std::string * error) {
    const std::size_t input_count = checked_count(shape.points, shape.input_channels, error);
    const std::size_t weight_count = checked_count(
        static_cast<std::size_t>(shape.output_channels), shape.input_channels, error);
    const std::size_t output_count = checked_count(shape.points, shape.output_channels, error);
    if (input_count == 0 || weight_count == 0 || output_count == 0) return false;

    std::vector<float> input_values;
    pixal3d::SparseTensorF32 input = make_input(shape, input_values);
    std::vector<float> weight(weight_count);
    std::vector<float> bias(shape.with_bias ? static_cast<std::size_t>(shape.output_channels) : 0);
    fill_values(weight, 0x0f1e2d3c4b5a6978ULL, 0.00055f);
    if (shape.with_bias) fill_values(bias, 0xa5a5a5a55a5a5a5aULL, 0.0008f);
    if (fixture) {
        fill_fixture_values(input_values, weight, bias);
        input.feats = input_values;
    }

    std::vector<float> cpu_output(output_count, 0.0f);
    std::vector<double> cpu_samples;
    const std::size_t cpu_warmup = fixture ? 1 : options.warmup;
    const std::size_t cpu_iterations = fixture ? 2 : options.iterations;
    for (std::size_t index = 0; index < cpu_warmup; ++index) {
        cpu_reference_linear(input_values, weight, bias, shape.with_bias,
                             shape.points, shape.input_channels,
                             shape.output_channels, cpu_output);
    }
    cpu_samples.reserve(cpu_iterations);
    for (std::size_t index = 0; index < cpu_iterations; ++index) {
        const auto begin = Clock::now();
        cpu_reference_linear(input_values, weight, bias, shape.with_bias,
                             shape.points, shape.input_channels,
                             shape.output_channels, cpu_output);
        cpu_samples.push_back(elapsed_ms(begin, Clock::now()));
    }

    GpuLinearGraph graph;
    if (!graph.initialize(shape, error)) return false;

    if (fixture && shape.with_bias) {
        pixal3d::SparseTensorF32 production_output;
        if (!pixal3d::sparse_linear(input, weight.data(), bias.data(),
                                    shape.output_channels, production_output, error)) {
            return false;
        }
        if (production_output.coords != input.coords ||
            production_output.feats != cpu_output) {
            set_error(error, "benchmark CPU reference does not match production sparse_linear");
            return false;
        }
    }

    std::vector<double> compute_samples;
    if (!run_compute_only(graph, input_values, weight, bias,
                          fixture ? 1 : options.warmup,
                          fixture ? 2 : options.iterations,
                          compute_samples, error)) {
        return false;
    }

    std::vector<float> gpu_output(output_count, 0.0f);
    double correctness_download_ms = 0.0;
    if (!graph.download(gpu_output, correctness_download_ms, error)) return false;
    pixal3d::SparseTensorF32 gpu_sparse_output = input;
    gpu_sparse_output.channels = shape.output_channels;
    gpu_sparse_output.feats = gpu_output;
    const ErrorStats correctness = compare_results(input, gpu_sparse_output,
                                                   cpu_output, gpu_output);

    std::vector<IterationTiming> nonresident_samples;
    if (!run_nonresident_end_to_end(graph, input_values, weight, bias, output_count,
                                    fixture ? 1 : options.warmup,
                                    fixture ? 2 : options.iterations,
                                    nonresident_samples, error)) {
        return false;
    }

    std::vector<IterationTiming> resident_samples;
    double setup_weight_ms = 0.0;
    double setup_bias_ms = 0.0;
    if (!run_resident_end_to_end(graph, input_values, weight, bias, output_count,
                                 fixture ? 1 : options.warmup,
                                 fixture ? 2 : options.iterations,
                                 setup_weight_ms, setup_bias_ms,
                                 resident_samples, error)) {
        return false;
    }

    const PhaseStats cpu_stats = summarize(cpu_samples);
    const PhaseStats compute_stats = summarize(compute_samples);
    const PhaseStats nonresident_total = summarize(
        phase_values(nonresident_samples, &IterationTiming::total_ms));
    const PhaseStats resident_total = summarize(
        phase_values(resident_samples, &IterationTiming::total_ms));
    const double flops = 2.0 * static_cast<double>(shape.points) *
                         static_cast<double>(shape.input_channels) *
                         static_cast<double>(shape.output_channels);
    const double cpu_gflops = flops / (cpu_stats.median_ms * 1.0e6);
    const double compute_tflops = flops / (compute_stats.median_ms * 1.0e9);
    const double nonresident_tflops = flops / (nonresident_total.median_ms * 1.0e9);
    const double resident_tflops = flops / (resident_total.median_ms * 1.0e9);
    const double setup_common_ms = graph.graph_build_ms() + graph.scheduler_allocate_ms();
    const double nonresident_amortized_ms = nonresident_total.median_ms +
        setup_common_ms / static_cast<double>(std::max<std::size_t>(nonresident_samples.size(), 1));
    const double resident_amortized_ms = resident_total.median_ms +
        (setup_common_ms + setup_weight_ms + setup_bias_ms) /
        static_cast<double>(std::max<std::size_t>(resident_samples.size(), 1));

    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::cout << "benchmark=sparse_linear_cuda"
              << " shape=" << shape.name
              << " points=" << shape.points
              << " cin=" << shape.input_channels
              << " cout=" << shape.output_channels
              << " batch=" << shape.batch_size
              << " bias=" << (shape.with_bias ? 1 : 0)
              << " flops=" << std::fixed << flops << std::defaultfloat << '\n';
    print_layout(shape, graph);
    std::cout << "backend primary=" << graph.manager().primary_name()
              << " backend_init_ms=" << graph.backend_init_ms()
              << " graph_build_ms=" << graph.graph_build_ms()
              << " scheduler_allocate_ms=" << graph.scheduler_allocate_ms() << '\n';
#if defined(_OPENMP)
    std::cout << "cpu_openmp max_threads=" << omp_get_max_threads() << '\n';
#else
    std::cout << "cpu_openmp compiled=0\n";
#endif
    print_phase("cpu_reference_compute", "compute", cpu_samples);
    std::cout << "cpu_reference_compute gflops=" << cpu_gflops << '\n';
    print_phase("gpu_compute_only", "compute", compute_samples);
    std::cout << "gpu_compute_only tflops=" << compute_tflops
              << " speedup_vs_cpu=" << (cpu_stats.median_ms / compute_stats.median_ms) << '\n';

    print_phase("gpu_e2e_nonresident", "input_upload",
                phase_values(nonresident_samples, &IterationTiming::input_upload_ms));
    print_phase("gpu_e2e_nonresident", "weight_upload",
                phase_values(nonresident_samples, &IterationTiming::weight_upload_ms));
    print_phase("gpu_e2e_nonresident", "bias_upload",
                phase_values(nonresident_samples, &IterationTiming::bias_upload_ms));
    print_phase("gpu_e2e_nonresident", "gpu_compute",
                phase_values(nonresident_samples, &IterationTiming::gpu_compute_ms));
    print_phase("gpu_e2e_nonresident", "output_download",
                phase_values(nonresident_samples, &IterationTiming::output_download_ms));
    print_phase("gpu_e2e_nonresident", "total",
                phase_values(nonresident_samples, &IterationTiming::total_ms));
    std::cout << "gpu_e2e_nonresident effective_tflops=" << nonresident_tflops
              << " speedup_vs_cpu=" << (cpu_stats.median_ms / nonresident_total.median_ms)
              << " amortized_total_ms=" << nonresident_amortized_ms
              << " amortized_effective_tflops=" <<
                     (flops / (nonresident_amortized_ms * 1.0e9)) << '\n';

    std::cout << "gpu_e2e_resident setup_weight_upload_ms=" << setup_weight_ms
              << " setup_bias_upload_ms=" << setup_bias_ms << '\n';
    print_phase("gpu_e2e_resident", "input_upload",
                phase_values(resident_samples, &IterationTiming::input_upload_ms));
    print_phase("gpu_e2e_resident", "weight_upload",
                phase_values(resident_samples, &IterationTiming::weight_upload_ms));
    print_phase("gpu_e2e_resident", "bias_upload",
                phase_values(resident_samples, &IterationTiming::bias_upload_ms));
    print_phase("gpu_e2e_resident", "gpu_compute",
                phase_values(resident_samples, &IterationTiming::gpu_compute_ms));
    print_phase("gpu_e2e_resident", "output_download",
                phase_values(resident_samples, &IterationTiming::output_download_ms));
    print_phase("gpu_e2e_resident", "total",
                phase_values(resident_samples, &IterationTiming::total_ms));
    std::cout << "gpu_e2e_resident effective_tflops=" << resident_tflops
              << " speedup_vs_cpu=" << (cpu_stats.median_ms / resident_total.median_ms)
              << " amortized_total_ms=" << resident_amortized_ms
              << " amortized_effective_tflops=" <<
                     (flops / (resident_amortized_ms * 1.0e9)) << '\n';

    std::cout << "correctness coords_exact=" << (correctness.coords_exact ? 1 : 0)
              << " shape_exact=" << (correctness.shape_exact ? 1 : 0)
              << " cpu_nan=" << correctness.cpu_nan
              << " gpu_nan=" << correctness.gpu_nan
              << " cpu_inf=" << correctness.cpu_inf
              << " gpu_inf=" << correctness.gpu_inf
              << " max_abs=" << correctness.max_abs
              << " max_rel=" << correctness.max_rel
              << " mean_abs=" << correctness.mean_abs
              << " correctness_download_ms=" << correctness_download_ms << '\n';
    std::cout << "correctness_worst_abs point=" << correctness.worst_point
              << " channel=" << correctness.worst_channel
              << " cpu_value=" << correctness.cpu_value
              << " gpu_value=" << correctness.gpu_value
              << " abs=" << correctness.worst_abs
              << " rel=" << correctness.worst_rel << '\n';
    std::cout << "correctness_worst_rel point=" << correctness.worst_relative_point
              << " channel=" << correctness.worst_relative_channel
              << " cpu_value=" << correctness.worst_relative_cpu_value
              << " gpu_value=" << correctness.worst_relative_gpu_value
              << " abs=" << correctness.worst_relative_abs
              << " rel=" << correctness.worst_relative << '\n';

    if (fixture) {
        const bool pass = correctness.coords_exact && correctness.shape_exact &&
                          correctness.cpu_nan == 0 && correctness.gpu_nan == 0 &&
                          correctness.cpu_inf == 0 && correctness.gpu_inf == 0 &&
                          correctness.max_abs <= 1e-5 && correctness.max_rel <= 1e-5;
        std::cout << "fixture_result=" << (pass ? "PASS" : "FAIL")
                  << " tolerance_max_abs=1e-5 tolerance_max_rel=1e-5\n";
        if (!pass) {
            set_error(error, "sparse linear CUDA correctness fixture exceeded strict tolerance");
            return false;
        }
    }
    return true;
}

const Shape * find_shape(const std::string & name) {
    for (const Shape & shape : kHotShapes) {
        if (name == shape.name) return &shape;
    }
    return nullptr;
}

void print_usage(const char * program) {
    std::cerr << "usage: " << program << " [--fixture] [--warmup N] "
                 "[--iterations N] [--shape NAME]\n";
    std::cerr << "  without --fixture, runs the real profiling hotspot shapes; "
                 "--shape may be repeated\n";
}

bool parse_options(int argc, char ** argv, Options & options) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--fixture") {
            options.fixture = true;
        } else if (argument == "--warmup" && index + 1 < argc) {
            if (!parse_size(argv[++index], options.warmup)) return false;
        } else if (argument == "--iterations" && index + 1 < argc) {
            if (!parse_size(argv[++index], options.iterations)) return false;
        } else if (argument == "--shape" && index + 1 < argc) {
            options.selected_shapes.emplace_back(argv[++index]);
        } else if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            return false;
        }
    }
    return true;
}

int run(int argc, char ** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        print_usage(argv[0]);
        return 2;
    }

    std::string error;
    if (options.fixture) {
        for (const Shape & shape : kFixtureShapes) {
            if (!run_case(shape, options, true, &error)) {
                std::cerr << "sparse linear CUDA fixture failed for " << shape.name
                          << ": " << error << '\n';
                return 1;
            }
        }
        return 0;
    }

    std::vector<const Shape *> shapes;
    if (options.selected_shapes.empty()) {
        for (const Shape & shape : kHotShapes) shapes.push_back(&shape);
    } else {
        for (const std::string & name : options.selected_shapes) {
            const Shape * shape = find_shape(name);
            if (!shape) {
                std::cerr << "unknown shape: " << name << '\n';
                return 2;
            }
            shapes.push_back(shape);
        }
    }
    for (const Shape * shape : shapes) {
        if (!run_case(*shape, options, false, &error)) {
            std::cerr << "sparse linear CUDA benchmark failed for " << shape->name
                      << ": " << error << '\n';
            return 1;
        }
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    return run(argc, argv);
}
