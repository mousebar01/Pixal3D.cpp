#include "pixal3d/naf.h"

#include "pixal3d/pack.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pixal3d {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

bool prefix_name(const std::string & name, const char * prefix) {
    const std::size_t length = std::strlen(prefix);
    return name.size() >= length && name.compare(0, length, prefix) == 0;
}

bool checked_product(std::size_t a, std::size_t b, std::size_t & out) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) return false;
    out = a * b;
    return true;
}

int reflect_index(int index, int size) {
    if (size <= 1) return 0;
    while (index < 0 || index >= size) {
        if (index < 0) index = -index;
        else index = 2 * size - 2 - index;
    }
    return index;
}

float silu(float value) {
    return value / (1.0f + std::exp(-value));
}

struct TensorF32 {
    std::vector<std::int64_t> shape;
    std::vector<float> data;
};

// NAF's learned image stacks are ordinary 2-D convolutions.  ggml's CUDA
// backend exposes the convolution, reflection-padding, GroupNorm, and SiLU
// kernels needed here.  The surrounding adaptive pooling and neighborhood
// attention still use the scalar reference below; keeping this state lazy
// means CPU-only builds do not allocate a second copy of the checkpoint.
struct NafGpuState {
    ggml_context * weights_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    std::string backend_name;
    std::unordered_map<const float *, ggml_tensor *> tensors;
    bool attempted = false;

    ~NafGpuState() { close(); }

    void close() noexcept {
        tensors.clear();
        if (weights_buffer) {
            ggml_backend_buffer_free(weights_buffer);
            weights_buffer = nullptr;
        }
        if (backend) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
        if (weights_ctx) {
            ggml_free(weights_ctx);
            weights_ctx = nullptr;
        }
        backend_name.clear();
    }

    bool ready() const noexcept {
        return backend != nullptr && weights_buffer != nullptr;
    }

    ggml_tensor * tensor(const float * host) const noexcept {
        const auto found = tensors.find(host);
        return found == tensors.end() ? nullptr : found->second;
    }

    bool init(const std::unordered_map<std::string, TensorF32> & host,
              std::string * error) {
        if (ready()) return true;
        if (attempted) {
            if (error) *error = "NAF GPU initialization was unavailable";
            return false;
        }
        close();
        attempted = true;
        if (host.empty()) {
            if (error) *error = "NAF GPU initialization has no host tensors";
            return false;
        }
        if (host.size() > (std::numeric_limits<std::size_t>::max() /
                           ggml_tensor_overhead()) - 1) {
            if (error) *error = "NAF GPU tensor count overflows context size";
            return false;
        }
        ggml_init_params params{};
        params.mem_size = ggml_tensor_overhead() * (host.size() + 1) + 4096;
        params.no_alloc = true;
        weights_ctx = ggml_init(params);
        if (!weights_ctx) {
            if (error) *error = "failed to allocate NAF GPU tensor context";
            return false;
        }
        for (const auto & entry : host) {
            const TensorF32 & source = entry.second;
            if (source.shape.empty() || source.shape.size() > 4) {
                if (error) *error = "invalid NAF GPU tensor rank: " + entry.first;
                close();
                return false;
            }
            std::vector<int64_t> ne(source.shape.rbegin(), source.shape.rend());
            ggml_tensor * tensor = ggml_new_tensor(weights_ctx, GGML_TYPE_F32,
                                                   static_cast<int>(ne.size()), ne.data());
            if (!tensor || ggml_nbytes(tensor) != source.data.size() * sizeof(float)) {
                if (error) *error = "NAF GPU tensor shape mismatch: " + entry.first;
                close();
                return false;
            }
            ggml_set_name(tensor, entry.first.c_str());
            tensors.emplace(source.data.data(), tensor);
        }

        // Do not select a GPU that can merely allocate weights: every op in
        // run_conv_reflect must be supported by the selected backend.
        for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
            ggml_backend_dev_t device = ggml_backend_dev_get(index);
            if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            ggml_backend_t candidate = ggml_backend_dev_init(device, nullptr);
            if (!candidate) continue;
            ggml_init_params probe_params{};
            probe_params.mem_size = ggml_tensor_overhead() * 32 + 4096;
            probe_params.no_alloc = true;
            ggml_context * probe = ggml_init(probe_params);
            bool supported = false;
            if (probe) {
                ggml_tensor * weight = ggml_new_tensor_4d(probe, GGML_TYPE_F32,
                                                          3, 3, 1, 1);
                ggml_tensor * input = ggml_new_tensor_4d(probe, GGML_TYPE_F32,
                                                         8, 8, 1, 1);
                ggml_tensor * pad_x = weight && input
                    ? ggml_pad_reflect_1d(probe, input, 1, 1) : nullptr;
                ggml_tensor * swap = pad_x
                    ? ggml_cont(probe, ggml_permute(probe, pad_x, 1, 0, 2, 3)) : nullptr;
                ggml_tensor * pad_y = swap
                    ? ggml_pad_reflect_1d(probe, swap, 1, 1) : nullptr;
                ggml_tensor * padded = pad_y
                    ? ggml_cont(probe, ggml_permute(probe, pad_y, 1, 0, 2, 3)) : nullptr;
                ggml_tensor * conv = weight && padded
                    ? ggml_conv_2d_direct(probe, weight, padded,
                                          1, 1, 0, 0, 1, 1) : nullptr;
                ggml_tensor * bias = ggml_new_tensor_1d(probe, GGML_TYPE_F32, 1);
                ggml_tensor * biased = conv && bias
                    ? ggml_add(probe, conv, ggml_reshape_4d(probe, bias, 1, 1, 1, 1))
                    : nullptr;
                supported = pad_x && swap && pad_y && padded && conv && biased &&
                    ggml_backend_supports_op(candidate, pad_x) &&
                    ggml_backend_supports_op(candidate, pad_y) &&
                    ggml_backend_supports_op(candidate, conv) &&
                    ggml_backend_supports_op(candidate, biased);
                ggml_free(probe);
            }
            if (supported) {
                backend = candidate;
                const char * description = ggml_backend_dev_description(device);
                backend_name = description ? description : ggml_backend_dev_name(device);
                break;
            }
            ggml_backend_free(candidate);
        }
        if (!backend) {
            if (error) *error = "no GPU backend supports NAF convolution recipe";
            close();
            return false;
        }
        weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, backend);
        if (!weights_buffer) {
            if (error) *error = "failed to allocate NAF GPU weights on backend " + backend_name;
            close();
            return false;
        }
        ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const auto & entry : host) {
            ggml_backend_tensor_set(tensor(entry.second.data.data()), entry.second.data.data(),
                                    0, entry.second.data.size() * sizeof(float));
        }
        std::cerr << "pixal3d: NAF GPU convolution backend ready ("
                  << backend_name << ")" << std::endl;
        return true;
    }

    bool run_conv_reflect(const std::vector<float> & input,
                          int in_channels,
                          int height,
                          int width,
                          const TensorF32 & weight_host,
                          const TensorF32 & bias_host,
                          int kernel,
                          std::vector<float> & output,
                          std::string * error) const {
        if (!ready() || in_channels <= 0 || height <= 0 || width <= 0 ||
            kernel <= 0 || (kernel & 1) == 0 || height <= kernel / 2 || width <= kernel / 2) {
            if (error) *error = "invalid NAF GPU convolution dimensions";
            return false;
        }
        ggml_tensor * weight = tensor(weight_host.data.data());
        ggml_tensor * bias = tensor(bias_host.data.data());
        if (!weight || !bias || weight->ne[0] != kernel || weight->ne[1] != kernel ||
            weight->ne[2] != in_channels || bias->ne[0] != weight->ne[3]) {
            if (error) *error = "NAF GPU convolution tensor shape mismatch";
            return false;
        }
        const int out_channels = static_cast<int>(weight->ne[3]);
        if (input.size() != static_cast<std::size_t>(in_channels) * height * width) {
            if (error) *error = "NAF GPU convolution input size mismatch";
            return false;
        }
        ggml_init_params params{};
        params.mem_size = ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(128, false) + 4096;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            if (error) *error = "failed to allocate NAF GPU convolution graph";
            return false;
        }
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
        ggml_tensor * input_tensor = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F32, width, height, in_channels, 1);
        if (!graph || !input_tensor) {
            ggml_free(ctx);
            if (error) *error = "failed to allocate NAF GPU convolution input";
            return false;
        }
        ggml_set_input(input_tensor);
        ggml_tensor * pad_x = ggml_pad_reflect_1d(ctx, input_tensor, kernel / 2, kernel / 2);
        ggml_tensor * swapped = ggml_cont(ctx, ggml_permute(ctx, pad_x, 1, 0, 2, 3));
        ggml_tensor * pad_y = ggml_pad_reflect_1d(ctx, swapped, kernel / 2, kernel / 2);
        ggml_tensor * padded = ggml_cont(ctx, ggml_permute(ctx, pad_y, 1, 0, 2, 3));
        ggml_tensor * result = ggml_conv_2d_direct(ctx, weight, padded,
                                                   1, 1, 0, 0, 1, 1);
        ggml_tensor * bias_view = ggml_reshape_4d(ctx, bias, 1, 1, out_channels, 1);
        result = ggml_cont(ctx, ggml_add(ctx, result, bias_view));
        if (!pad_x || !swapped || !pad_y || !padded || !result) {
            ggml_free(ctx);
            if (error) *error = "failed to build NAF GPU convolution graph";
            return false;
        }
        ggml_set_output(result);
        ggml_build_forward_expand(graph, result);
        ggml_gallocr_t allocator = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
            if (allocator) ggml_gallocr_free(allocator);
            ggml_free(ctx);
            if (error) *error = "failed to allocate NAF GPU convolution graph";
            return false;
        }
        ggml_backend_tensor_set(input_tensor, input.data(), 0,
                                input.size() * sizeof(float));
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        bool ok = status == GGML_STATUS_SUCCESS;
        output.resize(static_cast<std::size_t>(out_channels) * height * width);
        if (ok) {
            ggml_backend_tensor_get(result, output.data(), 0,
                                    output.size() * sizeof(float));
            for (float value : output) {
                if (!std::isfinite(value)) {
                    if (error) *error = "NAF GPU convolution produced a non-finite value";
                    ok = false;
                    break;
                }
            }
        } else if (error) {
            *error = "NAF GPU convolution graph compute failed";
        }
        ggml_gallocr_free(allocator);
        ggml_free(ctx);
        return ok;
    }
};

bool conv_reflect(const std::vector<float> & input,
                  int in_channels,
                  int height,
                  int width,
                  const TensorF32 & weight,
                  const TensorF32 & bias,
                  int kernel,
                  std::vector<float> & output,
                  std::string * error) {
    if (weight.shape.size() != 4 || bias.shape.size() != 1 ||
        weight.shape[1] != in_channels || weight.shape[2] != kernel ||
        weight.shape[3] != kernel || bias.shape[0] != weight.shape[0]) {
        set_error(error, "NAF convolution tensor shape mismatch");
        return false;
    }
    const int out_channels = static_cast<int>(weight.shape[0]);
    const std::size_t output_size = static_cast<std::size_t>(out_channels) *
                                    static_cast<std::size_t>(height) *
                                    static_cast<std::size_t>(width);
    output.assign(output_size, 0.0f);
    const int radius = kernel / 2;
#if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float value = bias.data[static_cast<std::size_t>(oc)];
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int ky = 0; ky < kernel; ++ky) {
                        const int sy = reflect_index(y + ky - radius, height);
                        for (int kx = 0; kx < kernel; ++kx) {
                            const int sx = reflect_index(x + kx - radius, width);
                            const std::size_t input_index =
                                (static_cast<std::size_t>(ic) * height +
                                 static_cast<std::size_t>(sy)) * width +
                                static_cast<std::size_t>(sx);
                            const std::size_t weight_index =
                                (((static_cast<std::size_t>(oc) * in_channels + ic) * kernel + ky) *
                                 kernel) + kx;
                            value += input[input_index] * weight.data[weight_index];
                        }
                    }
                }
                output[(static_cast<std::size_t>(oc) * height +
                        static_cast<std::size_t>(y)) * width +
                       static_cast<std::size_t>(x)] = value;
            }
        }
    }
    return true;
}

bool group_norm(const std::vector<float> & input,
                int channels,
                int height,
                int width,
                int groups,
                const TensorF32 & gamma,
                const TensorF32 & beta,
                std::vector<float> & output,
                std::string * error) {
    if (groups <= 0 || channels <= 0 || channels % groups != 0 ||
        gamma.shape.size() != 1 || beta.shape.size() != 1 ||
        gamma.shape[0] != channels || beta.shape[0] != channels) {
        set_error(error, "NAF GroupNorm tensor shape mismatch");
        return false;
    }
    const int channels_per_group = channels / groups;
    const std::size_t plane = static_cast<std::size_t>(height) * width;
    output.resize(input.size());
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int group = 0; group < groups; ++group) {
        const int first = group * channels_per_group;
        const std::size_t count = static_cast<std::size_t>(channels_per_group) * plane;
        float mean = 0.0f;
        for (int channel = first; channel < first + channels_per_group; ++channel) {
            for (std::size_t pixel = 0; pixel < plane; ++pixel) {
                mean += input[static_cast<std::size_t>(channel) * plane + pixel];
            }
        }
        mean /= static_cast<float>(count);
        float variance = 0.0f;
        for (int channel = first; channel < first + channels_per_group; ++channel) {
            for (std::size_t pixel = 0; pixel < plane; ++pixel) {
                const float delta = input[static_cast<std::size_t>(channel) * plane + pixel] - mean;
                variance += delta * delta;
            }
        }
        variance /= static_cast<float>(count);
        const float inverse_std = 1.0f / std::sqrt(variance + 1e-5f);
        for (int channel = first; channel < first + channels_per_group; ++channel) {
            for (std::size_t pixel = 0; pixel < plane; ++pixel) {
                const std::size_t index = static_cast<std::size_t>(channel) * plane + pixel;
                output[index] = (input[index] - mean) * inverse_std *
                                gamma.data[static_cast<std::size_t>(channel)] +
                                beta.data[static_cast<std::size_t>(channel)];
            }
        }
    }
    return true;
}

bool adaptive_avg_pool(const std::vector<float> & input,
                       int channels,
                       int input_height,
                       int input_width,
                       int output_height,
                       int output_width,
                       std::vector<float> & output,
                       std::string * error) {
    if (input_height <= 0 || input_width <= 0 || output_height <= 0 || output_width <= 0) {
        set_error(error, "NAF adaptive-pool dimensions must be positive");
        return false;
    }
    const std::size_t input_plane = static_cast<std::size_t>(input_height) * input_width;
    const std::size_t output_plane = static_cast<std::size_t>(output_height) * output_width;
    output.assign(static_cast<std::size_t>(channels) * output_plane, 0.0f);
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int channel = 0; channel < channels; ++channel) {
        for (int oy = 0; oy < output_height; ++oy) {
            const int y0 = (oy * input_height) / output_height;
            const int y1 = (static_cast<int>((static_cast<std::int64_t>(oy + 1) * input_height +
                                               output_height - 1) / output_height));
            for (int ox = 0; ox < output_width; ++ox) {
                const int x0 = (ox * input_width) / output_width;
                const int x1 = (static_cast<int>((static_cast<std::int64_t>(ox + 1) * input_width +
                                               output_width - 1) / output_width));
                float sum = 0.0f;
                for (int iy = y0; iy < std::max(y1, y0 + 1); ++iy) {
                    for (int ix = x0; ix < std::max(x1, x0 + 1); ++ix) {
                        sum += input[static_cast<std::size_t>(channel) * input_plane +
                                     static_cast<std::size_t>(iy) * input_width + ix];
                    }
                }
                const int count = std::max(y1, y0 + 1) - y0;
                const int width_count = std::max(x1, x0 + 1) - x0;
                output[static_cast<std::size_t>(channel) * output_plane +
                       static_cast<std::size_t>(oy) * output_width + ox] =
                    sum / static_cast<float>(count * width_count);
            }
        }
    }
    return true;
}

bool bilinear_resize(const std::vector<float> & input,
                     int channels,
                     int input_height,
                     int input_width,
                     int output_height,
                     int output_width,
                     std::vector<float> & output,
                     std::string * error) {
    if (input_height <= 0 || input_width <= 0 || output_height <= 0 || output_width <= 0) {
        set_error(error, "NAF resize dimensions must be positive");
        return false;
    }
    const std::size_t input_plane = static_cast<std::size_t>(input_height) * input_width;
    const std::size_t output_plane = static_cast<std::size_t>(output_height) * output_width;
    output.assign(static_cast<std::size_t>(channels) * output_plane, 0.0f);
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int channel = 0; channel < channels; ++channel) {
        for (int oy = 0; oy < output_height; ++oy) {
            const float source_y = (static_cast<float>(oy) + 0.5f) * input_height /
                                   output_height - 0.5f;
            const float clamped_y = std::max(0.0f, std::min(source_y,
                                                    static_cast<float>(input_height - 1)));
            const int y0 = static_cast<int>(std::floor(clamped_y));
            const int y1 = std::min(y0 + 1, input_height - 1);
            const float wy = clamped_y - y0;
            for (int ox = 0; ox < output_width; ++ox) {
                const float source_x = (static_cast<float>(ox) + 0.5f) * input_width /
                                       output_width - 0.5f;
                const float clamped_x = std::max(0.0f, std::min(source_x,
                                                        static_cast<float>(input_width - 1)));
                const int x0 = static_cast<int>(std::floor(clamped_x));
                const int x1 = std::min(x0 + 1, input_width - 1);
                const float wx = clamped_x - x0;
                const auto at = [&](int y, int x) {
                    return input[static_cast<std::size_t>(channel) * input_plane +
                                 static_cast<std::size_t>(y) * input_width + x];
                };
                const float top = at(y0, x0) * (1.0f - wx) + at(y0, x1) * wx;
                const float bottom = at(y1, x0) * (1.0f - wx) + at(y1, x1) * wx;
                output[static_cast<std::size_t>(channel) * output_plane +
                       static_cast<std::size_t>(oy) * output_width + ox] =
                    top * (1.0f - wy) + bottom * wy;
            }
        }
    }
    return true;
}

} // namespace

bool NafOutputF32::valid(std::string * error) const {
    if (height <= 0 || width <= 0 || channels <= 0 ||
        features.size() != static_cast<std::size_t>(height) * width * channels) {
        set_error(error, "NAF output dimensions do not match payload");
        return false;
    }
    for (float value : features) {
        if (!std::isfinite(value)) {
            set_error(error, "NAF output contains a non-finite value");
            return false;
        }
    }
    return true;
}

struct NafModel::Impl {
    Pixal3DPackReader reader;
    NafHParams hp;
    std::unordered_map<std::string, TensorF32> tensors;
    mutable NafGpuState gpu;
    bool has_data = false;

    ~Impl() { close(); }

    void close() noexcept {
        tensors.clear();
        has_data = false;
        hp = NafHParams{};
        reader.close();
    }

    const TensorF32 * tensor(const std::string & name) const {
        const auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : &it->second;
    }

    bool init_gpu(std::string * error) const {
        if (!has_data) {
            if (error) *error = "NAF model was loaded metadata-only";
            return false;
        }
        return gpu.init(tensors, error);
    }
};

NafModel::~NafModel() { close(); }

void NafModel::close() noexcept {
    delete impl_;
    impl_ = nullptr;
}

bool NafModel::load(const std::string & path,
                    bool load_tensors,
                    std::string * error) {
    close();
    std::unique_ptr<Impl> impl(new Impl());
    if (!impl->reader.open(path, error)) return false;
    if (!impl->reader.info().has_tensor("naf.image_encoder.rope.periods")) {
        set_error(error, "GGUF pack does not contain NAF tensors");
        return false;
    }
    const std::string p = "pixal3d.naf.";
    std::uint32_t u = 0;
    if (!impl->reader.metadata_u32(p + "dim", u, true, error)) return false;
    impl->hp.dim = static_cast<int>(u);
    if (!impl->reader.metadata_u32(p + "in_channels", u, true, error)) return false;
    impl->hp.in_channels = static_cast<int>(u);
    if (!impl->reader.metadata_u32(p + "heads_attn", u, true, error)) return false;
    impl->hp.heads_attn = static_cast<int>(u);
    if (!impl->reader.metadata_u32(p + "heads_rope", u, true, error)) return false;
    impl->hp.heads_rope = static_cast<int>(u);
    if (!impl->reader.metadata_u32(p + "kernel_size", u, true, error)) return false;
    impl->hp.kernel_size = static_cast<int>(u);
    if (!impl->reader.metadata_u32(p + "img_layers", u, true, error)) return false;
    impl->hp.img_layers = static_cast<int>(u);
    if (!impl->reader.metadata_u32(p + "num_groups", u, true, error)) return false;
    impl->hp.num_groups = static_cast<int>(u);
    if (!impl->reader.metadata_f32(p + "rope_base", impl->hp.rope_base, true, error)) return false;
    if (impl->hp.dim <= 0 || impl->hp.dim % 2 != 0 || impl->hp.in_channels <= 0 ||
        impl->hp.heads_attn <= 0 || impl->hp.heads_rope <= 0 ||
        impl->hp.dim % impl->hp.heads_attn != 0 || impl->hp.dim % impl->hp.heads_rope != 0 ||
        (impl->hp.dim / impl->hp.heads_rope) % 4 != 0 || impl->hp.kernel_size <= 0 ||
        impl->hp.kernel_size % 2 == 0 || impl->hp.img_layers <= 0 ||
        impl->hp.num_groups <= 0 || (impl->hp.dim / 2) % impl->hp.num_groups != 0 ||
        !(impl->hp.rope_base > 0.0f) || !std::isfinite(impl->hp.rope_base)) {
        set_error(error, "invalid NAF hyperparameters");
        return false;
    }
    const int half = impl->hp.dim / 2;
    std::unordered_map<std::string, std::vector<std::int64_t>> expected;
    expected.emplace("image_encoder.encoder.0.weight",
                     std::vector<std::int64_t>{half, impl->hp.in_channels, 1, 1});
    expected.emplace("image_encoder.encoder.0.bias", std::vector<std::int64_t>{half});
    expected.emplace("image_encoder.sem_encoder.0.weight",
                     std::vector<std::int64_t>{half, impl->hp.in_channels, 3, 3});
    expected.emplace("image_encoder.sem_encoder.0.bias", std::vector<std::int64_t>{half});
    expected.emplace("image_encoder.rope.periods",
                     std::vector<std::int64_t>{impl->hp.dim / impl->hp.heads_rope / 4});
    for (const char * stack : {"encoder", "sem_encoder"}) {
        const int kernel = std::strcmp(stack, "encoder") == 0 ? 1 : 3;
        for (int layer = 1; layer <= impl->hp.img_layers; ++layer) {
            const std::string root = std::string("image_encoder.") + stack + "." +
                                     std::to_string(layer) + ".";
            expected.emplace(root + "norm1.weight", std::vector<std::int64_t>{half});
            expected.emplace(root + "norm1.bias", std::vector<std::int64_t>{half});
            expected.emplace(root + "conv1.weight",
                             std::vector<std::int64_t>{half, half, kernel, kernel});
            expected.emplace(root + "conv1.bias", std::vector<std::int64_t>{half});
            expected.emplace(root + "norm2.weight", std::vector<std::int64_t>{half});
            expected.emplace(root + "norm2.bias", std::vector<std::int64_t>{half});
            expected.emplace(root + "conv2.weight",
                             std::vector<std::int64_t>{half, half, kernel, kernel});
            expected.emplace(root + "conv2.bias", std::vector<std::int64_t>{half});
        }
    }
    for (const Pixal3DTensorInfo & info : impl->reader.info().tensors) {
        if (!prefix_name(info.name, "naf.")) continue;
        const std::string local_name = info.name.substr(4);
        const auto expected_it = expected.find(local_name);
        if (expected_it == expected.end()) {
            set_error(error, "unexpected NAF tensor: " + info.name);
            return false;
        }
        if (info.ggml_type != GGML_TYPE_F32 || info.n_bytes % sizeof(float) != 0) {
            set_error(error, "NAF currently requires F32 tensors: " + info.name);
            return false;
        }
        std::vector<std::int64_t> source_shape;
        source_shape.reserve(static_cast<std::size_t>(info.n_dims));
        for (int dim = info.n_dims - 1; dim >= 0; --dim) {
            source_shape.push_back(info.ne[dim]);
        }
        if (source_shape != expected_it->second) {
            set_error(error, "NAF tensor shape mismatch: " + info.name);
            return false;
        }
        expected.erase(expected_it);
        if (!load_tensors) continue;
        std::vector<std::uint8_t> payload;
        if (!impl->reader.read_tensor(info.name, payload, error)) return false;
        TensorF32 tensor;
        tensor.shape = source_shape;
        tensor.data.resize(info.n_bytes / sizeof(float));
        std::memcpy(tensor.data.data(), payload.data(), info.n_bytes);
        for (float value : tensor.data) {
            if (!std::isfinite(value)) {
                set_error(error, "NAF tensor contains a non-finite value: " + info.name);
                return false;
            }
        }
        impl->tensors.emplace(local_name, std::move(tensor));
    }
    if (!expected.empty()) {
        set_error(error, "NAF GGUF is missing tensor: " + expected.begin()->first);
        return false;
    }
    if (!load_tensors) {
        impl_ = impl.release();
        return true;
    }
    if (impl->tensors.empty()) {
        set_error(error, "NAF GGUF pack has no readable tensors");
        return false;
    }
    impl->has_data = true;
    impl_ = impl.release();
    return true;
}

bool NafModel::is_loaded() const noexcept { return impl_ != nullptr; }
bool NafModel::has_data() const noexcept { return impl_ && impl_->has_data; }
const NafHParams & NafModel::hparams() const noexcept {
    static const NafHParams empty;
    return impl_ ? impl_->hp : empty;
}
int NafModel::tensor_count() const noexcept {
    return impl_ ? static_cast<int>(impl_->reader.info().tensors.size()) : 0;
}
bool NafModel::has_tensor(const std::string & name) const noexcept {
    return impl_ && impl_->tensor(name) != nullptr;
}

bool NafModel::upsample(const float * image,
                        int image_height,
                        int image_width,
                        const float * low_resolution,
                        int low_height,
                        int low_width,
                        int low_channels,
                        int output_height,
                        int output_width,
                        NafOutputF32 & output,
                        std::string * error) const {
    output = NafOutputF32{};
    if (!impl_) {
        set_error(error, "NAF model is not loaded");
        return false;
    }
    if (!impl_->has_data) {
        set_error(error, "NAF model was loaded metadata-only");
        return false;
    }
    const NafHParams & hp = impl_->hp;
    if (!image || !low_resolution || image_height <= 0 || image_width <= 0 ||
        low_height <= 0 || low_width <= 0 || low_channels <= 0 || output_height <= 0 ||
        output_width <= 0 || image_height < low_height || image_width < low_width ||
        output_height < low_height || output_width < low_width ||
        output_height % low_height != 0 || output_width % low_width != 0 ||
        low_channels % hp.heads_attn != 0) {
        set_error(error, "invalid NAF image, feature-map, or output dimensions");
        return false;
    }
    std::size_t image_count = 0;
    if (!checked_product(static_cast<std::size_t>(image_height),
                         static_cast<std::size_t>(image_width), image_count) ||
        !checked_product(image_count, static_cast<std::size_t>(hp.in_channels), image_count)) {
        set_error(error, "NAF image size overflows size_t");
        return false;
    }
    std::size_t low_count = 0;
    if (!checked_product(static_cast<std::size_t>(low_height),
                         static_cast<std::size_t>(low_width), low_count) ||
        !checked_product(low_count, static_cast<std::size_t>(low_channels), low_count)) {
        set_error(error, "NAF feature-map size overflows size_t");
        return false;
    }
    for (std::size_t i = 0; i < image_count; ++i) {
        if (!std::isfinite(image[i])) {
            set_error(error, "NAF image contains a non-finite value");
            return false;
        }
    }
    for (std::size_t i = 0; i < low_count; ++i) {
        if (!std::isfinite(low_resolution[i])) {
            set_error(error, "NAF feature map contains a non-finite value");
            return false;
        }
    }
    auto T = [&](const std::string & name) -> const TensorF32 * {
        return impl_->tensor(name);
    };
    const int half = hp.dim / 2;
    std::vector<float> image_values(image, image + image_count);
    if (image_height > 4 * output_height || image_width > 4 * output_width) {
        const int resized_h = std::min(image_height, 4 * output_height);
        const int resized_w = std::min(image_width, 4 * output_width);
        std::vector<float> resized;
        if (!bilinear_resize(image_values, hp.in_channels, image_height, image_width,
                             resized_h, resized_w, resized, error)) return false;
        image_values.swap(resized);
        image_height = resized_h;
        image_width = resized_w;
    }

    std::string gpu_error;
    const char * backend_mode = std::getenv("PIXAL3D_NAF_BACKEND");
    bool use_gpu = !(backend_mode && std::string(backend_mode) == "cpu") &&
                   impl_->init_gpu(&gpu_error);

    auto run_stack_gpu = [&](const std::string & stack, int kernel,
                             std::vector<float> & stack_output) -> bool {
        const std::string root = "image_encoder." + stack + ".";
        const TensorF32 * first_w = T(root + "0.weight");
        const TensorF32 * first_b = T(root + "0.bias");
        if (!first_w || !first_b) {
            gpu_error = "missing NAF tensor: " + root + "0";
            return false;
        }
        const int height = image_height;
        const int width = image_width;
        if (!impl_->gpu.run_conv_reflect(image_values, hp.in_channels, height, width,
                                         *first_w, *first_b, kernel, stack_output,
                                         &gpu_error)) return false;
        for (int layer = 1; layer <= hp.img_layers; ++layer) {
            const std::string p = root + std::to_string(layer) + ".";
            const TensorF32 * n1w = T(p + "norm1.weight");
            const TensorF32 * n1b = T(p + "norm1.bias");
            const TensorF32 * c1w = T(p + "conv1.weight");
            const TensorF32 * c1b = T(p + "conv1.bias");
            const TensorF32 * n2w = T(p + "norm2.weight");
            const TensorF32 * n2b = T(p + "norm2.bias");
            const TensorF32 * c2w = T(p + "conv2.weight");
            const TensorF32 * c2b = T(p + "conv2.bias");
            if (!n1w || !n1b || !c1w || !c1b || !n2w || !n2b || !c2w || !c2b) {
                gpu_error = "missing NAF tensor in " + p;
                return false;
            }
            std::vector<float> normalized;
            if (!group_norm(stack_output, half, height, width, hp.num_groups,
                            *n1w, *n1b, normalized, &gpu_error)) return false;
            for (float & value : normalized) value = silu(value);
            std::vector<float> hidden;
            if (!impl_->gpu.run_conv_reflect(normalized, half, height, width,
                                             *c1w, *c1b, kernel, hidden,
                                             &gpu_error)) return false;
            if (!group_norm(hidden, half, height, width, hp.num_groups,
                            *n2w, *n2b, normalized, &gpu_error)) return false;
            for (float & value : normalized) value = silu(value);
            if (!impl_->gpu.run_conv_reflect(normalized, half, height, width,
                                             *c2w, *c2b, kernel, stack_output,
                                             &gpu_error)) return false;
        }
        return true;
    };

    auto run_stack = [&](const std::string & stack, int kernel,
                         std::vector<float> & stack_output) -> bool {
        const std::string root = "image_encoder." + stack + ".";
        const TensorF32 * first_w = T(root + "0.weight");
        const TensorF32 * first_b = T(root + "0.bias");
        if (!first_w || !first_b) {
            set_error(error, "missing NAF tensor: " + root + "0");
            return false;
        }
        int height = image_height;
        int width = image_width;
        if (!conv_reflect(image_values, hp.in_channels, height, width,
                          *first_w, *first_b, kernel, stack_output, error)) return false;
        for (int layer = 1; layer <= hp.img_layers; ++layer) {
            const std::string p = root + std::to_string(layer) + ".";
            const TensorF32 * n1w = T(p + "norm1.weight");
            const TensorF32 * n1b = T(p + "norm1.bias");
            const TensorF32 * c1w = T(p + "conv1.weight");
            const TensorF32 * c1b = T(p + "conv1.bias");
            const TensorF32 * n2w = T(p + "norm2.weight");
            const TensorF32 * n2b = T(p + "norm2.bias");
            const TensorF32 * c2w = T(p + "conv2.weight");
            const TensorF32 * c2b = T(p + "conv2.bias");
            if (!n1w || !n1b || !c1w || !c1b || !n2w || !n2b || !c2w || !c2b) {
                set_error(error, "missing NAF tensor in " + p);
                return false;
            }
            std::vector<float> normalized;
            if (!group_norm(stack_output, half, height, width, hp.num_groups,
                            *n1w, *n1b, normalized, error)) return false;
            for (float & value : normalized) value = silu(value);
            std::vector<float> hidden;
            if (!conv_reflect(normalized, half, height, width, *c1w, *c1b,
                              kernel, hidden, error)) return false;
            if (!group_norm(hidden, half, height, width, hp.num_groups,
                            *n2w, *n2b, normalized, error)) return false;
            for (float & value : normalized) value = silu(value);
            if (!conv_reflect(normalized, half, height, width, *c2w, *c2b,
                              kernel, stack_output, error)) return false;
        }
        return true;
    };

    std::vector<float> linear_stack;
    std::vector<float> semantic_stack;
    if (use_gpu) {
        use_gpu = run_stack_gpu("encoder", 1, linear_stack) &&
                  run_stack_gpu("sem_encoder", 3, semantic_stack);
        if (!use_gpu && std::getenv("PIXAL3D_SLAT_VERBOSE")) {
            std::cerr << "pixal3d: NAF GPU convolution unavailable; using CPU fallback: "
                      << gpu_error << std::endl;
        }
    }
    if (!use_gpu && (!run_stack("encoder", 1, linear_stack) ||
                     !run_stack("sem_encoder", 3, semantic_stack))) return false;
    const int encoder_height = image_height;
    const int encoder_width = image_width;
    const std::size_t encoder_plane = static_cast<std::size_t>(encoder_height) * encoder_width;
    std::vector<float> encoded(static_cast<std::size_t>(hp.dim) * encoder_plane);
    for (int channel = 0; channel < half; ++channel) {
        std::copy(linear_stack.begin() + static_cast<std::size_t>(channel) * encoder_plane,
                  linear_stack.begin() + static_cast<std::size_t>(channel + 1) * encoder_plane,
                  encoded.begin() + static_cast<std::size_t>(channel) * encoder_plane);
        std::copy(semantic_stack.begin() + static_cast<std::size_t>(channel) * encoder_plane,
                  semantic_stack.begin() + static_cast<std::size_t>(channel + 1) * encoder_plane,
                  encoded.begin() + static_cast<std::size_t>(channel + half) * encoder_plane);
    }
    std::vector<float> pooled;
    if (!adaptive_avg_pool(encoded, hp.dim, encoder_height, encoder_width,
                           output_height, output_width, pooled, error)) return false;

    const TensorF32 * periods = T("image_encoder.rope.periods");
    if (!periods || periods->shape.size() != 1 ||
        periods->shape[0] != hp.dim / hp.heads_rope / 4) {
        set_error(error, "NAF RoPE periods tensor shape mismatch");
        return false;
    }
    const int rope_head_dim = hp.dim / hp.heads_rope;
    const int rope_freqs = rope_head_dim / 4;
    const std::size_t output_plane = static_cast<std::size_t>(output_height) * output_width;
    std::vector<float> q = pooled;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int head = 0; head < hp.heads_rope; ++head) {
        for (int y = 0; y < output_height; ++y) {
            const float coord_y = 2.0f * ((static_cast<float>(y) + 0.5f) / output_height) - 1.0f;
            for (int x = 0; x < output_width; ++x) {
                const float coord_x = 2.0f * ((static_cast<float>(x) + 0.5f) / output_width) - 1.0f;
                for (int d = 0; d < rope_head_dim; ++d) {
                    const int base = d < 2 * rope_freqs ? d : d - 2 * rope_freqs;
                    const int axis = base / rope_freqs;
                    const int frequency = base % rope_freqs;
                    const float coordinate = axis == 0 ? coord_y : coord_x;
                    const float angle = 2.0f * 3.14159265358979323846f * coordinate /
                                        periods->data[static_cast<std::size_t>(frequency)];
                    const int channel = head * rope_head_dim + d;
                    const int rotated_channel = d < rope_head_dim / 2
                        ? channel + rope_head_dim / 2 : channel - rope_head_dim / 2;
                    const std::size_t index = static_cast<std::size_t>(channel) * output_plane +
                                              static_cast<std::size_t>(y) * output_width + x;
                    const std::size_t rotated_index = static_cast<std::size_t>(rotated_channel) *
                                                      output_plane +
                                                      static_cast<std::size_t>(y) * output_width + x;
                    const float rotated = d < rope_head_dim / 2
                        ? -pooled[rotated_index] : pooled[rotated_index];
                    q[index] = pooled[index] * std::cos(angle) + rotated * std::sin(angle);
                }
            }
        }
    }

    // The reference CrossAttention keys are an adaptive-average-pooled copy
    // of the HR image-encoder output.  Values remain the caller's LR feature
    // map and are sampled with nearest-exact HR->LR coordinates below.
    std::vector<float> k_lr;
    if (!adaptive_avg_pool(q, hp.dim, output_height, output_width,
                           low_height, low_width, k_lr, error)) return false;

    const int dilation_y = output_height / low_height;
    const int dilation_x = output_width / low_width;
    const int radius = hp.kernel_size / 2;
    const int query_head_dim = hp.dim / hp.heads_attn;
    const int value_head_dim = low_channels / hp.heads_attn;
    const float scale = 1.0f / std::sqrt(static_cast<float>(query_head_dim));
    const std::size_t low_plane = static_cast<std::size_t>(low_height) * low_width;
    output.height = output_height;
    output.width = output_width;
    output.channels = low_channels;
    output.features.assign(static_cast<std::size_t>(output_height) * output_width * low_channels, 0.0f);
#if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            // These workspaces are per output pixel.  Keeping them inside the
            // parallel region avoids cross-thread races in neighborhood
            // softmax while preserving the reference accumulation order.
            std::vector<float> scores(static_cast<std::size_t>(hp.kernel_size) *
                                      hp.kernel_size);
            std::vector<float> probabilities(scores.size());
            for (int head = 0; head < hp.heads_attn; ++head) {
                int score_index = 0;
                float maximum = -std::numeric_limits<float>::infinity();
                for (int ky = -radius; ky <= radius; ++ky) {
                    const int hy = y + ky * dilation_y;
                    for (int kx = -radius; kx <= radius; ++kx, ++score_index) {
                        const int hx = x + kx * dilation_x;
                        float score = 0.0f;
                        if (hy >= 0 && hy < output_height && hx >= 0 && hx < output_width) {
                            const int ly = hy / dilation_y;
                            const int lx = hx / dilation_x;
                            for (int d = 0; d < query_head_dim; ++d) {
                                const std::size_t q_index =
                                    static_cast<std::size_t>(head * query_head_dim + d) * output_plane +
                                    static_cast<std::size_t>(y) * output_width + x;
                                const std::size_t k_index =
                                    static_cast<std::size_t>(head * query_head_dim + d) * low_plane +
                                    static_cast<std::size_t>(ly) * low_width + lx;
                                score += q[q_index] * k_lr[k_index];
                            }
                            score *= scale;
                        }
                        scores[static_cast<std::size_t>(score_index)] = score;
                        maximum = std::max(maximum, score);
                    }
                }
                float normalizer = 0.0f;
                for (std::size_t index = 0; index < scores.size(); ++index) {
                    probabilities[index] = std::exp(scores[index] - maximum);
                    normalizer += probabilities[index];
                }
                for (float & probability : probabilities) probability /= normalizer;
                for (int value = 0; value < value_head_dim; ++value) {
                    float accumulated = 0.0f;
                    score_index = 0;
                    for (int ky = -radius; ky <= radius; ++ky) {
                        const int hy = y + ky * dilation_y;
                        for (int kx = -radius; kx <= radius; ++kx, ++score_index) {
                            const int hx = x + kx * dilation_x;
                            if (hy < 0 || hy >= output_height || hx < 0 || hx >= output_width) continue;
                            const int ly = hy / dilation_y;
                            const int lx = hx / dilation_x;
                            const std::size_t v_index =
                                (static_cast<std::size_t>(ly) * low_width + lx) * low_channels +
                                static_cast<std::size_t>(head * value_head_dim + value);
                            accumulated += probabilities[static_cast<std::size_t>(score_index)] *
                                           low_resolution[v_index];
                        }
                    }
                    output.features[(static_cast<std::size_t>(y) * output_width + x) * low_channels +
                                    static_cast<std::size_t>(head * value_head_dim + value)] = accumulated;
                }
            }
        }
    }
    return output.valid(error);
}

} // namespace pixal3d
