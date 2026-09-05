#include "pixal3d/ss_decoder.h"

#include "pixal3d/pack.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

bool checked_cube(int resolution, std::size_t & points, std::string * error) {
    if (resolution <= 0) {
        set_error(error, "decoder resolution must be positive");
        return false;
    }
    const std::size_t r = static_cast<std::size_t>(resolution);
    if (r > std::numeric_limits<std::size_t>::max() / r ||
        r * r > std::numeric_limits<std::size_t>::max() / r) {
        set_error(error, "decoder resolution^3 overflows size_t");
        return false;
    }
    points = r * r * r;
    return true;
}

bool sparse_dense_to_channel_major(const SparseTensorF32 & latent,
                                   int resolution,
                                   int channels,
                                   std::vector<float> & output,
                                   std::string * error) {
    if (!latent.valid(error)) return false;
    std::size_t points = 0;
    if (!checked_cube(resolution, points, error)) return false;
    if (latent.batch_size != 1 || latent.channels != channels ||
        latent.spatial_x != resolution || latent.spatial_y != resolution ||
        latent.spatial_z != resolution || latent.points() != points) {
        set_error(error, "sparse SS latent shape does not match decoder input grid");
        return false;
    }
    output.resize(points * static_cast<std::size_t>(channels));
    for (std::size_t point = 0; point < points; ++point) {
        const std::size_t coordinate_base = point * 4;
        const std::int32_t expected_x = static_cast<std::int32_t>(
            point / static_cast<std::size_t>(resolution * resolution));
        const std::int32_t expected_y = static_cast<std::int32_t>(
            (point / static_cast<std::size_t>(resolution)) %
            static_cast<std::size_t>(resolution));
        const std::int32_t expected_z = static_cast<std::int32_t>(
            point % static_cast<std::size_t>(resolution));
        if (latent.coords[coordinate_base] != 0 ||
            latent.coords[coordinate_base + 1] != expected_x ||
            latent.coords[coordinate_base + 2] != expected_y ||
            latent.coords[coordinate_base + 3] != expected_z) {
            set_error(error, "sparse SS latent coordinates are not lexicographic");
            return false;
        }
        for (int channel = 0; channel < channels; ++channel) {
            output[static_cast<std::size_t>(channel) * points + point] =
                latent.feats[point * static_cast<std::size_t>(channels) +
                             static_cast<std::size_t>(channel)];
        }
    }
    return true;
}

ggml_backend_t init_best_backend(std::string & name_out) {
    for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) continue;

        // The SS decoder graph contains 3-D convolutions.  The direct Conv3D
        // op is not implemented by every ggml GPU backend, but the equivalent
        // im2col-3d plus matmul recipe is widely available.  Probe both
        // primitive nodes before committing the model weights to a GPU
        // buffer; otherwise fall back cleanly to the CPU backend.
        ggml_init_params params{};
        params.mem_size = ggml_tensor_overhead() * 4 + 4096;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_context * context = ggml_init(params);
        bool supports_conv3d = false;
        if (context) {
            ggml_tensor * weight = ggml_new_tensor_4d(context, GGML_TYPE_F32,
                                                       3, 3, 3, 1);
            ggml_tensor * input = ggml_new_tensor_4d(context, GGML_TYPE_F32,
                                                      4, 4, 4, 1);
            ggml_tensor * im2col = weight && input
                ? ggml_im2col_3d(context, weight, input, 1,
                                 1, 1, 1, 0, 0, 0, 1, 1, 1, GGML_TYPE_F32)
                : nullptr;
            ggml_tensor * im2col_matrix = im2col
                ? ggml_reshape_2d(context, im2col, im2col->ne[0],
                                  im2col->ne[1] * im2col->ne[2] * im2col->ne[3])
                : nullptr;
            ggml_tensor * weight_matrix = weight
                ? ggml_reshape_2d(context, weight,
                                  weight->ne[0] * weight->ne[1] * weight->ne[2],
                                  weight->ne[3])
                : nullptr;
            ggml_tensor * product = weight_matrix && im2col_matrix
                ? ggml_mul_mat(context, weight_matrix, im2col_matrix) : nullptr;
            supports_conv3d = im2col && product &&
                ggml_backend_supports_op(backend, im2col) &&
                ggml_backend_supports_op(backend, product);
            ggml_free(context);
        }
        if (supports_conv3d) {
            const char * description = ggml_backend_dev_description(device);
            name_out = description ? description : ggml_backend_dev_name(device);
            return backend;
        }
        ggml_backend_free(backend);
    }
    name_out = "CPU";
    return ggml_backend_cpu_init();
}

bool read_hparams(const Pixal3DPackReader & reader,
                  SSDecoderHParams & hp,
                  std::string * error) {
    const std::string prefix = "pixal3d.ss_decoder.";
    std::uint32_t value = 0;
    if (!reader.metadata_u32(prefix + "out_channels", value, true, error)) return false;
    hp.out_channels = static_cast<int>(value);
    if (!reader.metadata_u32(prefix + "latent_channels", value, true, error)) return false;
    hp.latent_channels = static_cast<int>(value);
    if (!reader.metadata_u32(prefix + "num_res_blocks", value, true, error)) return false;
    hp.num_res_blocks = static_cast<int>(value);
    if (!reader.metadata_u32(prefix + "num_res_blocks_middle", value, true, error)) return false;
    hp.num_res_blocks_middle = static_cast<int>(value);
    if (!reader.metadata_u32(prefix + "n_levels", value, true, error)) return false;
    const int n_levels = static_cast<int>(value);
    if (!reader.metadata_u32(prefix + "resolution", value, false, error)) return false;
    hp.resolution = value == 0 ? 16 : static_cast<int>(value);
    if (!reader.metadata_f32(prefix + "norm_eps", hp.norm_eps, false, error)) return false;
    if (hp.norm_eps == 0.0f) hp.norm_eps = 1e-5f;
    if (!reader.metadata_string(prefix + "norm_type", hp.norm_type, false, error)) return false;
    if (hp.norm_type.empty()) hp.norm_type = "layer";

    if (n_levels <= 0 || n_levels > 8 || hp.out_channels <= 0 ||
        hp.latent_channels <= 0 || hp.num_res_blocks <= 0 ||
        hp.num_res_blocks_middle <= 0 || hp.norm_type != "layer" ||
        !(hp.norm_eps > 0.0f) || !std::isfinite(hp.norm_eps)) {
        set_error(error, "invalid sparse-structure decoder hyperparameters");
        return false;
    }
    hp.channels.clear();
    hp.channels.reserve(static_cast<std::size_t>(n_levels));
    for (int level = 0; level < n_levels; ++level) {
        if (!reader.metadata_u32(prefix + "channels." + std::to_string(level),
                                 value, true, error)) return false;
        if (value == 0 || value > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
            set_error(error, "invalid decoder channel count");
            return false;
        }
        hp.channels.push_back(static_cast<int>(value));
    }
    std::size_t ignored = 0;
    return checked_cube(hp.resolution, ignored, error);
}

} // namespace

bool ss_occupancy_to_coords_f32(const float * occupancy_logits,
                                int out_channels,
                                int output_resolution,
                                float threshold,
                                int target_resolution,
                                std::vector<std::int32_t> & coords,
                                std::string * error) {
    coords.clear();
    if (!occupancy_logits || out_channels != 1 || output_resolution <= 0 ||
        !std::isfinite(threshold)) {
        set_error(error, "invalid occupancy grid arguments");
        return false;
    }
    std::size_t output_points = 0;
    if (!checked_cube(output_resolution, output_points, error)) return false;
    const int target = target_resolution <= 0 ? output_resolution : target_resolution;
    if (target <= 0 || target > output_resolution || output_resolution % target != 0) {
        set_error(error, "occupancy target resolution must divide decoder resolution");
        return false;
    }
    const int factor = output_resolution / target;
    const std::size_t target_points = static_cast<std::size_t>(target) *
                                      static_cast<std::size_t>(target) *
                                      static_cast<std::size_t>(target);
    coords.reserve(target_points * 4);
    for (int x = 0; x < target; ++x) {
        for (int y = 0; y < target; ++y) {
            for (int z = 0; z < target; ++z) {
                bool occupied = false;
                for (int dx = 0; dx < factor && !occupied; ++dx) {
                    for (int dy = 0; dy < factor && !occupied; ++dy) {
                        for (int dz = 0; dz < factor; ++dz) {
                            const int source_x = x * factor + dx;
                            const int source_y = y * factor + dy;
                            const int source_z = z * factor + dz;
                            const std::size_t index =
                                (static_cast<std::size_t>(source_x) *
                                 static_cast<std::size_t>(output_resolution) +
                                 static_cast<std::size_t>(source_y)) *
                                    static_cast<std::size_t>(output_resolution) +
                                static_cast<std::size_t>(source_z);
                            const float value = occupancy_logits[index];
                            if (!std::isfinite(value)) {
                                set_error(error, "occupancy grid contains a non-finite value");
                                coords.clear();
                                return false;
                            }
                            if (value > threshold) {
                                occupied = true;
                                break;
                            }
                        }
                    }
                }
                if (occupied) {
                    coords.insert(coords.end(), {0, static_cast<std::int32_t>(x),
                                                 static_cast<std::int32_t>(y),
                                                 static_cast<std::int32_t>(z)});
                }
            }
        }
    }
    (void) output_points;
    return true;
}

struct SSDecoderModel::Impl {
    Pixal3DPackReader reader;
    SSDecoderHParams hp;
    ggml_context * weights_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    std::string backend_name;
    bool has_data = false;
    std::unordered_map<std::string, ggml_tensor *> tensors;

    ~Impl() { close(); }

    void close() noexcept {
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
        tensors.clear();
        backend_name.clear();
        has_data = false;
        hp = SSDecoderHParams{};
        reader.close();
    }

    ggml_tensor * weight(const std::string & local_name) const {
        const auto it = tensors.find("ss_decoder." + local_name);
        return it == tensors.end() ? nullptr : it->second;
    }
};

SSDecoderModel::~SSDecoderModel() {
    delete impl_;
    impl_ = nullptr;
}

void SSDecoderModel::close() noexcept {
    delete impl_;
    impl_ = nullptr;
}

bool SSDecoderModel::load(const std::string & path,
                          bool load_tensors,
                          std::string * error) {
    close();
    std::unique_ptr<Impl> impl(new Impl());
    if (!impl->reader.open(path, error)) return false;
    if (!impl->reader.info().has_tensor("ss_decoder.input_layer.weight")) {
        set_error(error, "GGUF pack does not contain ss-decoder tensors");
        return false;
    }
    if (!read_hparams(impl->reader, impl->hp, error)) return false;

    std::size_t tensor_count = 0;
    for (const Pixal3DTensorInfo & info : impl->reader.info().tensors) {
        if (prefix_name(info.name, "ss_decoder.")) ++tensor_count;
    }
    if (tensor_count == 0 || tensor_count >
        (std::numeric_limits<std::size_t>::max() / ggml_tensor_overhead()) - 1) {
        set_error(error, "invalid ss-decoder tensor count");
        return false;
    }
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * (tensor_count + 1) + 4096;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    impl->weights_ctx = ggml_init(params);
    if (!impl->weights_ctx) {
        set_error(error, "failed to allocate ss-decoder tensor context");
        return false;
    }
    for (const Pixal3DTensorInfo & info : impl->reader.info().tensors) {
        if (!prefix_name(info.name, "ss_decoder.")) continue;
        ggml_tensor * tensor = ggml_new_tensor(
            impl->weights_ctx, static_cast<ggml_type>(info.ggml_type),
            info.n_dims, info.ne);
        if (!tensor) {
            set_error(error, "failed to create tensor descriptor: " + info.name);
            return false;
        }
        ggml_set_name(tensor, info.name.c_str());
        if (ggml_nbytes(tensor) != info.n_bytes) {
            set_error(error, "GGUF tensor byte size mismatch: " + info.name);
            return false;
        }
        impl->tensors.emplace(info.name, tensor);
    }
    if (load_tensors) {
        impl->backend = init_best_backend(impl->backend_name);
        if (!impl->backend) {
            set_error(error, "failed to initialize ggml backend");
            return false;
        }
        impl->weights_buffer = ggml_backend_alloc_ctx_tensors(
            impl->weights_ctx, impl->backend);
        if (!impl->weights_buffer) {
            set_error(error, "failed to allocate ss-decoder weights on backend " +
                             impl->backend_name);
            return false;
        }
        ggml_backend_buffer_set_usage(impl->weights_buffer,
                                      GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<std::uint8_t> payload;
        for (const Pixal3DTensorInfo & info : impl->reader.info().tensors) {
            if (!prefix_name(info.name, "ss_decoder.")) continue;
            if (!impl->reader.read_tensor(info.name, payload, error)) return false;
            ggml_backend_tensor_set(impl->tensors.at(info.name), payload.data(),
                                     0, payload.size());
        }
        impl->has_data = true;
    }
    impl_ = impl.release();
    return true;
}

bool SSDecoderModel::is_loaded() const noexcept { return impl_ != nullptr; }
bool SSDecoderModel::has_data() const noexcept { return impl_ && impl_->has_data; }

const SSDecoderHParams & SSDecoderModel::hparams() const noexcept {
    static const SSDecoderHParams empty;
    return impl_ ? impl_->hp : empty;
}

const std::string & SSDecoderModel::backend_name() const noexcept {
    static const std::string none = "none";
    return impl_ && !impl_->backend_name.empty() ? impl_->backend_name : none;
}

int SSDecoderModel::tensor_count() const noexcept {
    return impl_ ? static_cast<int>(impl_->tensors.size()) : 0;
}

bool SSDecoderModel::has_tensor(const std::string & name) const noexcept {
    return impl_ && impl_->tensors.find(name) != impl_->tensors.end();
}

bool SSDecoderModel::decode(const float * latent,
                            float * output,
                            std::string * error) {
    if (!impl_) {
        set_error(error, "ss-decoder model is not loaded");
        return false;
    }
    if (!impl_->has_data) {
        set_error(error, "ss-decoder model was loaded metadata-only");
        return false;
    }
    const SSDecoderHParams & hp = impl_->hp;
    std::size_t input_points = 0;
    if (!checked_cube(hp.resolution, input_points, error)) return false;
    const int output_resolution = hp.output_resolution();
    std::size_t output_points = 0;
    if (!checked_cube(output_resolution, output_points, error)) return false;
    if (!latent || !output) {
        set_error(error, "latent/output buffer is null");
        return false;
    }

    std::string missing;
    auto W = [&](const std::string & local_name) -> ggml_tensor * {
        ggml_tensor * tensor = impl_->weight(local_name);
        if (!tensor && missing.empty()) missing = "ss_decoder." + local_name;
        return tensor;
    };
    auto require = [&](const std::string & local_name) -> bool {
        return W(local_name) != nullptr;
    };
    auto require_resblock = [&](const std::string & prefix) {
        return require(prefix + ".norm1.weight") && require(prefix + ".norm1.bias") &&
               require(prefix + ".conv1.weight") && require(prefix + ".conv1.bias") &&
               require(prefix + ".norm2.weight") && require(prefix + ".norm2.bias") &&
               require(prefix + ".conv2.weight") && require(prefix + ".conv2.bias");
    };
    if (!require("input_layer.weight") || !require("input_layer.bias") ||
        !require("out_layer.0.weight") || !require("out_layer.0.bias") ||
        !require("out_layer.2.weight") || !require("out_layer.2.bias")) {
        set_error(error, "missing ss-decoder tensor: " + missing);
        return false;
    }
    for (int index = 0; index < hp.num_res_blocks_middle; ++index) {
        if (!require_resblock("middle_block." + std::to_string(index))) {
            set_error(error, "missing ss-decoder tensor: " + missing);
            return false;
        }
    }
    int block = 0;
    for (std::size_t level = 0; level < hp.channels.size(); ++level) {
        const int channels = hp.channels[level];
        if (channels <= 0) {
            set_error(error, "invalid ss-decoder channel count");
            return false;
        }
        for (int index = 0; index < hp.num_res_blocks; ++index) {
            if (!require_resblock("blocks." + std::to_string(block++))) {
                set_error(error, "missing ss-decoder tensor: " + missing);
                return false;
            }
        }
        if (level + 1 < hp.channels.size()) {
            const int next = hp.channels[level + 1];
            if (!require("blocks." + std::to_string(block) + ".conv.weight") ||
                !require("blocks." + std::to_string(block) + ".conv.bias")) {
                set_error(error, "missing ss-decoder tensor: " + missing);
                return false;
            }
            ++block;
            (void) next;
        }
    }

    const std::size_t graph_memory = ggml_tensor_overhead() * 8192 +
                                     ggml_graph_overhead_custom(8192, false);
    ggml_init_params graph_params{};
    graph_params.mem_size = graph_memory;
    graph_params.mem_buffer = nullptr;
    graph_params.no_alloc = true;
    ggml_context * ctx = ggml_init(graph_params);
    if (!ctx) {
        set_error(error, "failed to allocate ss-decoder graph context");
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    if (!graph) {
        ggml_free(ctx);
        set_error(error, "failed to allocate ss-decoder graph");
        return false;
    }
    const int input_channels = hp.latent_channels;
    ggml_tensor * input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                             hp.resolution, hp.resolution,
                                             hp.resolution, input_channels);
    if (!input) {
        ggml_free(ctx);
        set_error(error, "failed to allocate ss-decoder input");
        return false;
    }
    ggml_set_input(input);
    ggml_set_name(input, "z_s");

    auto conv = [&](ggml_tensor * in, const std::string & local_name,
                    int channels_in, int channels_out) {
        ggml_tensor * weight = W(local_name + ".weight");
        ggml_tensor * bias = W(local_name + ".bias");
        if (ggml_backend_is_cpu(impl_->backend)) {
            // The CPU backend has a dedicated Conv3d implementation which
            // accepts the native F16-kernel path.  Keep it for the validated
            // reference/fallback route; only CUDA needs the explicit recipe
            // below because its direct Conv3d op is not available.
            ggml_tensor * result = ggml_conv_3d(
                ctx, weight, in, channels_in, 1, 1, 1, 1, 1, 1, 1, 1, 1);
            return ggml_add(ctx, result,
                            ggml_reshape_4d(ctx, bias, 1, 1, 1, channels_out));
        }
        // Keep the im2col activation matrix in F32 even when the serialized
        // decoder kernel is F16.  The CUDA im2col kernel supports both types,
        // but F16 intermediate activations can overflow in the deep decoder
        // before the following LayerNorm brings them back to scale.  The
        // weight matrix remains in its stored type and ggml's matmul promotes
        // the accumulation to F32, matching the CPU Conv3d reference.
        ggml_tensor * im2col = ggml_im2col_3d(
            ctx, weight, in, channels_in, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            GGML_TYPE_F32);
        ggml_tensor * im2col_matrix = ggml_reshape_2d(
            ctx, im2col, im2col->ne[0],
            im2col->ne[3] * im2col->ne[2] * im2col->ne[1]);
        ggml_tensor * weight_matrix = ggml_reshape_2d(
            ctx, weight, weight->ne[0] * weight->ne[1] * weight->ne[2] *
            channels_in, weight->ne[3] / channels_in);
        ggml_tensor * result = ggml_mul_mat(ctx, im2col_matrix, weight_matrix);
        const int64_t output_depth = im2col->ne[3] / (in->ne[3] / channels_in);
        result = ggml_reshape_4d(ctx, result,
                                 im2col->ne[1] * im2col->ne[2],
                                 output_depth, in->ne[3] / channels_in,
                                 weight->ne[3] / channels_in);
        result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2));
        result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2],
                                 output_depth,
                                 (weight->ne[3] / channels_in) *
                                 (in->ne[3] / channels_in));
        return ggml_add(ctx, result,
                        ggml_reshape_4d(ctx, bias, 1, 1, 1, channels_out));
    };
    auto channel_norm = [&](ggml_tensor * in, const std::string & local_name) {
        ggml_tensor * permuted = ggml_cont(ctx, ggml_permute(ctx, in, 1, 2, 3, 0));
        permuted = ggml_norm(ctx, permuted, hp.norm_eps);
        permuted = ggml_add(ctx, ggml_mul(ctx, permuted,
                                           W(local_name + ".weight")),
                            W(local_name + ".bias"));
        return ggml_cont(ctx, ggml_permute(ctx, permuted, 3, 0, 1, 2));
    };
    auto resblock = [&](ggml_tensor * in, const std::string & local_name,
                        int channels) {
        ggml_tensor * h = channel_norm(in, local_name + ".norm1");
        h = ggml_silu(ctx, h);
        h = conv(h, local_name + ".conv1", channels, channels);
        h = channel_norm(h, local_name + ".norm2");
        h = ggml_silu(ctx, h);
        h = conv(h, local_name + ".conv2", channels, channels);
        return ggml_add(ctx, h, in);
    };
    auto pixel_shuffle = [&](ggml_tensor * tensor, int a0, int a1, int a2,
                             int channels_out) {
        tensor = ggml_reshape_4d(ctx, tensor, a0, a1 * a2, 2, channels_out * 4);
        tensor = ggml_cont(ctx, ggml_permute(ctx, tensor, 1, 2, 0, 3));
        tensor = ggml_reshape_4d(ctx, tensor, 2 * a0, a1, a2, channels_out * 4);
        tensor = ggml_cont(ctx, ggml_permute(ctx, tensor, 1, 0, 2, 3));
        tensor = ggml_reshape_4d(ctx, tensor, a1, 2 * a0 * a2, 2,
                                 channels_out * 2);
        tensor = ggml_cont(ctx, ggml_permute(ctx, tensor, 1, 2, 0, 3));
        tensor = ggml_reshape_4d(ctx, tensor, 2 * a1, 2 * a0, a2,
                                 channels_out * 2);
        tensor = ggml_cont(ctx, ggml_permute(ctx, tensor, 1, 0, 2, 3));
        tensor = ggml_cont(ctx, ggml_permute(ctx, tensor, 1, 2, 0, 3));
        tensor = ggml_reshape_4d(ctx, tensor, a2, 2 * a0 * 2 * a1, 2,
                                 channels_out);
        tensor = ggml_cont(ctx, ggml_permute(ctx, tensor, 1, 2, 0, 3));
        tensor = ggml_reshape_4d(ctx, tensor, 2 * a2, 2 * a0, 2 * a1,
                                 channels_out);
        return ggml_cont(ctx, ggml_permute(ctx, tensor, 2, 0, 1, 3));
    };

    ggml_tensor * h = conv(input, "input_layer", hp.latent_channels,
                           hp.channels.front());
    for (int index = 0; index < hp.num_res_blocks_middle; ++index) {
        h = resblock(h, "middle_block." + std::to_string(index), hp.channels.front());
    }
    int block_index = 0;
    int current_resolution = hp.resolution;
    for (std::size_t level = 0; level < hp.channels.size(); ++level) {
        const int channels = hp.channels[level];
        for (int index = 0; index < hp.num_res_blocks; ++index) {
            h = resblock(h, "blocks." + std::to_string(block_index++), channels);
        }
        if (level + 1 < hp.channels.size()) {
            const int next = hp.channels[level + 1];
            h = conv(h, "blocks." + std::to_string(block_index++) + ".conv",
                     channels, next * 8);
            h = pixel_shuffle(h, current_resolution, current_resolution,
                              current_resolution, next);
            current_resolution *= 2;
        }
    }
    h = channel_norm(h, "out_layer.0");
    h = ggml_silu(ctx, h);
    h = conv(h, "out_layer.2", hp.channels.back(), hp.out_channels);
    ggml_set_output(h);
    if (!missing.empty()) {
        set_error(error, "missing ss-decoder tensor: " + missing);
        ggml_free(ctx);
        return false;
    }
    ggml_build_forward_expand(graph, h);
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(impl_->backend));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        if (allocator) ggml_gallocr_free(allocator);
        ggml_free(ctx);
        set_error(error, "failed to allocate ss-decoder compute graph");
        return false;
    }
    if (ggml_backend_is_cpu(impl_->backend)) {
        ggml_backend_cpu_set_n_threads(impl_->backend, 4);
    }
    ggml_backend_tensor_set(input, latent, 0,
                            static_cast<std::size_t>(hp.latent_channels) *
                            input_points * sizeof(float));
    const ggml_status status = ggml_backend_graph_compute(impl_->backend, graph);
    bool ok = status == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_tensor_get(h, output, 0,
                                static_cast<std::size_t>(hp.out_channels) *
                                output_points * sizeof(float));
    } else {
        set_error(error, "ss-decoder graph compute failed");
    }
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return ok;
}

bool SSDecoderModel::decode_coords(const float * latent,
                                   float threshold,
                                   int target_resolution,
                                   std::vector<std::int32_t> & coords,
                                   std::string * error) {
    coords.clear();
    if (!impl_) {
        set_error(error, "ss-decoder model is not loaded");
        return false;
    }
    const int resolution = impl_->hp.output_resolution();
    std::size_t points = 0;
    if (!checked_cube(resolution, points, error)) return false;
    std::vector<float> logits(points * static_cast<std::size_t>(impl_->hp.out_channels));
    if (!decode(latent, logits.data(), error)) return false;
    return ss_occupancy_to_coords_f32(logits.data(), impl_->hp.out_channels,
                                      resolution, threshold, target_resolution,
                                      coords, error);
}

bool SSDecoderModel::decode_sparse_coords(const SparseTensorF32 & latent,
                                          float threshold,
                                          int target_resolution,
                                          std::vector<std::int32_t> & coords,
                                          std::string * error) {
    coords.clear();
    if (!impl_) {
        set_error(error, "ss-decoder model is not loaded");
        return false;
    }
    std::vector<float> dense_latent;
    if (!sparse_dense_to_channel_major(latent, impl_->hp.resolution,
                                       impl_->hp.latent_channels, dense_latent,
                                       error)) {
        return false;
    }
    return decode_coords(dense_latent.data(), threshold, target_resolution,
                         coords, error);
}

} // namespace pixal3d
