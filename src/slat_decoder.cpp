#include "pixal3d/slat_decoder.h"

#include "pixal3d/backend.h"
#include "pixal3d/pack.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pixal3d {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

bool require_pointer(const float * pointer, const char * name, std::string * error) {
    if (pointer) return true;
    set_error(error, std::string("missing SLat decoder tensor: ") + name);
    return false;
}

bool add_in_place(SparseTensorF32 & destination, const SparseTensorF32 & source,
                  std::string * error) {
    if (destination.batch_size != source.batch_size ||
        destination.channels != source.channels ||
        destination.spatial_x != source.spatial_x ||
        destination.spatial_y != source.spatial_y ||
        destination.spatial_z != source.spatial_z ||
        destination.coords != source.coords) {
        set_error(error, "SLat decoder residual shapes do not match");
        return false;
    }
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t index = 0; index < destination.feats.size(); ++index) {
        destination.feats[index] += source.feats[index];
    }
    return true;
}

void silu_in_place(SparseTensorF32 & input) {
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (float & value : input.feats) {
        value *= 1.0f / (1.0f + std::exp(-value));
    }
}

bool repeat_channels(const SparseTensorF32 & input, int output_channels,
                     SparseTensorF32 & output, std::string * error) {
    if (!input.valid(error) || output_channels <= 0 ||
        output_channels % input.channels != 0) {
        set_error(error, "SLat decoder skip channels are not repeatable");
        return false;
    }
    const int repeats = output_channels / input.channels;
    output = input;
    output.channels = output_channels;
    output.feats.resize(input.points() * static_cast<std::size_t>(output_channels));
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t point = 0; point < input.points(); ++point) {
        const float * source = input.feats.data() + point * input.channels;
        float * destination = output.feats.data() + point * output_channels;
        std::size_t index = 0;
        for (int channel = 0; channel < input.channels; ++channel) {
            for (int repeat = 0; repeat < repeats; ++repeat) {
                destination[index++] = source[channel];
            }
        }
    }
    return true;
}

bool valid_eps(float epsilon, std::string * error) {
    if (epsilon > 0.0f && std::isfinite(epsilon)) return true;
    set_error(error, "SLat decoder norm epsilon must be finite and positive");
    return false;
}

} // namespace

bool sparse_convnext_block_f32(const SparseTensorF32 & input,
                               const SLatDecoderBlockWeightsF32 & weights,
                               float norm_eps,
                               SparseTensorF32 & output,
                               std::string * error) {
    if (!input.valid(error) || !valid_eps(norm_eps, error)) return false;
    if (weights.kind != SLatDecoderBlockKind::convnext ||
        weights.channels != input.channels || weights.out_channels != input.channels ||
        weights.mlp_hidden <= 0) {
        set_error(error, "invalid SLat ConvNeXt block dimensions");
        return false;
    }
    if (!require_pointer(weights.norm_weight, "block.norm.weight", error) ||
        !require_pointer(weights.norm_bias, "block.norm.bias", error) ||
        !require_pointer(weights.conv1_weight, "block.conv.weight", error) ||
        !require_pointer(weights.conv1_bias, "block.conv.bias", error) ||
        !require_pointer(weights.mlp0_weight, "block.mlp.0.weight", error) ||
        !require_pointer(weights.mlp0_bias, "block.mlp.0.bias", error) ||
        !require_pointer(weights.mlp2_weight, "block.mlp.2.weight", error) ||
        !require_pointer(weights.mlp2_bias, "block.mlp.2.bias", error)) return false;

    SparseTensorF32 convolved;
    if (!sparse_submanifold_conv3d(input, weights.conv1_weight, weights.conv1_bias,
                                   input.channels, convolved, error)) return false;
    SparseTensorF32 normalized;
    if (!sparse_layer_norm(convolved, norm_eps, weights.norm_weight, weights.norm_bias,
                           normalized, error)) return false;
    SparseTensorF32 mlp_hidden;
    if (!sparse_linear(normalized, weights.mlp0_weight, weights.mlp0_bias,
                       weights.mlp_hidden, mlp_hidden, error)) return false;
    silu_in_place(mlp_hidden);
    SparseTensorF32 result;
    if (!sparse_linear(mlp_hidden, weights.mlp2_weight, weights.mlp2_bias,
                       input.channels, result, error) ||
        !add_in_place(result, input, error)) return false;
    output = std::move(result);
    return true;
}

bool sparse_resblock_c2s_f32(
    const SparseTensorF32 & input,
    const SLatDecoderBlockWeightsF32 & weights,
    float norm_eps,
    bool pred_subdiv,
    const SparseTensorF32 * subdivision,
    SparseTensorF32 & output,
    SparseTensorF32 * predicted_subdiv,
    std::string * error) {
    if (!input.valid(error) || !valid_eps(norm_eps, error)) return false;
    if (weights.kind != SLatDecoderBlockKind::channel_to_spatial ||
        weights.channels != input.channels || weights.out_channels <= 0 ||
        input.channels % 8 != 0) {
        set_error(error, "invalid SLat channel-to-spatial block dimensions");
        return false;
    }
    if (!require_pointer(weights.norm1_weight, "up.norm1.weight", error) ||
        !require_pointer(weights.norm1_bias, "up.norm1.bias", error) ||
        !require_pointer(weights.conv1_weight, "up.conv1.weight", error) ||
        !require_pointer(weights.conv1_bias, "up.conv1.bias", error) ||
        !require_pointer(weights.conv2_weight, "up.conv2.weight", error) ||
        !require_pointer(weights.conv2_bias, "up.conv2.bias", error)) return false;
    if (pred_subdiv &&
        (!require_pointer(weights.to_subdiv_weight, "up.to_subdiv.weight", error) ||
         !require_pointer(weights.to_subdiv_bias, "up.to_subdiv.bias", error))) return false;

    SparseTensorF32 predicted;
    const SparseTensorF32 * selected_subdivision = subdivision;
    if (pred_subdiv) {
        if (!sparse_linear(input, weights.to_subdiv_weight, weights.to_subdiv_bias,
                           8, predicted, error)) return false;
        selected_subdivision = &predicted;
        if (predicted_subdiv) *predicted_subdiv = predicted;
    } else if (predicted_subdiv) {
        *predicted_subdiv = SparseTensorF32{};
    }

    SparseTensorF32 normalized;
    if (!sparse_layer_norm(input, norm_eps, weights.norm1_weight, weights.norm1_bias,
                           normalized, error)) return false;
    silu_in_place(normalized);
    SparseTensorF32 packed;
    if (!sparse_submanifold_conv3d(normalized, weights.conv1_weight, weights.conv1_bias,
                                   weights.out_channels * 8, packed, error)) return false;
    SparseTensorF32 hidden_spatial;
    if (!sparse_channel_to_spatial(packed, 2, selected_subdivision,
                                   hidden_spatial, error)) return false;

    SparseTensorF32 skip_spatial;
    if (!sparse_channel_to_spatial(input, 2, selected_subdivision,
                                   skip_spatial, error)) return false;
    SparseTensorF32 skip;
    if (!repeat_channels(skip_spatial, weights.out_channels, skip, error)) return false;

    if (!sparse_layer_norm(hidden_spatial, norm_eps, nullptr, nullptr,
                           normalized, error)) return false;
    silu_in_place(normalized);
    SparseTensorF32 convolved;
    if (!sparse_submanifold_conv3d(normalized, weights.conv2_weight,
                                   weights.conv2_bias, weights.out_channels,
                                   convolved, error) ||
        !add_in_place(convolved, skip, error)) return false;
    output = std::move(convolved);
    return true;
}

bool slat_decoder_forward_f32(
    const SparseTensorF32 & input,
    const SLatDecoderConfig & config,
    const SLatDecoderWeightsF32 & weights,
    const std::vector<SparseTensorF32> * guide_subdivisions,
    SparseTensorF32 & output,
    std::vector<SparseTensorF32> * predicted_subdivisions,
    std::string * error) {
    if (!input.valid(error) || !valid_eps(config.norm_eps, error) ||
        config.latent_channels <= 0 || config.out_channels <= 0 ||
        config.model_channels.empty() ||
        config.model_channels.size() != config.num_blocks.size()) {
        set_error(error, "invalid SLat decoder configuration");
        return false;
    }
    if (input.channels != config.latent_channels) {
        set_error(error, "SLat decoder latent channel count does not match input");
        return false;
    }
    const std::size_t levels = config.model_channels.size();
    if (guide_subdivisions && guide_subdivisions->size() != levels - 1) {
        set_error(error, "SLat decoder guide subdivision count does not match levels");
        return false;
    }
    std::size_t expected_blocks = levels - 1;
    for (int count : config.num_blocks) {
        if (count < 0 || expected_blocks > std::numeric_limits<std::size_t>::max() -
                                     static_cast<std::size_t>(count)) {
            set_error(error, "invalid SLat decoder block count");
            return false;
        }
        expected_blocks += static_cast<std::size_t>(count);
    }
    if (weights.blocks.size() != expected_blocks) {
        set_error(error, "SLat decoder weight block count does not match configuration");
        return false;
    }
    if (!require_pointer(weights.from_latent_weight, "from_latent.weight", error) ||
        !require_pointer(weights.from_latent_bias, "from_latent.bias", error) ||
        !require_pointer(weights.output_weight, "output_layer.weight", error) ||
        !require_pointer(weights.output_bias, "output_layer.bias", error)) return false;

    SparseTensorF32 hidden;
    if (!sparse_linear(input, weights.from_latent_weight, weights.from_latent_bias,
                       config.model_channels.front(), hidden, error)) return false;
    if (predicted_subdivisions) predicted_subdivisions->clear();
    std::size_t block_index = 0;
    for (std::size_t level = 0; level < levels; ++level) {
        for (int block = 0; block < config.num_blocks[level]; ++block) {
            if (weights.blocks[block_index].kind != SLatDecoderBlockKind::convnext) {
                set_error(error, "SLat decoder expected ConvNeXt block");
                return false;
            }
            if (!sparse_convnext_block_f32(hidden, weights.blocks[block_index],
                                           config.norm_eps, hidden, error)) return false;
            ++block_index;
        }
        if (level + 1 < levels) {
            const SparseTensorF32 * guide = guide_subdivisions
                ? &(*guide_subdivisions)[level] : nullptr;
            SparseTensorF32 predicted;
            if (weights.blocks[block_index].kind != SLatDecoderBlockKind::channel_to_spatial) {
                set_error(error, "SLat decoder expected channel-to-spatial block");
                return false;
            }
            if (!sparse_resblock_c2s_f32(
                    hidden, weights.blocks[block_index], config.norm_eps,
                    config.pred_subdiv, guide, hidden,
                    config.pred_subdiv ? &predicted : nullptr, error)) return false;
            if (config.pred_subdiv && predicted_subdivisions) {
                predicted_subdivisions->push_back(std::move(predicted));
            }
            ++block_index;
        }
    }
    if (block_index != weights.blocks.size()) {
        set_error(error, "SLat decoder did not consume all block weights");
        return false;
    }
    SparseTensorF32 normalized;
    if (!sparse_layer_norm(hidden, config.norm_eps, nullptr, nullptr, normalized, error) ||
        !sparse_linear(normalized, weights.output_weight, weights.output_bias,
                       config.out_channels, output, error)) return false;
    return true;
}

bool slat_decoder_upsample_coords_f32(
    const SparseTensorF32 & input,
    const SLatDecoderConfig & config,
    const SLatDecoderWeightsF32 & weights,
    int upsample_times,
    SparseTensorF32 & output,
    std::string * error) {
    output = SparseTensorF32{};
    if (!input.valid(error) || !valid_eps(config.norm_eps, error) ||
        config.latent_channels <= 0 || config.out_channels <= 0 ||
        config.model_channels.empty() ||
        config.model_channels.size() != config.num_blocks.size() ||
        !config.pred_subdiv || upsample_times < 0 ||
        upsample_times >= static_cast<int>(config.model_channels.size())) {
        set_error(error, "invalid SLat decoder coordinate-upsample configuration");
        return false;
    }
    if (input.channels != config.latent_channels) {
        set_error(error, "SLat decoder coordinate-upsample latent channel mismatch");
        return false;
    }
    const std::size_t levels = config.model_channels.size();
    std::size_t expected_blocks = levels - 1;
    for (int count : config.num_blocks) {
        if (count < 0 || expected_blocks > std::numeric_limits<std::size_t>::max() -
                                     static_cast<std::size_t>(count)) {
            set_error(error, "invalid SLat decoder block count");
            return false;
        }
        expected_blocks += static_cast<std::size_t>(count);
    }
    if (weights.blocks.size() != expected_blocks ||
        !require_pointer(weights.from_latent_weight, "from_latent.weight", error) ||
        !require_pointer(weights.from_latent_bias, "from_latent.bias", error)) return false;

    SparseTensorF32 hidden;
    if (!sparse_linear(input, weights.from_latent_weight, weights.from_latent_bias,
                       config.model_channels.front(), hidden, error)) return false;
    std::size_t block_index = 0;
    for (std::size_t level = 0; level < levels; ++level) {
        if (static_cast<int>(level) == upsample_times) {
            output = std::move(hidden);
            return true;
        }
        const int channels = config.model_channels[level];
        for (int block = 0; block < config.num_blocks[level]; ++block) {
            if (block_index >= weights.blocks.size() ||
                weights.blocks[block_index].kind != SLatDecoderBlockKind::convnext) {
                set_error(error, "SLat decoder coordinate-upsample expected ConvNeXt block");
                return false;
            }
            if (!sparse_convnext_block_f32(hidden, weights.blocks[block_index],
                                           config.norm_eps, hidden, error)) return false;
            ++block_index;
        }
        if (level + 1 >= levels || block_index >= weights.blocks.size() ||
            weights.blocks[block_index].kind != SLatDecoderBlockKind::channel_to_spatial) {
            set_error(error, "SLat decoder coordinate-upsample expected C2S block");
            return false;
        }
        if (!sparse_resblock_c2s_f32(
                hidden, weights.blocks[block_index], config.norm_eps, true,
                nullptr, hidden, nullptr, error)) return false;
        ++block_index;
    }
    set_error(error, "SLat decoder coordinate-upsample did not reach target level");
    return false;
}

namespace {

bool has_prefix(const std::string & value, const std::string & prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool read_tensor_f32(const Pixal3DPackReader & reader,
                     const Pixal3DTensorInfo & info,
                     std::vector<float> & output,
                     std::string * error) {
    std::vector<std::uint8_t> payload;
    if (!reader.read_tensor(info.name, payload, error)) return false;
    const ggml_type type = static_cast<ggml_type>(info.ggml_type);
    std::size_t element_size = 0;
    if (type == GGML_TYPE_F32) element_size = sizeof(float);
    else if (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) element_size = sizeof(std::uint16_t);
    else {
        set_error(error, "unsupported SLat decoder tensor type: " + info.name);
        return false;
    }
    if (payload.size() % element_size != 0) {
        set_error(error, "SLat decoder tensor payload is not element aligned: " + info.name);
        return false;
    }
    const std::size_t count = payload.size() / element_size;
    if (count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        set_error(error, "SLat decoder tensor is too large: " + info.name);
        return false;
    }
    output.resize(count);
    if (type == GGML_TYPE_F32) {
        std::memcpy(output.data(), payload.data(), payload.size());
    } else if (type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(payload.data()),
                              output.data(), static_cast<std::int64_t>(count));
    } else {
        ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(payload.data()),
                              output.data(), static_cast<std::int64_t>(count));
    }
    for (float value : output) {
        if (!std::isfinite(value)) {
            set_error(error, "SLat decoder tensor contains a non-finite value: " + info.name);
            return false;
        }
    }
    return true;
}

// The released SLat decoder is sparse: its 3x3x3 convolution keeps the
// active coordinate set instead of materialising a dense voxel grid.  ggml's
// CUDA backend does not expose a sparse-convolution op, but it does provide
// device-side gather (GET_ROWS) and matrix multiplication.  A compact
// convolution can therefore be evaluated as 27 gathered neighbour matrices
// multiplied by 27 [out,in] slices.  Coordinates and the active mask remain
// host metadata; feature arithmetic stays on the selected GPU.
struct DecoderCoord {
    std::int32_t batch = 0;
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::int32_t z = 0;

    bool operator==(const DecoderCoord & other) const noexcept {
        return batch == other.batch && x == other.x && y == other.y && z == other.z;
    }
};

struct DecoderCoordHash {
    std::size_t operator()(const DecoderCoord & value) const noexcept {
        std::size_t result = static_cast<std::size_t>(value.batch);
        result = result * 1000003u + static_cast<std::size_t>(value.x);
        result = result * 1000003u + static_cast<std::size_t>(value.y);
        return result * 1000003u + static_cast<std::size_t>(value.z);
    }
};

struct SLatDecoderGpuState {
    ggml_context * weights_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    std::string backend_name;
    std::unordered_map<std::string, ggml_tensor *> tensors;
    std::unordered_map<const float *, ggml_tensor *> host_tensors;
    BackendManager * backend_manager = nullptr;
    bool attempted = false;

    ~SLatDecoderGpuState() { close(); }

    void close() noexcept {
        host_tensors.clear();
        tensors.clear();
        if (weights_buffer) {
            ggml_backend_buffer_free(weights_buffer);
            weights_buffer = nullptr;
        }
        backend = nullptr;
        backend_manager = nullptr;
        if (weights_ctx) {
            ggml_free(weights_ctx);
            weights_ctx = nullptr;
        }
        backend_name.clear();
    }

    bool ready() const noexcept { return backend != nullptr && weights_buffer != nullptr; }

    ggml_tensor * weight(const float * host) const noexcept {
        const auto it = host_tensors.find(host);
        return it == host_tensors.end() ? nullptr : it->second;
    }

    bool init(const Pixal3DPackReader & reader,
              const std::string & component,
              const std::unordered_map<std::string, std::vector<float>> & host,
              ggml_backend_t selected_backend,
              const std::string & selected_name,
              BackendManager * manager,
              std::string * error) {
        if (ready()) return true;
        if (attempted) {
            set_error(error, "SLat decoder GPU initialization was unavailable");
            return false;
        }
        close();
        attempted = true;
        const std::string prefix = component + ".";
        std::size_t count = 0;
        for (const Pixal3DTensorInfo & info : reader.info().tensors) {
            if (has_prefix(info.name, prefix)) ++count;
        }
        if (count == 0) {
            set_error(error, "SLat decoder GPU component has no tensors");
            return false;
        }
        ggml_init_params params{};
        params.mem_size = ggml_tensor_overhead() * (count + 1) + 4096;
        params.no_alloc = true;
        weights_ctx = ggml_init(params);
        if (!weights_ctx) {
            set_error(error, "failed to allocate SLat decoder GPU tensor context");
            return false;
        }
        for (const Pixal3DTensorInfo & info : reader.info().tensors) {
            if (!has_prefix(info.name, prefix)) continue;
            const auto found = host.find(info.name);
            if (found == host.end()) {
                set_error(error, "SLat decoder GPU host tensor is missing: " + info.name);
                close();
                return false;
            }
            // Decoder sparse CUDA convolution uses F32 gathered activations and
            // F32 accumulation.  Keep decoder weights F32 on this path as well;
            // flow weights retain their native reduced storage, but the large
            // decoder view/CONT/MUL_MAT combination is not stable for native F16
            // tensors at production token counts.
            const ggml_type type = GGML_TYPE_F32;
            ggml_tensor * tensor = ggml_new_tensor(weights_ctx, type,
                                                   info.n_dims, info.ne);
            if (!tensor || ggml_nelements(tensor) != static_cast<int64_t>(found->second.size())) {
                set_error(error, "SLat decoder GPU tensor shape mismatch: " + info.name);
                close();
                return false;
            }
            ggml_set_name(tensor, info.name.c_str());
            tensors.emplace(info.name, tensor);
        }

        backend = selected_backend;
        backend_name = selected_name;
        backend_manager = manager;
        if (!backend) {
            set_error(error, "no GPU backend supports SLat sparse gather/matmul");
            close();
            return false;
        }
        weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, backend);
        if (!weights_buffer) {
            set_error(error, "failed to allocate SLat decoder weights on GPU backend " +
                             backend_name);
            close();
            return false;
        }
        ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        backend_manager->log_buffer(
            "weights_allocated", backend, ggml_backend_buffer_get_size(weights_buffer));
        for (const auto & entry : tensors) {
            const auto found = host.find(entry.first);
            ggml_backend_tensor_set(entry.second, found->second.data(), 0,
                                    found->second.size() * sizeof(float));
            host_tensors.emplace(found->second.data(), entry.second);
        }
        std::cerr << "pixal3d: " << component << " SLat GPU decoder ready ("
                  << backend_name << ")" << std::endl;
        return true;
    }

    static bool channel_major(const SparseTensorF32 & input,
                              std::vector<float> & values,
                              std::string * error) {
        if (!input.valid(error)) return false;
        values.resize(input.points() * static_cast<std::size_t>(input.channels));
        for (std::size_t point = 0; point < input.points(); ++point) {
            for (int channel = 0; channel < input.channels; ++channel) {
                // ggml tensors shaped [channels, points] are contiguous in
                // point-major rows: ne[0] (channels) is the innermost
                // dimension.  This is the same byte order as SparseTensor's
                // public [point, channel] feature rows.
                values[point * static_cast<std::size_t>(input.channels) +
                       static_cast<std::size_t>(channel)] =
                    input.feats[point * static_cast<std::size_t>(input.channels) +
                                static_cast<std::size_t>(channel)];
            }
        }
        return true;
    }

    static bool finish_sparse(const SparseTensorF32 & input,
                              int channels,
                              const std::vector<float> & channel_values,
                              SparseTensorF32 & output,
                              std::string * error) {
        const std::size_t points = input.points();
        if (channel_values.size() != points * static_cast<std::size_t>(channels)) {
            set_error(error, "SLat decoder GPU output size mismatch");
            return false;
        }
        output = input;
        output.channels = channels;
        output.feats.resize(channel_values.size());
        for (std::size_t point = 0; point < points; ++point) {
            for (int channel = 0; channel < channels; ++channel) {
                output.feats[point * static_cast<std::size_t>(channels) +
                             static_cast<std::size_t>(channel)] =
                    channel_values[point * static_cast<std::size_t>(channels) +
                                   static_cast<std::size_t>(channel)];
            }
        }
        return output.valid(error);
    }

    bool run_conv(const SparseTensorF32 & input,
                  const float * host_weight,
                  const float * host_bias,
                  int out_channels,
                  SparseTensorF32 & output,
                  std::string * error) const {
        if (!ready() || !input.valid(error) || out_channels <= 0) return false;
        ggml_tensor * weight_tensor = weight(host_weight);
        ggml_tensor * bias_tensor = weight(host_bias);
        if (!weight_tensor || !bias_tensor || weight_tensor->ne[0] != input.channels ||
            weight_tensor->ne[1] != out_channels || weight_tensor->ne[2] != 27 ||
            bias_tensor->ne[0] != out_channels) {
            set_error(error, "SLat decoder GPU convolution tensor shape mismatch");
            return false;
        }
        const std::size_t points = input.points();
        if (points == 0 || points > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
            set_error(error, "SLat decoder GPU convolution point count is invalid");
            return false;
        }
        std::vector<float> values;
        if (!channel_major(input, values, error)) return false;
        // Append one explicit zero column so missing sparse neighbours can be
        // gathered without a backend-dependent broadcast mask operation.
        const std::size_t input_columns = points + 1;
        std::vector<float> padded_values(
            static_cast<std::size_t>(input.channels) * input_columns, 0.0f);
        for (std::size_t point = 0; point < points; ++point) {
            std::copy_n(values.data() + point * static_cast<std::size_t>(input.channels),
                        input.channels,
                        padded_values.data() + point * static_cast<std::size_t>(input.channels));
        }
        std::unordered_map<DecoderCoord, std::size_t, DecoderCoordHash> indices;
        indices.reserve(points);
        for (std::size_t point = 0; point < points; ++point) {
            indices.emplace(DecoderCoord{input.coords[point * 4 + 0], input.coords[point * 4 + 1],
                                  input.coords[point * 4 + 2], input.coords[point * 4 + 3]}, point);
        }

        const std::size_t graph_memory = ggml_tensor_overhead() * 4096 +
                                         ggml_graph_overhead_custom(4096, false);
        ggml_init_params params{};
        params.mem_size = graph_memory;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            set_error(error, "failed to allocate SLat decoder GPU convolution graph");
            return false;
        }
        if (bias_tensor->type != GGML_TYPE_F32) {
            bias_tensor = ggml_cast(ctx, bias_tensor, GGML_TYPE_F32);
        }
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);
        ggml_tensor * input_tensor = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, input.channels, static_cast<int64_t>(input_columns));
        if (!graph || !input_tensor) {
            ggml_free(ctx);
            set_error(error, "failed to allocate SLat decoder GPU convolution inputs");
            return false;
        }
        ggml_set_input(input_tensor);
        ggml_tensor * accumulator = nullptr;
        std::vector<ggml_tensor *> index_tensors;
        index_tensors.reserve(27);
        for (int kd = 0; kd < 3; ++kd) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    const int kernel = kd * 9 + kh * 3 + kw;
                    ggml_tensor * index_tensor = ggml_new_tensor_1d(
                        ctx, GGML_TYPE_I32, static_cast<int64_t>(points));
                    if (!index_tensor) {
                        ggml_free(ctx);
                        set_error(error, "failed to allocate SLat decoder GPU convolution gather");
                        return false;
                    }
                    ggml_set_input(index_tensor);
                    index_tensors.push_back(index_tensor);
                    ggml_tensor * gathered = ggml_get_rows(ctx, input_tensor, index_tensor);
                    ggml_tensor * kernel_weight_view = ggml_view_2d(
                        ctx, weight_tensor, input.channels, out_channels,
                            weight_tensor->nb[1],
                            static_cast<std::size_t>(kernel) * weight_tensor->nb[2]);

                    // Materialise the slice before GEMM.  This keeps the
                    // operation valid on CUDA backends that require a
                    // strictly contiguous matrix source for MUL_MAT.
                    ggml_tensor * kernel_weight = ggml_cont(ctx, kernel_weight_view);
                    ggml_tensor * part = ggml_mul_mat(ctx, kernel_weight, gathered);
                    if (part) ggml_mul_mat_set_prec(part, GGML_PREC_F32);
                    if (!gathered || !kernel_weight || !part) {
                        ggml_free(ctx);
                        set_error(error, "failed to build SLat decoder GPU convolution graph");
                        return false;
                    }
                    accumulator = accumulator ? ggml_add(ctx, accumulator, part) : part;
                }
            }
        }
        accumulator = ggml_add(ctx, accumulator, bias_tensor);
        accumulator = ggml_cont(ctx, accumulator);
        ggml_set_output(accumulator);
        ggml_build_forward_expand(graph, accumulator);
        std::string scheduler_error;
        BackendScheduler scheduler(*backend_manager, 4096, false, true,
                                   &scheduler_error, "SLat decoder");
        if (!scheduler.valid() ||
            (backend_manager->requires_primary_backend() &&
             !scheduler.require_primary_graph(graph, &scheduler_error)) ||
            !scheduler.allocate_graph(graph, &scheduler_error)) {
            ggml_free(ctx);
            set_error(error, scheduler_error.empty()
                ? "failed to allocate SLat decoder GPU convolution graph" : scheduler_error);
            return false;
        }
        ggml_backend_tensor_set(input_tensor, padded_values.data(), 0,
                                padded_values.size() * sizeof(float));
        for (std::size_t index = 0; index < index_tensors.size(); ++index) {
            // Reconstructing the small gather arrays here keeps their lifetime
            // independent from the graph descriptors and avoids host-side
            // pointers being retained by a backend.
            std::vector<std::int32_t> index_values(points, 0);
            const int kd = static_cast<int>(index / 9);
            const int kh = static_cast<int>((index / 3) % 3);
            const int kw = static_cast<int>(index % 3);
            for (std::size_t point = 0; point < points; ++point) {
                const DecoderCoord center{input.coords[point * 4 + 0], input.coords[point * 4 + 1],
                                   input.coords[point * 4 + 2], input.coords[point * 4 + 3]};
                const DecoderCoord neighbor{center.batch, center.x + kd - 1,
                                     center.y + kh - 1, center.z + kw - 1};
                const auto found = indices.find(neighbor);
                if (found != indices.end()) {
                    index_values[point] = static_cast<std::int32_t>(found->second);
                } else {
                    index_values[point] = static_cast<std::int32_t>(points);
                }
            }
            ggml_backend_tensor_set(index_tensors[index], index_values.data(), 0,
                                    index_values.size() * sizeof(std::int32_t));
        }
        const ggml_status status = scheduler.compute(graph, &scheduler_error);
        bool ok = status == GGML_STATUS_SUCCESS;
        std::vector<float> result_values(points * static_cast<std::size_t>(out_channels));
        if (ok) {
            ggml_backend_tensor_get(accumulator, result_values.data(), 0,
                                    result_values.size() * sizeof(float));
            ok = finish_sparse(input, out_channels, result_values, output, error);
        } else {
            set_error(error, "SLat decoder GPU convolution graph compute failed" +
                      (scheduler_error.empty() ? std::string{} : ": " + scheduler_error));
        }
        ggml_free(ctx);
        return ok;
    }
};

} // namespace

struct SLatDecoderModel::Impl {
    Pixal3DPackReader reader;
    SLatDecoderHParams hp;
    std::string component;
    std::unordered_map<std::string, std::vector<float>> tensors;
    BackendManager backend_manager;
    SLatDecoderGpuState gpu;
    int tensor_count = 0;
    bool has_data = false;

    const float * tensor(const std::string & local_name, std::string * error) const {
        const std::string full_name = component + "." + local_name;
        const auto it = tensors.find(full_name);
        if (it != tensors.end()) return it->second.data();
        set_error(error, "missing SLat decoder tensor: " + full_name);
        return nullptr;
    }

    bool make_weights(SLatDecoderWeightsF32 & weights, std::string * error) const;
    bool decode_gpu(const SparseTensorF32 & input,
                    const std::vector<SparseTensorF32> * guide_subdivisions,
                    SparseTensorF32 & output,
                    std::vector<SparseTensorF32> * predicted_subdivisions,
                    std::string * error);
    bool upsample_coords_gpu(const SparseTensorF32 & input,
                             int upsample_times,
                             SparseTensorF32 & output,
                             std::string * error);

    bool init_gpu(std::string * error) {
        if (!has_data) {
            set_error(error, "SLat decoder model was loaded metadata-only");
            return false;
        }
        if (gpu.ready()) return true;
        if (!backend_manager.initialize_from_environment("PIXAL3D_SLAT_DECODER_BACKEND", error)) {
            return false;
        }
        if (backend_manager.policy().kind == BackendPolicyKind::cpu ||
            !backend_manager.primary() || backend_manager.devices().empty() ||
            (backend_manager.devices().front().type != GGML_BACKEND_DEVICE_TYPE_GPU &&
             backend_manager.devices().front().type != GGML_BACKEND_DEVICE_TYPE_IGPU)) {
            set_error(error, "SLat decoder backend policy did not select a GPU backend");
            backend_manager.close();
            return false;
        }
        return gpu.init(reader, component, tensors,
                        backend_manager.primary(), backend_manager.primary_name(),
                        &backend_manager, error);
    }
};

bool SLatDecoderModel::Impl::make_weights(SLatDecoderWeightsF32 & weights,
                                          std::string * error) const {
    weights = SLatDecoderWeightsF32{};
    weights.from_latent_weight = tensor("from_latent.weight", error);
    weights.from_latent_bias = tensor("from_latent.bias", error);
    weights.output_weight = tensor("output_layer.weight", error);
    weights.output_bias = tensor("output_layer.bias", error);
    if (!weights.from_latent_weight || !weights.from_latent_bias ||
        !weights.output_weight || !weights.output_bias) return false;
    const std::size_t levels = hp.model_channels.size();
    weights.blocks.reserve(levels - 1);
    for (std::size_t level = 0; level < levels; ++level) {
        const int channels = hp.model_channels[level];
        for (int block = 0; block < hp.num_blocks[level]; ++block) {
            SLatDecoderBlockWeightsF32 current;
            current.kind = SLatDecoderBlockKind::convnext;
            current.channels = channels;
            current.out_channels = channels;
            current.mlp_hidden = channels * 4;
            const std::string prefix = "blocks." + std::to_string(level) + "." +
                                       std::to_string(block);
            current.norm_weight = tensor(prefix + ".norm.weight", error);
            current.norm_bias = tensor(prefix + ".norm.bias", error);
            current.conv1_weight = tensor(prefix + ".conv.weight", error);
            current.conv1_bias = tensor(prefix + ".conv.bias", error);
            current.mlp0_weight = tensor(prefix + ".mlp.0.weight", error);
            current.mlp0_bias = tensor(prefix + ".mlp.0.bias", error);
            current.mlp2_weight = tensor(prefix + ".mlp.2.weight", error);
            current.mlp2_bias = tensor(prefix + ".mlp.2.bias", error);
            if (!current.norm_weight || !current.norm_bias || !current.conv1_weight ||
                !current.conv1_bias || !current.mlp0_weight || !current.mlp0_bias ||
                !current.mlp2_weight || !current.mlp2_bias) return false;
            weights.blocks.push_back(current);
        }
        if (level + 1 < levels) {
            SLatDecoderBlockWeightsF32 current;
            current.kind = SLatDecoderBlockKind::channel_to_spatial;
            current.channels = channels;
            current.out_channels = hp.model_channels[level + 1];
            const int block = hp.num_blocks[level];
            const std::string prefix = "blocks." + std::to_string(level) + "." +
                                       std::to_string(block);
            current.norm1_weight = tensor(prefix + ".norm1.weight", error);
            current.norm1_bias = tensor(prefix + ".norm1.bias", error);
            current.conv1_weight = tensor(prefix + ".conv1.weight", error);
            current.conv1_bias = tensor(prefix + ".conv1.bias", error);
            current.conv2_weight = tensor(prefix + ".conv2.weight", error);
            current.conv2_bias = tensor(prefix + ".conv2.bias", error);
            if (hp.pred_subdiv) {
                current.to_subdiv_weight = tensor(prefix + ".to_subdiv.weight", error);
                current.to_subdiv_bias = tensor(prefix + ".to_subdiv.bias", error);
            }
            if (!current.norm1_weight || !current.norm1_bias || !current.conv1_weight ||
                !current.conv1_bias || !current.conv2_weight || !current.conv2_bias ||
                (hp.pred_subdiv && (!current.to_subdiv_weight ||
                                    !current.to_subdiv_bias))) return false;
            weights.blocks.push_back(current);
        }
    }
    return true;
}

bool SLatDecoderModel::Impl::decode_gpu(
    const SparseTensorF32 & input,
    const std::vector<SparseTensorF32> * guide_subdivisions,
    SparseTensorF32 & output,
    std::vector<SparseTensorF32> * predicted_subdivisions,
    std::string * error) {
    if (!gpu.ready()) {
        set_error(error, "SLat decoder GPU state is not initialized");
        return false;
    }
    SLatDecoderWeightsF32 weights;
    if (!make_weights(weights, error)) return false;
    SLatDecoderConfig config;
    config.latent_channels = hp.latent_channels;
    config.out_channels = hp.out_channels;
    config.norm_eps = hp.norm_eps;
    config.pred_subdiv = hp.pred_subdiv;
    config.model_channels = hp.model_channels;
    config.num_blocks = hp.num_blocks;
    if (!input.valid(error) || input.channels != config.latent_channels ||
        config.model_channels.empty() || config.model_channels.size() != config.num_blocks.size()) {
        set_error(error, "invalid SLat decoder GPU input or configuration");
        return false;
    }
    if (guide_subdivisions && guide_subdivisions->size() != config.model_channels.size() - 1) {
        set_error(error, "SLat decoder GPU guide subdivision count does not match levels");
        return false;
    }
    if (!config.pred_subdiv && !guide_subdivisions) {
        set_error(error, "SLat decoder GPU texture path requires guide subdivisions");
        return false;
    }
    SparseTensorF32 hidden;
    if (!sparse_linear(input, weights.from_latent_weight, weights.from_latent_bias,
                       config.model_channels.front(), hidden, error)) return false;
    if (predicted_subdivisions) predicted_subdivisions->clear();
    std::size_t block_index = 0;
    for (std::size_t level = 0; level < config.model_channels.size(); ++level) {
        for (int block = 0; block < config.num_blocks[level]; ++block) {
            const SLatDecoderBlockWeightsF32 & current = weights.blocks[block_index];
            SparseTensorF32 convolved;
            if (!gpu.run_conv(hidden, current.conv1_weight, current.conv1_bias,
                              hidden.channels, convolved, error)) return false;
            SparseTensorF32 normalized;
            if (!sparse_layer_norm(convolved, config.norm_eps, current.norm_weight,
                                   current.norm_bias, normalized, error)) return false;
            SparseTensorF32 mlp_hidden;
            if (!sparse_linear(normalized, current.mlp0_weight, current.mlp0_bias,
                               current.mlp_hidden, mlp_hidden, error)) return false;
            silu_in_place(mlp_hidden);
            SparseTensorF32 result;
            if (!sparse_linear(mlp_hidden, current.mlp2_weight, current.mlp2_bias,
                               hidden.channels, result, error) ||
                !add_in_place(result, hidden, error)) return false;
            hidden = std::move(result);
            ++block_index;
        }
        if (level + 1 >= config.model_channels.size()) continue;
        const SLatDecoderBlockWeightsF32 & current = weights.blocks[block_index];
        const SparseTensorF32 * guide = guide_subdivisions
            ? &(*guide_subdivisions)[level] : nullptr;
        SparseTensorF32 predicted;
        if (config.pred_subdiv &&
            !sparse_linear(hidden, current.to_subdiv_weight, current.to_subdiv_bias,
                           8, predicted, error)) return false;
        const SparseTensorF32 * selected = config.pred_subdiv ? &predicted : guide;
        SparseTensorF32 normalized;
        if (!sparse_layer_norm(hidden, config.norm_eps, current.norm1_weight,
                               current.norm1_bias, normalized, error)) return false;
        silu_in_place(normalized);
        SparseTensorF32 packed;
        if (!gpu.run_conv(normalized, current.conv1_weight, current.conv1_bias,
                          current.out_channels * 8, packed, error)) return false;
        SparseTensorF32 hidden_spatial;
        if (!sparse_channel_to_spatial(packed, 2, selected, hidden_spatial, error)) return false;
        SparseTensorF32 skip_spatial;
        if (!sparse_channel_to_spatial(hidden, 2, selected, skip_spatial, error)) return false;
        SparseTensorF32 skip;
        if (!repeat_channels(skip_spatial, current.out_channels, skip, error)) return false;
        if (!sparse_layer_norm(hidden_spatial, config.norm_eps, nullptr, nullptr,
                               normalized, error)) return false;
        silu_in_place(normalized);
        if (!gpu.run_conv(normalized, current.conv2_weight, current.conv2_bias,
                          current.out_channels, hidden, error) ||
            !add_in_place(hidden, skip, error)) return false;
        if (config.pred_subdiv && predicted_subdivisions) {
            predicted_subdivisions->push_back(std::move(predicted));
        }
        ++block_index;
    }
    if (block_index != weights.blocks.size()) {
        set_error(error, "SLat decoder GPU did not consume all block weights");
        return false;
    }
    SparseTensorF32 normalized;
    if (!sparse_layer_norm(hidden, config.norm_eps, nullptr, nullptr, normalized, error) ||
        !sparse_linear(normalized, weights.output_weight, weights.output_bias,
                       config.out_channels, output, error)) return false;
    return true;
}

bool SLatDecoderModel::Impl::upsample_coords_gpu(
    const SparseTensorF32 & input,
    int upsample_times,
    SparseTensorF32 & output,
    std::string * error) {
    if (!gpu.ready() || upsample_times < 0 ||
        upsample_times >= static_cast<int>(hp.model_channels.size())) {
        set_error(error, "invalid SLat decoder GPU coordinate-upsample request");
        return false;
    }
    SLatDecoderWeightsF32 weights;
    if (!make_weights(weights, error)) return false;
    SLatDecoderConfig config;
    config.latent_channels = hp.latent_channels;
    config.out_channels = hp.out_channels;
    config.norm_eps = hp.norm_eps;
    config.pred_subdiv = hp.pred_subdiv;
    config.model_channels = hp.model_channels;
    config.num_blocks = hp.num_blocks;
    if (!config.pred_subdiv || !input.valid(error) || input.channels != config.latent_channels) {
        set_error(error, "invalid SLat decoder GPU coordinate-upsample input");
        return false;
    }
    SparseTensorF32 hidden;
    if (!sparse_linear(input, weights.from_latent_weight, weights.from_latent_bias,
                       config.model_channels.front(), hidden, error)) return false;
    std::size_t block_index = 0;
    for (std::size_t level = 0; level < config.model_channels.size(); ++level) {
        if (static_cast<int>(level) == upsample_times) {
            output = std::move(hidden);
            return true;
        }
        for (int block = 0; block < config.num_blocks[level]; ++block) {
            const SLatDecoderBlockWeightsF32 & current = weights.blocks[block_index++];
            SparseTensorF32 convolved;
            if (!gpu.run_conv(hidden, current.conv1_weight, current.conv1_bias,
                              hidden.channels, convolved, error)) return false;
            SparseTensorF32 normalized;
            if (!sparse_layer_norm(convolved, config.norm_eps, current.norm_weight,
                                   current.norm_bias, normalized, error)) return false;
            SparseTensorF32 mlp_hidden;
            if (!sparse_linear(normalized, current.mlp0_weight, current.mlp0_bias,
                               current.mlp_hidden, mlp_hidden, error)) return false;
            silu_in_place(mlp_hidden);
            SparseTensorF32 result;
            if (!sparse_linear(mlp_hidden, current.mlp2_weight, current.mlp2_bias,
                               hidden.channels, result, error) ||
                !add_in_place(result, hidden, error)) return false;
            hidden = std::move(result);
        }
        if (level + 1 >= config.model_channels.size()) break;
        const SLatDecoderBlockWeightsF32 & current = weights.blocks[block_index++];
        SparseTensorF32 predicted;
        if (!sparse_linear(hidden, current.to_subdiv_weight, current.to_subdiv_bias,
                           8, predicted, error)) return false;
        SparseTensorF32 normalized;
        if (!sparse_layer_norm(hidden, config.norm_eps, current.norm1_weight,
                               current.norm1_bias, normalized, error)) return false;
        silu_in_place(normalized);
        SparseTensorF32 packed;
        if (!gpu.run_conv(normalized, current.conv1_weight, current.conv1_bias,
                          current.out_channels * 8, packed, error)) return false;
        SparseTensorF32 hidden_spatial;
        if (!sparse_channel_to_spatial(packed, 2, &predicted, hidden_spatial, error)) return false;
        SparseTensorF32 skip_spatial;
        if (!sparse_channel_to_spatial(hidden, 2, &predicted, skip_spatial, error)) return false;
        SparseTensorF32 skip;
        if (!repeat_channels(skip_spatial, current.out_channels, skip, error)) return false;
        if (!sparse_layer_norm(hidden_spatial, config.norm_eps, nullptr, nullptr,
                               normalized, error)) return false;
        silu_in_place(normalized);
        if (!gpu.run_conv(normalized, current.conv2_weight, current.conv2_bias,
                          current.out_channels, hidden, error) ||
            !add_in_place(hidden, skip, error)) return false;
    }
    set_error(error, "SLat decoder GPU coordinate-upsample did not reach target level");
    return false;
}

SLatDecoderModel::~SLatDecoderModel() {
    close();
}

void SLatDecoderModel::close() noexcept {
    delete impl_;
    impl_ = nullptr;
}

bool SLatDecoderModel::load(const std::string & path,
                            const std::string & component,
                            bool load_tensors,
                            std::string * error) {
    close();
    if (component.empty() || component.find('.') != std::string::npos) {
        set_error(error, "invalid SLat decoder component name");
        return false;
    }
    std::unique_ptr<Impl> impl(new Impl());
    if (!impl->reader.open(path, error)) return false;
    impl->component = component;
    const std::string prefix = component + ".";
    if (!impl->reader.info().has_tensor(prefix + "from_latent.weight")) {
        set_error(error, "GGUF pack does not contain SLat decoder component: " + component);
        return false;
    }
    const std::string metadata_prefix = "pixal3d." + component + ".";
    std::uint32_t value = 0;
    if (!impl->reader.metadata_string(metadata_prefix + "model_class", impl->hp.model_class,
                                      true, error) ||
        !impl->reader.metadata_u32(metadata_prefix + "resolution", value, false, error)) return false;
    impl->hp.resolution = static_cast<int>(value);
    if (!impl->reader.metadata_u32(metadata_prefix + "latent_channels", value, true, error)) return false;
    impl->hp.latent_channels = static_cast<int>(value);
    if (!impl->reader.metadata_u32(metadata_prefix + "out_channels", value, true, error)) return false;
    impl->hp.out_channels = static_cast<int>(value);
    if (!impl->reader.metadata_u32(metadata_prefix + "n_levels", value, true, error)) return false;
    const int levels = static_cast<int>(value);
    float norm_eps = 0.0f;
    if (!impl->reader.metadata_f32(metadata_prefix + "norm_eps", norm_eps, false, error)) {
        return false;
    }
    impl->hp.norm_eps = norm_eps > 0.0f && std::isfinite(norm_eps) ? norm_eps : 1e-6f;
    if (!impl->reader.metadata_bool(metadata_prefix + "pred_subdiv", impl->hp.pred_subdiv,
                                    false, error)) return false;
    if (impl->hp.resolution < 0 || impl->hp.latent_channels <= 0 ||
        impl->hp.out_channels <= 0 || levels <= 0 || levels > 16) {
        set_error(error, "invalid SLat decoder metadata");
        return false;
    }
    impl->hp.component = component;
    impl->hp.model_channels.resize(static_cast<std::size_t>(levels));
    impl->hp.num_blocks.resize(static_cast<std::size_t>(levels));
    for (int level = 0; level < levels; ++level) {
        if (!impl->reader.metadata_u32(metadata_prefix + "model_channels." +
                                           std::to_string(level), value, true, error)) return false;
        impl->hp.model_channels[static_cast<std::size_t>(level)] = static_cast<int>(value);
        if (!impl->reader.metadata_u32(metadata_prefix + "num_blocks." +
                                           std::to_string(level), value, true, error)) return false;
        impl->hp.num_blocks[static_cast<std::size_t>(level)] = static_cast<int>(value);
        if (impl->hp.model_channels[static_cast<std::size_t>(level)] <= 0 ||
            impl->hp.num_blocks[static_cast<std::size_t>(level)] < 0) {
            set_error(error, "invalid SLat decoder level metadata");
            return false;
        }
    }
    for (const Pixal3DTensorInfo & info : impl->reader.info().tensors) {
        if (has_prefix(info.name, prefix)) ++impl->tensor_count;
    }
    if (impl->tensor_count <= 0) {
        set_error(error, "SLat decoder component has no tensors: " + component);
        return false;
    }
    if (load_tensors) {
        for (const Pixal3DTensorInfo & info : impl->reader.info().tensors) {
            if (!has_prefix(info.name, prefix)) continue;
            std::vector<float> values;
            if (!read_tensor_f32(impl->reader, info, values, error)) return false;
            impl->tensors.emplace(info.name, std::move(values));
        }
        impl->has_data = true;
    }
    impl_ = impl.release();
    return true;
}

bool SLatDecoderModel::is_loaded() const noexcept {
    return impl_ != nullptr;
}

bool SLatDecoderModel::has_data() const noexcept {
    return impl_ && impl_->has_data;
}

const SLatDecoderHParams & SLatDecoderModel::hparams() const noexcept {
    static const SLatDecoderHParams empty;
    return impl_ ? impl_->hp : empty;
}

int SLatDecoderModel::tensor_count() const noexcept {
    return impl_ ? impl_->tensor_count : 0;
}

bool SLatDecoderModel::has_tensor(const std::string & name) const noexcept {
    if (!impl_) return false;
    const std::string full_name = has_prefix(name, impl_->component + ".")
        ? name : impl_->component + "." + name;
    // Metadata-only loads intentionally keep no payload vectors, but tensor
    // presence is still useful to callers inspecting a pack before deciding
    // whether to pay the memory cost of load_tensors=true.
    return impl_->reader.info().has_tensor(full_name);
}

bool SLatDecoderModel::decode(
    const SparseTensorF32 & input,
    const std::vector<SparseTensorF32> * guide_subdivisions,
    SparseTensorF32 & output,
    std::vector<SparseTensorF32> * predicted_subdivisions,
    std::string * error) {
    if (!impl_) {
        set_error(error, "SLat decoder model is not loaded");
        return false;
    }
    if (!impl_->has_data) {
        set_error(error, "SLat decoder model was loaded metadata-only");
        return false;
    }
    if (!input.valid(error) || input.channels != impl_->hp.latent_channels) {
        set_error(error, "SLat decoder latent shape does not match metadata");
        return false;
    }
    if (!impl_->hp.pred_subdiv &&
        (!guide_subdivisions || guide_subdivisions->size() != impl_->hp.model_channels.size() - 1)) {
        set_error(error, "texture SLat decoder requires one guide subdivision per upsample level");
        return false;
    }

    BackendPolicy policy = BackendPolicy::from_environment(
        "PIXAL3D_SLAT_DECODER_BACKEND", nullptr, error);
    if (error && !error->empty()) return false;
    const bool force_cpu = policy.kind == BackendPolicyKind::cpu;
    const bool force_gpu = policy.kind == BackendPolicyKind::gpu;
    const bool verbose = std::getenv("PIXAL3D_SLAT_VERBOSE") != nullptr;
    if (!force_cpu) {
        std::string gpu_error;
        if (impl_->init_gpu(&gpu_error)) {
            SparseTensorF32 gpu_output;
            std::vector<SparseTensorF32> gpu_subdivisions;
            if (impl_->decode_gpu(input, guide_subdivisions, gpu_output,
                                  predicted_subdivisions ? &gpu_subdivisions : nullptr,
                                  &gpu_error)) {
                output = std::move(gpu_output);
                if (predicted_subdivisions) *predicted_subdivisions = std::move(gpu_subdivisions);
                return true;
            }
        }
        if (force_gpu) {
            set_error(error, "SLat decoder forced GPU execution failed" +
                             (gpu_error.empty() ? std::string{} : ": " + gpu_error));
            impl_->gpu.close();
            impl_->backend_manager.close();
            return false;
        }
        if (verbose && !gpu_error.empty()) {
            std::cerr << "pixal3d: SLat decoder GPU unavailable; using CPU fallback: "
                      << gpu_error << std::endl;
        }
        impl_->gpu.close();
        impl_->backend_manager.close();
    }

    SLatDecoderWeightsF32 weights;
    if (!impl_->make_weights(weights, error)) return false;
    SLatDecoderConfig config;
    config.latent_channels = impl_->hp.latent_channels;
    config.out_channels = impl_->hp.out_channels;
    config.norm_eps = impl_->hp.norm_eps;
    config.pred_subdiv = impl_->hp.pred_subdiv;
    config.model_channels = impl_->hp.model_channels;
    config.num_blocks = impl_->hp.num_blocks;
    return slat_decoder_forward_f32(input, config, weights, guide_subdivisions,
                                    output, predicted_subdivisions, error);
}

bool SLatDecoderModel::decode_shape_mesh(
    const SparseTensorF32 & input,
    const std::vector<SparseTensorF32> * guide_subdivisions,
    std::vector<DualGridMeshF32> & meshes,
    std::vector<SparseTensorF32> * predicted_subdivisions,
    int resolution_override,
    float voxel_margin,
    std::string * error) {
    meshes.clear();
    if (!impl_ || impl_->component != "shape_decoder" ||
        impl_->hp.model_class != "FlexiDualGridVaeDecoder" ||
        impl_->hp.out_channels != 7) {
        set_error(error, "decode_shape_mesh requires a FlexiDualGrid shape decoder");
        return false;
    }
    const int resolution = resolution_override > 0 ? resolution_override : impl_->hp.resolution;
    if (resolution <= 0) {
        set_error(error, "shape decoder resolution is unspecified");
        return false;
    }
    SparseTensorF32 decoded;
    if (!decode(input, guide_subdivisions, decoded, predicted_subdivisions, error)) return false;
    return flexi_dual_grid_decode_mesh_f32(decoded, resolution, voxel_margin, meshes, error);
}

bool SLatDecoderModel::upsample_coords(const SparseTensorF32 & input,
                                       int upsample_times,
                                       SparseTensorF32 & output,
                                       std::string * error) {
    output = SparseTensorF32{};
    if (!impl_ || impl_->component != "shape_decoder" ||
        impl_->hp.model_class != "FlexiDualGridVaeDecoder" ||
        !impl_->hp.pred_subdiv) {
        set_error(error, "upsample_coords requires a pred_subdiv shape decoder");
        return false;
    }
    if (!impl_->has_data) {
        set_error(error, "SLat decoder model was loaded metadata-only");
        return false;
    }
    if (!input.valid(error) || input.channels != impl_->hp.latent_channels) {
        set_error(error, "SLat decoder upsample latent shape does not match metadata");
        return false;
    }
    BackendPolicy policy = BackendPolicy::from_environment(
        "PIXAL3D_SLAT_DECODER_BACKEND", nullptr, error);
    if (error && !error->empty()) return false;
    const bool force_cpu = policy.kind == BackendPolicyKind::cpu;
    const bool force_gpu = policy.kind == BackendPolicyKind::gpu;
    const bool verbose = std::getenv("PIXAL3D_SLAT_VERBOSE") != nullptr;
    if (!force_cpu) {
        std::string gpu_error;
        if (impl_->init_gpu(&gpu_error) &&
            impl_->upsample_coords_gpu(input, upsample_times, output, &gpu_error)) {
            return true;
        }
        if (force_gpu) {
            set_error(error, "SLat decoder forced GPU coordinate upsample failed" +
                             (gpu_error.empty() ? std::string{} : ": " + gpu_error));
            impl_->gpu.close();
            impl_->backend_manager.close();
            return false;
        }
        if (verbose && !gpu_error.empty()) {
            std::cerr << "pixal3d: SLat decoder coordinate upsample GPU unavailable; "
                      << "using CPU fallback: " << gpu_error << std::endl;
        }
        impl_->gpu.close();
        impl_->backend_manager.close();
    }
    SLatDecoderWeightsF32 weights;
    if (!impl_->make_weights(weights, error)) return false;
    SLatDecoderConfig config;
    config.latent_channels = impl_->hp.latent_channels;
    config.out_channels = impl_->hp.out_channels;
    config.norm_eps = impl_->hp.norm_eps;
    config.pred_subdiv = impl_->hp.pred_subdiv;
    config.model_channels = impl_->hp.model_channels;
    config.num_blocks = impl_->hp.num_blocks;
    return slat_decoder_upsample_coords_f32(input, config, weights,
                                            upsample_times, output, error);
}

} // namespace pixal3d
