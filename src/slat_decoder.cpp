#include "pixal3d/slat_decoder.h"

#include "pixal3d/backend.h"
#include "pixal3d/pack.h"

#include "sparse_profile.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <map>
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

bool validate_decoder_sparse_tensor(const SparseTensorF32 & input,
                                    const char * label,
                                    std::string * error);

bool require_pointer(const float * pointer, const char * name, std::string * error) {
    if (pointer) return true;
    set_error(error, std::string("missing SLat decoder tensor: ") + name);
    return false;
}

bool add_in_place(SparseTensorF32 & destination, const SparseTensorF32 & source,
                  std::string * error) {
    auto timer = detail::make_sparse_profile_timer(
        &detail::SparseProfileStats::add_residual_calls,
        &detail::SparseProfileStats::add_residual_ms);
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
    auto timer = detail::make_sparse_profile_timer(
        &detail::SparseProfileStats::silu_calls,
        &detail::SparseProfileStats::silu_ms);
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (float & value : input.feats) {
        value *= 1.0f / (1.0f + std::exp(-value));
    }
}

bool repeat_channels(const SparseTensorF32 & input, int output_channels,
                     SparseTensorF32 & output, std::string * error) {
    auto timer = detail::make_sparse_profile_timer(
        &detail::SparseProfileStats::repeat_channels_calls,
        &detail::SparseProfileStats::repeat_channels_ms);
    if (!validate_decoder_sparse_tensor(input, "repeat_channels.input", error) ||
        output_channels <= 0 ||
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

bool validate_decoder_sparse_tensor(const SparseTensorF32 & input,
                                    const char * label,
                                    std::string * error) {
    detail::SparseValidationProfileLabelScope label_scope(label);
    return input.valid(error);
}

bool run_labeled_sparse_linear(const SparseTensorF32 & input,
                               const float * weight,
                               const float * bias,
                               int out_channels,
                               SparseTensorF32 & output,
                               const std::string & label,
                               std::string * error) {
    detail::SparseLinearProfileLabelScope label_scope(label.c_str());
    return sparse_linear(input, weight, bias, out_channels, output, error);
}

bool environment_flag(const char * name, bool default_value = false) noexcept {
    const char * value = std::getenv(name);
    if (!value || !*value) return default_value;
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "off") != 0;
}

bool slat_decoder_trace_enabled() noexcept {
    return environment_flag("PIXAL3D_SLAT_DECODER_TRACE");
}

bool slat_decoder_profile_enabled() noexcept {
    return slat_decoder_trace_enabled() ||
           environment_flag("PIXAL3D_SLAT_DECODER_PROFILE") ||
           environment_flag("PIXAL3D_VALIDATE_SPARSE_MAP_CACHE");
}

bool slat_decoder_neighbor_cache_enabled() noexcept {
    return environment_flag("PIXAL3D_SLAT_DECODER_NEIGHBOR_CACHE", true);
}

bool slat_decoder_neighbor_cache_validation_enabled() noexcept {
    return environment_flag("PIXAL3D_VALIDATE_SPARSE_MAP_CACHE");
}

// GPU-first is the normal CUDA rollout path.  Set this to 0 only to reproduce
// the pre-GPU-first hybrid decoder while debugging a placement or parity
// regression; a failed GPU-first graph is never silently replaced by the
// legacy partial graph.
bool slat_decoder_gpu_first_enabled() noexcept {
    return environment_flag("PIXAL3D_SLAT_DECODER_GPU_FIRST", true);
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
    if (!sparse_layer_norm(hidden, k_slat_decoder_final_layer_norm_eps,
                           nullptr, nullptr,
                           normalized, error) ||
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

std::uint64_t decoder_coords_fingerprint(
    const std::vector<std::int32_t> & coords) noexcept {
    constexpr std::uint64_t kOffsetBasis = 1469598103934665603ULL;
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    std::uint64_t hash = kOffsetBasis;
    const auto * bytes = reinterpret_cast<const std::uint8_t *>(coords.data());
    const std::size_t byte_count = coords.size() * sizeof(std::int32_t);
    for (std::size_t index = 0; index < byte_count; ++index) {
        hash ^= bytes[index];
        hash *= kPrime;
    }
    return hash;
}

struct NeighborGatherMap {
    std::size_t points = 0;
    int batch_size = 0;
    int spatial_x = 0;
    int spatial_y = 0;
    int spatial_z = 0;
    std::uint64_t coordinate_fingerprint = 0;

    // Layout is [offset * points + output_point].  Each value is an input
    // point row, or points for the explicit zero row appended to the input.
    std::vector<std::int32_t> gather_indices;
};

struct SLatDecoderTopologyProfile {
    std::uint64_t generation = 0;
    std::size_t points = 0;
    int batch_size = 0;
    int spatial_x = 0;
    int spatial_y = 0;
    int spatial_z = 0;
    std::uint64_t coordinate_fingerprint = 0;
    std::size_t conv_calls = 0;
    std::size_t cache_hits = 0;
    std::size_t cache_misses = 0;
};

struct SLatDecoderProfile {
    bool neighbor_cache_enabled = true;
    bool validate_neighbor_cache = false;
    std::size_t total_sparse_conv_calls = 0;
    std::size_t neighbor_map_builds = 0;
    std::size_t cache_hits = 0;
    std::size_t cache_misses = 0;
    std::size_t cache_validation_calls = 0;

    double input_validation_ms = 0.0;
    double padded_input_build_ms = 0.0;
    double coordinate_fingerprint_ms = 0.0;
    double coord_hashmap_build_ms = 0.0;
    double neighbor_index_build_ms = 0.0;
    double cache_validation_ms = 0.0;
    double index_upload_ms = 0.0;
    double graph_build_ms = 0.0;
    double scheduler_allocate_ms = 0.0;
    double input_upload_ms = 0.0;
    double gpu_compute_ms = 0.0;
    double output_download_ms = 0.0;
    double finish_sparse_ms = 0.0;
    double output_validation_ms = 0.0;

    std::vector<SLatDecoderTopologyProfile> topologies;
};

void decoder_profile_add_phase(SLatDecoderProfile * profile,
                               double SLatDecoderProfile::* phase,
                               double start_ms) noexcept {
    if (!profile) return;
    profile->*phase += backend_time_now_ms() - start_ms;
}

bool neighbor_map_shape_matches(const NeighborGatherMap & map,
                                const SparseTensorF32 & input,
                                bool compare_fingerprint,
                                std::uint64_t coordinate_fingerprint) noexcept {
    return map.points == input.points() && map.batch_size == input.batch_size &&
           map.spatial_x == input.spatial_x && map.spatial_y == input.spatial_y &&
           map.spatial_z == input.spatial_z &&
           (!compare_fingerprint || map.coordinate_fingerprint == coordinate_fingerprint);
}

bool build_neighbor_gather_map(const SparseTensorF32 & input,
                               std::uint64_t coordinate_fingerprint,
                               NeighborGatherMap & map,
                               SLatDecoderProfile * profile,
                               std::string * error) {
    const std::size_t points = input.points();
    if (points == 0 || points > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        points > std::numeric_limits<std::size_t>::max() / 27) {
        set_error(error, "SLat decoder neighbor map point count is invalid");
        return false;
    }

    map = NeighborGatherMap{};
    map.points = points;
    map.batch_size = input.batch_size;
    map.spatial_x = input.spatial_x;
    map.spatial_y = input.spatial_y;
    map.spatial_z = input.spatial_z;
    map.coordinate_fingerprint = coordinate_fingerprint;

    const double coord_hashmap_begin = profile ? backend_time_now_ms() : 0.0;
    std::unordered_map<DecoderCoord, std::size_t, DecoderCoordHash> indices;
    indices.reserve(points);
    for (std::size_t point = 0; point < points; ++point) {
        indices.emplace(DecoderCoord{input.coords[point * 4 + 0], input.coords[point * 4 + 1],
                                     input.coords[point * 4 + 2], input.coords[point * 4 + 3]}, point);
    }
    if (profile) {
        profile->coord_hashmap_build_ms +=
            backend_time_now_ms() - coord_hashmap_begin;
    }

    const double neighbor_index_begin = profile ? backend_time_now_ms() : 0.0;
    map.gather_indices.assign(points * 27, static_cast<std::int32_t>(points));
    for (int kd = 0; kd < 3; ++kd) {
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                const std::size_t offset = static_cast<std::size_t>(kd * 9 + kh * 3 + kw);
                std::int32_t * destination = map.gather_indices.data() + offset * points;
                for (std::size_t point = 0; point < points; ++point) {
                    const DecoderCoord center{
                        input.coords[point * 4 + 0], input.coords[point * 4 + 1],
                        input.coords[point * 4 + 2], input.coords[point * 4 + 3]};
                    const DecoderCoord neighbor{center.batch, center.x + kd - 1,
                                                 center.y + kh - 1, center.z + kw - 1};
                    const auto found = indices.find(neighbor);
                    if (found != indices.end()) {
                        destination[point] = static_cast<std::int32_t>(found->second);
                    }
                }
            }
        }
    }
    if (profile) {
        profile->neighbor_index_build_ms +=
            backend_time_now_ms() - neighbor_index_begin;
        ++profile->neighbor_map_builds;
    }
    return true;
}

bool build_reference_neighbor_gather_map(const SparseTensorF32 & input,
                                           NeighborGatherMap & map,
                                           std::string * error) {
    const std::size_t points = input.points();
    if (points == 0 || points > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        points > std::numeric_limits<std::size_t>::max() / 27) {
        set_error(error, "SLat decoder reference neighbor map point count is invalid");
        return false;
    }
    map = NeighborGatherMap{};
    map.points = points;
    map.batch_size = input.batch_size;
    map.spatial_x = input.spatial_x;
    map.spatial_y = input.spatial_y;
    map.spatial_z = input.spatial_z;
    map.gather_indices.resize(points * 27);

    std::unordered_map<DecoderCoord, std::size_t, DecoderCoordHash> indices;
    indices.reserve(points);
    for (std::size_t point = 0; point < points; ++point) {
        indices.emplace(DecoderCoord{input.coords[point * 4 + 0], input.coords[point * 4 + 1],
                                     input.coords[point * 4 + 2], input.coords[point * 4 + 3]}, point);
    }
    for (std::size_t index = 0; index < 27; ++index) {
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
        std::copy(index_values.begin(), index_values.end(),
                  map.gather_indices.begin() + index * points);
    }
    return true;
}

bool validate_neighbor_gather_map(const SparseTensorF32 & input,
                                  const NeighborGatherMap & cached,
                                  std::string * error) {
    NeighborGatherMap reference;
    if (!build_reference_neighbor_gather_map(input, reference, error)) return false;
    if (cached.points != reference.points || cached.batch_size != reference.batch_size ||
        cached.spatial_x != reference.spatial_x || cached.spatial_y != reference.spatial_y ||
        cached.spatial_z != reference.spatial_z ||
        cached.gather_indices.size() != reference.gather_indices.size()) {
        set_error(error, "SLat decoder neighbor map cache metadata mismatch");
        return false;
    }
    for (std::size_t index = 0; index < cached.gather_indices.size(); ++index) {
        if (cached.gather_indices[index] == reference.gather_indices[index]) continue;
        const std::size_t offset = index / cached.points;
        const std::size_t point = index % cached.points;
        const int kd = static_cast<int>(offset / 9);
        const int kh = static_cast<int>((offset / 3) % 3);
        const int kw = static_cast<int>(offset % 3);
        const DecoderCoord center{
            input.coords[point * 4 + 0], input.coords[point * 4 + 1],
            input.coords[point * 4 + 2], input.coords[point * 4 + 3]};
        const DecoderCoord neighbor{center.batch, center.x + kd - 1,
                                    center.y + kh - 1, center.z + kw - 1};
        set_error(error,
                  "SLat decoder neighbor map cache mismatch: offset=" +
                  std::to_string(offset) + " output_point=" + std::to_string(point) +
                  " cached_input_row=" + std::to_string(cached.gather_indices[index]) +
                  " reference_input_row=" + std::to_string(reference.gather_indices[index]) +
                  " center=(" + std::to_string(center.batch) + "," +
                  std::to_string(center.x) + "," + std::to_string(center.y) + "," +
                  std::to_string(center.z) + ") expected_neighbor=(" +
                  std::to_string(neighbor.batch) + "," + std::to_string(neighbor.x) + "," +
                  std::to_string(neighbor.y) + "," + std::to_string(neighbor.z) + ")");
        return false;
    }
    return true;
}

bool record_decoder_conv(SLatDecoderProfile * profile,
                         std::uint64_t generation,
                         const SparseTensorF32 & input,
                         std::uint64_t coordinate_fingerprint,
                         bool cache_hit,
                         std::string * error) {
    if (!profile) return true;
    SLatDecoderTopologyProfile * topology = nullptr;
    for (SLatDecoderTopologyProfile & candidate : profile->topologies) {
        if (candidate.generation == generation) {
            topology = &candidate;
            break;
        }
    }
    if (!topology) {
        profile->topologies.push_back(SLatDecoderTopologyProfile{});
        topology = &profile->topologies.back();
        topology->generation = generation;
        topology->points = input.points();
        topology->batch_size = input.batch_size;
        topology->spatial_x = input.spatial_x;
        topology->spatial_y = input.spatial_y;
        topology->spatial_z = input.spatial_z;
        topology->coordinate_fingerprint = coordinate_fingerprint;
    } else if (topology->coordinate_fingerprint != coordinate_fingerprint) {
        set_error(error, "SLat decoder coordinates changed without topology invalidation");
        return false;
    }
    ++profile->total_sparse_conv_calls;
    ++topology->conv_calls;
    if (cache_hit) {
        ++profile->cache_hits;
        ++topology->cache_hits;
    } else {
        ++profile->cache_misses;
        ++topology->cache_misses;
    }
    std::cerr << "pixal3d: SLat-decoder-conv conv_index="
              << profile->total_sparse_conv_calls
              << " points=" << input.points()
              << " batch_size=" << input.batch_size
              << " spatial=" << input.spatial_x << 'x' << input.spatial_y << 'x'
              << input.spatial_z
              << " coords=0x" << std::hex << coordinate_fingerprint << std::dec
              << " topology_generation=" << generation
              << " cache=" << (cache_hit ? "hit" : "miss") << std::endl;
    return true;
}

// The GPU-first decoder keeps public SparseTensorF32 storage on the host only
// at a topology boundary.  Inside one level, features use ggml's [channel,
// point] layout, which is byte-identical to the public point-major [point,
// channel] vector.  These helpers deliberately use existing ggml operations;
// no decoder-specific CUDA kernels or weight relayout are introduced.
ggml_tensor * name_gpu_tensor(ggml_tensor * tensor, const std::string & name) {
    if (tensor && !name.empty()) ggml_set_name(tensor, name.c_str());
    return tensor;
}

ggml_tensor * gpu_linear_node(ggml_context * ctx,
                              ggml_tensor * input,
                              ggml_tensor * weight,
                              ggml_tensor * bias,
                              const std::string & name) {
    if (!ctx || !input || !weight || !bias) return nullptr;
    ggml_tensor * product = ggml_mul_mat(ctx, weight, input);
    if (!product) return nullptr;
    ggml_mul_mat_set_prec(product, GGML_PREC_F32);
    ggml_tensor * biased = ggml_add(ctx, product, bias);
    if (!biased) return nullptr;
    return name_gpu_tensor(ggml_cont(ctx, biased), name);
}

ggml_tensor * gpu_layer_norm_affine_node(ggml_context * ctx,
                                          ggml_tensor * input,
                                          ggml_tensor * gamma,
                                          ggml_tensor * beta,
                                          float epsilon,
                                          const std::string & name) {
    if (!ctx || !input || !gamma || !beta) return nullptr;
    ggml_tensor * normalized = ggml_norm(ctx, input, epsilon);
    if (!normalized) return nullptr;
    ggml_tensor * scaled = ggml_mul(ctx, normalized, gamma);
    if (!scaled) return nullptr;
    ggml_tensor * shifted = ggml_add(ctx, scaled, beta);
    if (!shifted) return nullptr;
    return name_gpu_tensor(ggml_cont(ctx, shifted), name);
}

ggml_tensor * gpu_layer_norm_node(ggml_context * ctx,
                                   ggml_tensor * input,
                                   float epsilon,
                                   const std::string & name) {
    if (!ctx || !input) return nullptr;
    ggml_tensor * normalized = ggml_norm(ctx, input, epsilon);
    if (!normalized) return nullptr;
    return name_gpu_tensor(ggml_cont(ctx, normalized), name);
}

ggml_tensor * gpu_sparse_conv_node(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * zero_column,
    ggml_tensor * weight,
    ggml_tensor * bias,
    const std::vector<ggml_tensor *> & index_tensors,
    int input_channels,
    int output_channels,
    const std::string & name) {
    if (!ctx || !input || !zero_column || !weight || !bias ||
        index_tensors.size() != 27 || input_channels <= 0 || output_channels <= 0) {
        return nullptr;
    }
    // Append one explicit zero point on device.  GET_ROWS then uses the same
    // sentinel convention as the existing sparse-conv implementation without
    // downloading or rebuilding an intermediate feature matrix.
    ggml_tensor * padded = ggml_concat(ctx, input, zero_column, 1);
    if (!padded) return nullptr;

    ggml_tensor * accumulator = nullptr;
    for (int kernel = 0; kernel < 27; ++kernel) {
        ggml_tensor * gathered = ggml_get_rows(ctx, padded, index_tensors[
            static_cast<std::size_t>(kernel)]);
        if (!gathered) return nullptr;
        ggml_tensor * weight_view = ggml_view_2d(
            ctx, weight, input_channels, output_channels, weight->nb[1],
            static_cast<std::size_t>(kernel) * weight->nb[2]);
        ggml_tensor * contiguous_weight = weight_view
            ? ggml_cont(ctx, weight_view) : nullptr;
        ggml_tensor * part = contiguous_weight
            ? ggml_mul_mat(ctx, contiguous_weight, gathered) : nullptr;
        if (!part) return nullptr;
        ggml_mul_mat_set_prec(part, GGML_PREC_F32);
        accumulator = accumulator
            ? ggml_add(ctx, accumulator, part) : part;
        if (!accumulator) return nullptr;
    }
    ggml_tensor * biased = ggml_add(ctx, accumulator, bias);
    if (!biased) return nullptr;
    return name_gpu_tensor(ggml_cont(ctx, biased), name);
}

ggml_tensor * gpu_convnext_block_node(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * zero_column,
    const std::vector<ggml_tensor *> & index_tensors,
    ggml_tensor * norm_weight,
    ggml_tensor * norm_bias,
    ggml_tensor * conv_weight,
    ggml_tensor * conv_bias,
    ggml_tensor * mlp0_weight,
    ggml_tensor * mlp0_bias,
    ggml_tensor * mlp2_weight,
    ggml_tensor * mlp2_bias,
    int channels,
    int mlp_hidden,
    float epsilon,
    const std::string & name) {
    if (!ctx || !input || !zero_column || !norm_weight || !norm_bias ||
        !conv_weight || !conv_bias || !mlp0_weight || !mlp0_bias ||
        !mlp2_weight || !mlp2_bias || channels <= 0 || mlp_hidden <= 0) {
        return nullptr;
    }
    ggml_tensor * convolved = gpu_sparse_conv_node(
        ctx, input, zero_column, conv_weight, conv_bias, index_tensors,
        channels, channels, name + ".conv");
    if (!convolved) return nullptr;
    ggml_tensor * normalized = gpu_layer_norm_affine_node(
        ctx, convolved, norm_weight, norm_bias, epsilon, name + ".norm");
    if (!normalized) return nullptr;
    ggml_tensor * hidden = gpu_linear_node(
        ctx, normalized, mlp0_weight, mlp0_bias, name + ".mlp0");
    if (!hidden) return nullptr;
    hidden = name_gpu_tensor(ggml_silu(ctx, hidden), name + ".silu");
    if (!hidden) return nullptr;
    ggml_tensor * projected = gpu_linear_node(
        ctx, hidden, mlp2_weight, mlp2_bias, name + ".mlp2");
    if (!projected) return nullptr;
    ggml_tensor * residual = ggml_add(ctx, projected, input);
    return name_gpu_tensor(ggml_cont(ctx, residual), name);
}

void log_decoder_profile(const SLatDecoderProfile & profile,
                         const detail::SparseProfileStats & sparse) {
    std::cerr << "pixal3d: SLat-decoder-profile cache_enabled="
              << (profile.neighbor_cache_enabled ? 1 : 0)
              << " validate_cache=" << (profile.validate_neighbor_cache ? 1 : 0)
              << " total_sparse_conv_calls=" << profile.total_sparse_conv_calls
              << " unique_topologies=" << profile.topologies.size()
              << " neighbor_map_builds=" << profile.neighbor_map_builds
              << " cache_hits=" << profile.cache_hits
              << " cache_misses=" << profile.cache_misses
              << " cache_validation_calls=" << profile.cache_validation_calls << std::endl;
    const auto phase = [&profile](const char * name,
                                  double SLatDecoderProfile::* value) {
        std::cerr << "pixal3d: SLat-decoder-profile phase=" << name
                  << " elapsed_ms=" << profile.*value << std::endl;
    };
    phase("input_validation", &SLatDecoderProfile::input_validation_ms);
    phase("padded_input_build", &SLatDecoderProfile::padded_input_build_ms);
    phase("coordinate_fingerprint", &SLatDecoderProfile::coordinate_fingerprint_ms);
    phase("coord_hashmap_build", &SLatDecoderProfile::coord_hashmap_build_ms);
    phase("neighbor_index_build", &SLatDecoderProfile::neighbor_index_build_ms);
    phase("cache_validation", &SLatDecoderProfile::cache_validation_ms);
    phase("index_upload", &SLatDecoderProfile::index_upload_ms);
    phase("graph_build", &SLatDecoderProfile::graph_build_ms);
    phase("scheduler_allocate", &SLatDecoderProfile::scheduler_allocate_ms);
    phase("input_upload", &SLatDecoderProfile::input_upload_ms);
    phase("gpu_compute", &SLatDecoderProfile::gpu_compute_ms);
    phase("output_download", &SLatDecoderProfile::output_download_ms);
    phase("finish_sparse", &SLatDecoderProfile::finish_sparse_ms);
    phase("output_validation", &SLatDecoderProfile::output_validation_ms);

    std::cerr << "pixal3d: SLat-decoder-profile sparse_valid calls=" << sparse.valid_calls
              << " elapsed_ms=" << sparse.valid_ms
              << " coord_hashmap_ms=" << sparse.valid_coord_hashmap_ms
              << " feature_scan_ms=" << sparse.valid_feature_scan_ms
              << " recorded_calls=" << sparse.validation_records.size() << std::endl;
    std::cerr << "pixal3d: SLat-decoder-profile copy_shape calls=" << sparse.copy_shape_calls
              << " bytes=" << sparse.copy_shape_bytes
              << " elapsed_ms=" << sparse.copy_shape_ms << std::endl;
    const auto sparse_op = [&sparse](const char * name, std::uint64_t calls,
                                     double elapsed_ms) {
        std::cerr << "pixal3d: SLat-decoder-profile sparse_op=" << name
                  << " calls=" << calls << " elapsed_ms=" << elapsed_ms << std::endl;
    };
    sparse_op("sparse_linear", sparse.sparse_linear_calls, sparse.sparse_linear_ms);
    sparse_op("sparse_layer_norm", sparse.sparse_layer_norm_calls,
              sparse.sparse_layer_norm_ms);
    sparse_op("sparse_channel_to_spatial", sparse.sparse_channel_to_spatial_calls,
              sparse.sparse_channel_to_spatial_ms);
    sparse_op("sparse_submanifold_conv3d", sparse.sparse_submanifold_conv3d_calls,
              sparse.sparse_submanifold_conv3d_ms);
    sparse_op("add_residual", sparse.add_residual_calls, sparse.add_residual_ms);
    sparse_op("SiLU", sparse.silu_calls, sparse.silu_ms);
    sparse_op("repeat_channels", sparse.repeat_channels_calls, sparse.repeat_channels_ms);

    std::cerr << "pixal3d: SLat-decoder-profile sparse_linear_breakdown"
              << " inclusive_ms=" << sparse.sparse_linear_ms
              << " input_validation_ms=" << sparse.sparse_linear_input_validation_ms
              << " copy_shape_ms=" << sparse.sparse_linear_copy_shape_ms
              << " output_allocation_ms=" << sparse.sparse_linear_output_allocation_ms
              << " actual_compute_ms=" << sparse.sparse_linear_actual_compute_ms
              << " bias_ms=" << sparse.sparse_linear_bias_ms
              << " output_validation_ms=" << sparse.sparse_linear_output_validation_ms
              << " other_ms=" << sparse.sparse_linear_other_ms
              << " output_validation_in_linear=0" << std::endl;

#if defined(NDEBUG)
    constexpr int build_release = 1;
#else
    constexpr int build_release = 0;
#endif
#if defined(__OPTIMIZE__)
    constexpr int compiler_optimized = 1;
#else
    constexpr int compiler_optimized = 0;
#endif
#if defined(_OPENMP)
    constexpr int openmp_compiled = 1;
#else
    constexpr int openmp_compiled = 0;
#endif
#if defined(__AVX512F__)
    constexpr int avx512f = 1;
#else
    constexpr int avx512f = 0;
#endif
#if defined(__AVX2__)
    constexpr int avx2 = 1;
#else
    constexpr int avx2 = 0;
#endif
    std::cerr << "pixal3d: SLat-decoder-profile sparse_linear_environment"
              << " build_release=" << build_release
              << " compiler_optimized=" << compiler_optimized
              << " openmp_compiled=" << openmp_compiled
              << " omp_schedule=static"
              << " avx2=" << avx2
              << " avx512f=" << avx512f
              << " OMP_NUM_THREADS="
              << (std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS") : "unset")
              << " OMP_DYNAMIC="
              << (std::getenv("OMP_DYNAMIC") ? std::getenv("OMP_DYNAMIC") : "unset")
              << std::endl;

    std::vector<std::size_t> linear_order(sparse.sparse_linear_records.size());
    for (std::size_t index = 0; index < linear_order.size(); ++index) {
        linear_order[index] = index;
    }
    std::sort(linear_order.begin(), linear_order.end(), [&sparse](std::size_t left,
                                                                    std::size_t right) {
        return sparse.sparse_linear_records[left].total_ms >
               sparse.sparse_linear_records[right].total_ms;
    });
    for (std::size_t rank = 0; rank < linear_order.size(); ++rank) {
        const detail::SparseLinearProfileRecord & record =
            sparse.sparse_linear_records[linear_order[rank]];
        std::cerr << "pixal3d: SLat-decoder-profile sparse_linear_call rank=" << (rank + 1)
                  << " label=" << record.label
                  << " points=" << record.points
                  << " cin=" << record.input_channels
                  << " cout=" << record.output_channels
                  << " total_ms=" << record.total_ms
                  << " input_validation_ms=" << record.input_validation_ms
                  << " copy_shape_ms=" << record.copy_shape_ms
                  << " output_allocation_ms=" << record.output_allocation_ms
                  << " compute_ms=" << record.actual_compute_ms
                  << " bias_ms=" << record.bias_ms
                  << " output_validation_ms=" << record.output_validation_ms
                  << " other_ms=" << record.other_ms
                  << " omp_max_threads=" << record.omp_max_threads
                  << " omp_actual_threads=" << record.omp_actual_threads
                  << " profile_split_bias=" << (record.profile_split_bias ? 1 : 0)
                  << " success=" << (record.success ? 1 : 0) << std::endl;
    }

    struct LinearShapeKey {
        std::size_t points = 0;
        int input_channels = 0;
        int output_channels = 0;

        bool operator<(const LinearShapeKey & other) const noexcept {
            if (points != other.points) return points < other.points;
            if (input_channels != other.input_channels) {
                return input_channels < other.input_channels;
            }
            return output_channels < other.output_channels;
        }
    };
    struct LinearShapeAggregate {
        std::size_t calls = 0;
        std::size_t successful_calls = 0;
        double total_ms = 0.0;
        double compute_ms = 0.0;
        double bias_ms = 0.0;
        double input_validation_ms = 0.0;
        double copy_shape_ms = 0.0;
        double output_allocation_ms = 0.0;
        double output_validation_ms = 0.0;
        double other_ms = 0.0;
    };
    std::map<LinearShapeKey, LinearShapeAggregate> linear_shapes;
    for (const detail::SparseLinearProfileRecord & record : sparse.sparse_linear_records) {
        LinearShapeAggregate & aggregate = linear_shapes[
            LinearShapeKey{record.points, record.input_channels, record.output_channels}];
        ++aggregate.calls;
        if (record.success) ++aggregate.successful_calls;
        aggregate.total_ms += record.total_ms;
        aggregate.compute_ms += record.actual_compute_ms;
        aggregate.bias_ms += record.bias_ms;
        aggregate.input_validation_ms += record.input_validation_ms;
        aggregate.copy_shape_ms += record.copy_shape_ms;
        aggregate.output_allocation_ms += record.output_allocation_ms;
        aggregate.output_validation_ms += record.output_validation_ms;
        aggregate.other_ms += record.other_ms;
    }
    std::vector<std::pair<LinearShapeKey, LinearShapeAggregate>> linear_shape_order(
        linear_shapes.begin(), linear_shapes.end());
    std::sort(linear_shape_order.begin(), linear_shape_order.end(),
              [](const auto & left, const auto & right) {
                  return left.second.total_ms > right.second.total_ms;
              });
    for (const auto & entry : linear_shape_order) {
        const LinearShapeKey & key = entry.first;
        const LinearShapeAggregate & aggregate = entry.second;
        std::cerr << "pixal3d: SLat-decoder-profile sparse_linear_shape"
                  << " points=" << key.points
                  << " cin=" << key.input_channels
                  << " cout=" << key.output_channels
                  << " calls=" << aggregate.calls
                  << " successful_calls=" << aggregate.successful_calls
                  << " total_ms=" << aggregate.total_ms
                  << " compute_ms=" << aggregate.compute_ms
                  << " bias_ms=" << aggregate.bias_ms
                  << " input_validation_ms=" << aggregate.input_validation_ms
                  << " copy_shape_ms=" << aggregate.copy_shape_ms
                  << " output_allocation_ms=" << aggregate.output_allocation_ms
                  << " output_validation_ms=" << aggregate.output_validation_ms
                  << " other_ms=" << aggregate.other_ms << std::endl;
    }

    const auto validation_category = [](const std::string & label) {
        if (label.find("decoder.") == 0) return "A_stage_boundary";
        if (label.find("sparse_channel_to_spatial") != std::string::npos) {
            return "B_topology_change";
        }
        if (label.find("sparse_conv_gpu") != std::string::npos ||
            label.find("sparse_submanifold_conv3d") != std::string::npos) {
            return "D_sparse_conv";
        }
        if (label.find("sparse_linear") != std::string::npos ||
            label.find("sparse_layer_norm") != std::string::npos ||
            label.find("repeat_channels") != std::string::npos) {
            return "C_pointwise_hot_path";
        }
        if (label.find("test") != std::string::npos ||
            label.find("debug") != std::string::npos) {
            return "E_debug_or_test";
        }
        return "other";
    };
    struct ValidationAggregate {
        std::size_t calls = 0;
        std::size_t failures = 0;
        std::size_t max_points = 0;
        int max_channels = 0;
        double total_ms = 0.0;
        double coord_hashmap_ms = 0.0;
        double feature_scan_ms = 0.0;
    };
    std::map<std::string, ValidationAggregate> validations;
    for (const detail::SparseValidationProfileRecord & record : sparse.validation_records) {
        ValidationAggregate & aggregate = validations[record.label];
        ++aggregate.calls;
        if (!record.success) ++aggregate.failures;
        aggregate.max_points = std::max(aggregate.max_points, record.points);
        aggregate.max_channels = std::max(aggregate.max_channels, record.channels);
        aggregate.total_ms += record.elapsed_ms;
        aggregate.coord_hashmap_ms += record.coord_hashmap_ms;
        aggregate.feature_scan_ms += record.feature_scan_ms;
    }
    std::vector<std::pair<std::string, ValidationAggregate>> validation_order(
        validations.begin(), validations.end());
    std::sort(validation_order.begin(), validation_order.end(),
              [](const auto & left, const auto & right) {
                  return left.second.total_ms > right.second.total_ms;
              });
    for (const auto & entry : validation_order) {
        const ValidationAggregate & aggregate = entry.second;
        std::cerr << "pixal3d: SLat-decoder-profile valid_group"
                  << " category=" << validation_category(entry.first)
                  << " label=" << entry.first
                  << " calls=" << aggregate.calls
                  << " failures=" << aggregate.failures
                  << " max_points=" << aggregate.max_points
                  << " max_channels=" << aggregate.max_channels
                  << " total_ms=" << aggregate.total_ms
                  << " coord_hashmap_ms=" << aggregate.coord_hashmap_ms
                  << " feature_scan_ms=" << aggregate.feature_scan_ms << std::endl;
    }
    for (const SLatDecoderTopologyProfile & topology : profile.topologies) {
        std::cerr << "pixal3d: SLat-decoder-profile topology generation="
                  << topology.generation << " points=" << topology.points
                  << " batch_size=" << topology.batch_size
                  << " spatial=" << topology.spatial_x << 'x' << topology.spatial_y << 'x'
                  << topology.spatial_z
                  << " coords=0x" << std::hex << topology.coordinate_fingerprint << std::dec
                  << " conv_calls=" << topology.conv_calls
                  << " reused_by_convs=" << (topology.conv_calls > 0
                      ? topology.conv_calls - 1 : 0)
                  << " cache_hits=" << topology.cache_hits
                  << " cache_misses=" << topology.cache_misses << std::endl;
    }
}

struct SLatDecoderGpuState {
    ggml_context * weights_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    std::string backend_name;
    std::unordered_map<std::string, ggml_tensor *> tensors;
    std::unordered_map<const float *, ggml_tensor *> host_tensors;
    BackendManager * backend_manager = nullptr;
    bool attempted = false;

    // The map is deliberately invocation-scoped.  It is invalidated only at
    // the explicit channel-to-spatial topology boundary; no global hash cache
    // or model-parameter-based guess is involved.
    NeighborGatherMap current_neighbor_map;
    bool has_neighbor_map = false;
    std::uint64_t topology_generation = 0;
    bool neighbor_cache_enabled = true;
    bool validate_neighbor_cache = false;
    SLatDecoderProfile * profile = nullptr;

    ~SLatDecoderGpuState() { close(); }

    void close() noexcept {
        profile = nullptr;
        has_neighbor_map = false;
        current_neighbor_map = NeighborGatherMap{};
        topology_generation = 0;
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

    void begin_invocation(SLatDecoderProfile * active_profile,
                          bool cache_enabled,
                          bool validate_cache) {
        profile = active_profile;
        neighbor_cache_enabled = cache_enabled;
        validate_neighbor_cache = validate_cache;
        has_neighbor_map = false;
        current_neighbor_map = NeighborGatherMap{};
        topology_generation = 0;
    }

    void end_invocation(const detail::SparseProfileStats & sparse) noexcept {
        if (profile) log_decoder_profile(*profile, sparse);
        profile = nullptr;
        has_neighbor_map = false;
        current_neighbor_map = NeighborGatherMap{};
        topology_generation = 0;
    }

    void invalidate_neighbor_map() {
        has_neighbor_map = false;
        current_neighbor_map = NeighborGatherMap{};
        ++topology_generation;
    }

    bool ready() const noexcept { return backend != nullptr && weights_buffer != nullptr; }

    ggml_tensor * weight(const float * host) const noexcept {
        const auto it = host_tensors.find(host);
        return it == host_tensors.end() ? nullptr : it->second;
    }

    bool prepare_neighbor_map(const SparseTensorF32 & input,
                              NeighborGatherMap & local_map,
                              NeighborGatherMap *& gather_map,
                              bool & cache_hit,
                              std::uint64_t & coordinate_fingerprint,
                              std::string * error);

    bool run_convnext_level(
        const SparseTensorF32 & input,
        const SLatDecoderBlockWeightsF32 * blocks,
        std::size_t block_count,
        float norm_eps,
        const float * prefix_weight,
        const float * prefix_bias,
        int prefix_out_channels,
        bool final_layer_norm,
        const float * output_weight,
        const float * output_bias,
        int output_channels,
        const std::string & label,
        SparseTensorF32 & output,
        std::string * error);

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

    bool finish_sparse(const SparseTensorF32 & input,
                       int channels,
                       const std::vector<float> & channel_values,
                       SparseTensorF32 & output,
                       std::string * error) {
        const double finish_begin = profile ? backend_time_now_ms() : 0.0;
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
        decoder_profile_add_phase(profile, &SLatDecoderProfile::finish_sparse_ms,
                                  finish_begin);
        const double validation_begin = profile ? backend_time_now_ms() : 0.0;
        const bool valid = validate_decoder_sparse_tensor(
            output, "sparse_conv_gpu.output", error);
        decoder_profile_add_phase(profile, &SLatDecoderProfile::output_validation_ms,
                                  validation_begin);
        return valid;
    }

    bool run_conv(const SparseTensorF32 & input,
                  const float * host_weight,
                  const float * host_bias,
                  int out_channels,
                  SparseTensorF32 & output,
                  std::string * error) {
        if (!ready()) {
            set_error(error, "SLat decoder GPU state is not initialized");
            return false;
        }
        const double input_validation_begin = profile ? backend_time_now_ms() : 0.0;
        const bool input_valid = validate_decoder_sparse_tensor(
            input, "sparse_conv_gpu.input", error);
        decoder_profile_add_phase(profile, &SLatDecoderProfile::input_validation_ms,
                                  input_validation_begin);
        if (!input_valid || out_channels <= 0) return false;
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

        // The public feature rows use ggml's contiguous [channel, point] byte
        // order, so append the explicit zero column without a transpose.
        const double padded_input_begin = profile ? backend_time_now_ms() : 0.0;
        const std::size_t input_columns = points + 1;
        std::vector<float> padded_values(
            static_cast<std::size_t>(input.channels) * input_columns, 0.0f);
        std::copy(input.feats.begin(), input.feats.end(), padded_values.begin());
        decoder_profile_add_phase(profile, &SLatDecoderProfile::padded_input_build_ms,
                                  padded_input_begin);

        NeighborGatherMap local_map;
        NeighborGatherMap * gather_map = nullptr;
        bool cache_hit = false;
        std::uint64_t coordinate_fingerprint = 0;
        if (!prepare_neighbor_map(input, local_map, gather_map, cache_hit,
                                  coordinate_fingerprint, error)) {
            return false;
        }

        const double graph_build_begin = profile ? backend_time_now_ms() : 0.0;
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
        decoder_profile_add_phase(profile, &SLatDecoderProfile::graph_build_ms,
                                  graph_build_begin);

        const bool trace = slat_decoder_trace_enabled();
        const double allocation_begin = profile ? backend_time_now_ms() : 0.0;
        std::string scheduler_error;
        BackendScheduler scheduler(*backend_manager, 4096, false, true,
                                   &scheduler_error, "SLat decoder");
        if (!scheduler.valid() ||
            (backend_manager->requires_primary_backend() &&
             !scheduler.require_primary_graph(graph, &scheduler_error)) ||
            !scheduler.allocate_graph(graph, &scheduler_error)) {
            scheduler.synchronize();
            scheduler = BackendScheduler{};
            ggml_free(ctx);
            set_error(error, scheduler_error.empty()
                ? "failed to allocate SLat decoder GPU convolution graph" : scheduler_error);
            return false;
        }
        decoder_profile_add_phase(profile, &SLatDecoderProfile::scheduler_allocate_ms,
                                  allocation_begin);
        if (trace) {
            backend_log_timing("SLat-decoder-conv", "scheduler_allocate",
                               backend_time_now_ms() - allocation_begin);
        }
        const double input_upload_begin = profile ? backend_time_now_ms() : 0.0;
        ggml_backend_tensor_set(input_tensor, padded_values.data(), 0,
                                padded_values.size() * sizeof(float));
        decoder_profile_add_phase(profile, &SLatDecoderProfile::input_upload_ms,
                                  input_upload_begin);
        if (trace) {
            backend_log_timing("SLat-decoder-conv", "input_upload",
                               backend_time_now_ms() - input_upload_begin);
        }
        const double index_upload_begin = profile ? backend_time_now_ms() : 0.0;
        for (std::size_t index = 0; index < index_tensors.size(); ++index) {
            const std::int32_t * index_values =
                gather_map->gather_indices.data() + index * points;
            ggml_backend_tensor_set(index_tensors[index], index_values, 0,
                                    points * sizeof(std::int32_t));
        }
        decoder_profile_add_phase(profile, &SLatDecoderProfile::index_upload_ms,
                                  index_upload_begin);
        if (trace) {
            backend_log_timing("SLat-decoder-conv", "index_upload",
                               backend_time_now_ms() - index_upload_begin);
        }
        const double compute_begin = profile ? backend_time_now_ms() : 0.0;
        const ggml_status status = scheduler.compute(graph, &scheduler_error);
        decoder_profile_add_phase(profile, &SLatDecoderProfile::gpu_compute_ms, compute_begin);
        if (trace) {
            backend_log_timing("SLat-decoder-conv", "gpu_compute",
                               backend_time_now_ms() - compute_begin);
        }
        bool ok = status == GGML_STATUS_SUCCESS;
        std::vector<float> result_values(points * static_cast<std::size_t>(out_channels));
        if (ok) {
            const double output_download_begin = profile ? backend_time_now_ms() : 0.0;
            ggml_backend_tensor_get(accumulator, result_values.data(), 0,
                                    result_values.size() * sizeof(float));
            decoder_profile_add_phase(profile, &SLatDecoderProfile::output_download_ms,
                                      output_download_begin);
            if (trace) {
                backend_log_timing("SLat-decoder-conv", "output_download",
                                   backend_time_now_ms() - output_download_begin);
                std::cerr << "pixal3d: SLat-decoder-conv points=" << points
                          << " in_channels=" << input.channels
                          << " out_channels=" << out_channels
                          << " input_bytes=" << padded_values.size() * sizeof(float)
                          << " index_bytes=" << index_tensors.size() * points * sizeof(std::int32_t)
                          << " output_bytes=" << result_values.size() * sizeof(float)
                          << std::endl;
            }
            ok = finish_sparse(input, out_channels, result_values, output, error);
        } else {
            set_error(error, "SLat decoder GPU convolution graph compute failed" +
                      (scheduler_error.empty() ? std::string{} : ": " + scheduler_error));
        }
        scheduler.synchronize();
        scheduler = BackendScheduler{};
        ggml_free(ctx);
        return ok;
    }
};

bool SLatDecoderGpuState::prepare_neighbor_map(
    const SparseTensorF32 & input,
    NeighborGatherMap & local_map,
    NeighborGatherMap *& gather_map,
    bool & cache_hit,
    std::uint64_t & coordinate_fingerprint,
    std::string * error) {
    gather_map = nullptr;
    cache_hit = false;
    coordinate_fingerprint = 0;
    if (profile) {
        const double fingerprint_begin = backend_time_now_ms();
        coordinate_fingerprint = decoder_coords_fingerprint(input.coords);
        profile->coordinate_fingerprint_ms +=
            backend_time_now_ms() - fingerprint_begin;
    }

    if (neighbor_cache_enabled && has_neighbor_map) {
        if (!neighbor_map_shape_matches(current_neighbor_map, input, profile != nullptr,
                                        coordinate_fingerprint)) {
            set_error(error,
                      "SLat decoder neighbor map topology changed without invalidation "
                      "(generation=" + std::to_string(topology_generation) + ")");
            return false;
        }
        cache_hit = true;
        if (validate_neighbor_cache) {
            if (profile) ++profile->cache_validation_calls;
            const double validation_begin = profile ? backend_time_now_ms() : 0.0;
            const bool valid = validate_neighbor_gather_map(input, current_neighbor_map, error);
            decoder_profile_add_phase(profile, &SLatDecoderProfile::cache_validation_ms,
                                      validation_begin);
            if (!valid) return false;
        }
        gather_map = &current_neighbor_map;
    } else {
        if (!build_neighbor_gather_map(input, coordinate_fingerprint, local_map, profile,
                                       error)) {
            return false;
        }
        if (neighbor_cache_enabled) {
            current_neighbor_map = std::move(local_map);
            has_neighbor_map = true;
            gather_map = &current_neighbor_map;
        } else {
            gather_map = &local_map;
        }
    }
    return record_decoder_conv(profile, topology_generation, input,
                               coordinate_fingerprint, cache_hit, error);
}

bool SLatDecoderGpuState::run_convnext_level(
    const SparseTensorF32 & input,
    const SLatDecoderBlockWeightsF32 * blocks,
    std::size_t block_count,
    float norm_eps,
    const float * prefix_weight,
    const float * prefix_bias,
    int prefix_out_channels,
    bool final_layer_norm,
    const float * output_weight,
    const float * output_bias,
    int output_channels,
    const std::string & label,
    SparseTensorF32 & output,
    std::string * error) {
    output = SparseTensorF32{};
    if (!ready() || !backend_manager || !backend_manager->initialized()) {
        set_error(error, label + ": GPU state is not initialized");
        return false;
    }
    if (!valid_eps(norm_eps, error)) return false;
    if (block_count > 0 && !blocks) {
        set_error(error, label + ": ConvNeXt block weights are missing");
        return false;
    }
    const bool has_prefix = prefix_weight || prefix_bias;
    if (has_prefix && (!prefix_weight || !prefix_bias || prefix_out_channels <= 0)) {
        set_error(error, label + ": incomplete prefix linear specification");
        return false;
    }
    const bool has_output = output_weight || output_bias;
    if (has_output && (!output_weight || !output_bias || output_channels <= 0)) {
        set_error(error, label + ": incomplete output linear specification");
        return false;
    }

    const double input_validation_begin = profile ? backend_time_now_ms() : 0.0;
    const bool input_valid = validate_decoder_sparse_tensor(
        input, (label + ".input").c_str(), error);
    decoder_profile_add_phase(profile, &SLatDecoderProfile::input_validation_ms,
                              input_validation_begin);
    if (!input_valid) return false;
    const std::size_t points = input.points();
    if (points == 0 || points > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
        set_error(error, label + ": invalid point count");
        return false;
    }

    const int feature_channels = has_prefix
        ? prefix_out_channels
        : (block_count > 0 ? blocks[0].channels : input.channels);
    if (feature_channels <= 0) {
        set_error(error, label + ": invalid feature channel count");
        return false;
    }
    if (!has_prefix && block_count == 0 && input.channels != feature_channels) {
        set_error(error, label + ": input feature channel count is inconsistent");
        return false;
    }
    for (std::size_t index = 0; index < block_count; ++index) {
        const SLatDecoderBlockWeightsF32 & block = blocks[index];
        if (block.kind != SLatDecoderBlockKind::convnext ||
            block.channels != feature_channels ||
            block.out_channels != feature_channels || block.mlp_hidden <= 0) {
            set_error(error, label + ": ConvNeXt block channel layout is inconsistent");
            return false;
        }
    }

    NeighborGatherMap local_graph_map;
    NeighborGatherMap * graph_map = nullptr;
    if (block_count > 0) {
        for (std::size_t index = 0; index < block_count; ++index) {
            NeighborGatherMap local_map;
            NeighborGatherMap * map = nullptr;
            bool cache_hit = false;
            std::uint64_t coordinate_fingerprint = 0;
            if (index == 0) {
                if (!prepare_neighbor_map(input, local_graph_map, map, cache_hit,
                                          coordinate_fingerprint, error)) {
                    return false;
                }
                graph_map = map;
            } else if (!prepare_neighbor_map(input, local_map, map, cache_hit,
                                             coordinate_fingerprint, error)) {
                return false;
            }
        }
        if (!graph_map || graph_map->gather_indices.size() != points * 27) {
            set_error(error, label + ": invalid GPU neighbor map");
            return false;
        }
    }

    auto lookup_weight = [this, &label, error](const float * host,
                                                const std::string & name) -> ggml_tensor * {
        if (!host) {
            set_error(error, label + ": missing weight " + name);
            return nullptr;
        }
        ggml_tensor * tensor = weight(host);
        if (!tensor) {
            set_error(error, label + ": weight is not resident on GPU: " + name);
            return nullptr;
        }
        return tensor;
    };

    ggml_tensor * prefix_weight_tensor = nullptr;
    ggml_tensor * prefix_bias_tensor = nullptr;
    if (has_prefix) {
        prefix_weight_tensor = lookup_weight(prefix_weight, "prefix.weight");
        prefix_bias_tensor = lookup_weight(prefix_bias, "prefix.bias");
        if (!prefix_weight_tensor || !prefix_bias_tensor) return false;
        if (prefix_weight_tensor->type != GGML_TYPE_F32 ||
            prefix_weight_tensor->ne[0] != input.channels ||
            prefix_weight_tensor->ne[1] != prefix_out_channels ||
            prefix_bias_tensor->type != GGML_TYPE_F32 ||
            prefix_bias_tensor->ne[0] != prefix_out_channels) {
            set_error(error, label + ": prefix linear tensor shape mismatch");
            return false;
        }
    }

    ggml_tensor * output_weight_tensor = nullptr;
    ggml_tensor * output_bias_tensor = nullptr;
    if (has_output) {
        output_weight_tensor = lookup_weight(output_weight, "output.weight");
        output_bias_tensor = lookup_weight(output_bias, "output.bias");
        if (!output_weight_tensor || !output_bias_tensor) return false;
        if (output_weight_tensor->type != GGML_TYPE_F32 ||
            output_weight_tensor->ne[0] != feature_channels ||
            output_weight_tensor->ne[1] != output_channels ||
            output_bias_tensor->type != GGML_TYPE_F32 ||
            output_bias_tensor->ne[0] != output_channels) {
            set_error(error, label + ": output linear tensor shape mismatch");
            return false;
        }
    }

    constexpr std::size_t kGraphBase = 1024;
    constexpr std::size_t kPerBlockCapacity = 256;
    if (block_count > (std::numeric_limits<std::size_t>::max() - kGraphBase) /
                          kPerBlockCapacity) {
        set_error(error, label + ": GPU graph size overflows the ggml limits");
        return false;
    }
    const std::size_t graph_capacity = kGraphBase + block_count * kPerBlockCapacity;
    const std::size_t tensor_capacity = graph_capacity;
    if (graph_capacity > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        tensor_capacity > std::numeric_limits<std::size_t>::max() /
                              ggml_tensor_overhead()) {
        set_error(error, label + ": GPU graph size overflows the ggml limits");
        return false;
    }
    const double graph_build_begin = profile ? backend_time_now_ms() : 0.0;
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * tensor_capacity +
                      ggml_graph_overhead_custom(static_cast<int>(graph_capacity), false) +
                      4096;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        set_error(error, label + ": failed to allocate GPU graph context");
        return false;
    }

    ggml_tensor * input_tensor = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, input.channels, static_cast<int64_t>(points));
    if (!input_tensor) {
        ggml_free(ctx);
        set_error(error, label + ": failed to allocate GPU feature input");
        return false;
    }
    ggml_set_name(input_tensor, (label + ".input").c_str());
    ggml_set_input(input_tensor);

    ggml_tensor * zero_column = nullptr;
    std::vector<float> zero_values;
    std::vector<ggml_tensor *> index_tensors;
    if (block_count > 0) {
        zero_column = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, feature_channels, 1);
        if (!zero_column) {
            ggml_free(ctx);
            set_error(error, label + ": failed to allocate GPU zero sentinel");
            return false;
        }
        ggml_set_name(zero_column, (label + ".zero_column").c_str());
        ggml_set_input(zero_column);
        zero_values.assign(static_cast<std::size_t>(feature_channels), 0.0f);

        index_tensors.reserve(27);
        for (int kernel = 0; kernel < 27; ++kernel) {
            ggml_tensor * index_tensor = ggml_new_tensor_1d(
                ctx, GGML_TYPE_I32, static_cast<int64_t>(points));
            if (!index_tensor) {
                ggml_free(ctx);
                set_error(error, label + ": failed to allocate GPU neighbor index");
                return false;
            }
            ggml_set_name(index_tensor,
                          (label + ".neighbor_index." + std::to_string(kernel)).c_str());
            ggml_set_input(index_tensor);
            index_tensors.push_back(index_tensor);
        }
    }

    ggml_tensor * feature = input_tensor;
    if (has_prefix) {
        feature = gpu_linear_node(ctx, feature, prefix_weight_tensor,
                                  prefix_bias_tensor, label + ".prefix");
        if (!feature) {
            ggml_free(ctx);
            set_error(error, label + ": failed to build prefix linear graph");
            return false;
        }
    }

    for (std::size_t index = 0; index < block_count; ++index) {
        const SLatDecoderBlockWeightsF32 & block = blocks[index];
        const std::string block_label = label + ".block=" + std::to_string(index);
        ggml_tensor * norm_weight = lookup_weight(block.norm_weight,
                                                   block_label + ".norm.weight");
        ggml_tensor * norm_bias = lookup_weight(block.norm_bias,
                                                 block_label + ".norm.bias");
        ggml_tensor * conv_weight = lookup_weight(block.conv1_weight,
                                                  block_label + ".conv.weight");
        ggml_tensor * conv_bias = lookup_weight(block.conv1_bias,
                                                block_label + ".conv.bias");
        ggml_tensor * mlp0_weight = lookup_weight(block.mlp0_weight,
                                                  block_label + ".mlp.0.weight");
        ggml_tensor * mlp0_bias = lookup_weight(block.mlp0_bias,
                                                block_label + ".mlp.0.bias");
        ggml_tensor * mlp2_weight = lookup_weight(block.mlp2_weight,
                                                  block_label + ".mlp.2.weight");
        ggml_tensor * mlp2_bias = lookup_weight(block.mlp2_bias,
                                                block_label + ".mlp.2.bias");
        if (!norm_weight || !norm_bias || !conv_weight || !conv_bias ||
            !mlp0_weight || !mlp0_bias || !mlp2_weight || !mlp2_bias) {
            ggml_free(ctx);
            return false;
        }
        if (norm_weight->type != GGML_TYPE_F32 || norm_weight->ne[0] != feature_channels ||
            norm_bias->type != GGML_TYPE_F32 || norm_bias->ne[0] != feature_channels ||
            conv_weight->type != GGML_TYPE_F32 || conv_weight->ne[0] != feature_channels ||
            conv_weight->ne[1] != feature_channels || conv_weight->ne[2] != 27 ||
            conv_bias->type != GGML_TYPE_F32 || conv_bias->ne[0] != feature_channels ||
            mlp0_weight->type != GGML_TYPE_F32 || mlp0_weight->ne[0] != feature_channels ||
            mlp0_weight->ne[1] != block.mlp_hidden ||
            mlp0_bias->type != GGML_TYPE_F32 || mlp0_bias->ne[0] != block.mlp_hidden ||
            mlp2_weight->type != GGML_TYPE_F32 || mlp2_weight->ne[0] != block.mlp_hidden ||
            mlp2_weight->ne[1] != feature_channels ||
            mlp2_bias->type != GGML_TYPE_F32 || mlp2_bias->ne[0] != feature_channels) {
            ggml_free(ctx);
            set_error(error, label + ": ConvNeXt GPU tensor shape mismatch at block " +
                              std::to_string(index));
            return false;
        }
        feature = gpu_convnext_block_node(
            ctx, feature, zero_column, index_tensors,
            norm_weight, norm_bias, conv_weight, conv_bias,
            mlp0_weight, mlp0_bias, mlp2_weight, mlp2_bias,
            feature_channels, block.mlp_hidden, norm_eps, block_label);
        if (!feature) {
            ggml_free(ctx);
            set_error(error, label + ": failed to build ConvNeXt GPU graph at block " +
                              std::to_string(index));
            return false;
        }
    }

    if (final_layer_norm) {
        feature = gpu_layer_norm_node(ctx, feature,
                                      k_slat_decoder_final_layer_norm_eps,
                                      label + ".final_norm");
        if (!feature) {
            ggml_free(ctx);
            set_error(error, label + ": failed to build final LayerNorm graph");
            return false;
        }
    }
    if (has_output) {
        feature = gpu_linear_node(ctx, feature, output_weight_tensor,
                                  output_bias_tensor, label + ".output");
        if (!feature) {
            ggml_free(ctx);
            set_error(error, label + ": failed to build output linear graph");
            return false;
        }
    }
    ggml_set_output(feature);
    ggml_cgraph * graph = ggml_new_graph_custom(
        ctx, static_cast<int>(graph_capacity), false);
    if (!graph) {
        ggml_free(ctx);
        set_error(error, label + ": failed to allocate GPU graph");
        return false;
    }
    ggml_build_forward_expand(graph, feature);
    decoder_profile_add_phase(profile, &SLatDecoderProfile::graph_build_ms,
                              graph_build_begin);

    const bool trace = slat_decoder_trace_enabled();
    const double allocation_begin = profile ? backend_time_now_ms() : 0.0;
    std::string scheduler_error;
    BackendScheduler scheduler(*backend_manager, graph_capacity, false, true,
                               &scheduler_error, label.c_str());
    if (!scheduler.valid()) {
        scheduler_error = scheduler_error.empty()
            ? "failed to initialize GPU scheduler" : scheduler_error;
    }

    std::size_t cpu_fallback_ops = 0;
    const auto pin_if_gpu_supported = [&](ggml_tensor * node, bool operation) -> bool {
        if (!node) {
            set_error(error, label + ": cannot place a null graph tensor");
            return false;
        }
        std::string support_error;
        if (backend_manager->primary_supports_op(node, &support_error)) {
            return scheduler.require_primary(node, error);
        }
        if (backend_manager->policy().kind == BackendPolicyKind::gpu) {
            set_error(error, label + ": forced GPU cannot place " +
                              (operation ? std::string(ggml_op_name(node->op))
                                         : std::string("input tensor")) +
                              " (" + support_error + ")");
            return false;
        }
        ggml_backend_t fallback = backend_manager->backend_for_op(node);
        if (!fallback) {
            set_error(error, label + ": no configured backend supports " +
                              std::string(ggml_op_name(node->op)));
            return false;
        }
        if (operation) {
            ++cpu_fallback_ops;
            std::cerr << "pixal3d: GPU-first placement fallback stage=" << label
                      << " op=" << ggml_op_name(node->op)
                      << " tensor=" << ggml_get_name(node)
                      << " backend=" << ggml_backend_name(fallback)
                      << " reason=" << support_error << std::endl;
        }
        return true;
    };

    bool placement_ok = scheduler.valid();
    if (placement_ok) placement_ok = pin_if_gpu_supported(input_tensor, false);
    if (placement_ok && zero_column) {
        placement_ok = pin_if_gpu_supported(zero_column, false);
    }
    for (ggml_tensor * index_tensor : index_tensors) {
        if (placement_ok) placement_ok = pin_if_gpu_supported(index_tensor, false);
    }
    if (placement_ok) {
        for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
            ggml_tensor * node = ggml_graph_node(graph, index);
            if (!node || node->op == GGML_OP_NONE) continue;
            if (!pin_if_gpu_supported(node, true)) {
                placement_ok = false;
                break;
            }
        }
    }
    if (placement_ok && !scheduler.allocate_graph(graph, &scheduler_error)) {
        placement_ok = false;
    }
    decoder_profile_add_phase(profile, &SLatDecoderProfile::scheduler_allocate_ms,
                              allocation_begin);
    if (!placement_ok) {
        scheduler.synchronize();
        scheduler = BackendScheduler{};
        ggml_free(ctx);
        set_error(error, label + ": GPU graph placement/allocation failed" +
                          (scheduler_error.empty() ? std::string{} : ": " + scheduler_error));
        return false;
    }

    const double input_upload_begin = profile ? backend_time_now_ms() : 0.0;
    ggml_backend_tensor_set(input_tensor, input.feats.data(), 0,
                            input.feats.size() * sizeof(float));
    if (zero_column) {
        ggml_backend_tensor_set(zero_column, zero_values.data(), 0,
                                zero_values.size() * sizeof(float));
    }
    decoder_profile_add_phase(profile, &SLatDecoderProfile::input_upload_ms,
                              input_upload_begin);

    const double index_upload_begin = profile ? backend_time_now_ms() : 0.0;
    for (std::size_t index = 0; index < index_tensors.size(); ++index) {
        const std::int32_t * values = graph_map->gather_indices.data() + index * points;
        ggml_backend_tensor_set(index_tensors[index], values, 0,
                                points * sizeof(std::int32_t));
    }
    decoder_profile_add_phase(profile, &SLatDecoderProfile::index_upload_ms,
                              index_upload_begin);

    const double compute_begin = profile ? backend_time_now_ms() : 0.0;
    const ggml_status status = scheduler.compute(graph, &scheduler_error);
    decoder_profile_add_phase(profile, &SLatDecoderProfile::gpu_compute_ms, compute_begin);
    if (status != GGML_STATUS_SUCCESS) {
        scheduler.synchronize();
        scheduler = BackendScheduler{};
        ggml_free(ctx);
        set_error(error, label + ": GPU graph compute failed" +
                          (scheduler_error.empty() ? std::string{} : ": " + scheduler_error));
        return false;
    }

    const int result_channels = has_output ? output_channels : feature_channels;
    const std::size_t result_count = points * static_cast<std::size_t>(result_channels);
    std::vector<float> result_values(result_count, 0.0f);
    const double output_download_begin = profile ? backend_time_now_ms() : 0.0;
    ggml_backend_tensor_get(feature, result_values.data(), 0,
                            result_values.size() * sizeof(float));
    decoder_profile_add_phase(profile, &SLatDecoderProfile::output_download_ms,
                              output_download_begin);

    if (trace) {
        scheduler.log_trace(graph, "gpu-first-computed");
        std::cerr << "pixal3d: GPU-first level=" << label
                  << " points=" << points
                  << " blocks=" << block_count
                  << " input_channels=" << input.channels
                  << " output_channels=" << result_channels
                  << " h2d_inputs=" << (zero_column ? 2 : 1)
                  << " h2d_indices=" << index_tensors.size()
                  << " d2h_outputs=1"
                  << " weight_uploads=0"
                  << " cpu_fallback_ops=" << cpu_fallback_ops << std::endl;
    }
    const bool finished = finish_sparse(input, result_channels, result_values, output, error);
    scheduler.synchronize();
    scheduler = BackendScheduler{};
    ggml_free(ctx);
    return finished;
}

struct SLatDecoderInvocation {
    SLatDecoderGpuState & gpu;
    SLatDecoderProfile profile;
    detail::SparseProfileStats sparse;
    detail::SparseProfileScope sparse_scope;

    SLatDecoderInvocation(SLatDecoderGpuState & state,
                          bool profiling,
                          bool cache_enabled,
                          bool validate_cache)
        : gpu(state),
          sparse_scope(profiling ? &sparse : nullptr) {
        profile.neighbor_cache_enabled = cache_enabled;
        profile.validate_neighbor_cache = validate_cache;
        profile.topologies.reserve(8);
        gpu.begin_invocation(profiling ? &profile : nullptr,
                             cache_enabled, validate_cache);
    }

    ~SLatDecoderInvocation() {
        gpu.end_invocation(sparse);
    }

    SLatDecoderInvocation(const SLatDecoderInvocation &) = delete;
    SLatDecoderInvocation & operator=(const SLatDecoderInvocation &) = delete;
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
    bool decode_gpu_first(const SparseTensorF32 & input,
                          const std::vector<SparseTensorF32> * guide_subdivisions,
                          SparseTensorF32 & output,
                          std::vector<SparseTensorF32> * predicted_subdivisions,
                          std::string * error);
    bool upsample_coords_gpu(const SparseTensorF32 & input,
                             int upsample_times,
                             SparseTensorF32 & output,
                             std::string * error);
    bool upsample_coords_gpu_first(const SparseTensorF32 & input,
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

bool SLatDecoderModel::Impl::decode_gpu_first(
    const SparseTensorF32 & input,
    const std::vector<SparseTensorF32> * guide_subdivisions,
    SparseTensorF32 & output,
    std::vector<SparseTensorF32> * predicted_subdivisions,
    std::string * error) {
    SLatDecoderInvocation invocation(
        gpu, slat_decoder_profile_enabled(),
        slat_decoder_neighbor_cache_enabled(),
        slat_decoder_neighbor_cache_validation_enabled());
    SLatDecoderWeightsF32 weights;
    if (!make_weights(weights, error)) return false;

    SLatDecoderConfig config;
    config.latent_channels = hp.latent_channels;
    config.out_channels = hp.out_channels;
    config.norm_eps = hp.norm_eps;
    config.pred_subdiv = hp.pred_subdiv;
    config.model_channels = hp.model_channels;
    config.num_blocks = hp.num_blocks;
    const std::size_t levels = config.model_channels.size();
    if (levels == 0 || config.num_blocks.size() != levels ||
        (guide_subdivisions && guide_subdivisions->size() != levels - 1) ||
        (!config.pred_subdiv && !guide_subdivisions)) {
        set_error(error, "invalid SLat decoder GPU-first configuration");
        return false;
    }
    if (predicted_subdivisions) predicted_subdivisions->clear();

    std::cerr << "pixal3d: " << component
              << " SLat GPU-first decoder path enabled"
              << " (features stay on GPU within each level; C2S remains host boundary)"
              << std::endl;

    SparseTensorF32 hidden;
    std::size_t block_index = 0;
    for (std::size_t level = 0; level < levels; ++level) {
        const std::size_t block_count = static_cast<std::size_t>(config.num_blocks[level]);
        const bool final_level = level + 1 == levels;
        SparseTensorF32 level_output;
        const SLatDecoderBlockWeightsF32 * level_blocks = block_count > 0
            ? weights.blocks.data() + block_index : nullptr;
        if (!gpu.run_convnext_level(
                level == 0 ? input : hidden,
                level_blocks, block_count, config.norm_eps,
                level == 0 ? weights.from_latent_weight : nullptr,
                level == 0 ? weights.from_latent_bias : nullptr,
                level == 0 ? config.model_channels.front() : 0,
                final_level,
                final_level ? weights.output_weight : nullptr,
                final_level ? weights.output_bias : nullptr,
                final_level ? config.out_channels : 0,
                "decode.level=" + std::to_string(level), level_output, error)) {
            return false;
        }
        block_index += block_count;
        if (final_level) {
            output = std::move(level_output);
            break;
        }
        hidden = std::move(level_output);

        if (block_index >= weights.blocks.size() ||
            weights.blocks[block_index].kind != SLatDecoderBlockKind::channel_to_spatial) {
            set_error(error, "SLat decoder GPU-first expected channel-to-spatial block");
            return false;
        }
        const SLatDecoderBlockWeightsF32 & current = weights.blocks[block_index];
        const SparseTensorF32 * guide = guide_subdivisions
            ? &(*guide_subdivisions)[level] : nullptr;
        SparseTensorF32 predicted;
        if (config.pred_subdiv &&
            !run_labeled_sparse_linear(hidden, current.to_subdiv_weight,
                                       current.to_subdiv_bias, 8, predicted,
                                       "decode.level=" + std::to_string(level) +
                                           ".transition.to_subdiv", error)) {
            return false;
        }
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
        gpu.invalidate_neighbor_map();
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
        set_error(error, "SLat decoder GPU-first did not consume all block weights");
        return false;
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
    if (slat_decoder_gpu_first_enabled()) {
        return decode_gpu_first(input, guide_subdivisions, output,
                                predicted_subdivisions, error);
    }
    SLatDecoderInvocation invocation(
        gpu, slat_decoder_profile_enabled(),
        slat_decoder_neighbor_cache_enabled(),
        slat_decoder_neighbor_cache_validation_enabled());
    SLatDecoderWeightsF32 weights;
    if (!make_weights(weights, error)) return false;
    SLatDecoderConfig config;
    config.latent_channels = hp.latent_channels;
    config.out_channels = hp.out_channels;
    config.norm_eps = hp.norm_eps;
    config.pred_subdiv = hp.pred_subdiv;
    config.model_channels = hp.model_channels;
    config.num_blocks = hp.num_blocks;
    const double input_validation_begin = gpu.profile ? backend_time_now_ms() : 0.0;
    const bool input_valid = validate_decoder_sparse_tensor(
        input, "decoder.decode.input", error);
    decoder_profile_add_phase(gpu.profile, &SLatDecoderProfile::input_validation_ms,
                              input_validation_begin);
    if (!input_valid || input.channels != config.latent_channels ||
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
    if (!run_labeled_sparse_linear(input, weights.from_latent_weight,
                                   weights.from_latent_bias,
                                   config.model_channels.front(), hidden,
                                   "decode.from_latent", error)) return false;
    if (predicted_subdivisions) predicted_subdivisions->clear();
    std::size_t block_index = 0;
    for (std::size_t level = 0; level < config.model_channels.size(); ++level) {
        for (int block = 0; block < config.num_blocks[level]; ++block) {
            const SLatDecoderBlockWeightsF32 & current = weights.blocks[block_index];
            const std::string block_label = "decode.level=" + std::to_string(level) +
                                            ".block=" + std::to_string(block);
            SparseTensorF32 convolved;
            if (!gpu.run_conv(hidden, current.conv1_weight, current.conv1_bias,
                              hidden.channels, convolved, error)) return false;
            SparseTensorF32 normalized;
            if (!sparse_layer_norm(convolved, config.norm_eps, current.norm_weight,
                                   current.norm_bias, normalized, error)) return false;
            SparseTensorF32 mlp_hidden;
            if (!run_labeled_sparse_linear(normalized, current.mlp0_weight,
                                           current.mlp0_bias, current.mlp_hidden, mlp_hidden,
                                           block_label + ".mlp0", error)) return false;
            silu_in_place(mlp_hidden);
            SparseTensorF32 result;
            if (!run_labeled_sparse_linear(mlp_hidden, current.mlp2_weight,
                                           current.mlp2_bias, hidden.channels, result,
                                           block_label + ".mlp2", error) ||
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
            !run_labeled_sparse_linear(hidden, current.to_subdiv_weight,
                                       current.to_subdiv_bias, 8, predicted,
                                       "decode.level=" + std::to_string(level) +
                                           ".transition.to_subdiv", error)) return false;
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
        gpu.invalidate_neighbor_map();
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
    if (!sparse_layer_norm(hidden, k_slat_decoder_final_layer_norm_eps,
                           nullptr, nullptr,
                           normalized, error) ||
        !run_labeled_sparse_linear(normalized, weights.output_weight, weights.output_bias,
                                   config.out_channels, output, "decode.output", error)) return false;
    return true;
}

bool SLatDecoderModel::Impl::upsample_coords_gpu_first(
    const SparseTensorF32 & input,
    int upsample_times,
    SparseTensorF32 & output,
    std::string * error) {
    SLatDecoderInvocation invocation(
        gpu, slat_decoder_profile_enabled(),
        slat_decoder_neighbor_cache_enabled(),
        slat_decoder_neighbor_cache_validation_enabled());
    SLatDecoderWeightsF32 weights;
    if (!make_weights(weights, error)) return false;

    SLatDecoderConfig config;
    config.latent_channels = hp.latent_channels;
    config.out_channels = hp.out_channels;
    config.norm_eps = hp.norm_eps;
    config.pred_subdiv = hp.pred_subdiv;
    config.model_channels = hp.model_channels;
    config.num_blocks = hp.num_blocks;
    if (!config.pred_subdiv || upsample_times < 0 ||
        upsample_times >= static_cast<int>(config.model_channels.size())) {
        set_error(error, "invalid SLat decoder GPU-first coordinate-upsample request");
        return false;
    }

    SparseTensorF32 hidden;
    std::size_t block_index = 0;
    for (std::size_t level = 0; level < config.model_channels.size(); ++level) {
        const std::size_t block_count = static_cast<std::size_t>(config.num_blocks[level]);
        const SLatDecoderBlockWeightsF32 * level_blocks = block_count > 0
            ? weights.blocks.data() + block_index : nullptr;
        SparseTensorF32 level_output;
        if (!gpu.run_convnext_level(
                level == 0 ? input : hidden,
                level_blocks, block_count, config.norm_eps,
                level == 0 ? weights.from_latent_weight : nullptr,
                level == 0 ? weights.from_latent_bias : nullptr,
                level == 0 ? config.model_channels.front() : 0,
                false, nullptr, nullptr, 0,
                "upsample.level=" + std::to_string(level), level_output, error)) {
            return false;
        }
        block_index += block_count;
        hidden = std::move(level_output);
        if (static_cast<int>(level) == upsample_times) {
            output = std::move(hidden);
            return true;
        }
        if (level + 1 >= config.model_channels.size() ||
            block_index >= weights.blocks.size() ||
            weights.blocks[block_index].kind != SLatDecoderBlockKind::channel_to_spatial) {
            set_error(error, "SLat decoder GPU-first coordinate upsample expected "
                              "channel-to-spatial block");
            return false;
        }
        const SLatDecoderBlockWeightsF32 & current = weights.blocks[block_index];
        SparseTensorF32 predicted;
        if (!run_labeled_sparse_linear(hidden, current.to_subdiv_weight,
                                       current.to_subdiv_bias, 8, predicted,
                                       "upsample.level=" + std::to_string(level) +
                                           ".transition.to_subdiv", error)) {
            return false;
        }
        SparseTensorF32 normalized;
        if (!sparse_layer_norm(hidden, config.norm_eps, current.norm1_weight,
                               current.norm1_bias, normalized, error)) return false;
        silu_in_place(normalized);
        SparseTensorF32 packed;
        if (!gpu.run_conv(normalized, current.conv1_weight, current.conv1_bias,
                          current.out_channels * 8, packed, error)) return false;
        SparseTensorF32 hidden_spatial;
        if (!sparse_channel_to_spatial(packed, 2, &predicted, hidden_spatial, error)) {
            return false;
        }
        SparseTensorF32 skip_spatial;
        if (!sparse_channel_to_spatial(hidden, 2, &predicted, skip_spatial, error)) {
            return false;
        }
        gpu.invalidate_neighbor_map();
        SparseTensorF32 skip;
        if (!repeat_channels(skip_spatial, current.out_channels, skip, error)) return false;
        if (!sparse_layer_norm(hidden_spatial, config.norm_eps, nullptr, nullptr,
                               normalized, error)) return false;
        silu_in_place(normalized);
        if (!gpu.run_conv(normalized, current.conv2_weight, current.conv2_bias,
                          current.out_channels, hidden, error) ||
            !add_in_place(hidden, skip, error)) return false;
        ++block_index;
    }
    set_error(error, "SLat decoder GPU-first coordinate upsample did not reach target level");
    return false;
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
    if (slat_decoder_gpu_first_enabled()) {
        return upsample_coords_gpu_first(input, upsample_times, output, error);
    }
    SLatDecoderInvocation invocation(
        gpu, slat_decoder_profile_enabled(),
        slat_decoder_neighbor_cache_enabled(),
        slat_decoder_neighbor_cache_validation_enabled());
    SLatDecoderWeightsF32 weights;
    if (!make_weights(weights, error)) return false;
    SLatDecoderConfig config;
    config.latent_channels = hp.latent_channels;
    config.out_channels = hp.out_channels;
    config.norm_eps = hp.norm_eps;
    config.pred_subdiv = hp.pred_subdiv;
    config.model_channels = hp.model_channels;
    config.num_blocks = hp.num_blocks;
    const double input_validation_begin = gpu.profile ? backend_time_now_ms() : 0.0;
    const bool input_valid = validate_decoder_sparse_tensor(
        input, "decoder.upsample.input", error);
    decoder_profile_add_phase(gpu.profile, &SLatDecoderProfile::input_validation_ms,
                              input_validation_begin);
    if (!config.pred_subdiv || !input_valid || input.channels != config.latent_channels) {
        set_error(error, "invalid SLat decoder GPU coordinate-upsample input");
        return false;
    }
    SparseTensorF32 hidden;
    if (!run_labeled_sparse_linear(input, weights.from_latent_weight,
                                   weights.from_latent_bias,
                                   config.model_channels.front(), hidden,
                                   "upsample.from_latent", error)) return false;
    std::size_t block_index = 0;
    for (std::size_t level = 0; level < config.model_channels.size(); ++level) {
        if (static_cast<int>(level) == upsample_times) {
            output = std::move(hidden);
            return true;
        }
        for (int block = 0; block < config.num_blocks[level]; ++block) {
            const SLatDecoderBlockWeightsF32 & current = weights.blocks[block_index++];
            const std::string block_label = "upsample.level=" + std::to_string(level) +
                                            ".block=" + std::to_string(block);
            SparseTensorF32 convolved;
            if (!gpu.run_conv(hidden, current.conv1_weight, current.conv1_bias,
                              hidden.channels, convolved, error)) return false;
            SparseTensorF32 normalized;
            if (!sparse_layer_norm(convolved, config.norm_eps, current.norm_weight,
                                   current.norm_bias, normalized, error)) return false;
            SparseTensorF32 mlp_hidden;
            if (!run_labeled_sparse_linear(normalized, current.mlp0_weight,
                                           current.mlp0_bias, current.mlp_hidden, mlp_hidden,
                                           block_label + ".mlp0", error)) return false;
            silu_in_place(mlp_hidden);
            SparseTensorF32 result;
            if (!run_labeled_sparse_linear(mlp_hidden, current.mlp2_weight,
                                           current.mlp2_bias, hidden.channels, result,
                                           block_label + ".mlp2", error) ||
                !add_in_place(result, hidden, error)) return false;
            hidden = std::move(result);
        }
        if (level + 1 >= config.model_channels.size()) break;
        const SLatDecoderBlockWeightsF32 & current = weights.blocks[block_index++];
        SparseTensorF32 predicted;
        if (!run_labeled_sparse_linear(hidden, current.to_subdiv_weight,
                                       current.to_subdiv_bias, 8, predicted,
                                       "upsample.level=" + std::to_string(level) +
                                           ".transition.to_subdiv", error)) return false;
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
        gpu.invalidate_neighbor_map();
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
    const auto decode_start = std::chrono::steady_clock::now();
    const auto elapsed = [start = decode_start]() {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
    };
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
                std::cerr << "pixal3d: " << impl_->component
                          << " SLat GPU decode took " << elapsed() << " s"
                          << std::endl;
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
        // A silent fallback would turn an hour-long CPU decode into an
        // unexplained gap, so the reason is always reported.
        std::cerr << "pixal3d: " << impl_->component
                  << " SLat GPU decode failed; using CPU fallback"
                  << (gpu_error.empty() ? std::string{} : ": " + gpu_error)
                  << std::endl;
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
    if (verbose) {
        std::cerr << "pixal3d: " << impl_->component
                  << " SLat CPU reference decode starting (points="
                  << input.points() << ")" << std::endl;
    }
    const bool decoded = slat_decoder_forward_f32(input, config, weights,
                                                  guide_subdivisions,
                                                  output, predicted_subdivisions,
                                                  error);
    if (verbose) {
        std::cerr << "pixal3d: " << impl_->component
                  << " SLat CPU reference decode finished in " << elapsed()
                  << " s" << std::endl;
    }
    return decoded;
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
