#include "pixal3d/pipeline.h"
#include "pixal3d/mesh_topology.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <iostream>
#include <set>
#include <string>

namespace pixal3d {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

bool slat_decoder_verbose() {
    const char * value = std::getenv("PIXAL3D_SLAT_DECODER_VERBOSE");
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

std::uint64_t fnv1a_bytes(const void * data, std::size_t size) {
    const auto * bytes = static_cast<const std::uint8_t *>(data);
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::uint64_t sparse_coords_fingerprint(const std::vector<std::int32_t> & coords) {
    return fnv1a_bytes(coords.data(), coords.size() * sizeof(std::int32_t));
}

struct SubdivisionStats {
    std::size_t positive = 0;
    std::size_t negative = 0;
    std::size_t near_zero = 0;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
};

SubdivisionStats subdivision_stats(const SparseTensorF32 & subdivision) {
    SubdivisionStats stats;
    for (float value : subdivision.feats) {
        if (value > 0.0f) {
            ++stats.positive;
        } else {
            ++stats.negative;
        }
        if (std::fabs(value) <= 1.0e-4f) ++stats.near_zero;
        stats.minimum = std::min(stats.minimum, value);
        stats.maximum = std::max(stats.maximum, value);
    }
    return stats;
}

bool cascade_spatial_trace_enabled() {
    const char * value = std::getenv("PIXAL3D_CASCADE_SPATIAL_TRACE");
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

void log_spatial_buckets(const char * label,
                         const std::vector<std::int32_t> & coords,
                         int resolution) {
    if (!cascade_spatial_trace_enabled() || resolution <= 0 || coords.size() % 4 != 0) return;
    constexpr int kBuckets = 8;
    std::array<std::size_t, kBuckets> x{};
    std::array<std::size_t, kBuckets> y{};
    std::array<std::size_t, kBuckets> z{};
    for (std::size_t point = 0; point < coords.size() / 4; ++point) {
        for (int axis = 0; axis < 3; ++axis) {
            const int value = coords[point * 4 + 1 + static_cast<std::size_t>(axis)];
            const int bucket = std::max(0, std::min(kBuckets - 1,
                value * kBuckets / std::max(1, resolution)));
            (axis == 0 ? x : axis == 1 ? y : z)[static_cast<std::size_t>(bucket)]++;
        }
    }
    const auto print = [](const std::array<std::size_t, kBuckets> & values) {
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (index != 0) std::cerr << ',';
            std::cerr << values[index];
        }
    };
    std::cerr << "pixal3d: cascade_spatial label=" << label
              << " points=" << coords.size() / 4 << " resolution=" << resolution
              << " x=";
    print(x);
    std::cerr << " y=";
    print(y);
    std::cerr << " z=";
    print(z);
    std::cerr << std::endl;
}

void log_decoder_diagnostics(const char * label,
                             const SparseTensorF32 & input,
                             const std::vector<SparseTensorF32> & subdivisions,
                             const SparseTensorF32 & output) {
    if (!slat_decoder_verbose()) return;
    std::cerr << "pixal3d: " << label << " input_points=" << input.points()
              << " input_coords=0x" << std::hex << sparse_coords_fingerprint(input.coords)
              << std::dec << " subdivisions=" << subdivisions.size() << std::endl;
    for (std::size_t level = 0; level < subdivisions.size(); ++level) {
        const SparseTensorF32 & subdivision = subdivisions[level];
        const SubdivisionStats stats = subdivision_stats(subdivision);
        std::cerr << "pixal3d: " << label << " subdiv[" << level << "] points="
                  << subdivision.points() << " positive=" << stats.positive
                  << " negative=" << stats.negative << " near_zero=" << stats.near_zero
                  << " min=" << stats.minimum << " max=" << stats.maximum
                  << " coords=0x" << std::hex << sparse_coords_fingerprint(subdivision.coords)
                  << std::dec << std::endl;
    }
    std::cerr << "pixal3d: " << label << " output_points=" << output.points()
              << " output_coords=0x" << std::hex << sparse_coords_fingerprint(output.coords)
              << std::dec << std::endl;
}

bool write_sparse_dump(const std::filesystem::path & directory,
                       const std::string & name,
                       const SparseTensorF32 & tensor,
                       std::string * error) {
    std::error_code filesystem_error;
    std::filesystem::create_directories(directory, filesystem_error);
    if (filesystem_error) {
        set_error(error, "failed to create cascade dump directory: " + filesystem_error.message());
        return false;
    }
    std::ofstream stream(directory / (name + ".bin"), std::ios::binary | std::ios::trunc);
    if (!stream) {
        set_error(error, "failed to open cascade dump: " + (directory / (name + ".bin")).string());
        return false;
    }
    const std::uint32_t version = 1;
    const std::int32_t shape[5] = {
        static_cast<std::int32_t>(tensor.batch_size),
        static_cast<std::int32_t>(tensor.channels),
        static_cast<std::int32_t>(tensor.spatial_x),
        static_cast<std::int32_t>(tensor.spatial_y),
        static_cast<std::int32_t>(tensor.spatial_z),
    };
    const std::uint64_t points = tensor.points();
    const std::uint64_t feature_count = tensor.feats.size();
    stream.write(reinterpret_cast<const char *>(&version), sizeof(version));
    stream.write(reinterpret_cast<const char *>(shape), sizeof(shape));
    stream.write(reinterpret_cast<const char *>(&points), sizeof(points));
    stream.write(reinterpret_cast<const char *>(&feature_count), sizeof(feature_count));
    stream.write(reinterpret_cast<const char *>(tensor.coords.data()),
                 static_cast<std::streamsize>(tensor.coords.size() * sizeof(std::int32_t)));
    stream.write(reinterpret_cast<const char *>(tensor.feats.data()),
                 static_cast<std::streamsize>(tensor.feats.size() * sizeof(float)));
    if (!stream) {
        set_error(error, "failed while writing cascade dump: " + (directory / (name + ".bin")).string());
        return false;
    }
    return true;
}

bool write_i32_dump(const std::filesystem::path & directory,
                    const std::string & name,
                    const std::vector<std::int32_t> & values,
                    std::string * error) {
    std::error_code filesystem_error;
    std::filesystem::create_directories(directory, filesystem_error);
    if (filesystem_error) {
        set_error(error, "failed to create cascade dump directory: " + filesystem_error.message());
        return false;
    }
    std::ofstream stream(directory / (name + ".i32"), std::ios::binary | std::ios::trunc);
    if (!stream) {
        set_error(error, "failed to open cascade coordinate dump: " +
                          (directory / (name + ".i32")).string());
        return false;
    }
    const std::uint32_t version = 1;
    const std::uint64_t count = values.size();
    stream.write(reinterpret_cast<const char *>(&version), sizeof(version));
    stream.write(reinterpret_cast<const char *>(&count), sizeof(count));
    stream.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(std::int32_t)));
    if (!stream) {
        set_error(error, "failed while writing cascade coordinate dump: " +
                          (directory / (name + ".i32")).string());
        return false;
    }
    return true;
}

bool write_condition_f32_dump(const std::filesystem::path & directory,
                              const std::string & name,
                              const std::vector<float> & values,
                              std::string * error) {
    std::error_code filesystem_error;
    std::filesystem::create_directories(directory, filesystem_error);
    if (filesystem_error) {
        set_error(error, "failed to create condition dump directory: " +
                          filesystem_error.message());
        return false;
    }
    std::ofstream stream(directory / (name + ".f32"),
                        std::ios::binary | std::ios::trunc);
    if (!stream) {
        set_error(error, "failed to open condition dump: " +
                          (directory / (name + ".f32")).string());
        return false;
    }
    stream.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!stream) {
        set_error(error, "failed while writing condition dump: " +
                          (directory / (name + ".f32")).string());
        return false;
    }
    return true;
}

bool dump_ss_condition(const Pixal3DImageConditionF32 & condition,
                       std::string * error) {
    const char * directory = std::getenv("PIXAL3D_CONDITION_DUMP_DIR");
    if (!directory || directory[0] == '\0') return true;
    const std::filesystem::path root(directory);
    return write_condition_f32_dump(root, "ss_global", condition.global.feats, error) &&
           write_condition_f32_dump(root, "ss_projection", condition.projection.feats, error);
}

bool dump_cascade_ss_coords(const Pixal3DCascadeOutputF32 & output,
                            std::string * error) {
    const char * directory = std::getenv("PIXAL3D_CASCADE_DUMP_DIR");
    if (!directory || directory[0] == '\0') return true;
    const std::filesystem::path root(directory);
    return write_sparse_dump(root, "ss_flow", output.sparse_structure.flow.samples, error) &&
           write_i32_dump(root, "ss_coords", output.sparse_structure.coords, error);
}

bool dump_cascade_shape_tensors(const Pixal3DCascadeOutputF32 & output,
                                std::string * error) {
    const char * directory = std::getenv("PIXAL3D_CASCADE_DUMP_DIR");
    if (!directory || directory[0] == '\0') return true;
    const std::filesystem::path root(directory);
    return write_sparse_dump(root, "shape_slat_low", output.shape_slat_low.latent, error) &&
           write_sparse_dump(root, "shape_upsampled", output.shape_upsampled, error) &&
           write_i32_dump(root, "high_coords", output.high_coords, error) &&
           write_sparse_dump(root, "shape_slat_high", output.shape_slat_high.latent, error);
}

bool dump_cascade_decoder_tensors(const Pixal3DCascadeOutputF32 & output,
                                  std::string * error) {
    const char * directory = std::getenv("PIXAL3D_CASCADE_DUMP_DIR");
    if (!directory || directory[0] == '\0') return true;
    const std::filesystem::path root(directory);
    if (!write_sparse_dump(root, "shape_decoded", output.shape_decoded, error)) return false;
    for (std::size_t level = 0; level < output.shape_subdivisions.size(); ++level) {
        if (!write_sparse_dump(root, "shape_subdiv_" + std::to_string(level),
                               output.shape_subdivisions[level], error)) return false;
    }
    return write_sparse_dump(root, "texture_slat", output.texture_slat.latent, error) &&
           write_sparse_dump(root, "texture_decoded", output.texture_decoded, error);
}

bool valid_normalization(const SLatNormalizationF32 & normalization,
                         int channels, std::string * error) {
    if (channels <= 0 || normalization.mean.size() != static_cast<std::size_t>(channels) ||
        normalization.std.size() != static_cast<std::size_t>(channels)) {
        set_error(error, "SLat normalization channel count does not match model output");
        return false;
    }
    for (int channel = 0; channel < channels; ++channel) {
        const float mean = normalization.mean[static_cast<std::size_t>(channel)];
        const float standard_deviation = normalization.std[static_cast<std::size_t>(channel)];
        if (!std::isfinite(mean) || !std::isfinite(standard_deviation) ||
            standard_deviation == 0.0f) {
            set_error(error, "SLat normalization contains a non-finite or zero standard deviation");
            return false;
        }
    }
    return true;
}

bool checked_grid_resolution(int resolution, int & grid, std::string * error) {
    if (resolution != 1024) {
        set_error(error, "SLat cascade resolution must be 1024");
        return false;
    }
    grid = resolution / 16;
    return grid > 0;
}

bool quantize_at_resolution(const SparseTensorF32 & input,
                            int low_resolution,
                            int resolution,
                            std::set<std::array<std::int32_t, 4>> & unique,
                            std::string * error) {
    int grid = 0;
    if (!checked_grid_resolution(resolution, grid, error)) return false;
    if (low_resolution <= 0 || !std::isfinite(static_cast<float>(low_resolution))) {
        set_error(error, "SLat cascade low resolution must be positive");
        return false;
    }
    unique.clear();
    for (std::size_t point = 0; point < input.points(); ++point) {
        const std::size_t base = point * 4;
        const std::int32_t batch = input.coords[base + 0];
        std::array<std::int32_t, 4> quantized{batch, 0, 0, 0};
        for (int axis = 0; axis < 3; ++axis) {
            const std::int32_t source = input.coords[base + 1 + static_cast<std::size_t>(axis)];
            const double mapped = (static_cast<double>(source) + 0.5) /
                                  static_cast<double>(low_resolution) *
                                  static_cast<double>(grid - 1);
            const long long rounded = std::llround(mapped);
            if (rounded < 0 || rounded >= grid ||
                rounded > std::numeric_limits<std::int32_t>::max()) {
                set_error(error, "SLat decoder coordinate is outside cascade quantization range");
                return false;
            }
            quantized[static_cast<std::size_t>(axis + 1)] =
                static_cast<std::int32_t>(rounded);
        }
        unique.insert(quantized);
    }
    return true;
}

} // namespace

bool normalize_slat_f32(
    const SparseTensorF32 & input,
    const SLatNormalizationF32 & normalization,
    SparseTensorF32 & output,
    std::string * error) {
    output = SparseTensorF32{};
    if (!input.valid(error) || !valid_normalization(normalization, input.channels, error)) {
        return false;
    }
    output = input;
    for (std::size_t point = 0; point < output.points(); ++point) {
        float * row = output.feats.data() + point * static_cast<std::size_t>(output.channels);
        for (int channel = 0; channel < output.channels; ++channel) {
            const std::size_t index = static_cast<std::size_t>(channel);
            const float value = (row[channel] - normalization.mean[index]) /
                                normalization.std[index];
            if (!std::isfinite(value)) {
                output = SparseTensorF32{};
                set_error(error, "SLat normalization produced a non-finite value");
                return false;
            }
            row[channel] = value;
        }
    }
    return true;
}

bool run_slat_stage_f32(
    SLatFlowModel & flow_model,
    const SparseTensorF32 & noise,
    const Pixal3DImageConditionF32 & condition,
    const FlowEulerSamplerConfig & sampler_config,
    const SLatNormalizationF32 & normalization,
    SLatStageOutputF32 & output,
    std::string * error,
    const SparseTensorF32 * concat_condition) {
    output = SLatStageOutputF32{};
    if (!flow_model.is_loaded() || !flow_model.has_data()) {
        set_error(error, "SLat flow model must be loaded with tensor data");
        return false;
    }
    if (!noise.valid(error) || !condition.global.valid(error) ||
        (concat_condition && !concat_condition->valid(error))) return false;
    const SLatFlowHParams & hp = flow_model.hparams();
    if (noise.batch_size != condition.global.batch_size ||
        condition.global.channels != hp.cond_channels ||
        condition.global.tokens() == 0) {
        set_error(error, "SLat stage noise or global condition does not match model metadata");
        return false;
    }
    if (concat_condition) {
        if (concat_condition->batch_size != noise.batch_size ||
            concat_condition->spatial_x != noise.spatial_x ||
            concat_condition->spatial_y != noise.spatial_y ||
            concat_condition->spatial_z != noise.spatial_z ||
            concat_condition->coords != noise.coords ||
            concat_condition->channels <= 0 ||
            noise.channels > std::numeric_limits<int>::max() - concat_condition->channels ||
            noise.channels + concat_condition->channels != hp.in_channels) {
            set_error(error, "SLat stage concat condition does not match model input metadata");
            return false;
        }
    } else if (noise.channels != hp.in_channels) {
        set_error(error, "SLat stage noise channels do not match model input metadata");
        return false;
    }
    const SparseTensorF32 * projection = nullptr;
    if (condition.has_projection()) {
        if (!condition.projection.valid(error) ||
            condition.projection.batch_size != noise.batch_size ||
            condition.projection.coords != noise.coords) {
            set_error(error, "SLat stage projection condition coordinates do not match noise");
            return false;
        }
        projection = &condition.projection;
    }
    if (hp.image_attn_mode == "proj" &&
        (!projection || projection->channels != hp.proj_in_channels)) {
        set_error(error, "SLat stage projection condition is required by model metadata");
        return false;
    }
    if (!valid_normalization(normalization, hp.out_channels, error)) return false;
    const auto sampling_start = std::chrono::steady_clock::now();
    if (!flow_model.sample(noise, sampler_config, condition.global, projection,
                           output.flow, error, concat_condition)) {
        return false;
    }
    std::cerr << "pixal3d: SLat flow sampling (" << noise.points()
              << " points) took "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - sampling_start).count()
              << " s" << std::endl;
    output.latent = output.flow.samples;
    for (std::size_t point = 0; point < output.latent.points(); ++point) {
        float * row = output.latent.feats.data() +
                      point * static_cast<std::size_t>(output.latent.channels);
        for (int channel = 0; channel < output.latent.channels; ++channel) {
            row[channel] = row[channel] * normalization.std[static_cast<std::size_t>(channel)] +
                           normalization.mean[static_cast<std::size_t>(channel)];
            if (!std::isfinite(row[channel])) {
                output = SLatStageOutputF32{};
                set_error(error, "SLat stage denormalization produced a non-finite value");
                return false;
            }
        }
    }
    return true;
}

bool quantize_slat_coords_f32(
    const SparseTensorF32 & upsampled_coords,
    int low_resolution,
    int requested_resolution,
    std::size_t max_num_tokens,
    std::vector<std::int32_t> & coords,
    int & actual_resolution,
    std::string * error) {
    coords.clear();
    actual_resolution = 0;
    if (!upsampled_coords.valid(error) || low_resolution <= 0) {
        set_error(error, "invalid SLat cascade coordinate quantization input");
        return false;
    }
    if (requested_resolution != 1024) {
        set_error(error, "SLat cascade resolution must be 1024");
        return false;
    }
    std::set<std::array<std::int32_t, 4>> unique;
    if (!quantize_at_resolution(upsampled_coords, low_resolution, 1024,
                                unique, error)) return false;
    if (max_num_tokens != 0 && unique.size() > max_num_tokens) {
        set_error(error, "quantized SLat coordinate count exceeds max_num_tokens at resolution 1024");
        return false;
    }
    actual_resolution = 1024;
    coords.reserve(unique.size() * 4);
    for (const auto & coord : unique) {
        coords.insert(coords.end(), coord.begin(), coord.end());
    }
    return true;
}

bool full_grid_coords(int resolution, std::vector<std::int32_t> & coords,
                      std::string * error) {
    coords.clear();
    if (resolution <= 0) {
        set_error(error, "cascade grid resolution must be positive");
        return false;
    }
    const std::size_t r = static_cast<std::size_t>(resolution);
    if (r > std::numeric_limits<std::size_t>::max() / r ||
        r * r > std::numeric_limits<std::size_t>::max() / r) {
        set_error(error, "cascade grid resolution^3 overflows size_t");
        return false;
    }
    const std::size_t points = r * r * r;
    if (points > std::numeric_limits<std::size_t>::max() / 4) {
        set_error(error, "cascade coordinate buffer overflows size_t");
        return false;
    }
    coords.reserve(points * 4);
    for (int x = 0; x < resolution; ++x) {
        for (int y = 0; y < resolution; ++y) {
            for (int z = 0; z < resolution; ++z) {
                coords.insert(coords.end(), {0, static_cast<std::int32_t>(x),
                                             static_cast<std::int32_t>(y),
                                             static_cast<std::int32_t>(z)});
            }
        }
    }
    return true;
}

bool build_noise(const Pixal3DNoiseBuilderF32 & builder,
                 Pixal3DCascadeStage stage,
                 const std::vector<std::int32_t> & coords,
                 int channels,
                 int grid_resolution,
                 SparseTensorF32 & output,
                 std::string * error) {
    output = SparseTensorF32{};
    if (!builder) {
        set_error(error, "Pixal3D cascade noise builder callback is empty");
        return false;
    }
    if (!builder(stage, coords, channels, grid_resolution, output, error)) return false;
    if (!output.valid(error) || output.batch_size != 1 || output.channels != channels ||
        output.spatial_x != grid_resolution || output.spatial_y != grid_resolution ||
        output.spatial_z != grid_resolution || output.coords != coords ||
        output.points() == 0) {
        set_error(error, "cascade noise builder returned a tensor with the wrong shape");
        return false;
    }
    return true;
}

bool build_condition(const Pixal3DConditionBuilderF32 & builder,
                     Pixal3DCascadeStage stage,
                     const std::vector<std::int32_t> & coords,
                     int grid_resolution,
                     Pixal3DImageConditionF32 & output,
                     std::string * error) {
    output = Pixal3DImageConditionF32{};
    if (!builder) {
        set_error(error, "Pixal3D cascade condition builder callback is empty");
        return false;
    }
    if (!builder(stage, coords, grid_resolution, output, error) ||
        !output.global.valid(error) || output.global.batch_size != 1 ||
        output.global.tokens() == 0) {
        if (!error || error->empty()) {
            set_error(error, "cascade condition builder returned an invalid global condition");
        }
        return false;
    }
    if (output.has_projection()) {
        if (!output.projection.valid(error) || output.projection.batch_size != 1 ||
            output.projection.coords != coords ||
            output.projection.spatial_x != grid_resolution ||
            output.projection.spatial_y != grid_resolution ||
            output.projection.spatial_z != grid_resolution) {
            set_error(error, "cascade condition projection does not match requested coordinates");
            return false;
        }
    }
    return true;
}

bool model_ready(const SLatFlowModel & model, const char * name, std::string * error) {
    if (model.is_loaded() && model.has_data()) return true;
    set_error(error, std::string(name) + " must be loaded with tensor data");
    return false;
}

bool model_ready(const SLatDecoderModel & model, const char * name, std::string * error) {
    if (model.is_loaded() && model.has_data()) return true;
    set_error(error, std::string(name) + " must be loaded with tensor data");
    return false;
}

bool model_ready(const SSFlowModel & model, const char * name, std::string * error) {
    if (model.is_loaded() && model.has_data()) return true;
    set_error(error, std::string(name) + " must be loaded with tensor data");
    return false;
}

bool model_ready(const SSDecoderModel & model, const char * name, std::string * error) {
    if (model.is_loaded() && model.has_data()) return true;
    set_error(error, std::string(name) + " must be loaded with tensor data");
    return false;
}

bool run_cascade_models_ready(
    SSFlowModel & ss_flow,
    SSDecoderModel & ss_decoder,
    SLatFlowModel & shape_flow_low,
    SLatFlowModel & shape_flow_high,
    SLatFlowModel & texture_flow,
    SLatDecoderModel & shape_decoder,
    SLatDecoderModel & texture_decoder,
    std::string * error) {
    return model_ready(ss_flow, "SS-flow model", error) &&
           model_ready(ss_decoder, "SS decoder model", error) &&
           model_ready(shape_flow_low, "low-resolution shape flow model", error) &&
           model_ready(shape_flow_high, "high-resolution shape flow model", error) &&
           model_ready(texture_flow, "texture flow model", error) &&
           model_ready(shape_decoder, "shape SLat decoder", error) &&
           model_ready(texture_decoder, "texture SLat decoder", error);
}

bool run_pixal3d_cascade_f32(
    SSFlowModel & ss_flow,
    SSDecoderModel & ss_decoder,
    SLatFlowModel & shape_flow_low,
    SLatFlowModel & shape_flow_high,
    SLatFlowModel & texture_flow,
    SLatDecoderModel & shape_decoder,
    SLatDecoderModel & texture_decoder,
    const Pixal3DCascadeConfig & config,
    const Pixal3DNoiseBuilderF32 & noise_builder,
    const Pixal3DConditionBuilderF32 & condition_builder,
    Pixal3DCascadeOutputF32 & output,
    std::string * error) {
    output = Pixal3DCascadeOutputF32{};
    if (!run_cascade_models_ready(ss_flow, ss_decoder, shape_flow_low,
                                  shape_flow_high, texture_flow, shape_decoder,
                                  texture_decoder, error)) return false;
    if (!std::isfinite(config.voxel_margin) || config.voxel_margin < 0.0f ||
        config.decoder_upsample_times < 0 || config.requested_resolution != 1024) {
        set_error(error, "invalid Pixal3D cascade configuration: final resolution must be 1024");
        return false;
    }
    const SSFlowHParams & ss_hp = ss_flow.hparams();
    const SSDecoderHParams & ss_dec_hp = ss_decoder.hparams();
    const SLatFlowHParams & shape_low_hp = shape_flow_low.hparams();
    const SLatFlowHParams & shape_high_hp = shape_flow_high.hparams();
    const SLatFlowHParams & texture_hp = texture_flow.hparams();
    const SLatDecoderHParams & shape_dec_hp = shape_decoder.hparams();
    const SLatDecoderHParams & texture_dec_hp = texture_decoder.hparams();
    if (ss_hp.resolution != ss_dec_hp.resolution ||
        ss_hp.out_channels != ss_dec_hp.latent_channels ||
        shape_dec_hp.component != "shape_decoder" || !shape_dec_hp.pred_subdiv ||
        texture_dec_hp.component != "texture_decoder" || texture_dec_hp.pred_subdiv ||
        shape_low_hp.out_channels != shape_dec_hp.latent_channels ||
        shape_high_hp.out_channels != shape_dec_hp.latent_channels ||
        texture_hp.out_channels != texture_dec_hp.latent_channels ||
        texture_dec_hp.out_channels != 6) {
        set_error(error, "cascade model metadata is not compatible across stages or texture decoder does not expose six PBR channels");
        return false;
    }
    if (config.sparse_structure.target_resolution > 0 &&
        config.sparse_structure.target_resolution != shape_low_hp.resolution) {
        set_error(error, "sparse structure target resolution must equal low shape grid resolution");
        return false;
    }

    std::vector<std::int32_t> ss_coords;
    if (!full_grid_coords(ss_hp.resolution, ss_coords, error)) return false;
    SparseTensorF32 ss_noise;
    if (!build_noise(noise_builder, Pixal3DCascadeStage::sparse_structure,
                     ss_coords, ss_hp.in_channels, ss_hp.resolution, ss_noise, error)) {
        return false;
    }
    Pixal3DImageConditionF32 ss_condition;
    if (!build_condition(condition_builder, Pixal3DCascadeStage::sparse_structure,
                         ss_coords, ss_hp.resolution, ss_condition, error)) return false;
    if (!dump_ss_condition(ss_condition, error)) return false;
    SparseStructureStageConfig structure_config = config.sparse_structure;
    if (structure_config.target_resolution <= 0) {
        structure_config.target_resolution = shape_low_hp.resolution;
    }
    if (!run_sparse_structure_stage_f32(ss_flow, ss_decoder, ss_noise, ss_condition,
                                        structure_config, output.sparse_structure, error)) {
        return false;
    }
    if (output.sparse_structure.coords.empty()) {
        set_error(error, "sparse structure stage produced no active coordinates");
        return false;
    }
    if (!dump_cascade_ss_coords(output, error)) return false;
    log_spatial_buckets("ss_coords", output.sparse_structure.coords,
                        structure_config.target_resolution);
    if (config.max_structure_points > 0 &&
        output.sparse_structure.coords.size() / 4 > config.max_structure_points) {
        if (config.max_structure_points >
            std::numeric_limits<std::size_t>::max() / 4) {
            set_error(error, "max_structure_points overflows coordinate count");
            return false;
        }
        const std::size_t point_count = output.sparse_structure.coords.size() / 4;
        if (config.max_structure_points == 1) {
            // A lexicographic prefix is almost always a corner of the volume
            // and makes tiny diagnostic runs look empty after shape-decoder
            // subdivision.  Keep the representative point nearest the center
            // instead; production runs leave this cap disabled.
            int max_x = 0;
            int max_y = 0;
            int max_z = 0;
            for (std::size_t point = 0; point < point_count; ++point) {
                max_x = std::max(max_x, output.sparse_structure.coords[point * 4 + 1]);
                max_y = std::max(max_y, output.sparse_structure.coords[point * 4 + 2]);
                max_z = std::max(max_z, output.sparse_structure.coords[point * 4 + 3]);
            }
            const int center_x = max_x / 2;
            const int center_y = max_y / 2;
            const int center_z = max_z / 2;
            std::size_t best = 0;
            std::int64_t best_distance = std::numeric_limits<std::int64_t>::max();
            for (std::size_t point = 0; point < point_count; ++point) {
                const std::int64_t dx = output.sparse_structure.coords[point * 4 + 1] - center_x;
                const std::int64_t dy = output.sparse_structure.coords[point * 4 + 2] - center_y;
                const std::int64_t dz = output.sparse_structure.coords[point * 4 + 3] - center_z;
                const std::int64_t distance = dx * dx + dy * dy + dz * dz;
                if (distance < best_distance) {
                    best_distance = distance;
                    best = point;
                }
            }
            const std::array<std::int32_t, 4> selected = {
                output.sparse_structure.coords[best * 4 + 0],
                output.sparse_structure.coords[best * 4 + 1],
                output.sparse_structure.coords[best * 4 + 2],
                output.sparse_structure.coords[best * 4 + 3],
            };
            output.sparse_structure.coords.assign(selected.begin(), selected.end());
        } else {
            output.sparse_structure.coords.resize(config.max_structure_points * 4);
        }
    }

    SparseTensorF32 shape_low_noise;
    if (!build_noise(noise_builder, Pixal3DCascadeStage::shape_slat_low,
                     output.sparse_structure.coords, shape_low_hp.in_channels,
                     shape_low_hp.resolution, shape_low_noise, error)) return false;
    Pixal3DImageConditionF32 shape_low_condition;
    if (!build_condition(condition_builder, Pixal3DCascadeStage::shape_slat_low,
                         output.sparse_structure.coords, shape_low_hp.resolution,
                         shape_low_condition, error)) return false;
    std::cerr << "pixal3d: shape_512 SLat flow points="
              << output.sparse_structure.coords.size() / 4 << std::endl;
    if (!run_slat_stage_f32(shape_flow_low, shape_low_noise, shape_low_condition,
                            config.shape_sampler, config.shape_normalization,
                            output.shape_slat_low, error)) return false;
    if (!shape_decoder.upsample_coords(output.shape_slat_low.latent,
                                       config.decoder_upsample_times,
                                       output.shape_upsampled, error)) return false;
    if (slat_decoder_verbose()) {
        std::cerr << "pixal3d: shape_decoder upsample_times="
                  << config.decoder_upsample_times
                  << " points=" << output.shape_upsampled.points()
                  << " coords=0x" << std::hex
                  << sparse_coords_fingerprint(output.shape_upsampled.coords)
                  << std::dec << std::endl;
    }
    log_spatial_buckets("shape_upsampled", output.shape_upsampled.coords,
                        output.shape_upsampled.spatial_x);
    if (!quantize_slat_coords_f32(
            output.shape_upsampled, output.shape_upsampled.spatial_x,
            config.requested_resolution, config.max_num_tokens,
            output.high_coords, output.resolution, error)) return false;
    if (output.high_coords.empty()) {
        set_error(error, "shape decoder upsample produced no high-resolution coordinates");
        return false;
    }
    if (slat_decoder_verbose()) {
        std::cerr << "pixal3d: shape_high quantized_points="
                  << output.high_coords.size() / 4
                  << " resolution=" << output.resolution
                  << " coords=0x" << std::hex
                  << sparse_coords_fingerprint(output.high_coords)
                  << std::dec << std::endl;
    }
    log_spatial_buckets("high_coords", output.high_coords, output.resolution / 16);
    const int high_grid_resolution = output.resolution / 16;
    SparseTensorF32 shape_high_noise;
    if (!build_noise(noise_builder, Pixal3DCascadeStage::shape_slat_high,
                     output.high_coords, shape_high_hp.in_channels,
                     high_grid_resolution, shape_high_noise, error)) return false;
    Pixal3DImageConditionF32 shape_high_condition;
    if (!build_condition(condition_builder, Pixal3DCascadeStage::shape_slat_high,
                         output.high_coords, high_grid_resolution,
                         shape_high_condition, error)) return false;
    std::cerr << "pixal3d: shape_1024 SLat flow points="
              << output.high_coords.size() / 4 << std::endl;
    if (!run_slat_stage_f32(shape_flow_high, shape_high_noise, shape_high_condition,
                            config.shape_sampler, config.shape_normalization,
                            output.shape_slat_high, error)) return false;
    if (!dump_cascade_shape_tensors(output, error)) return false;

    const int texture_noise_channels = texture_hp.in_channels -
                                       output.shape_slat_high.latent.channels;
    if (texture_noise_channels <= 0) {
        set_error(error, "texture flow input has no channels left for fresh noise");
        return false;
    }
    SparseTensorF32 texture_noise;
    if (!build_noise(noise_builder, Pixal3DCascadeStage::texture_slat,
                     output.high_coords, texture_noise_channels,
                     high_grid_resolution, texture_noise, error)) return false;
    Pixal3DImageConditionF32 texture_condition;
    if (!build_condition(condition_builder, Pixal3DCascadeStage::texture_slat,
                         output.high_coords, high_grid_resolution,
                         texture_condition, error)) return false;
    std::cerr << "pixal3d: texture_1024 SLat flow points="
              << output.high_coords.size() / 4 << std::endl;
    SparseTensorF32 texture_shape_condition;
    if (!normalize_slat_f32(output.shape_slat_high.latent,
                            config.shape_normalization,
                            texture_shape_condition, error)) {
        return false;
    }
    if (!run_slat_stage_f32(texture_flow, texture_noise, texture_condition,
                            config.texture_sampler, config.texture_normalization,
                            output.texture_slat, error,
                            &texture_shape_condition)) return false;

    if (!shape_decoder.decode(output.shape_slat_high.latent, nullptr,
                              output.shape_decoded, &output.shape_subdivisions, error)) {
        return false;
    }
    log_decoder_diagnostics("shape_decoder", output.shape_slat_high.latent,
                           output.shape_subdivisions, output.shape_decoded);
    log_spatial_buckets("shape_decoded", output.shape_decoded.coords,
                        output.shape_decoded.spatial_x);
    if (!texture_decoder.decode(output.texture_slat.latent, &output.shape_subdivisions,
                                output.texture_decoded, nullptr, error)) return false;
    log_decoder_diagnostics("texture_decoder", output.texture_slat.latent,
                           std::vector<SparseTensorF32>{}, output.texture_decoded);
    // Pixal3DImageTo3DPipeline.decode_tex_slat() maps decoder channels from
    // the trained [-1, 1] range to material/voxel attributes in [0, 1].
    for (float & value : output.texture_decoded.feats) {
        value = value * 0.5f + 0.5f;
        if (!std::isfinite(value)) {
            output = Pixal3DCascadeOutputF32{};
            set_error(error, "texture decoder rescaling produced a non-finite value");
            return false;
        }
    }
    if (!dump_cascade_decoder_tensors(output, error)) return false;
    if (!flexi_dual_grid_decode_mesh_f32(
            output.shape_decoded, output.resolution, config.voxel_margin,
            output.meshes, error)) return false;
    for (DualGridMeshF32 & mesh : output.meshes) {
        std::string hole_error;
        if (!fill_mesh_holes_f32(mesh, 3.0e-2f, &hole_error)) {
            set_error(error, "mesh hole filling failed: " + hole_error);
            return false;
        }
    }
    return true;
}

bool run_sparse_structure_stage_f32(
    SSFlowModel & flow_model,
    SSDecoderModel & decoder_model,
    const SparseTensorF32 & noise,
    const Pixal3DImageConditionF32 & condition,
    const SparseStructureStageConfig & config,
    SparseStructureStageOutputF32 & output,
    std::string * error) {
    output = SparseStructureStageOutputF32{};
    if (!flow_model.is_loaded() || !flow_model.has_data()) {
        set_error(error, "SS-flow model must be loaded with tensor data");
        return false;
    }
    if (!decoder_model.is_loaded() || !decoder_model.has_data()) {
        set_error(error, "SS decoder model must be loaded with tensor data");
        return false;
    }
    const SSFlowHParams & flow_hp = flow_model.hparams();
    const SSDecoderHParams & decoder_hp = decoder_model.hparams();
    if (flow_hp.resolution != decoder_hp.resolution ||
        flow_hp.out_channels != decoder_hp.latent_channels) {
        set_error(error, "SS-flow output and decoder latent metadata do not match");
        return false;
    }
    if (!condition.global.valid(error) || condition.global.batch_size != 1 ||
        condition.global.channels != flow_hp.cond_channels ||
        condition.global.tokens() == 0) {
        set_error(error, "SS stage global condition does not match flow metadata");
        return false;
    }
    const SparseTensorF32 * projection = nullptr;
    if (flow_hp.image_attn_mode == "proj") {
        if (!condition.has_projection()) {
            set_error(error, "SS stage projection condition is required");
            return false;
        }
        projection = &condition.projection;
    }
    if (!flow_model.sample(noise, config.sampler, condition.global,
                           projection, output.flow, error)) {
        return false;
    }
    if (!decoder_model.decode_sparse_coords(output.flow.samples,
                                            config.occupancy_threshold,
                                            config.target_resolution,
                                            output.coords, error)) {
        output = SparseStructureStageOutputF32{};
        return false;
    }
    return true;
}

} // namespace pixal3d
