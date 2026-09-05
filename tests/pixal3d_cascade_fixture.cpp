#include "pixal3d/pipeline.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void emit_f32(const char * name, const std::vector<float> & values) {
    std::cout << name << " " << values.size();
    for (float value : values) std::cout << " " << value;
    std::cout << "\n";
}

void emit_i32(const char * name, const std::vector<std::int32_t> & values) {
    std::cout << name << " " << values.size();
    for (std::int32_t value : values) std::cout << " " << value;
    std::cout << "\n";
}

float stage_base(pixal3d::Pixal3DCascadeStage stage) {
    switch (stage) {
    case pixal3d::Pixal3DCascadeStage::sparse_structure: return -0.31f;
    case pixal3d::Pixal3DCascadeStage::shape_slat_low: return -0.23f;
    case pixal3d::Pixal3DCascadeStage::shape_slat_high: return -0.17f;
    case pixal3d::Pixal3DCascadeStage::texture_slat: return -0.11f;
    }
    return 0.0f;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 8) {
        std::cerr << "usage: " << argv[0]
                  << " <ss-flow> <ss-decoder> <shape-flow-low> <shape-flow-high>"
                  << " <texture-flow> <shape-decoder> <texture-decoder>\n";
        return 2;
    }
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    pixal3d::SSFlowModel ss_flow;
    pixal3d::SSDecoderModel ss_decoder;
    pixal3d::SLatFlowModel shape_flow_low;
    pixal3d::SLatFlowModel shape_flow_high;
    pixal3d::SLatFlowModel texture_flow;
    pixal3d::SLatDecoderModel shape_decoder;
    pixal3d::SLatDecoderModel texture_decoder;
    std::string error;
    if (!ss_flow.load(argv[1], true, &error) ||
        !ss_decoder.load(argv[2], true, &error) ||
        !shape_flow_low.load(argv[3], "shape_flow_512", true, &error) ||
        !shape_flow_high.load(argv[4], "shape_flow_1024", true, &error) ||
        !texture_flow.load(argv[5], "texture_flow_1024", true, &error) ||
        !shape_decoder.load(argv[6], "shape_decoder", true, &error) ||
        !texture_decoder.load(argv[7], "texture_decoder", true, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    const pixal3d::Pixal3DNoiseBuilderF32 noise_builder =
        [](pixal3d::Pixal3DCascadeStage stage,
           const std::vector<std::int32_t> & coords,
           int channels,
           int grid_resolution,
           pixal3d::SparseTensorF32 & output,
           std::string * error) {
            if (channels <= 0 || grid_resolution <= 0 || coords.empty()) {
                if (error) *error = "invalid fixture noise request";
                return false;
            }
            output = pixal3d::SparseTensorF32{};
            output.batch_size = 1;
            output.channels = channels;
            output.spatial_x = output.spatial_y = output.spatial_z = grid_resolution;
            output.coords = coords;
            output.feats.resize(output.points() * static_cast<std::size_t>(channels));
            const float base = stage_base(stage);
            for (std::size_t index = 0; index < output.feats.size(); ++index) {
                output.feats[index] = base + 0.0017f * static_cast<float>(index);
            }
            return true;
        };

    const pixal3d::Pixal3DConditionBuilderF32 condition_builder =
        [](pixal3d::Pixal3DCascadeStage stage,
           const std::vector<std::int32_t> & coords,
           int grid_resolution,
           pixal3d::Pixal3DImageConditionF32 & output,
           std::string * error) {
            if (grid_resolution <= 0 || coords.empty()) {
                if (error) *error = "invalid fixture condition request";
                return false;
            }
            output = pixal3d::Pixal3DImageConditionF32{};
            output.global.batch_size = 1;
            output.global.channels = 5;
            output.global.offsets = {0, 3};
            output.global.feats.resize(15);
            const float base = stage_base(stage) * 0.5f;
            for (std::size_t index = 0; index < output.global.feats.size(); ++index) {
                output.global.feats[index] = base + 0.0023f * static_cast<float>(index);
            }
            output.projection.batch_size = 1;
            output.projection.channels = 3;
            output.projection.spatial_x = output.projection.spatial_y =
                output.projection.spatial_z = grid_resolution;
            output.projection.coords = coords;
            output.projection.feats.resize(output.projection.points() * 3);
            for (std::size_t index = 0; index < output.projection.feats.size(); ++index) {
                output.projection.feats[index] = base - 0.0011f * static_cast<float>(index);
            }
            return true;
        };

    pixal3d::Pixal3DCascadeConfig config;
    config.sparse_structure.sampler.steps = 1;
    config.sparse_structure.sampler.sigma_min = 0.031f;
    config.sparse_structure.sampler.rescale_t = 1.0f;
    config.sparse_structure.sampler.guidance_strength = 1.0f;
    config.sparse_structure.occupancy_threshold = -1.0e9f;
    config.sparse_structure.target_resolution = 4;
    config.shape_sampler.steps = 1;
    config.shape_sampler.sigma_min = 0.031f;
    config.shape_sampler.rescale_t = 1.0f;
    config.shape_sampler.guidance_strength = 1.0f;
    config.texture_sampler = config.shape_sampler;
    config.shape_normalization.mean = {0.11f, -0.07f, 0.23f};
    config.shape_normalization.std = {1.3f, 0.8f, 1.7f};
    config.texture_normalization.mean = {-0.13f, 0.19f, 0.03f};
    config.texture_normalization.std = {0.9f, 1.4f, 1.1f};
    config.requested_resolution = 1024;
    config.max_num_tokens = 0;
    config.decoder_upsample_times = 1;
    config.voxel_margin = 0.5f;

    pixal3d::Pixal3DCascadeOutputF32 output;
    if (!pixal3d::run_pixal3d_cascade_f32(
            ss_flow, ss_decoder, shape_flow_low, shape_flow_high, texture_flow,
            shape_decoder, texture_decoder, config, noise_builder,
            condition_builder, output, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    emit_i32("cascade_structure_coords", output.sparse_structure.coords);
    emit_f32("cascade_structure_sample", output.sparse_structure.flow.samples.feats);
    emit_f32("cascade_shape_low", output.shape_slat_low.latent.feats);
    emit_i32("cascade_upsampled_coords", output.shape_upsampled.coords);
    emit_i32("cascade_high_coords", output.high_coords);
    emit_f32("cascade_shape_high", output.shape_slat_high.latent.feats);
    emit_f32("cascade_texture", output.texture_slat.latent.feats);
    emit_f32("cascade_shape_decoded", output.shape_decoded.feats);
    emit_f32("cascade_texture_decoded", output.texture_decoded.feats);
    emit_f32("cascade_mesh_vertices", output.meshes.empty()
                                      ? std::vector<float>{}
                                      : output.meshes[0].vertices);
    emit_i32("cascade_mesh_faces", output.meshes.empty()
                                ? std::vector<std::int32_t>{}
                                : output.meshes[0].faces);
    std::cout << "cascade_resolution " << output.resolution << "\n";
    return 0;
}
