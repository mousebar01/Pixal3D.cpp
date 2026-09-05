#include "pixal3d/pipeline.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void emit(const char * name, const std::vector<float> & values) {
    std::cout << name << " " << values.size();
    for (float value : values) std::cout << " " << value;
    std::cout << "\n";
}

void emit_ints(const char * name, const std::vector<std::int32_t> & values) {
    std::cout << name << " " << values.size();
    for (std::int32_t value : values) std::cout << " " << value;
    std::cout << "\n";
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <ss-flow.gguf> <ss-decoder.gguf>\n";
        return 2;
    }
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);

    pixal3d::SSFlowModel flow;
    pixal3d::SSDecoderModel decoder;
    std::string error;
    if (!flow.load(argv[1], true, &error) || !decoder.load(argv[2], true, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    const auto & flow_hp = flow.hparams();
    const auto & decoder_hp = decoder.hparams();
    const std::size_t points = static_cast<std::size_t>(flow_hp.resolution) *
                               static_cast<std::size_t>(flow_hp.resolution) *
                               static_cast<std::size_t>(flow_hp.resolution);
    pixal3d::SparseTensorF32 noise;
    noise.batch_size = 1;
    noise.channels = flow_hp.in_channels;
    noise.spatial_x = noise.spatial_y = noise.spatial_z = flow_hp.resolution;
    noise.coords.reserve(points * 4);
    noise.feats.resize(points * static_cast<std::size_t>(noise.channels));
    for (std::size_t point = 0; point < points; ++point) {
        const std::int32_t x = static_cast<std::int32_t>(
            point / static_cast<std::size_t>(flow_hp.resolution * flow_hp.resolution));
        const std::int32_t y = static_cast<std::int32_t>(
            (point / static_cast<std::size_t>(flow_hp.resolution)) %
            static_cast<std::size_t>(flow_hp.resolution));
        const std::int32_t z = static_cast<std::int32_t>(
            point % static_cast<std::size_t>(flow_hp.resolution));
        noise.coords.insert(noise.coords.end(), {0, x, y, z});
        for (int channel = 0; channel < noise.channels; ++channel) {
            noise.feats[point * static_cast<std::size_t>(noise.channels) +
                        static_cast<std::size_t>(channel)] =
                -0.27f + 0.031f * static_cast<float>(channel * points + point);
        }
    }
    pixal3d::Pixal3DImageConditionF32 condition;
    condition.global.batch_size = 1;
    condition.global.channels = flow_hp.cond_channels;
    condition.global.offsets = {0, 3};
    condition.global.feats.resize(3 * static_cast<std::size_t>(flow_hp.cond_channels));
    for (std::size_t index = 0; index < condition.global.feats.size(); ++index) {
        condition.global.feats[index] = -0.19f + 0.017f * static_cast<float>(index);
    }
    condition.projection.batch_size = 1;
    condition.projection.channels = flow_hp.proj_in_channels;
    condition.projection.spatial_x = condition.projection.spatial_y =
        condition.projection.spatial_z = flow_hp.resolution;
    condition.projection.coords = noise.coords;
    condition.projection.feats.resize(points * static_cast<std::size_t>(flow_hp.proj_in_channels));
    for (std::size_t index = 0; index < condition.projection.feats.size(); ++index) {
        condition.projection.feats[index] = 0.11f - 0.013f * static_cast<float>(index);
    }

    pixal3d::SparseStructureStageConfig config;
    config.sampler.steps = 4;
    config.sampler.sigma_min = 0.031f;
    config.sampler.rescale_t = 2.7f;
    config.sampler.guidance_strength = 2.0f;
    config.sampler.guidance_rescale = 0.23f;
    config.sampler.guidance_interval_min = 0.34f;
    config.sampler.guidance_interval_max = 0.91f;
    config.sampler.record_trajectory = false;
    config.occupancy_threshold = 0.0f;
    config.target_resolution = decoder_hp.output_resolution() / 2;

    pixal3d::SparseStructureStageOutputF32 output;
    if (!pixal3d::run_sparse_structure_stage_f32(
            flow, decoder, noise, condition, config, output, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    emit("ss_stage1_sample", output.flow.samples.feats);
    emit_ints("ss_stage1_coords", output.coords);
    return 0;
}
