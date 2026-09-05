#include "pixal3d/slat_flow.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

std::vector<std::int32_t> coords() {
    return {
        0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 1, 0,  0, 1, 1, 1,  0, 2, 2, 2,
        1, 0, 0, 0,  1, 0, 1, 0,  1, 1, 1, 1,  1, 2, 1, 2,  1, 3, 3, 3,
    };
}

void emit(const char * name, const std::vector<float> & values) {
    std::cout << name << " " << values.size();
    for (float value : values) std::cout << " " << value;
    std::cout << "\n";
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <texture-flow-test.gguf>\n";
        return 2;
    }
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    pixal3d::SLatFlowModel model;
    std::string error;
    if (!model.load(argv[1], "texture_flow_1024", true, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    const auto & hp = model.hparams();
    constexpr int base_channels = 12;
    const int concat_channels = hp.in_channels - base_channels;
    if (concat_channels <= 0) {
        std::cerr << "texture fixture metadata must have concat channels\n";
        return 1;
    }
    const auto coordinate_values = coords();
    pixal3d::SparseTensorF32 input;
    input.batch_size = 2;
    input.channels = base_channels;
    input.spatial_x = input.spatial_y = input.spatial_z = hp.resolution;
    input.coords = coordinate_values;
    input.feats.resize(input.points() * static_cast<std::size_t>(input.channels));
    for (std::size_t index = 0; index < input.feats.size(); ++index) {
        input.feats[index] = -0.27f + 0.013f * static_cast<float>(index);
    }
    pixal3d::SparseTensorF32 concat;
    concat.batch_size = input.batch_size;
    concat.channels = concat_channels;
    concat.spatial_x = concat.spatial_y = concat.spatial_z = hp.resolution;
    concat.coords = coordinate_values;
    concat.feats.resize(concat.points() * static_cast<std::size_t>(concat.channels));
    for (std::size_t index = 0; index < concat.feats.size(); ++index) {
        concat.feats[index] = 0.07f - 0.011f * static_cast<float>(index);
    }
    pixal3d::SparseTensorF32 projected;
    projected.batch_size = input.batch_size;
    projected.channels = hp.proj_in_channels;
    projected.spatial_x = projected.spatial_y = projected.spatial_z = hp.resolution;
    projected.coords = coordinate_values;
    projected.feats.resize(projected.points() * static_cast<std::size_t>(projected.channels));
    for (std::size_t index = 0; index < projected.feats.size(); ++index) {
        projected.feats[index] = 0.19f - 0.017f * static_cast<float>(index);
    }
    pixal3d::VarLenTensorF32 global;
    global.batch_size = 2;
    global.channels = hp.cond_channels;
    global.offsets = {0, 3, 5};
    global.feats.resize(5 * static_cast<std::size_t>(global.channels));
    for (std::size_t index = 0; index < global.feats.size(); ++index) {
        global.feats[index] = -0.11f + 0.021f * static_cast<float>(index);
    }
    const std::vector<float> timesteps = {0.17f, -0.23f};
    pixal3d::SparseTensorF32 output;
    if (!model.forward(input, timesteps.data(), timesteps.size(), global, &projected,
                       output, &error, &concat)) {
        std::cerr << error << "\n";
        return 1;
    }
    emit("slat_flow_concat_output", output.feats);

    pixal3d::FlowEulerSamplerConfig sampler;
    sampler.steps = 3;
    sampler.sigma_min = 0.031f;
    sampler.rescale_t = 2.7f;
    sampler.guidance_strength = 1.0f;
    sampler.record_trajectory = false;
    pixal3d::FlowEulerSampleF32 sampled;
    if (!model.sample(input, sampler, global, &projected, sampled, &error, &concat)) {
        std::cerr << error << "\n";
        return 1;
    }
    emit("slat_flow_concat_sample", sampled.samples.feats);
    return 0;
}
