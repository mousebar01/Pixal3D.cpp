#include "pixal3d/ss_flow.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void emit(const char * name, const std::vector<float> & values) {
    std::cout << name << " " << values.size();
    for (float value : values) {
        std::cout << " " << value;
    }
    std::cout << "\n";
}

std::vector<float> values(std::size_t count, float offset, float step) {
    std::vector<float> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        result[i] = offset + step * static_cast<float>(i);
    }
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: " << argv[0] << " <ss-flow-test.gguf> [--sample]\n";
        return 2;
    }
    const bool run_sampler = argc == 3 && std::string(argv[2]) == "--sample";
    if (argc == 3 && !run_sampler) {
        std::cerr << "unknown fixture mode: " << argv[2] << "\n";
        return 2;
    }
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);

    pixal3d::SSFlowModel model;
    std::string error;
    if (!model.load(argv[1], true, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    const pixal3d::SSFlowHParams & hp = model.hparams();
    const std::size_t points = static_cast<std::size_t>(hp.resolution) *
                               static_cast<std::size_t>(hp.resolution) *
                               static_cast<std::size_t>(hp.resolution);
    const int cond_tokens = 3;
    const std::vector<float> x = values(
        static_cast<std::size_t>(hp.in_channels) * points, -0.27f, 0.031f);
    const std::vector<float> cond = values(
        static_cast<std::size_t>(cond_tokens) * hp.cond_channels, -0.19f, 0.017f);
    const std::vector<float> projected = values(
        points * static_cast<std::size_t>(hp.proj_in_channels), 0.11f, -0.013f);
    std::vector<float> output(static_cast<std::size_t>(hp.out_channels) * points, 0.0f);
    if (!model.forward(x.data(), 0.75f, cond.data(), cond_tokens, hp.cond_channels,
                       projected.data(), static_cast<int>(points),
                       hp.proj_in_channels, output.data(), &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    std::cout << "backend " << model.backend_name() << "\n";
    emit("ss_flow_output", output);

    if (run_sampler) {
        pixal3d::SparseTensorF32 noise;
        noise.batch_size = 1;
        noise.channels = hp.in_channels;
        noise.spatial_x = noise.spatial_y = noise.spatial_z = hp.resolution;
        noise.coords.reserve(points * 4);
        noise.feats.resize(points * static_cast<std::size_t>(hp.in_channels));
        for (std::size_t point = 0; point < points; ++point) {
            const std::int32_t x_coord = static_cast<std::int32_t>(
                point / static_cast<std::size_t>(hp.resolution * hp.resolution));
            const std::int32_t y_coord = static_cast<std::int32_t>(
                (point / static_cast<std::size_t>(hp.resolution)) %
                static_cast<std::size_t>(hp.resolution));
            const std::int32_t z_coord = static_cast<std::int32_t>(
                point % static_cast<std::size_t>(hp.resolution));
            noise.coords.insert(noise.coords.end(), {0, x_coord, y_coord, z_coord});
            for (int channel = 0; channel < hp.in_channels; ++channel) {
                noise.feats[point * static_cast<std::size_t>(hp.in_channels) +
                            static_cast<std::size_t>(channel)] =
                    x[static_cast<std::size_t>(channel) * points + point];
            }
        }
        pixal3d::VarLenTensorF32 global;
        global.batch_size = 1;
        global.channels = hp.cond_channels;
        global.offsets = {0, static_cast<std::size_t>(cond_tokens)};
        global.feats = cond;
        pixal3d::SparseTensorF32 projected_sparse;
        projected_sparse.batch_size = 1;
        projected_sparse.channels = hp.proj_in_channels;
        projected_sparse.spatial_x = projected_sparse.spatial_y =
            projected_sparse.spatial_z = hp.resolution;
        projected_sparse.coords = noise.coords;
        projected_sparse.feats = projected;

        pixal3d::FlowEulerSamplerConfig sampler;
        sampler.steps = 4;
        sampler.sigma_min = 0.031f;
        sampler.rescale_t = 2.7f;
        sampler.guidance_strength = 2.0f;
        sampler.guidance_rescale = 0.23f;
        sampler.guidance_interval_min = 0.34f;
        sampler.guidance_interval_max = 0.91f;
        sampler.record_trajectory = true;
        pixal3d::FlowEulerSampleF32 sampled;
        if (!model.sample(noise, sampler, global, &projected_sparse, sampled, &error)) {
            std::cerr << error << "\n";
            return 1;
        }
        emit("ss_flow_sample", sampled.samples.feats);
        emit("ss_flow_sample_x0_last", sampled.pred_x_0.back().feats);
    }
    return 0;
}
