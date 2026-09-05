#include "pixal3d/flow_sampler.h"

#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace {

pixal3d::SparseTensorF32 make_noise() {
    pixal3d::SparseTensorF32 noise;
    noise.batch_size = 2;
    noise.channels = 3;
    noise.spatial_x = 4;
    noise.spatial_y = 3;
    noise.spatial_z = 2;
    noise.coords = {
        0, 0, 0, 0,  0, 0, 1, 0,  0, 2, 1, 1,
        1, 0, 0, 1,  1, 1, 2, 0,
    };
    noise.feats.resize(noise.points() * static_cast<std::size_t>(noise.channels));
    for (std::size_t index = 0; index < noise.feats.size(); ++index) {
        noise.feats[index] = -0.73f + 0.091f * static_cast<float>(index);
    }
    return noise;
}

void emit(const char * name, const std::vector<float> & values) {
    std::cout << name << " " << values.size();
    for (float value : values) std::cout << " " << value;
    std::cout << "\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    const pixal3d::SparseTensorF32 noise = make_noise();
    pixal3d::FlowEulerSamplerConfig config;
    config.steps = 6;
    config.sigma_min = 0.031f;
    config.rescale_t = 2.7f;
    config.guidance_strength = 2.35f;
    config.guidance_rescale = 0.41f;
    config.guidance_interval_min = 0.34f;
    config.guidance_interval_max = 0.91f;
    config.record_trajectory = true;

    pixal3d::FlowVelocityFn model = [](const pixal3d::SparseTensorF32 & state,
                                       float timestep, bool conditional,
                                       pixal3d::SparseTensorF32 & velocity,
                                       std::string *) {
        velocity = state;
        const float condition = conditional ? 0.19f : -0.07f;
        for (std::size_t index = 0; index < velocity.feats.size(); ++index) {
            const float channel = static_cast<float>(index %
                static_cast<std::size_t>(state.channels));
            velocity.feats[index] = 0.043f * state.feats[index] +
                                    0.00037f * timestep +
                                    0.011f * channel + condition;
        }
        return true;
    };

    pixal3d::FlowEulerSampleF32 result;
    std::string error;
    if (!pixal3d::flow_euler_sample_f32(noise, config, model, result, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    emit("flow_sampler_samples", result.samples.feats);
    if (result.pred_x_t.size() != static_cast<std::size_t>(config.steps) ||
        result.pred_x_0.size() != static_cast<std::size_t>(config.steps)) {
        std::cerr << "unexpected flow sampler trajectory length\n";
        return 1;
    }
    emit("flow_sampler_pred_x_t_last", result.pred_x_t.back().feats);
    emit("flow_sampler_pred_x_0_last", result.pred_x_0.back().feats);
    return 0;
}
