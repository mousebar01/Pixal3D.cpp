#include "pixal3d/pipeline.h"

#include <iomanip>
#include <iostream>
#include <limits>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

std::vector<std::int32_t> coords() {
    return {
        0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 1, 0,  0, 1, 1, 1,  0, 2, 2, 2,
        1, 0, 0, 0,  1, 0, 1, 0,  1, 1, 1, 1,  1, 2, 1, 2,  1, 3, 3, 3,
    };
}

std::vector<std::int32_t> generated_coords(std::size_t count, int resolution) {
    std::vector<std::int32_t> result;
    result.reserve(count * 4);
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t plane = static_cast<std::size_t>(resolution) * resolution;
        const int z = static_cast<int>((index / plane) % static_cast<std::size_t>(resolution));
        const std::size_t rem = index % plane;
        const int y = static_cast<int>(rem / static_cast<std::size_t>(resolution));
        const int x = static_cast<int>(rem % static_cast<std::size_t>(resolution));
        result.insert(result.end(), {0, x, y, z});
    }
    return result;
}

void emit(const char * name, const std::vector<float> & values) {
    std::cout << name << " " << values.size();
    for (float value : values) std::cout << " " << value;
    std::cout << "\n";
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: pixal3d_slat_flow_loader_fixture <pack> [--sample|--stage|--gpu]\n";
        return 2;
    }
    const bool run_sampler = argc == 3 && std::string(argv[2]) == "--sample";
    const bool run_stage = argc == 3 && std::string(argv[2]) == "--stage";
    // --gpu switches the fixture to batch-1 so SLatFlowModel selects its
    // CUDA graph; the default two-batch mode continues to exercise the CPU
    // reference path used by compare_slat_flow.py.
    const bool run_gpu = argc == 3 && std::string(argv[2]) == "--gpu";
    if (argc == 3 && !run_sampler && !run_stage && !run_gpu) {
        std::cerr << "unknown fixture mode: " << argv[2] << "\n";
        return 2;
    }
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    pixal3d::SLatFlowModel model;
    std::string error;
    if (!model.load(argv[1], "shape_flow_512", true, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    const pixal3d::SLatFlowHParams & hp = model.hparams();
    pixal3d::SparseTensorF32 input;
    input.batch_size = run_gpu ? 1 : 2;
    input.channels = hp.in_channels;
    input.spatial_x = input.spatial_y = input.spatial_z = hp.resolution;
    input.coords = coords();
    if (run_gpu) {
        input.coords.resize(5 * 4);
        if (const char * count_text = std::getenv("PIXAL3D_SLAT_FIXTURE_POINTS")) {
            const std::size_t count = static_cast<std::size_t>(std::strtoull(count_text, nullptr, 10));
            if (count > 0) input.coords = generated_coords(count, hp.resolution);
        }
    }
    input.feats.resize(input.points() * static_cast<std::size_t>(input.channels));
    for (std::size_t index = 0; index < input.feats.size(); ++index) {
        input.feats[index] = -0.27f + 0.013f * static_cast<float>(index);
    }
    pixal3d::SparseTensorF32 projected;
    projected.batch_size = input.batch_size;
    projected.channels = hp.proj_in_channels;
    projected.spatial_x = projected.spatial_y = projected.spatial_z = hp.resolution;
    projected.coords = input.coords;
    projected.feats.resize(input.points() * static_cast<std::size_t>(projected.channels));
    for (std::size_t index = 0; index < projected.feats.size(); ++index) {
        projected.feats[index] = 0.19f - 0.017f * static_cast<float>(index);
    }
    pixal3d::VarLenTensorF32 global;
    global.batch_size = input.batch_size;
    global.channels = hp.cond_channels;
    global.offsets = run_gpu ? std::vector<std::size_t>{0, 5}
                             : std::vector<std::size_t>{0, 3, 5};
    global.feats.resize(5 * static_cast<std::size_t>(global.channels));
    for (std::size_t index = 0; index < global.feats.size(); ++index) {
        global.feats[index] = -0.11f + 0.021f * static_cast<float>(index);
    }
    const std::vector<float> timesteps = run_gpu ? std::vector<float>{0.17f}
                                                 : std::vector<float>{0.17f, -0.23f};
    pixal3d::SparseTensorF32 output;
    if (!model.forward(input, timesteps.data(), timesteps.size(), global,
                       &projected, output, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    emit("slat_flow_coords", std::vector<float>(output.coords.begin(), output.coords.end()));
    emit("slat_flow_output", output.feats);
    if (run_sampler || run_stage) {
        pixal3d::FlowEulerSamplerConfig sampler;
        sampler.steps = 4;
        sampler.sigma_min = 0.031f;
        sampler.rescale_t = 2.7f;
        sampler.guidance_strength = 2.0f;
        sampler.guidance_rescale = 0.23f;
        sampler.guidance_interval_min = 0.34f;
        sampler.guidance_interval_max = 0.91f;
        sampler.record_trajectory = true;
        const float probe_t = sampler.rescale_t * 0.75f /
                              (1.0f + (sampler.rescale_t - 1.0f) * 0.75f);
        const std::vector<float> probe_timesteps = {1000.0f * probe_t,
                                                    1000.0f * probe_t};
        pixal3d::SparseTensorF32 probe;
        if (!model.forward(input, probe_timesteps.data(), probe_timesteps.size(),
                           global, &projected, probe, &error)) {
            std::cerr << error << "\n";
            return 1;
        }
        emit("slat_flow_probe", probe.feats);
        if (run_sampler) {
            pixal3d::FlowEulerSampleF32 sampled;
            if (!model.sample(input, sampler, global, &projected, sampled, &error)) {
                std::cerr << error << "\n";
                return 1;
            }
            emit("slat_flow_sample", sampled.samples.feats);
            emit("slat_flow_sample_x0_last", sampled.pred_x_0.back().feats);
        } else {
            pixal3d::Pixal3DImageConditionF32 condition;
            condition.global = global;
            condition.projection = projected;
            pixal3d::SLatNormalizationF32 normalization;
            normalization.mean.resize(static_cast<std::size_t>(hp.out_channels));
            normalization.std.resize(static_cast<std::size_t>(hp.out_channels));
            for (int channel = 0; channel < hp.out_channels; ++channel) {
                normalization.mean[static_cast<std::size_t>(channel)] =
                    0.041f + 0.007f * static_cast<float>(channel);
                normalization.std[static_cast<std::size_t>(channel)] =
                    0.73f + 0.013f * static_cast<float>(channel);
            }
            pixal3d::SLatStageOutputF32 stage;
            if (!pixal3d::run_slat_stage_f32(
                    model, input, condition, sampler, normalization, stage, &error)) {
                std::cerr << error << "\n";
                return 1;
            }
            emit("slat_flow_stage_latent", stage.latent.feats);
        }
    }
    return 0;
}
