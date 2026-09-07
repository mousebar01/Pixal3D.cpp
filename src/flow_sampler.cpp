#include "pixal3d/flow_sampler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace pixal3d {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

bool flow_diagnostics_enabled() {
    const char * value = std::getenv("PIXAL3D_FLOW_NONFINITE_TRACE");
    return value && *value && std::string(value) != "0";
}

bool trace_features(const SparseTensorF32 & tensor,
                    const char * label,
                    const char * branch,
                    int step,
                    float timestep,
                    std::string * error) {
    std::size_t nonfinite_count = 0;
    std::size_t first_nonfinite = 0;
    float first_value = 0.0f;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    float maximum_abs = 0.0f;
    long double sum = 0.0L;
    for (std::size_t index = 0; index < tensor.feats.size(); ++index) {
        const float value = tensor.feats[index];
        if (!std::isfinite(value)) {
            if (nonfinite_count == 0) {
                first_nonfinite = index;
                first_value = value;
            }
            ++nonfinite_count;
            continue;
        }
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        maximum_abs = std::max(maximum_abs, std::fabs(value));
        sum += static_cast<long double>(value);
    }
    const double mean = tensor.feats.empty()
        ? 0.0
        : static_cast<double>(sum / static_cast<long double>(tensor.feats.size()));
    std::cerr << "pixal3d: flow trace step=" << step
              << " timestep=" << timestep
              << " branch=" << (branch && *branch ? branch : "unknown")
              << " tensor=" << (label && *label ? label : "features")
              << " points=" << tensor.points()
              << " values=" << tensor.feats.size()
              << " min=" << (tensor.feats.empty() ? 0.0f : minimum)
              << " max=" << (tensor.feats.empty() ? 0.0f : maximum)
              << " max_abs=" << maximum_abs
              << " mean=" << mean
              << " nonfinite=" << nonfinite_count;
    if (nonfinite_count != 0) {
        std::cerr << " first_bad_index=" << first_nonfinite
                  << " first_bad_value=" << first_value;
    }
    std::cerr << std::endl;
    if (nonfinite_count != 0) {
        set_error(error, std::string("flow ") +
                  (label && *label ? label : "features") +
                  " contains non-finite values at step " + std::to_string(step) +
                  " (branch=" + (branch && *branch ? branch : "unknown") +
                  ", first index=" + std::to_string(first_nonfinite) + ")");
        return false;
    }
    return true;
}

bool same_shape(const SparseTensorF32 & left, const SparseTensorF32 & right) {
    return left.batch_size == right.batch_size && left.channels == right.channels &&
           left.spatial_x == right.spatial_x && left.spatial_y == right.spatial_y &&
           left.spatial_z == right.spatial_z && left.coords == right.coords &&
           left.feats.size() == right.feats.size();
}

void clear_tensor_for_callback(SparseTensorF32 & tensor) {
    tensor.batch_size = 0;
    tensor.channels = 0;
    tensor.spatial_x = 0;
    tensor.spatial_y = 0;
    tensor.spatial_z = 0;
    tensor.coords.clear();
    tensor.feats.clear();
}

struct FlowSamplerWorkspace {
    SparseTensorF32 prediction;
    SparseTensorF32 positive;
    SparseTensorF32 negative;
    SparseTensorF32 x0_positive;
    SparseTensorF32 x0_cfg;
    std::vector<float> std_positive;
    std::vector<float> std_cfg;
    std::vector<std::size_t> counts;
    std::vector<float> sum;
    std::vector<float> sum_squared;
};

bool valid_config(const FlowEulerSamplerConfig & config, std::string * error) {
    if (config.steps <= 0 || !(config.sigma_min >= 0.0f) ||
        !(config.sigma_min < 1.0f) || !std::isfinite(config.sigma_min) ||
        !(config.rescale_t > 0.0f) || !std::isfinite(config.rescale_t) ||
        !std::isfinite(config.guidance_strength) ||
        !std::isfinite(config.guidance_rescale) || config.guidance_rescale < 0.0f ||
        !std::isfinite(config.guidance_interval_min) ||
        !std::isfinite(config.guidance_interval_max) ||
        config.guidance_interval_min > config.guidance_interval_max) {
        set_error(error, "invalid flow Euler sampler configuration");
        return false;
    }
    return true;
}

bool check_model_output(const SparseTensorF32 & state,
                        const SparseTensorF32 & velocity,
                        std::string * error) {
    if (!velocity.valid(error)) return false;
    if (!same_shape(state, velocity)) {
        set_error(error, "flow model velocity shape or coordinates changed during sampling");
        return false;
    }
    return true;
}

void pred_to_xstart(const SparseTensorF32 & state, float timestep, float sigma_min,
                    const SparseTensorF32 & prediction, SparseTensorF32 & x0) {
    x0 = prediction;
    const float a = 1.0f - sigma_min;
    const float b = sigma_min + a * timestep;
    for (std::size_t index = 0; index < state.feats.size(); ++index) {
        x0.feats[index] = a * state.feats[index] - b * prediction.feats[index];
    }
}

void xstart_to_pred(const SparseTensorF32 & state, float timestep, float sigma_min,
                    const SparseTensorF32 & x0, SparseTensorF32 & prediction) {
    prediction = x0;
    const float a = 1.0f - sigma_min;
    const float b = sigma_min + a * timestep;
    for (std::size_t index = 0; index < state.feats.size(); ++index) {
        prediction.feats[index] = (a * state.feats[index] - x0.feats[index]) / b;
    }
}

// SparseTensor inherits VarLenTensor.std() from the reference implementation.
// For a SparseTensor and dim=[1], the first reduction is over channels and the
// segment reduction is over points.  Its mean2 path squares features before
// reducing, so this is the population standard deviation over all feature
// values in each batch.
bool sparse_batch_std(const SparseTensorF32 & input, std::vector<float> & output,
                      std::vector<std::size_t> & counts,
                      std::vector<float> & sum,
                      std::vector<float> & sum_squared,
                      std::string * error) {
    if (!input.valid(error)) return false;
    const std::size_t batch_count = static_cast<std::size_t>(input.batch_size);
    output.assign(batch_count, 0.0f);
    counts.assign(batch_count, 0);
    sum.assign(batch_count, 0.0f);
    sum_squared.assign(batch_count, 0.0f);
    for (std::size_t point = 0; point < input.points(); ++point) {
        const int batch = input.coords[point * 4];
        const float * values = input.feats.data() + point * static_cast<std::size_t>(input.channels);
        for (int channel = 0; channel < input.channels; ++channel) {
            sum[static_cast<std::size_t>(batch)] += values[channel];
            sum_squared[static_cast<std::size_t>(batch)] += values[channel] * values[channel];
            ++counts[static_cast<std::size_t>(batch)];
        }
    }
    for (int batch = 0; batch < input.batch_size; ++batch) {
        const std::size_t count = counts[static_cast<std::size_t>(batch)];
        if (count == 0) {
            set_error(error, "flow CFG rescale requires at least one point per batch");
            return false;
        }
        const float mean = sum[static_cast<std::size_t>(batch)] /
                           static_cast<float>(count);
        const float mean_squared = sum_squared[static_cast<std::size_t>(batch)] /
                                   static_cast<float>(count);
        output[static_cast<std::size_t>(batch)] =
            std::sqrt(std::max(mean_squared - mean * mean, 0.0f));
    }
    return true;
}

void blend_guidance(const SparseTensorF32 & positive, const SparseTensorF32 & negative,
                    float guidance_strength, SparseTensorF32 & output) {
    output = positive;
    const float negative_weight = 1.0f - guidance_strength;
    for (std::size_t index = 0; index < output.feats.size(); ++index) {
        output.feats[index] = guidance_strength * positive.feats[index] +
                              negative_weight * negative.feats[index];
    }
}

bool apply_guidance_rescale(const SparseTensorF32 & state, float timestep,
                            const FlowEulerSamplerConfig & config,
                            const SparseTensorF32 & positive,
                            SparseTensorF32 & prediction,
                            FlowSamplerWorkspace & workspace,
                            std::string * error) {
    pred_to_xstart(state, timestep, config.sigma_min, positive, workspace.x0_positive);
    pred_to_xstart(state, timestep, config.sigma_min, prediction, workspace.x0_cfg);
    if (!sparse_batch_std(workspace.x0_positive, workspace.std_positive,
                          workspace.counts, workspace.sum, workspace.sum_squared, error) ||
        !sparse_batch_std(workspace.x0_cfg, workspace.std_cfg,
                          workspace.counts, workspace.sum, workspace.sum_squared, error)) {
        return false;
    }
    SparseTensorF32 & x0_cfg = workspace.x0_cfg;
    const std::vector<float> & std_positive = workspace.std_positive;
    const std::vector<float> & std_cfg = workspace.std_cfg;
    for (std::size_t point = 0; point < x0_cfg.points(); ++point) {
        const int batch = x0_cfg.coords[point * 4];
        const float denominator = std_cfg[static_cast<std::size_t>(batch)];
        const float ratio = denominator != 0.0f
            ? std_positive[static_cast<std::size_t>(batch)] / denominator : 1.0f;
        float * values = x0_cfg.feats.data() + point * static_cast<std::size_t>(x0_cfg.channels);
        for (int channel = 0; channel < x0_cfg.channels; ++channel) {
            const float rescaled = values[channel] * ratio;
            values[channel] = config.guidance_rescale * rescaled +
                              (1.0f - config.guidance_rescale) * values[channel];
        }
    }
    xstart_to_pred(state, timestep, config.sigma_min, x0_cfg, prediction);
    return true;
}

} // namespace

bool flow_euler_sample_f32(
    const SparseTensorF32 & noise,
    const FlowEulerSamplerConfig & config,
    const FlowVelocityFn & model,
    FlowEulerSampleF32 & output,
    std::string * error) {
    output = FlowEulerSampleF32{};
    if (!noise.valid(error) || !valid_config(config, error)) return false;
    if (!model) {
        set_error(error, "flow Euler sampler model callback is empty");
        return false;
    }

    SparseTensorF32 state = noise;
    FlowSamplerWorkspace workspace;
    const bool diagnostics = flow_diagnostics_enabled();
    if (diagnostics && !trace_features(state, "state_before", "initial", -1, 1.0f, error)) {
        return false;
    }
    if (config.record_trajectory) {
        output.pred_x_t.reserve(static_cast<std::size_t>(config.steps));
        output.pred_x_0.reserve(static_cast<std::size_t>(config.steps));
    }
    const float steps = static_cast<float>(config.steps);
    for (int step = 0; step < config.steps; ++step) {
        const float linear = 1.0f - static_cast<float>(step) / steps;
        const float denominator = 1.0f + (config.rescale_t - 1.0f) * linear;
        if (!(denominator > 0.0f) || !std::isfinite(denominator)) {
            set_error(error, "flow Euler timestep schedule is not finite");
            return false;
        }
        const float timestep = config.rescale_t * linear / denominator;
        const float next_linear = 1.0f - static_cast<float>(step + 1) / steps;
        const float next_denominator = 1.0f + (config.rescale_t - 1.0f) * next_linear;
        if (!(next_denominator > 0.0f) || !std::isfinite(next_denominator)) {
            set_error(error, "flow Euler timestep schedule is not finite");
            return false;
        }
        const float next_timestep = config.rescale_t * next_linear / next_denominator;
        const bool in_interval = timestep >= config.guidance_interval_min &&
                                 timestep <= config.guidance_interval_max;
        const float guidance = in_interval ? config.guidance_strength : 1.0f;

        clear_tensor_for_callback(workspace.prediction);
        if (guidance == 1.0f) {
            if (!model(state, 1000.0f * timestep, true, workspace.prediction, error) ||
                !check_model_output(state, workspace.prediction, error)) return false;
            if (diagnostics && !trace_features(workspace.prediction, "prediction", "conditional", step,
                                               timestep, error)) return false;
        } else if (guidance == 0.0f) {
            if (!model(state, 1000.0f * timestep, false, workspace.prediction, error) ||
                !check_model_output(state, workspace.prediction, error)) return false;
            if (diagnostics && !trace_features(workspace.prediction, "prediction", "unconditional", step,
                                               timestep, error)) return false;
        } else {
            clear_tensor_for_callback(workspace.positive);
            clear_tensor_for_callback(workspace.negative);
            if (!model(state, 1000.0f * timestep, true, workspace.positive, error) ||
                !check_model_output(state, workspace.positive, error)) return false;
            if (diagnostics && !trace_features(workspace.positive, "prediction", "conditional", step,
                                               timestep, error)) return false;
            if (!model(state, 1000.0f * timestep, false, workspace.negative, error) ||
                !check_model_output(state, workspace.negative, error)) return false;
            if (diagnostics && !trace_features(workspace.negative, "prediction", "unconditional", step,
                                               timestep, error)) return false;
            blend_guidance(workspace.positive, workspace.negative, guidance, workspace.prediction);
            if (diagnostics && !trace_features(workspace.prediction, "cfg_blend", "guided", step,
                                               timestep, error)) return false;
            if (config.guidance_rescale > 0.0f &&
                !apply_guidance_rescale(state, timestep, config, workspace.positive,
                                        workspace.prediction, workspace, error)) {
                return false;
            }
            if (diagnostics && config.guidance_rescale > 0.0f &&
                !trace_features(workspace.prediction, "guidance_rescale", "guided", step,
                                timestep, error)) return false;
        }

        SparseTensorF32 x0;
        if (config.record_trajectory) {
            pred_to_xstart(state, timestep, config.sigma_min, workspace.prediction, x0);
            if (diagnostics && !trace_features(x0, "x0", "guided", step, timestep, error)) {
                return false;
            }
        }
        const float delta = timestep - next_timestep;
        for (std::size_t index = 0; index < state.feats.size(); ++index) {
            state.feats[index] -= delta * workspace.prediction.feats[index];
        }
        if (diagnostics && !trace_features(state, "state_after", "euler", step,
                                           next_timestep, error)) return false;
        if (config.record_trajectory) {
            output.pred_x_t.push_back(state);
            output.pred_x_0.push_back(std::move(x0));
        }
    }
    output.samples = std::move(state);
    return true;
}

} // namespace pixal3d
