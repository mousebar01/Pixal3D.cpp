#pragma once

#include "pixal3d/sparse.h"

#include <functional>
#include <string>
#include <vector>

namespace pixal3d {

// Parameters for the flow-matching Euler sampler used by Pixal3D.  The
// defaults mirror the generic FlowEulerSampler; pipeline-specific defaults
// (for example 12 steps and guidance 7.5) are selected by the caller.
struct FlowEulerSamplerConfig {
    int steps = 50;
    float sigma_min = 1e-5f;
    float rescale_t = 1.0f;
    float guidance_strength = 1.0f;
    float guidance_rescale = 0.0f;
    float guidance_interval_min = 0.0f;
    float guidance_interval_max = 1.0f;
    bool record_trajectory = false;
};

// The callback receives the current sparse state and model timestep.  The
// timestep is 1000*t, matching FlowEulerSampler._inference_model.  The
// conditional flag selects the positive or negative CFG condition; it is
// always true for an unguided call.
using FlowVelocityFn = std::function<bool(const SparseTensorF32 & state,
                                           float model_timestep,
                                           bool conditional,
                                           SparseTensorF32 & velocity,
                                           std::string * error)>;

struct FlowEulerSampleF32 {
    SparseTensorF32 samples;
    std::vector<SparseTensorF32> pred_x_t;
    std::vector<SparseTensorF32> pred_x_0;
};

// Run flow-Euler integration from t=1 to t=0.  The sparse coordinates remain
// fixed throughout the loop; only feature values are integrated.  CFG
// rescaling follows Pixal3D's SparseTensor semantics (population standard
// deviation of per-point channel means, independently for each batch).
bool flow_euler_sample_f32(
    const SparseTensorF32 & noise,
    const FlowEulerSamplerConfig & config,
    const FlowVelocityFn & model,
    FlowEulerSampleF32 & output,
    std::string * error = nullptr);

} // namespace pixal3d
