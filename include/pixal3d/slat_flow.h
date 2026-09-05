#pragma once

#include "pixal3d/flow_sampler.h"
#include "pixal3d/sparse_transformer.h"

#include <cstddef>
#include <string>
#include <vector>

namespace pixal3d {

struct SLatFlowHParams {
    std::string component;
    std::string model_class;
    int resolution = 0;
    int in_channels = 0;
    int out_channels = 0;
    int model_channels = 0;
    int cond_channels = 0;
    int num_blocks = 0;
    int num_heads = 0;
    int proj_in_channels = 0;
    float mlp_ratio = 0.0f;
    float norm_eps = 1e-6f;
    float rope_freq_min = 1.0f;
    float rope_freq_base = 10000.0f;
    std::string pe_mode = "rope";
    std::string image_attn_mode = "cross";
    bool share_mod = false;
    bool qk_rms_norm = false;
    bool qk_rms_norm_cross = false;

    int head_dim() const noexcept {
        return num_heads > 0 ? model_channels / num_heads : 0;
    }
};

struct SLatFlowWeightsF32 {
    const float * input_weight = nullptr;
    const float * input_bias = nullptr;
    const float * timestep_weight0 = nullptr;
    const float * timestep_bias0 = nullptr;
    const float * timestep_weight2 = nullptr;
    const float * timestep_bias2 = nullptr;
    const float * modulation_weight = nullptr;
    const float * modulation_bias = nullptr;
    const float * output_weight = nullptr;
    const float * output_bias = nullptr;
    std::vector<SparseTransformerBlockWeightsF32> blocks;
};

// Run a complete sparse SLat flow model using CPU/F32 primitives.  timesteps
// contains one value per input batch; global_context is variable-length per
// batch; projection_context must share input coordinates in proj mode.
bool slat_flow_forward_f32(
    const SparseTensorF32 & input,
    const float * timesteps,
    std::size_t timestep_count,
    const VarLenTensorF32 & global_context,
    const SparseTensorF32 * projection_context,
    const SLatFlowHParams & hparams,
    const SLatFlowWeightsF32 & weights,
    SparseTensorF32 & output,
    std::string * error = nullptr,
    const SparseTensorF32 * concat_condition = nullptr);

// F32 loader for one shape_flow_512, shape_flow_1024, or texture_flow_1024
// component in a grouped Pixal3D GGUF pack. Compact-v1 aliases
// (sh512/sh1024/tx1024) are resolved internally. Batch-1 forward calls use a
// ggml GPU graph when a CUDA backend is available and fall back to the
// validated host implementation when it is not.
class SLatFlowModel {
public:
    SLatFlowModel() = default;
    ~SLatFlowModel();

    SLatFlowModel(const SLatFlowModel &) = delete;
    SLatFlowModel & operator=(const SLatFlowModel &) = delete;

    bool load(const std::string & path,
              const std::string & component,
              bool load_tensors = true,
              std::string * error = nullptr);
    void close() noexcept;

    bool is_loaded() const noexcept;
    bool has_data() const noexcept;
    const SLatFlowHParams & hparams() const noexcept;
    int tensor_count() const noexcept;
    bool has_tensor(const std::string & name) const noexcept;

    bool forward(const SparseTensorF32 & input,
                 const float * timesteps,
                 std::size_t timestep_count,
                 const VarLenTensorF32 & global_context,
                 const SparseTensorF32 * projection_context,
                 SparseTensorF32 & output,
                 std::string * error = nullptr,
                 const SparseTensorF32 * concat_condition = nullptr);

    // Run the flow-Euler sampler with this model.  The negative CFG condition
    // is synthesized as zeros_like(global_context) and, in projection mode,
    // zeros_like(projection_context), matching Pixal3D inference.
    bool sample(const SparseTensorF32 & noise,
                const FlowEulerSamplerConfig & sampler_config,
                const VarLenTensorF32 & global_context,
                const SparseTensorF32 * projection_context,
                FlowEulerSampleF32 & output,
                std::string * error = nullptr,
                const SparseTensorF32 * concat_condition = nullptr);

private:
    struct Impl;
    Impl * impl_ = nullptr;
};

} // namespace pixal3d
