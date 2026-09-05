#pragma once

#include "pixal3d/flow_sampler.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <string>

namespace pixal3d {

// Runtime hyperparameters read from pixal3d.ss_flow.* GGUF metadata.
struct SSFlowHParams {
    int resolution = 0;
    int in_channels = 0;
    int out_channels = 0;
    int model_channels = 0;
    int cond_channels = 0;
    int num_blocks = 0;
    int num_heads = 0;
    float mlp_ratio = 0.0f;
    bool share_mod = false;
    bool qk_rms_norm = false;
    bool qk_rms_norm_cross = false;
    float rope_freq_min = 1.0f;
    float rope_freq_base = 10000.0f;
    std::string pe_mode = "rope";
    std::string image_attn_mode = "cross";
    int proj_in_channels = 0;

    int head_dim() const noexcept {
        return num_heads > 0 ? model_channels / num_heads : 0;
    }
};

// Dense sparse-structure flow model (stage 1).  This is deliberately a
// Pixal3D-owned API rather than a trellis2cpp ABI.  The implementation follows
// the validated trellis2cpp ggml graph while adapting compact-v1 names and the
// Pixal3D projection-attention condition.
class SSFlowModel {
public:
    SSFlowModel() = default;
    ~SSFlowModel();

    SSFlowModel(const SSFlowModel &) = delete;
    SSFlowModel & operator=(const SSFlowModel &) = delete;

    // Parse metadata and descriptors.  When load_tensors is true, the SS-flow
    // tensors are streamed into a ggml backend buffer one tensor at a time.
    bool load(const std::string & path,
              bool load_tensors = true,
              std::string * error = nullptr);
    void close() noexcept;

    bool is_loaded() const noexcept;
    bool has_data() const noexcept;
    const SSFlowHParams & hparams() const noexcept;
    const std::string & backend_name() const noexcept;
    int tensor_count() const noexcept;
    bool has_tensor(const std::string & name) const noexcept;

    // Run one dense SS-flow velocity prediction.  x/out are channel-major
    // [channels, resolution^3].  cond is token-major [tokens, cond_channels].
    // For image_attn_mode="proj", projected is token-major
    // [resolution^3, proj_in_channels] and must contain one projected feature
    // row for every spatial token.  The graph itself consumes the equivalent
    // ggml [channels, tokens] views, matching PyTorch's contiguous layout.
    bool forward(const float * x,
                 float timestep,
                 const float * cond,
                 int cond_tokens,
                 int cond_channels,
                 const float * projected,
                 int projected_points,
                 int projected_channels,
                 float * out,
                 std::string * error = nullptr);

    // Sample a complete dense stage-1 latent with the reference Flow-Euler
    // loop.  The noise is represented as a full-grid SparseTensor so it can
    // share the sampler contract with SLat stages; it must contain exactly
    // one batch in lexicographic [x,y,z] order.  global_context is the
    // token-major image condition.  For image_attn_mode="proj", the optional
    // projection context must contain one feature row per grid point.
    // Classifier-free negative conditions are synthesized as zero tensors,
    // matching Pixal3D inference.py.
    bool sample(const SparseTensorF32 & noise,
                const FlowEulerSamplerConfig & sampler_config,
                const VarLenTensorF32 & global_context,
                const SparseTensorF32 * projection_context,
                FlowEulerSampleF32 & output,
                std::string * error = nullptr);

private:
    struct Impl;
    Impl * impl_ = nullptr;
};

} // namespace pixal3d
