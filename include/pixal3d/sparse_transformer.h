#pragma once

#include "pixal3d/sparse.h"

#include <string>

namespace pixal3d {

// Configuration for one Pixal3D ModulatedSparseTransformerCrossBlock.
// The current reference models use the shared-modulation path, RoPE, and
// projection attention; the flags keep the primitive useful for the cross
// variant as well.
struct SparseTransformerBlockConfig {
    int channels = 0;
    int num_heads = 0;
    int mlp_hidden = 0;
    int context_channels = 0;
    int proj_in_channels = 0;
    float norm_eps = 1e-6f;
    float rope_freq_min = 1.0f;
    float rope_freq_base = 10000.0f;
    bool use_rope = true;
    bool qk_rms_norm = true;
    bool qk_rms_norm_cross = true;
    bool use_projection = true;
};

// F32 view of the state_dict tensors for one block.  Linear weights use the
// PyTorch [out,in] layout; modulation and RMS gamma use head-major flat rows.
struct SparseTransformerBlockWeightsF32 {
    const float * modulation = nullptr; // [6 * channels]
    const float * self_to_qkv_weight = nullptr;
    const float * self_to_qkv_bias = nullptr;
    const float * self_q_gamma = nullptr;
    const float * self_k_gamma = nullptr;
    const float * self_to_out_weight = nullptr;
    const float * self_to_out_bias = nullptr;
    const float * norm2_weight = nullptr;
    const float * norm2_bias = nullptr;
    const float * cross_to_q_weight = nullptr;
    const float * cross_to_q_bias = nullptr;
    const float * cross_to_kv_weight = nullptr;
    const float * cross_to_kv_bias = nullptr;
    const float * cross_q_gamma = nullptr;
    const float * cross_k_gamma = nullptr;
    const float * cross_to_out_weight = nullptr;
    const float * cross_to_out_bias = nullptr;
    const float * proj_weight = nullptr;
    const float * proj_bias = nullptr;
    const float * mlp0_weight = nullptr;
    const float * mlp0_bias = nullptr;
    const float * mlp2_weight = nullptr;
    const float * mlp2_bias = nullptr;
};

// Evaluate one ModulatedSparseTransformerCrossBlock.  timestep_modulation is
// [batch_size, 6 * channels] and is combined with the block's shared
// modulation parameter.  global_context contains variable-length K/V tokens;
// projected_context, when enabled, has one row per query coordinate.
bool sparse_transformer_cross_block_f32(
    const SparseTensorF32 & input,
    const float * timestep_modulation,
    const VarLenTensorF32 & global_context,
    const SparseTensorF32 * projected_context,
    const SparseTransformerBlockConfig & config,
    const SparseTransformerBlockWeightsF32 & weights,
    SparseTensorF32 & output,
    std::string * error = nullptr);

} // namespace pixal3d
