#include "pixal3d/sparse_transformer.h"

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

namespace pixal3d {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

bool require_pointer(const float * pointer, const char * name, std::string * error) {
    if (pointer) return true;
    set_error(error, std::string("missing sparse transformer tensor: ") + name);
    return false;
}

bool same_coords(const SparseTensorF32 & left, const SparseTensorF32 & right) {
    return left.batch_size == right.batch_size && left.coords == right.coords;
}

SparseTensorF32 sparse_slice(const SparseTensorF32 & input, int offset, int channels) {
    SparseTensorF32 result = input;
    result.channels = channels;
    result.feats.resize(input.points() * static_cast<std::size_t>(channels));
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t point = 0; point < input.points(); ++point) {
        const float * source = input.feats.data() + point * input.channels + offset;
        float * destination = result.feats.data() + point * static_cast<std::size_t>(channels);
        for (int channel = 0; channel < channels; ++channel) destination[channel] = source[channel];
    }
    return result;
}

VarLenTensorF32 varlen_slice(const VarLenTensorF32 & input, int offset, int channels) {
    VarLenTensorF32 result = input;
    result.channels = channels;
    result.feats.resize(input.tokens() * static_cast<std::size_t>(channels));
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t token = 0; token < input.tokens(); ++token) {
        const float * source = input.feats.data() + token * input.channels + offset;
        float * destination = result.feats.data() + token * static_cast<std::size_t>(channels);
        for (int channel = 0; channel < channels; ++channel) destination[channel] = source[channel];
    }
    return result;
}

bool add_in_place(SparseTensorF32 & destination, const SparseTensorF32 & source,
                  std::string * error) {
    if (!same_coords(destination, source) || destination.channels != source.channels) {
        set_error(error, "sparse transformer residual shapes do not match");
        return false;
    }
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t index = 0; index < destination.feats.size(); ++index) {
        destination.feats[index] += source.feats[index];
    }
    return true;
}

bool affine_modulate(const SparseTensorF32 & input,
                     const float * block_modulation,
                     const float * timestep_modulation,
                     int channels,
                     int mod_offset,
                     SparseTensorF32 & output,
                     std::string * error) {
    if (!require_pointer(block_modulation, "modulation", error) ||
        !require_pointer(timestep_modulation, "timestep_modulation", error)) return false;
    output = input;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t point = 0; point < input.points(); ++point) {
        const int batch = input.coords[point * 4];
        const float * source = input.feats.data() + point * static_cast<std::size_t>(channels);
        float * destination = output.feats.data() + point * static_cast<std::size_t>(channels);
        const float * batch_mod = timestep_modulation +
            static_cast<std::size_t>(batch) * static_cast<std::size_t>(6 * channels);
        for (int channel = 0; channel < channels; ++channel) {
            const float scale = block_modulation[mod_offset + channels + channel] +
                                batch_mod[mod_offset + channels + channel];
            const float shift = block_modulation[mod_offset + channel] +
                                batch_mod[mod_offset + channel];
            destination[channel] = source[channel] * (1.0f + scale) + shift;
        }
    }
    return true;
}

bool gate_in_place(SparseTensorF32 & input, const float * block_modulation,
                   const float * timestep_modulation, int channels,
                   int offset, std::string * error) {
    if (!require_pointer(block_modulation, "modulation", error) ||
        !require_pointer(timestep_modulation, "timestep_modulation", error)) return false;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t point = 0; point < input.points(); ++point) {
        const int batch = input.coords[point * 4];
        float * values = input.feats.data() + point * static_cast<std::size_t>(channels);
        const float * batch_mod = timestep_modulation +
            static_cast<std::size_t>(batch) * static_cast<std::size_t>(6 * channels);
        for (int channel = 0; channel < channels; ++channel) {
            const float gate = block_modulation[offset + channel] + batch_mod[offset + channel];
            values[channel] *= gate;
        }
    }
    return true;
}

void gelu_tanh_in_place(SparseTensorF32 & input) {
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t index = 0; index < input.feats.size(); ++index) {
        float & value = input.feats[index];
        const float cubic = value * value * value;
        value = 0.5f * value * (1.0f + std::tanh(kSqrtTwoOverPi *
                                                  (value + 0.044715f * cubic)));
    }
}

} // namespace

bool sparse_transformer_cross_block_f32(
    const SparseTensorF32 & input,
    const float * timestep_modulation,
    const VarLenTensorF32 & global_context,
    const SparseTensorF32 * projected_context,
    const SparseTransformerBlockConfig & config,
    const SparseTransformerBlockWeightsF32 & weights,
    SparseTensorF32 & output,
    std::string * error) {
    if (!input.valid(error) || !global_context.valid(error)) return false;
    if (!timestep_modulation || config.channels <= 0 || config.num_heads <= 0 ||
        config.mlp_hidden <= 0 || config.context_channels <= 0 ||
        config.channels % config.num_heads != 0 ||
        input.channels != config.channels ||
        global_context.channels != config.context_channels ||
        global_context.batch_size != input.batch_size ||
        !(config.norm_eps > 0.0f) || !std::isfinite(config.norm_eps)) {
        set_error(error, "invalid sparse transformer block configuration");
        return false;
    }
    const int head_dim = config.channels / config.num_heads;
    if (config.use_projection) {
        if (config.proj_in_channels <= 0 || !projected_context ||
            projected_context->channels != config.proj_in_channels ||
            !same_coords(input, *projected_context)) {
            set_error(error, "sparse transformer projection shape does not match query");
            return false;
        }
    }
    if (!require_pointer(weights.modulation, "modulation", error) ||
        !require_pointer(weights.self_to_qkv_weight, "self_attn.to_qkv.weight", error) ||
        !require_pointer(weights.self_to_qkv_bias, "self_attn.to_qkv.bias", error) ||
        !require_pointer(weights.self_to_out_weight, "self_attn.to_out.weight", error) ||
        !require_pointer(weights.self_to_out_bias, "self_attn.to_out.bias", error) ||
        !require_pointer(weights.norm2_weight, "norm2.weight", error) ||
        !require_pointer(weights.norm2_bias, "norm2.bias", error) ||
        !require_pointer(weights.cross_to_q_weight, "cross_attn.to_q.weight", error) ||
        !require_pointer(weights.cross_to_q_bias, "cross_attn.to_q.bias", error) ||
        !require_pointer(weights.cross_to_kv_weight, "cross_attn.to_kv.weight", error) ||
        !require_pointer(weights.cross_to_kv_bias, "cross_attn.to_kv.bias", error) ||
        !require_pointer(weights.cross_to_out_weight, "cross_attn.to_out.weight", error) ||
        !require_pointer(weights.cross_to_out_bias, "cross_attn.to_out.bias", error) ||
        !require_pointer(weights.mlp0_weight, "mlp.mlp.0.weight", error) ||
        !require_pointer(weights.mlp0_bias, "mlp.mlp.0.bias", error) ||
        !require_pointer(weights.mlp2_weight, "mlp.mlp.2.weight", error) ||
        !require_pointer(weights.mlp2_bias, "mlp.mlp.2.bias", error)) return false;
    if (config.qk_rms_norm &&
        (!require_pointer(weights.self_q_gamma, "self_attn.q_rms_norm.gamma", error) ||
         !require_pointer(weights.self_k_gamma, "self_attn.k_rms_norm.gamma", error))) return false;
    if (config.qk_rms_norm_cross &&
        (!require_pointer(weights.cross_q_gamma, "cross_attn.q_rms_norm.gamma", error) ||
         !require_pointer(weights.cross_k_gamma, "cross_attn.k_rms_norm.gamma", error))) return false;
    if (config.use_projection &&
        (!require_pointer(weights.proj_weight, "cross_attn.proj_linear.weight", error) ||
         !require_pointer(weights.proj_bias, "cross_attn.proj_linear.bias", error))) return false;

    SparseTensorF32 normalized;
    if (!sparse_layer_norm(input, config.norm_eps, nullptr, nullptr, normalized, error)) return false;
    SparseTensorF32 modulated;
    if (!affine_modulate(normalized, weights.modulation, timestep_modulation,
                         config.channels, 0, modulated, error)) return false;
    SparseTensorF32 qkv;
    if (!sparse_linear(modulated, weights.self_to_qkv_weight, weights.self_to_qkv_bias,
                       3 * config.channels, qkv, error)) return false;
    SparseTensorF32 q = sparse_slice(qkv, 0, config.channels);
    SparseTensorF32 k = sparse_slice(qkv, config.channels, config.channels);
    SparseTensorF32 v = sparse_slice(qkv, 2 * config.channels, config.channels);
    if (config.qk_rms_norm) {
        SparseTensorF32 normalized_q, normalized_k;
        if (!sparse_multihead_rms_norm(q, config.num_heads, head_dim,
                                       weights.self_q_gamma, normalized_q, error) ||
            !sparse_multihead_rms_norm(k, config.num_heads, head_dim,
                                       weights.self_k_gamma, normalized_k, error)) return false;
        q = std::move(normalized_q);
        k = std::move(normalized_k);
    }
    if (config.use_rope) {
        SparseTensorF32 rotated_q, rotated_k;
        if (!sparse_rotary_position_embedding(q, config.num_heads, head_dim,
                                              config.rope_freq_min, config.rope_freq_base,
                                              rotated_q, error) ||
            !sparse_rotary_position_embedding(k, config.num_heads, head_dim,
                                              config.rope_freq_min, config.rope_freq_base,
                                              rotated_k, error)) return false;
        q = std::move(rotated_q);
        k = std::move(rotated_k);
    }
    SparseTensorF32 self_attention;
    if (!sparse_scaled_dot_product_attention(q, k, v, config.num_heads, head_dim,
                                             1.0f / std::sqrt(static_cast<float>(head_dim)),
                                             self_attention, error)) return false;
    SparseTensorF32 self_out;
    if (!sparse_linear(self_attention, weights.self_to_out_weight, weights.self_to_out_bias,
                       config.channels, self_out, error) ||
                       !gate_in_place(self_out, weights.modulation, timestep_modulation,
                       config.channels, 2 * config.channels, error)) return false;
    output = input;
    if (!add_in_place(output, self_out, error)) return false;

    SparseTensorF32 cross_input;
    if (!sparse_layer_norm(output, config.norm_eps, weights.norm2_weight,
                           weights.norm2_bias, cross_input, error)) return false;
    SparseTensorF32 cross_q;
    if (!sparse_linear(cross_input, weights.cross_to_q_weight, weights.cross_to_q_bias,
                       config.channels, cross_q, error)) return false;
    if (config.qk_rms_norm_cross) {
        SparseTensorF32 normalized_q;
        if (!sparse_multihead_rms_norm(cross_q, config.num_heads, head_dim,
                                       weights.cross_q_gamma, normalized_q, error)) return false;
        cross_q = std::move(normalized_q);
    }
    VarLenTensorF32 cross_kv;
    if (!varlen_linear(global_context, weights.cross_to_kv_weight,
                       weights.cross_to_kv_bias, 2 * config.channels,
                       cross_kv, error)) return false;
    VarLenTensorF32 cross_k = varlen_slice(cross_kv, 0, config.channels);
    VarLenTensorF32 cross_v = varlen_slice(cross_kv, config.channels, config.channels);
    if (config.qk_rms_norm_cross) {
        VarLenTensorF32 normalized_k;
        if (!varlen_multihead_rms_norm(cross_k, config.num_heads, head_dim,
                                       weights.cross_k_gamma, normalized_k, error)) return false;
        cross_k = std::move(normalized_k);
    }
    SparseTensorF32 cross_attention;
    if (!sparse_cross_attention(cross_q, cross_k, cross_v, config.num_heads,
                                head_dim, 1.0f / std::sqrt(static_cast<float>(head_dim)),
                                cross_attention, error)) return false;
    SparseTensorF32 cross_out;
    if (!sparse_linear(cross_attention, weights.cross_to_out_weight,
                       weights.cross_to_out_bias, config.channels,
                       cross_out, error)) return false;
    SparseTensorF32 projected_out;
    if (config.use_projection) {
        if (!sparse_linear(*projected_context, weights.proj_weight, weights.proj_bias,
                           config.channels, projected_out, error)) return false;
        if (!add_in_place(cross_out, projected_out, error)) return false;
    }
    if (!add_in_place(output, cross_out, error)) return false;

    SparseTensorF32 mlp_input;
    if (!sparse_layer_norm(output, config.norm_eps, nullptr, nullptr, normalized, error) ||
        !affine_modulate(normalized, weights.modulation, timestep_modulation,
                         config.channels, 3 * config.channels, mlp_input, error)) return false;
    SparseTensorF32 mlp_hidden;
    if (!sparse_linear(mlp_input, weights.mlp0_weight, weights.mlp0_bias,
                       config.mlp_hidden, mlp_hidden, error)) return false;
    gelu_tanh_in_place(mlp_hidden);
    SparseTensorF32 mlp_out;
    if (!sparse_linear(mlp_hidden, weights.mlp2_weight, weights.mlp2_bias,
                       config.channels, mlp_out, error) ||
        !gate_in_place(mlp_out, weights.modulation, timestep_modulation,
                       config.channels, 5 * config.channels, error) ||
                       !add_in_place(output, mlp_out, error)) return false;
    return true;
}

} // namespace pixal3d
