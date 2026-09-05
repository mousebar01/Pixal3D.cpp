#include "pixal3d/sparse_transformer.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kChannels = 12;
constexpr int kHeads = 2;
constexpr int kHeadDim = 6;
constexpr int kMlpHidden = 18;
constexpr int kContextChannels = 5;
constexpr int kProjectionChannels = 7;
constexpr int kPoints = 10;

std::vector<std::int32_t> coords() {
    return {
        0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 1, 0,  0, 1, 1, 1,  0, 2, 2, 2,
        1, 0, 0, 0,  1, 0, 1, 0,  1, 1, 1, 1,  1, 2, 1, 2,  1, 3, 3, 3,
    };
}

std::vector<float> sequence(std::size_t count, float offset, float step) {
    std::vector<float> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        result[index] = offset + step * static_cast<float>(index);
    }
    return result;
}

float stable_value(const std::string & name, std::size_t index) {
    std::size_t stable = 0;
    for (std::size_t i = 0; i < name.size(); ++i) stable += (i + 1) * name[i];
    float value = (static_cast<float>((index + stable) % 29) - 14.0f) * 0.018f;
    if (name.size() >= 5 && name.compare(name.size() - 5, 5, "gamma") == 0) {
        value = std::fabs(value) + 0.5f;
    }
    return value;
}

std::vector<float> parameter(const std::string & name, std::size_t count) {
    std::vector<float> result(count);
    for (std::size_t index = 0; index < count; ++index) result[index] = stable_value(name, index);
    return result;
}

void emit(const char * name, const std::vector<float> & values) {
    std::cout << name << " " << values.size();
    for (float value : values) std::cout << " " << value;
    std::cout << "\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    std::unordered_map<std::string, std::vector<float>> params;
    auto add = [&](const char * name, std::size_t count) -> const float * {
        auto inserted = params.emplace(name, parameter(name, count));
        return inserted.first->second.data();
    };

    pixal3d::SparseTransformerBlockWeightsF32 weights;
    weights.modulation = add("modulation", 6 * kChannels);
    weights.self_to_qkv_weight = add("self_attn.to_qkv.weight", 3 * kChannels * kChannels);
    weights.self_to_qkv_bias = add("self_attn.to_qkv.bias", 3 * kChannels);
    weights.self_q_gamma = add("self_attn.q_rms_norm.gamma", kChannels);
    weights.self_k_gamma = add("self_attn.k_rms_norm.gamma", kChannels);
    weights.self_to_out_weight = add("self_attn.to_out.weight", kChannels * kChannels);
    weights.self_to_out_bias = add("self_attn.to_out.bias", kChannels);
    weights.norm2_weight = add("norm2.weight", kChannels);
    weights.norm2_bias = add("norm2.bias", kChannels);
    weights.cross_to_q_weight = add("cross_attn.cross_attn_block.to_q.weight", kChannels * kChannels);
    weights.cross_to_q_bias = add("cross_attn.cross_attn_block.to_q.bias", kChannels);
    weights.cross_to_kv_weight = add("cross_attn.cross_attn_block.to_kv.weight", 2 * kChannels * kContextChannels);
    weights.cross_to_kv_bias = add("cross_attn.cross_attn_block.to_kv.bias", 2 * kChannels);
    weights.cross_q_gamma = add("cross_attn.cross_attn_block.q_rms_norm.gamma", kChannels);
    weights.cross_k_gamma = add("cross_attn.cross_attn_block.k_rms_norm.gamma", kChannels);
    weights.cross_to_out_weight = add("cross_attn.cross_attn_block.to_out.weight", kChannels * kChannels);
    weights.cross_to_out_bias = add("cross_attn.cross_attn_block.to_out.bias", kChannels);
    weights.proj_weight = add("cross_attn.proj_linear.weight", kChannels * kProjectionChannels);
    weights.proj_bias = add("cross_attn.proj_linear.bias", kChannels);
    weights.mlp0_weight = add("mlp.mlp.0.weight", kMlpHidden * kChannels);
    weights.mlp0_bias = add("mlp.mlp.0.bias", kMlpHidden);
    weights.mlp2_weight = add("mlp.mlp.2.weight", kChannels * kMlpHidden);
    weights.mlp2_bias = add("mlp.mlp.2.bias", kChannels);

    pixal3d::SparseTensorF32 input;
    input.batch_size = 2;
    input.channels = kChannels;
    input.spatial_x = input.spatial_y = input.spatial_z = 4;
    input.coords = coords();
    input.feats = sequence(kPoints * kChannels, -0.27f, 0.013f);

    pixal3d::SparseTensorF32 projected;
    projected.batch_size = 2;
    projected.channels = kProjectionChannels;
    projected.spatial_x = projected.spatial_y = projected.spatial_z = 4;
    projected.coords = input.coords;
    projected.feats = sequence(kPoints * kProjectionChannels, 0.19f, -0.017f);

    pixal3d::VarLenTensorF32 global;
    global.batch_size = 2;
    global.channels = kContextChannels;
    global.offsets = {0, 3, 5};
    global.feats = sequence(5 * kContextChannels, -0.11f, 0.021f);
    const std::vector<float> timestep_mod = sequence(2 * 6 * kChannels, 0.07f, -0.009f);

    pixal3d::SparseTransformerBlockConfig config;
    config.channels = kChannels;
    config.num_heads = kHeads;
    config.mlp_hidden = kMlpHidden;
    config.context_channels = kContextChannels;
    config.proj_in_channels = kProjectionChannels;
    config.use_rope = true;
    config.qk_rms_norm = true;
    config.qk_rms_norm_cross = true;
    config.use_projection = true;

    pixal3d::SparseTensorF32 output;
    std::string error;
    if (!pixal3d::sparse_transformer_cross_block_f32(
            input, timestep_mod.data(), global, &projected, config, weights,
            output, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    emit("sparse_transformer_output", output.feats);
    return 0;
}
