#include "pixal3d/slat_flow.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr int kInChannels = 12;
constexpr int kOutChannels = 12;
constexpr int kModelChannels = 12;
constexpr int kCondChannels = 5;
constexpr int kProjChannels = 7;
constexpr int kHeads = 2;
constexpr int kMlpHidden = 18;

std::vector<std::int32_t> coords() {
    return {
        0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 1, 0,  0, 1, 1, 1,  0, 2, 2, 2,
        1, 0, 0, 0,  1, 0, 1, 0,  1, 1, 1, 1,  1, 2, 1, 2,  1, 3, 3, 3,
    };
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
    std::map<std::string, std::vector<float>> params;
    auto add = [&](const std::string & name, std::size_t count) -> const float * {
        return params.emplace(name, parameter(name, count)).first->second.data();
    };
    pixal3d::SLatFlowWeightsF32 weights;
    weights.input_weight = add("input_layer.weight", kModelChannels * kInChannels);
    weights.input_bias = add("input_layer.bias", kModelChannels);
    weights.timestep_weight0 = add("t_embedder.mlp.0.weight", kModelChannels * 256);
    weights.timestep_bias0 = add("t_embedder.mlp.0.bias", kModelChannels);
    weights.timestep_weight2 = add("t_embedder.mlp.2.weight", kModelChannels * kModelChannels);
    weights.timestep_bias2 = add("t_embedder.mlp.2.bias", kModelChannels);
    weights.modulation_weight = add("adaLN_modulation.1.weight", 6 * kModelChannels * kModelChannels);
    weights.modulation_bias = add("adaLN_modulation.1.bias", 6 * kModelChannels);
    weights.output_weight = add("out_layer.weight", kOutChannels * kModelChannels);
    weights.output_bias = add("out_layer.bias", kOutChannels);

    pixal3d::SparseTransformerBlockWeightsF32 block;
    block.modulation = add("blocks.0.modulation", 6 * kModelChannels);
    block.self_to_qkv_weight = add("blocks.0.self_attn.to_qkv.weight", 3 * kModelChannels * kModelChannels);
    block.self_to_qkv_bias = add("blocks.0.self_attn.to_qkv.bias", 3 * kModelChannels);
    block.self_q_gamma = add("blocks.0.self_attn.q_rms_norm.gamma", kModelChannels);
    block.self_k_gamma = add("blocks.0.self_attn.k_rms_norm.gamma", kModelChannels);
    block.self_to_out_weight = add("blocks.0.self_attn.to_out.weight", kModelChannels * kModelChannels);
    block.self_to_out_bias = add("blocks.0.self_attn.to_out.bias", kModelChannels);
    block.norm2_weight = add("blocks.0.norm2.weight", kModelChannels);
    block.norm2_bias = add("blocks.0.norm2.bias", kModelChannels);
    block.cross_to_q_weight = add("blocks.0.cross_attn.cross_attn_block.to_q.weight", kModelChannels * kModelChannels);
    block.cross_to_q_bias = add("blocks.0.cross_attn.cross_attn_block.to_q.bias", kModelChannels);
    block.cross_to_kv_weight = add("blocks.0.cross_attn.cross_attn_block.to_kv.weight", 2 * kModelChannels * kCondChannels);
    block.cross_to_kv_bias = add("blocks.0.cross_attn.cross_attn_block.to_kv.bias", 2 * kModelChannels);
    block.cross_q_gamma = add("blocks.0.cross_attn.cross_attn_block.q_rms_norm.gamma", kModelChannels);
    block.cross_k_gamma = add("blocks.0.cross_attn.cross_attn_block.k_rms_norm.gamma", kModelChannels);
    block.cross_to_out_weight = add("blocks.0.cross_attn.cross_attn_block.to_out.weight", kModelChannels * kModelChannels);
    block.cross_to_out_bias = add("blocks.0.cross_attn.cross_attn_block.to_out.bias", kModelChannels);
    block.proj_weight = add("blocks.0.cross_attn.proj_linear.weight", kModelChannels * kProjChannels);
    block.proj_bias = add("blocks.0.cross_attn.proj_linear.bias", kModelChannels);
    block.mlp0_weight = add("blocks.0.mlp.mlp.0.weight", kMlpHidden * kModelChannels);
    block.mlp0_bias = add("blocks.0.mlp.mlp.0.bias", kMlpHidden);
    block.mlp2_weight = add("blocks.0.mlp.mlp.2.weight", kModelChannels * kMlpHidden);
    block.mlp2_bias = add("blocks.0.mlp.mlp.2.bias", kModelChannels);
    weights.blocks.push_back(block);

    pixal3d::SLatFlowHParams hp;
    hp.component = "shape_flow_512";
    hp.model_class = "ElasticSLatFlowModel";
    hp.resolution = 4;
    hp.in_channels = kInChannels;
    hp.out_channels = kOutChannels;
    hp.model_channels = kModelChannels;
    hp.cond_channels = kCondChannels;
    hp.num_blocks = 1;
    hp.num_heads = kHeads;
    hp.proj_in_channels = kProjChannels;
    hp.mlp_ratio = 1.5f;
    hp.pe_mode = "rope";
    hp.image_attn_mode = "proj";
    hp.share_mod = true;
    hp.qk_rms_norm = true;
    hp.qk_rms_norm_cross = true;

    pixal3d::SparseTensorF32 input;
    input.batch_size = 2;
    input.channels = kInChannels;
    input.spatial_x = input.spatial_y = input.spatial_z = 4;
    input.coords = coords();
    input.feats.resize(10 * kInChannels);
    for (std::size_t index = 0; index < input.feats.size(); ++index) {
        input.feats[index] = -0.27f + 0.013f * static_cast<float>(index);
    }
    pixal3d::SparseTensorF32 projected;
    projected.batch_size = 2;
    projected.channels = kProjChannels;
    projected.spatial_x = projected.spatial_y = projected.spatial_z = 4;
    projected.coords = input.coords;
    projected.feats.resize(10 * kProjChannels);
    for (std::size_t index = 0; index < projected.feats.size(); ++index) {
        projected.feats[index] = 0.19f - 0.017f * static_cast<float>(index);
    }
    pixal3d::VarLenTensorF32 global;
    global.batch_size = 2;
    global.channels = kCondChannels;
    global.offsets = {0, 3, 5};
    global.feats.resize(5 * kCondChannels);
    for (std::size_t index = 0; index < global.feats.size(); ++index) {
        global.feats[index] = -0.11f + 0.021f * static_cast<float>(index);
    }
    const std::vector<float> timesteps = {0.17f, -0.23f};
    pixal3d::SparseTensorF32 output;
    std::string error;
    if (!pixal3d::slat_flow_forward_f32(
            input, timesteps.data(), timesteps.size(), global, &projected,
            hp, weights, output, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    emit("slat_flow_coords", std::vector<float>(output.coords.begin(), output.coords.end()));
    emit("slat_flow_output", output.feats);

    pixal3d::FlowEulerSamplerConfig sampler;
    sampler.steps = 4;
    sampler.sigma_min = 0.031f;
    sampler.rescale_t = 2.7f;
    sampler.guidance_strength = 2.0f;
    sampler.guidance_rescale = 0.23f;
    sampler.guidance_interval_min = 0.34f;
    sampler.guidance_interval_max = 0.91f;
    sampler.record_trajectory = true;
    pixal3d::VarLenTensorF32 negative_global = global;
    std::fill(negative_global.feats.begin(), negative_global.feats.end(), 0.0f);
    pixal3d::SparseTensorF32 negative_projected = projected;
    std::fill(negative_projected.feats.begin(), negative_projected.feats.end(), 0.0f);
    pixal3d::FlowVelocityFn callback = [&](const pixal3d::SparseTensorF32 & state,
                                            float timestep, bool conditional,
                                            pixal3d::SparseTensorF32 & velocity,
                                            std::string * callback_error) {
        const pixal3d::VarLenTensorF32 & context = conditional ? global : negative_global;
        const pixal3d::SparseTensorF32 * condition = conditional
            ? &projected : &negative_projected;
        std::vector<float> batch_t(static_cast<std::size_t>(state.batch_size), timestep);
        return pixal3d::slat_flow_forward_f32(
            state, batch_t.data(), batch_t.size(), context, condition,
            hp, weights, velocity, callback_error);
    };
    pixal3d::FlowEulerSampleF32 sampled;
    if (!pixal3d::flow_euler_sample_f32(input, sampler, callback, sampled, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    emit("slat_flow_sample_direct", sampled.samples.feats);
    emit("slat_flow_sample_direct_x0_last", sampled.pred_x_0.back().feats);
    return 0;
}
