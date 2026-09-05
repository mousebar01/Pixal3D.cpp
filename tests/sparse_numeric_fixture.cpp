#include "pixal3d/sparse.h"

#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kInChannels = 3;
constexpr int kOutChannels = 4;
constexpr int kKernelElements = 27;

void emit(const char * name, const std::vector<float> & values) {
    std::cout << name << " " << values.size();
    for (float value : values) std::cout << " " << value;
    std::cout << "\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    const std::vector<std::int32_t> coords = {
        0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 1, 0,  0, 1, 1, 1,  0, 2, 2, 2,
        1, 0, 0, 0,  1, 0, 1, 0,  1, 1, 1, 1,  1, 2, 1, 2,  1, 3, 3, 3,
    };
    const std::size_t points = coords.size() / 4;
    std::vector<float> features(points * kInChannels);
    for (std::size_t point = 0; point < points; ++point) {
        for (int channel = 0; channel < kInChannels; ++channel) {
            features[point * kInChannels + static_cast<std::size_t>(channel)] =
                -0.31f + 0.047f * static_cast<float>(point * kInChannels + channel);
        }
    }
    std::vector<float> linear_weight(kOutChannels * kInChannels);
    for (std::size_t index = 0; index < linear_weight.size(); ++index) {
        linear_weight[index] = -0.11f + 0.013f * static_cast<float>(index);
    }
    const std::vector<float> linear_bias = {-0.07f, 0.02f, 0.11f, 0.20f};
    std::vector<float> conv_weight(static_cast<std::size_t>(kKernelElements) *
                                   kOutChannels * kInChannels);
    for (std::size_t index = 0; index < conv_weight.size(); ++index) {
        conv_weight[index] = -0.021f + 0.0017f * static_cast<float>(index);
    }
    const std::vector<float> conv_bias = {-0.03f, 0.01f, 0.05f, 0.09f};

    pixal3d::SparseTensorF32 input;
    input.batch_size = 2;
    input.channels = kInChannels;
    input.spatial_x = 4;
    input.spatial_y = 4;
    input.spatial_z = 4;
    input.coords = coords;
    input.feats = features;
    std::string error;
    pixal3d::SparseTensorF32 linear_output;
    if (!pixal3d::sparse_linear(input, linear_weight.data(), linear_bias.data(),
                                kOutChannels, linear_output, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    pixal3d::SparseTensorF32 conv_output;
    if (!pixal3d::sparse_submanifold_conv3d(input, conv_weight.data(), conv_bias.data(),
                                            kOutChannels, conv_output, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    const std::vector<float> norm_gamma = {0.8f, 1.1f, 1.4f};
    const std::vector<float> norm_beta = {-0.2f, 0.03f, 0.17f};
    pixal3d::SparseTensorF32 norm_output;
    if (!pixal3d::sparse_layer_norm(input, 1e-5f, norm_gamma.data(), norm_beta.data(),
                                    norm_output, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    pixal3d::SparseTensorF32 downsample_output;
    if (!pixal3d::sparse_downsample_mean(input, 2, downsample_output, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    pixal3d::SparseTensorF32 attention_query = input;
    pixal3d::SparseTensorF32 key = input;
    pixal3d::SparseTensorF32 value = input;
    attention_query.channels = key.channels = value.channels = 12;
    attention_query.feats.resize(points * 12);
    key.feats.resize(points * 12);
    value.feats.resize(points * 12);
    for (std::size_t index = 0; index < attention_query.feats.size(); ++index) {
        attention_query.feats[index] = -0.23f + 0.017f * static_cast<float>(index);
        key.feats[index] = 0.17f - 0.021f * static_cast<float>(index);
        value.feats[index] = -0.13f + 0.031f * static_cast<float>(index);
    }
    pixal3d::SparseTensorF32 attention_output;
    if (!pixal3d::sparse_scaled_dot_product_attention(
            attention_query, key, value, 2, 6, 0.37f, attention_output, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    pixal3d::SparseTensorF32 rope_output;
    if (!pixal3d::sparse_rotary_position_embedding(
            attention_query, 2, 6, 1.0f, 10000.0f, rope_output, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    std::vector<float> rms_gamma(12);
    for (std::size_t index = 0; index < rms_gamma.size(); ++index) {
        rms_gamma[index] = 0.7f + 0.013f * static_cast<float>(index);
    }
    pixal3d::SparseTensorF32 rms_output;
    if (!pixal3d::sparse_multihead_rms_norm(
            attention_query, 2, 6, rms_gamma.data(), rms_output, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    pixal3d::VarLenTensorF32 context_key;
    context_key.batch_size = 2;
    context_key.channels = 12;
    context_key.offsets = {0, 3, 5};
    context_key.feats.resize(5 * 12);
    pixal3d::VarLenTensorF32 context_value = context_key;
    for (std::size_t index = 0; index < context_key.feats.size(); ++index) {
        context_key.feats[index] = 0.09f - 0.014f * static_cast<float>(index);
        context_value.feats[index] = -0.18f + 0.022f * static_cast<float>(index);
    }
    pixal3d::SparseTensorF32 cross_output;
    if (!pixal3d::sparse_cross_attention(
            attention_query, context_key, context_value, 2, 6, 0.29f,
            cross_output, &error)) {
        std::cerr << error << "\n";
        return 1;
    }

    pixal3d::SparseTensorF32 packed;
    packed.batch_size = 2;
    packed.channels = 16;
    packed.spatial_x = packed.spatial_y = packed.spatial_z = 2;
    packed.coords = {0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0};
    packed.feats.resize(3 * 16);
    for (std::size_t index = 0; index < packed.feats.size(); ++index) {
        packed.feats[index] = -0.17f + 0.011f * static_cast<float>(index);
    }
    pixal3d::SparseTensorF32 subdivision = packed;
    subdivision.channels = 8;
    subdivision.feats = {
        1, 0, 1, 0, 0, 1, 0, 0,
        0, 1, 1, 0, 1, 0, 0, 1,
        1, 1, 0, 0, 0, 0, 1, 0,
    };
    pixal3d::SparseTensorF32 spatial_output;
    if (!pixal3d::sparse_channel_to_spatial(
            packed, 2, &subdivision, spatial_output, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    emit("sparse_linear_output", linear_output.feats);
    emit("sparse_conv_output", conv_output.feats);
    emit("sparse_norm_output", norm_output.feats);
    emit("sparse_downsample_coords", std::vector<float>(downsample_output.coords.begin(),
                                                          downsample_output.coords.end()));
    emit("sparse_downsample_output", downsample_output.feats);
    emit("sparse_attention_output", attention_output.feats);
    emit("sparse_rope_output", rope_output.feats);
    emit("sparse_rms_output", rms_output.feats);
    emit("sparse_cross_output", cross_output.feats);
    emit("sparse_channel_to_spatial_coords",
         std::vector<float>(spatial_output.coords.begin(), spatial_output.coords.end()));
    emit("sparse_channel_to_spatial_output", spatial_output.feats);
    return 0;
}
