#include "pixal3d/slat_decoder.h"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr int kLatentChannels = 3;
constexpr int kOutChannels = 3;
constexpr int kLevel0Channels = 8;
constexpr int kLevel1Channels = 4;
constexpr int kMlp0Hidden = 32;

float stable_value(const std::string & name, std::size_t index) {
    std::size_t stable = 0;
    for (std::size_t i = 0; i < name.size(); ++i) stable += (i + 1) * name[i];
    float value = (static_cast<float>((index + stable) % 29) - 14.0f) * 0.018f;
    if (name.size() >= 11 && name.compare(name.size() - 11, 11, "norm.weight") == 0) {
        value += 1.0f;
    }
    return value;
}

std::vector<float> parameter(const std::string & name, std::size_t count) {
    std::vector<float> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        result[index] = stable_value(name, index);
    }
    return result;
}

std::vector<float> conv_parameter(const std::string & name, int out_channels,
                                  int in_channels) {
    std::vector<float> result(static_cast<std::size_t>(27) * out_channels * in_channels);
    for (int kd = 0; kd < 3; ++kd) {
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                const int kernel = kd * 9 + kh * 3 + kw;
                for (int out = 0; out < out_channels; ++out) {
                    for (int in = 0; in < in_channels; ++in) {
                        const std::size_t source_index =
                            ((static_cast<std::size_t>(out) * 3 + kd) * 3 + kh) * 3 *
                                static_cast<std::size_t>(in_channels) +
                            static_cast<std::size_t>(kw) * in_channels + in;
                        result[(static_cast<std::size_t>(kernel) * out_channels + out) *
                                   in_channels + in] = stable_value(name, source_index);
                    }
                }
            }
        }
    }
    return result;
}

void emit(const char * name, const std::vector<float> & values) {
    std::cout << name << " " << values.size();
    for (float value : values) std::cout << " " << value;
    std::cout << "\n";
}

void bind_convnext(pixal3d::SLatDecoderBlockWeightsF32 & weights,
                   const std::string & prefix, int channels, int mlp_hidden,
                   std::map<std::string, std::vector<float>> & params) {
    weights.kind = pixal3d::SLatDecoderBlockKind::convnext;
    weights.channels = channels;
    weights.out_channels = channels;
    weights.mlp_hidden = mlp_hidden;
    params.emplace(prefix + ".norm.weight", parameter(prefix + ".norm.weight", channels));
    params.emplace(prefix + ".norm.bias", parameter(prefix + ".norm.bias", channels));
    params.emplace(prefix + ".conv.weight", conv_parameter(prefix + ".conv.weight", channels, channels));
    params.emplace(prefix + ".conv.bias", parameter(prefix + ".conv.bias", channels));
    params.emplace(prefix + ".mlp.0.weight", parameter(prefix + ".mlp.0.weight",
                                                         static_cast<std::size_t>(mlp_hidden) * channels));
    params.emplace(prefix + ".mlp.0.bias", parameter(prefix + ".mlp.0.bias", mlp_hidden));
    params.emplace(prefix + ".mlp.2.weight", parameter(prefix + ".mlp.2.weight",
                                                         static_cast<std::size_t>(channels) * mlp_hidden));
    params.emplace(prefix + ".mlp.2.bias", parameter(prefix + ".mlp.2.bias", channels));
    weights.norm_weight = params.at(prefix + ".norm.weight").data();
    weights.norm_bias = params.at(prefix + ".norm.bias").data();
    weights.conv1_weight = params.at(prefix + ".conv.weight").data();
    weights.conv1_bias = params.at(prefix + ".conv.bias").data();
    weights.mlp0_weight = params.at(prefix + ".mlp.0.weight").data();
    weights.mlp0_bias = params.at(prefix + ".mlp.0.bias").data();
    weights.mlp2_weight = params.at(prefix + ".mlp.2.weight").data();
    weights.mlp2_bias = params.at(prefix + ".mlp.2.bias").data();
}

void bind_c2s(pixal3d::SLatDecoderBlockWeightsF32 & weights,
              const std::string & prefix, int channels, int out_channels,
              std::map<std::string, std::vector<float>> & params) {
    weights.kind = pixal3d::SLatDecoderBlockKind::channel_to_spatial;
    weights.channels = channels;
    weights.out_channels = out_channels;
    params.emplace(prefix + ".norm1.weight", parameter(prefix + ".norm1.weight", channels));
    params.emplace(prefix + ".norm1.bias", parameter(prefix + ".norm1.bias", channels));
    params.emplace(prefix + ".conv1.weight", conv_parameter(prefix + ".conv1.weight", out_channels * 8, channels));
    params.emplace(prefix + ".conv1.bias", parameter(prefix + ".conv1.bias", out_channels * 8));
    params.emplace(prefix + ".conv2.weight", conv_parameter(prefix + ".conv2.weight", out_channels, out_channels));
    params.emplace(prefix + ".conv2.bias", parameter(prefix + ".conv2.bias", out_channels));
    params.emplace(prefix + ".to_subdiv.weight", parameter(prefix + ".to_subdiv.weight",
                                                             static_cast<std::size_t>(8) * channels));
    params.emplace(prefix + ".to_subdiv.bias", parameter(prefix + ".to_subdiv.bias", 8));
    weights.norm1_weight = params.at(prefix + ".norm1.weight").data();
    weights.norm1_bias = params.at(prefix + ".norm1.bias").data();
    weights.conv1_weight = params.at(prefix + ".conv1.weight").data();
    weights.conv1_bias = params.at(prefix + ".conv1.bias").data();
    weights.conv2_weight = params.at(prefix + ".conv2.weight").data();
    weights.conv2_bias = params.at(prefix + ".conv2.bias").data();
    weights.to_subdiv_weight = params.at(prefix + ".to_subdiv.weight").data();
    weights.to_subdiv_bias = params.at(prefix + ".to_subdiv.bias").data();
}

} // namespace

int main(int argc, char ** argv) {
    if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--upsample")) {
        std::cerr << "usage: pixal3d_slat_decoder_fixture [--upsample]\n";
        return 2;
    }
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    std::map<std::string, std::vector<float>> params;
    auto add = [&](const std::string & name, std::size_t count) -> const float * {
        return params.emplace(name, parameter(name, count)).first->second.data();
    };

    pixal3d::SLatDecoderWeightsF32 weights;
    weights.from_latent_weight = add("from_latent.weight",
                                     static_cast<std::size_t>(kLevel0Channels) * kLatentChannels);
    weights.from_latent_bias = add("from_latent.bias", kLevel0Channels);
    weights.output_weight = add("output_layer.weight",
                                static_cast<std::size_t>(kOutChannels) * kLevel1Channels);
    weights.output_bias = add("output_layer.bias", kOutChannels);
    weights.blocks.resize(3);
    bind_convnext(weights.blocks[0], "blocks.0.0", kLevel0Channels, kMlp0Hidden, params);
    bind_c2s(weights.blocks[1], "blocks.0.1", kLevel0Channels, kLevel1Channels, params);
    bind_convnext(weights.blocks[2], "blocks.1.0", kLevel1Channels, 16, params);

    pixal3d::SLatDecoderConfig config;
    config.latent_channels = kLatentChannels;
    config.out_channels = kOutChannels;
    config.norm_eps = 1e-6f;
    config.pred_subdiv = true;
    config.model_channels = {kLevel0Channels, kLevel1Channels};
    config.num_blocks = {1, 1};

    pixal3d::SparseTensorF32 input;
    input.batch_size = 1;
    input.channels = kLatentChannels;
    input.spatial_x = input.spatial_y = input.spatial_z = 2;
    input.coords = {0, 0, 0, 0, 0, 1, 0, 1};
    input.feats.resize(2 * kLatentChannels);
    for (std::size_t index = 0; index < input.feats.size(); ++index) {
        input.feats[index] = -0.21f + 0.037f * static_cast<float>(index);
    }

    pixal3d::SparseTensorF32 output;
    std::vector<pixal3d::SparseTensorF32> subdivisions;
    std::string error;
    if (!pixal3d::slat_decoder_forward_f32(input, config, weights, nullptr,
                                           output, &subdivisions, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    emit("slat_decoder_coords",
         std::vector<float>(output.coords.begin(), output.coords.end()));
    emit("slat_decoder_output", output.feats);
    if (subdivisions.size() != 1) {
        std::cerr << "unexpected subdivision count\n";
        return 1;
    }
    emit("slat_decoder_subdiv_coords",
         std::vector<float>(subdivisions[0].coords.begin(), subdivisions[0].coords.end()));
    emit("slat_decoder_subdiv_output", subdivisions[0].feats);
    if (argc == 2) {
        pixal3d::SparseTensorF32 upsampled;
        if (!pixal3d::slat_decoder_upsample_coords_f32(
                input, config, weights, 1, upsampled, &error)) {
            std::cerr << error << "\n";
            return 1;
        }
        emit("slat_decoder_upsample_coords",
             std::vector<float>(upsampled.coords.begin(), upsampled.coords.end()));
    }
    return 0;
}
