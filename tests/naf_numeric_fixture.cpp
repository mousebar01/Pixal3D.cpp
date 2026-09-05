#include "pixal3d/naf.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: naf_numeric_fixture <naf.gguf>\n";
        return 2;
    }

    pixal3d::NafModel model;
    std::string error;
    if (!model.load(argv[1], true, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    const auto & hp = model.hparams();
    // Non-square/non-divisible source dimensions exercise reflect padding and
    // adaptive pooling, while the 2x output/feature ratio exercises the
    // nearest-exact LR sampling used by neighborhood attention.
    constexpr int image_height = 5;
    constexpr int image_width = 7;
    constexpr int low_height = 2;
    constexpr int low_width = 3;
    constexpr int output_height = 4;
    constexpr int output_width = 6;
    constexpr int low_channels = 8;
    if (hp.in_channels != 3 || low_channels % hp.heads_attn != 0) {
        std::cerr << "fixture expects in_channels=3 and compatible low channels\n";
        return 1;
    }
    std::vector<float> image(static_cast<std::size_t>(hp.in_channels) * image_height * image_width);
    for (std::size_t i = 0; i < image.size(); ++i) {
        image[i] = 0.35f + 0.2f * static_cast<float>((i * 13) % 17) / 16.0f;
    }
    std::vector<float> low(static_cast<std::size_t>(low_height) * low_width * low_channels);
    for (std::size_t i = 0; i < low.size(); ++i) {
        low[i] = -0.25f + 0.07f * static_cast<float>((i * 7) % 19);
    }
    pixal3d::NafOutputF32 output;
    if (!model.upsample(image.data(), image_height, image_width, low.data(), low_height,
                        low_width, low_channels, output_height, output_width, output,
                        &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    std::cout << std::setprecision(9) << "output " << output.features.size() << "\n";
    for (float value : output.features) std::cout << value << ' ';
    std::cout << "\n";
    return 0;
}
