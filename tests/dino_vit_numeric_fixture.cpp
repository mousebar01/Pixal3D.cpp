#include "pixal3d/dino_vit.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: dino_vit_numeric_fixture <dino.gguf> [image_size]\n";
        return 2;
    }
    pixal3d::DinoV3Model model;
    std::string error;
    if (!model.load(argv[1], true, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    const auto & hp = model.hparams();
    const int image_size = argc == 3 ? std::atoi(argv[2]) : 4;
    if (hp.num_channels != 3 || image_size <= 0 || image_size % hp.patch_size != 0) {
        std::cerr << "fixture expects a 3-channel image divisible by patch_size\n";
        return 1;
    }
    const std::size_t pixels_count = static_cast<std::size_t>(hp.num_channels) *
                                     image_size * image_size;
    std::vector<float> pixels(pixels_count);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = 0.15f * std::sin(static_cast<float>(i + 1)) +
                    0.01f * static_cast<float>(i);
    }
    pixal3d::DinoV3FeaturesF32 features;
    if (!model.encode(pixels.data(), image_size, image_size, features, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    std::cout << std::setprecision(9) << "global " << features.global.feats.size() << "\n";
    for (float value : features.global.feats) std::cout << value << ' ';
    std::cout << "\npatch " << features.patch_map.features.size() << "\n";
    for (float value : features.patch_map.features) std::cout << value << ' ';
    std::cout << "\n";
    return 0;
}
