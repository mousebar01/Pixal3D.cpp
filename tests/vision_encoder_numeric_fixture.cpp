#include "pixal3d/vision_condition.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_values(const char * label, const std::vector<float> & values) {
    std::cout << label << ' ' << values.size() << '\n';
    std::cout << std::setprecision(9);
    for (float value : values) std::cout << value << ' ';
    std::cout << '\n';
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: vision_encoder_numeric_fixture <dino.gguf> [naf.gguf] [image_size]\n";
        return 2;
    }
    pixal3d::DinoV3Model dino;
    pixal3d::NafModel naf;
    std::string error;
    if (!dino.load(argv[1], true, &error)) {
        std::cerr << error << '\n';
        return 1;
    }
    const bool use_naf = argc >= 3;
    if (use_naf && !naf.load(argv[2], true, &error)) {
        std::cerr << error << '\n';
        return 1;
    }
    const int image_size = argc == 4 ? std::atoi(argv[3]) : 4;
    if (image_size <= 0 || image_size % dino.hparams().patch_size != 0) {
        std::cerr << "image_size must be positive and divisible by the DINO patch size\n";
        return 1;
    }
    std::vector<float> image(static_cast<std::size_t>(3 * image_size * image_size));
    for (std::size_t index = 0; index < image.size(); ++index) {
        image[index] = 0.1f + 0.8f * static_cast<float>((index * 7) % 23) / 22.0f;
    }
    const pixal3d::Pixal3DVisionStageInputF32 input{
        "shape_512", image_size, use_naf ? image_size : 0,
        image.data(), image.size(),
    };
    pixal3d::Pixal3DConditionBundleF32 bundle;
    if (!pixal3d::encode_pixal3d_condition_bundle_f32(
            dino, use_naf ? &naf : nullptr, {input}, bundle, &error) ||
        bundle.stages.size() != 1) {
        std::cerr << error << '\n';
        return 1;
    }
    const auto & stage = bundle.stages.front();
    std::cout << "stage " << stage.name << ' ' << stage.dino_map.height << ' '
              << stage.dino_map.width << ' ' << stage.dino_map.channels << '\n';
    print_values("global", stage.global.feats);
    print_values("dino", stage.dino_map.features);
    if (stage.has_naf()) print_values("naf", stage.naf_map.features);
    return 0;
}
