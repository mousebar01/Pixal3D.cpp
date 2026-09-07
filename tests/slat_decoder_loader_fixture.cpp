#include "pixal3d/slat_decoder.h"

#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void emit(const std::string & name, const std::vector<float> & values) {
    std::cout << name << " " << values.size();
    for (float value : values) std::cout << " " << value;
    std::cout << "\n";
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--upsample")) {
        std::cerr << "usage: pixal3d_slat_decoder_loader_fixture <pack> [--upsample]\n";
        return 2;
    }
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    pixal3d::SLatDecoderModel model;
    std::string error;
    if (!model.load(argv[1], "shape_decoder", true, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    pixal3d::SparseTensorF32 input;
    input.batch_size = 1;
    input.channels = model.hparams().latent_channels;
    input.spatial_x = input.spatial_y = input.spatial_z = 2;
    input.coords = {0, 0, 0, 0, 0, 1, 0, 1};
    input.feats.resize(2 * static_cast<std::size_t>(input.channels));
    for (std::size_t index = 0; index < input.feats.size(); ++index) {
        input.feats[index] = -0.21f + 0.037f * static_cast<float>(index);
    }
    pixal3d::SparseTensorF32 output;
    std::vector<pixal3d::SparseTensorF32> subdivisions;
    if (!model.decode(input, nullptr, output, &subdivisions, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    if (subdivisions.size() != model.hparams().model_channels.size() - 1) {
        std::cerr << "unexpected subdivision count: " << subdivisions.size() << "\n";
        return 1;
    }
    emit("slat_decoder_input_coords",
         std::vector<float>(input.coords.begin(), input.coords.end()));
    emit("slat_decoder_coords",
         std::vector<float>(output.coords.begin(), output.coords.end()));
    emit("slat_decoder_output", output.feats);
    for (std::size_t level = 0; level < subdivisions.size(); ++level) {
        const std::string prefix = "slat_decoder_subdiv_" + std::to_string(level);
        emit(prefix + "_coords",
             std::vector<float>(subdivisions[level].coords.begin(), subdivisions[level].coords.end()));
        emit(prefix + "_output", subdivisions[level].feats);
        std::vector<float> active(subdivisions[level].feats.size(), 0.0f);
        for (std::size_t index = 0; index < active.size(); ++index) {
            active[index] = subdivisions[level].feats[index] > 0.0f ? 1.0f : 0.0f;
        }
        emit(prefix + "_active", active);
    }
    if (argc == 3) {
        for (int level = 0; level < static_cast<int>(model.hparams().model_channels.size()); ++level) {
            pixal3d::SparseTensorF32 upsampled;
            if (!model.upsample_coords(input, level, upsampled, &error)) {
                std::cerr << error << "\n";
                return 1;
            }
            emit("slat_decoder_upsample_coords_" + std::to_string(level),
                 std::vector<float>(upsampled.coords.begin(), upsampled.coords.end()));
        }
    }
    return 0;
}
