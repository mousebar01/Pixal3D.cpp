#include "pixal3d/ss_decoder.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void emit(const char * name, const std::vector<float> & values) {
    std::cout << name << " " << values.size();
    for (float value : values) std::cout << " " << value;
    std::cout << "\n";
}

void emit_coords(const char * name, const std::vector<std::int32_t> & coords) {
    std::cout << name << " " << coords.size();
    for (std::int32_t value : coords) std::cout << " " << value;
    std::cout << "\n";
}

std::vector<float> values(std::size_t count, float offset, float step) {
    std::vector<float> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        result[i] = offset + step * static_cast<float>(i);
    }
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: " << argv[0] << " <ss-decoder-test.gguf> [--coords]\n";
        return 2;
    }
    const bool run_coords = argc == 3 && std::string(argv[2]) == "--coords";
    if (argc == 3 && !run_coords) {
        std::cerr << "unknown fixture mode: " << argv[2] << "\n";
        return 2;
    }
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);

    pixal3d::SSDecoderModel model;
    std::string error;
    if (!model.load(argv[1], true, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    const pixal3d::SSDecoderHParams & hp = model.hparams();
    const std::size_t input_points = static_cast<std::size_t>(hp.resolution) *
                                     static_cast<std::size_t>(hp.resolution) *
                                     static_cast<std::size_t>(hp.resolution);
    const std::size_t output_points = static_cast<std::size_t>(hp.output_resolution()) *
                                      static_cast<std::size_t>(hp.output_resolution()) *
                                      static_cast<std::size_t>(hp.output_resolution());
    const std::vector<float> latent = values(
        static_cast<std::size_t>(hp.latent_channels) * input_points, -0.21f, 0.037f);
    std::vector<float> output(static_cast<std::size_t>(hp.out_channels) * output_points,
                              0.0f);
    if (!model.decode(latent.data(), output.data(), &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    std::cout << "backend " << model.backend_name() << "\n";
    emit("ss_decoder_output", output);
    if (run_coords) {
        std::vector<std::int32_t> coords;
        if (!model.decode_coords(latent.data(), 0.0f, hp.output_resolution(),
                                 coords, &error)) {
            std::cerr << error << "\n";
            return 1;
        }
        emit_coords("ss_decoder_coords", coords);
    }
    return 0;
}
