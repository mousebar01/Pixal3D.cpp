#include "pixal3d/ss_flow.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr char kMagic[] = "PXSSREF1";

template <typename T>
bool read_exact(std::ifstream & file, T * destination, std::size_t bytes) {
    return static_cast<bool>(file.read(reinterpret_cast<char *>(destination),
                                       static_cast<std::streamsize>(bytes)));
}

bool read_floats(std::ifstream & file, std::vector<float> & values) {
    if (values.empty()) return true;
    return read_exact(file, values.data(), values.size() * sizeof(float));
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "usage: %s <pixal3d-base-flow-f32.gguf> "
                     "<PXSSREF1.bin> [relative_l2_tolerance]\n", argv[0]);
        return 2;
    }
    const double tolerance = argc == 4 ? std::atof(argv[3]) : 2e-3;
    std::ifstream file(argv[2], std::ios::binary);
    char magic[sizeof(kMagic) - 1] = {};
    if (!file || !read_exact(file, magic, sizeof(magic)) ||
        std::memcmp(magic, kMagic, sizeof(magic)) != 0) {
        std::fprintf(stderr, "error: invalid SS-flow reference fixture\n");
        return 1;
    }
    std::int32_t dimensions[6] = {};
    float timestep = 0.0f;
    if (!read_exact(file, dimensions, sizeof(dimensions)) ||
        !read_exact(file, &timestep, sizeof(timestep))) {
        std::fprintf(stderr, "error: truncated SS-flow reference header\n");
        return 1;
    }
    const int resolution = dimensions[0];
    const int in_channels = dimensions[1];
    const int out_channels = dimensions[2];
    const int cond_tokens = dimensions[3];
    const int cond_channels = dimensions[4];
    const int projected_channels = dimensions[5];
    if (resolution <= 0 || in_channels <= 0 || out_channels <= 0 ||
        cond_tokens <= 0 || cond_channels <= 0 || projected_channels <= 0) {
        std::fprintf(stderr, "error: invalid SS-flow reference dimensions\n");
        return 1;
    }
    const std::size_t points = static_cast<std::size_t>(resolution) *
                               static_cast<std::size_t>(resolution) *
                               static_cast<std::size_t>(resolution);
    std::vector<float> input(static_cast<std::size_t>(in_channels) * points);
    std::vector<float> cond(static_cast<std::size_t>(cond_tokens) * cond_channels);
    std::vector<float> projected(points * static_cast<std::size_t>(projected_channels));
    std::vector<float> reference(static_cast<std::size_t>(out_channels) * points);
    if (!read_floats(file, input) || !read_floats(file, cond) ||
        !read_floats(file, projected) || !read_floats(file, reference)) {
        std::fprintf(stderr, "error: truncated SS-flow reference payload\n");
        return 1;
    }

    pixal3d::SSFlowModel model;
    std::string error;
    if (!model.load(argv[1], true, &error)) {
        std::fprintf(stderr, "load error: %s\n", error.c_str());
        return 1;
    }
    const pixal3d::SSFlowHParams & hp = model.hparams();
    if (hp.resolution != resolution || hp.in_channels != in_channels ||
        hp.out_channels != out_channels || hp.cond_channels != cond_channels ||
        hp.proj_in_channels != projected_channels) {
        std::fprintf(stderr, "error: reference dimensions do not match GGUF metadata\n");
        return 1;
    }
    std::vector<float> actual(reference.size(), 0.0f);
    if (!model.forward(input.data(), timestep, cond.data(), cond_tokens,
                       cond_channels, projected.data(), static_cast<int>(points),
                       projected_channels, actual.data(), &error)) {
        std::fprintf(stderr, "forward error: %s\n", error.c_str());
        return 1;
    }
    double max_abs = 0.0;
    double squared_error = 0.0;
    double squared_reference = 0.0;
    std::size_t max_index = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) {
            std::fprintf(stderr, "error: non-finite C++ output at %zu\n", i);
            return 1;
        }
        const double difference = static_cast<double>(actual[i]) - reference[i];
        const double absolute = std::fabs(difference);
        if (absolute > max_abs) {
            max_abs = absolute;
            max_index = i;
        }
        squared_error += difference * difference;
        squared_reference += static_cast<double>(reference[i]) * reference[i];
    }
    const double relative_l2 = std::sqrt(squared_error) /
                               (std::sqrt(squared_reference) + 1e-30);
    std::printf("backend: %s\n", model.backend_name().c_str());
    std::printf("shape: [%d, %d, %d, %d, %d, %d], t=%.6g\n", resolution,
                in_channels, out_channels, cond_tokens, cond_channels,
                projected_channels, timestep);
    std::printf("max_abs_error: %.6e at %zu (cpp=%.7g ref=%.7g)\n", max_abs,
                max_index, actual[max_index], reference[max_index]);
    std::printf("relative_l2_error: %.6e (tol %.6e)\n", relative_l2, tolerance);
    if (relative_l2 > tolerance) {
        std::printf("RESULT: FAIL\n");
        return 1;
    }
    std::printf("RESULT: PASS\n");
    return 0;
}
