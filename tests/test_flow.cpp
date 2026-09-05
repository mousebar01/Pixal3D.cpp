#include "pixal3d/flow.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main() {
    const float t[] = {0.0f, 1.0f};
    std::vector<float> output;
    std::string error;
    require(pixal3d::timestep_embedding(t, 2, 4, 10000, output, &error), error);
    require(output.size() == 8, "timestep embedding shape");
    require(std::fabs(output[0] - 1.0f) < 1e-6f &&
                std::fabs(output[1] - 1.0f) < 1e-6f &&
                std::fabs(output[2]) < 1e-6f &&
                std::fabs(output[3]) < 1e-6f,
            "zero timestep is cos=1 and sin=0");
    require(!pixal3d::timestep_embedding(t, 2, 0, 10000, output, &error),
            "invalid dimension is rejected");
    require(!pixal3d::timestep_embedding(t, 2, 4, 0, output, &error),
            "invalid max period is rejected");

    const float rows[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                          -1.0f, 0.0f, 1.0f, 0.5f, 2.0f};
    std::vector<float> normalized;
    require(pixal3d::layer_norm_rows(rows, 2, 1, 5, 1e-6f, nullptr,
                                     nullptr, normalized, &error), error);
    require(normalized.size() == 10, "layer norm shape");
    require(std::fabs(normalized[2]) < 1e-5f, "layer norm centers the row");
    const float shift_scale[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                                 -0.1f, -0.2f, -0.3f, -0.4f, -0.5f};
    std::vector<float> modulated;
    require(pixal3d::adaln_modulate_rows(normalized.data(), 2, 1, 5,
                                         shift_scale, modulated, &error), error);
    std::cout << "flow tests: PASS\n";
    return EXIT_SUCCESS;
}
