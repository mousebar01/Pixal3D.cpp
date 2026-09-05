#include "pixal3d/ss_decoder.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

int main() {
    constexpr int resolution = 4;
    std::vector<float> logits(static_cast<std::size_t>(resolution) * resolution * resolution,
                              -1.0f);
    auto at = [&](int x, int y, int z) -> float & {
        return logits[(static_cast<std::size_t>(x) * resolution +
                      static_cast<std::size_t>(y)) * resolution +
                     static_cast<std::size_t>(z)];
    };
    at(0, 0, 0) = 0.1f;
    at(1, 1, 1) = 0.2f;
    at(2, 3, 1) = 0.3f;
    at(3, 2, 2) = 0.4f;

    std::vector<std::int32_t> coords;
    std::string error;
    assert(pixal3d::ss_occupancy_to_coords_f32(
        logits.data(), 1, resolution, 0.0f, 2, coords, &error));
    const std::vector<std::int32_t> expected = {
        0, 0, 0, 0,
        0, 1, 1, 0,
        0, 1, 1, 1,
    };
    assert(coords == expected);

    assert(pixal3d::ss_occupancy_to_coords_f32(
        logits.data(), 1, resolution, 0.15f, 4, coords, &error));
    assert(coords.size() == 12);
    assert(coords[1] == 1 && coords[2] == 1 && coords[3] == 1);
    assert(coords[4] == 0 && coords[5] == 2 && coords[6] == 3 && coords[7] == 1);
    assert(coords[8] == 0 && coords[9] == 3 && coords[10] == 2 && coords[11] == 2);

    logits[0] = std::numeric_limits<float>::quiet_NaN();
    assert(!pixal3d::ss_occupancy_to_coords_f32(
        logits.data(), 1, resolution, 0.0f, 4, coords, &error));
    assert(!error.empty());
    return 0;
}
