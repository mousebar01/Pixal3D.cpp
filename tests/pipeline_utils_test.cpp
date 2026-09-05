#include "pixal3d/pipeline.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

int main() {
    pixal3d::SparseTensorF32 upsampled;
    upsampled.batch_size = 1;
    upsampled.channels = 2;
    upsampled.spatial_x = upsampled.spatial_y = upsampled.spatial_z = 512;
    // The first two rows intentionally quantize to the same 1024-grid cell.
    upsampled.coords = {
        0, 0, 0, 0,
        0, 1, 1, 1,
        0, 256, 256, 256,
        0, 511, 511, 511,
    };
    upsampled.feats.assign(upsampled.points() * 2, 0.0f);
    std::vector<std::int32_t> coords;
    int actual_resolution = 0;
    std::string error;
    assert(pixal3d::quantize_slat_coords_f32(
        upsampled, 512, 1024, 0, coords, actual_resolution, &error));
    assert(actual_resolution == 1024);
    const std::vector<std::int32_t> expected = {
        0, 0, 0, 0,
        0, 32, 32, 32,
        0, 63, 63, 63,
    };
    assert(coords == expected);

    coords.clear();
    actual_resolution = 0;
    assert(pixal3d::quantize_slat_coords_f32(
        upsampled, 512, 1536, 2, coords, actual_resolution, &error));
    // The token cap forces the same 128-step fallback used by the Python
    // cascade until its hard floor at 1024.
    assert(actual_resolution == 1024);
    assert(coords == expected);

    assert(!pixal3d::quantize_slat_coords_f32(
        upsampled, 512, 1000, 0, coords, actual_resolution, &error));
    assert(!error.empty());
    return 0;
}
