#include "pixal3d/moge_camera.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

// Deterministic synthetic scene matching the golden values produced with the
// reference moge.utils.geometry_torch.recover_focal_shift (torch solve):
// a pinhole view with focal 1.7 (half-diagonal units) and z shift 0.35.
constexpr int kWidth = 32;
constexpr int kHeight = 32;
constexpr double kGoldenFocal = 1.6999996901;
constexpr double kGoldenShift = 0.3499998748;
constexpr double kTolerance = 1.0e-4;

} // namespace

int main() {
    std::vector<float> points(static_cast<std::size_t>(kWidth) * kHeight * 3);
    std::vector<float> mask(static_cast<std::size_t>(kWidth) * kHeight);
    const double aspect = 1.0;
    const double span_x = aspect / std::sqrt(1.0 + aspect * aspect);
    const double span_y = 1.0 / std::sqrt(1.0 + aspect * aspect);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const double u = -span_x * (kWidth - 1) / kWidth +
                             2.0 * span_x * (kWidth - 1) / kWidth * x / (kWidth - 1);
            const double v = -span_y * (kHeight - 1) / kHeight +
                             2.0 * span_y * (kHeight - 1) / kHeight * y / (kHeight - 1);
            const double z = 1.25 + 0.3 * std::sin(0.7 * u) * std::cos(0.9 * v);
            const std::size_t index = static_cast<std::size_t>(y) * kWidth + x;
            points[index * 3 + 0] = static_cast<float>(u * (z + 0.35) / 1.7);
            points[index * 3 + 1] = static_cast<float>(v * (z + 0.35) / 1.7);
            points[index * 3 + 2] = static_cast<float>(z);
            mask[index] = (x >= 8 && x < 24 && y >= 8 && y < 24) ? 1.0f : 0.0f;
        }
    }

    pixal3d::MoGeCameraOptions options;
    options.recovery_resolution = 16;
    float focal = 0.0f;
    float shift = 0.0f;
    std::string error;
    if (!pixal3d::moge_recover_focal_shift_f32(
            points.data(), mask.data(), kWidth, kHeight, options,
            focal, shift, &error)) {
        std::fprintf(stderr, "moge_recover_focal_shift_f32 failed: %s\n", error.c_str());
        return 1;
    }
    if (std::fabs(focal - kGoldenFocal) > kTolerance ||
        std::fabs(shift - kGoldenShift) > kTolerance) {
        std::fprintf(stderr,
                     "focal recovery mismatch: focal=%.10f (golden %.10f) "
                     "shift=%.10f (golden %.10f)\n",
                     focal, kGoldenFocal, shift, kGoldenShift);
        return 1;
    }

    // v2.py conversion: fx = focal / 2 * sqrt(1 + a^2) / a (a = 1) and the
    // horizontal FOV from the width-normalized focal.
    const double fx = focal * 0.5 * std::sqrt(2.0);
    const double fov = 2.0 * std::atan(1.0 / (2.0 * fx));
    if (std::fabs(fov - 0.7883525132) > kTolerance) {
        std::fprintf(stderr, "fov mismatch: %.10f (golden 0.7883525132)\n", fov);
        return 1;
    }

    // inference.py distance_from_fov(): grid point (-0.5, 0, 0) projected to
    // the left image border; golden values from the reference formula.
    const double distance_cases[][3] = {
        {0.2, 1.0, 4.9833222116},
        {0.883420, 1.0, 1.0573703633},
        {0.883420, 2.0, 0.5286851817},
    };
    for (const auto & case_values : distance_cases) {
        const float distance = pixal3d::moge_distance_from_fov_f32(
            static_cast<float>(case_values[0]),
            static_cast<float>(case_values[1]), 512);
        if (std::fabs(distance - case_values[2]) > kTolerance) {
            std::fprintf(stderr,
                         "distance mismatch: fov=%.4f mesh_scale=%.1f -> %.10f "
                         "(golden %.10f)\n",
                         case_values[0], case_values[1], distance, case_values[2]);
            return 1;
        }
    }

    (void)pixal3d::moge_camera_supported();
    std::printf("moge_camera_test passed: focal=%.7f shift=%.7f\n", focal, shift);
    return 0;
}
