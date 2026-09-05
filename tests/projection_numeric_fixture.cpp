#include "pixal3d/projection.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {

std::vector<float> make_feature_map(float base) {
    std::vector<float> values(5 * 4 * 3);
    for (std::size_t i = 0; i < values.size(); ++i) {
        // Powers-of-two fractions keep the fixture bit-stable in Python and
        // C++ while still exercising non-constant bilinear samples.
        values[i] = base + 0.125f * static_cast<float>(i) +
                    0.03125f * static_cast<float>(i % 7);
    }
    return values;
}

void emit_floats(const char * name, const std::vector<float> & values) {
    std::cout << name << " " << values.size();
    for (float value : values) {
        std::cout << " " << value;
    }
    std::cout << "\n";
}

void emit_bytes(const char * name, const std::vector<std::uint8_t> & values) {
    std::cout << name << " " << values.size();
    for (std::uint8_t value : values) {
        std::cout << " " << static_cast<unsigned int>(value);
    }
    std::cout << "\n";
}

} // namespace

int main() {
    using pixal3d::ProjectionCamera;
    using pixal3d::ProjectionGridOptions;
    using pixal3d::ProjectionView;
    using pixal3d::ProjectedGrid;

    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);

    ProjectionGridOptions options;
    options.grid_resolution = 4;
    options.image_resolution = 16;

    const std::vector<float> map_a = make_feature_map(-0.5f);
    const std::vector<float> map_b = make_feature_map(0.75f);
    const ProjectionCamera camera_a = ProjectionCamera::front(0.8f, 2.25f, 1.5f);

    ProjectedGrid single;
    std::string error;
    if (!pixal3d::project_grid_features(
            map_a.data(), 5, 4, 3, options, camera_a, single, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    emit_floats("single_features", single.features);
    emit_bytes("single_valid", single.valid);

    auto translated = camera_a;
    translated.has_transform = true;
    translated.transform_matrix = {{
        1.0f, 0.0f, 0.0f, 0.5f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    }};

    const ProjectionView view_a{map_a.data(), 5, 4, 3, camera_a};
    const ProjectionView view_b{map_b.data(), 5, 4, 3, translated};
    ProjectedGrid average;
    if (!pixal3d::project_grid_features_average(
            {view_a, view_b}, options, average, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    emit_floats("average_features", average.features);
    emit_bytes("average_valid", average.valid);

    const std::array<float, 16> identity = {{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    }};
    std::vector<std::array<float, 16>> relative;
    if (!pixal3d::compute_relative_calc_matrices(
            {identity, translated.transform_matrix}, {2.25f, 2.25f},
            relative, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    std::vector<float> relative_flat;
    for (const auto & matrix : relative) {
        relative_flat.insert(relative_flat.end(), matrix.begin(), matrix.end());
    }
    emit_floats("relative_matrices", relative_flat);
    return 0;
}
