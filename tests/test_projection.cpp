#include "pixal3d/projection.h"
#include "pixal3d/sparse.h"

#include <cmath>
#include <cstdint>
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

void require_close(float actual, float expected, float tolerance,
                   const std::string & message) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << message << " actual=" << actual
                  << " expected=" << expected << "\n";
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main() {
    using pixal3d::ProjectionCamera;
    using pixal3d::ProjectionGridOptions;
    using pixal3d::ProjectionView;
    using pixal3d::ProjectedGrid;

    // A constant map makes the projection geometry independent of feature
    // values and validates the exact point count/order and border handling.
    const std::vector<float> constant_map(2 * 2 * 3, 4.25f);
    ProjectionGridOptions options;
    options.grid_resolution = 3;
    options.image_resolution = 8;
    ProjectedGrid projected;
    std::string error;
    require(pixal3d::project_grid_features(
                constant_map.data(), 2, 2, 3, options,
                ProjectionCamera::front(1.0f, 2.0f), projected, &error),
            error);
    require(projected.points() == 27, "3^3 projected points");
    require(projected.features.size() == 27 * 3, "projected feature shape");
    for (float value : projected.features) {
        require_close(value, 4.25f, 1e-6f, "constant border/bilinear sample");
    }

    // The sparse-coordinate fast path must be numerically identical to the
    // dense path at the requested cells.  This catches drift when projection
    // constants or border interpolation are changed independently in either
    // implementation, while also checking that caller order is preserved.
    const int sparse_height = 3;
    const int sparse_width = 4;
    const int sparse_channels = 2;
    std::vector<float> sparse_map(
        static_cast<std::size_t>(sparse_height * sparse_width * sparse_channels));
    for (int y = 0; y < sparse_height; ++y) {
        for (int x = 0; x < sparse_width; ++x) {
            for (int channel = 0; channel < sparse_channels; ++channel) {
                sparse_map[(static_cast<std::size_t>(y) * sparse_width + x) *
                               sparse_channels + channel] =
                    static_cast<float>(100 * y + 10 * x + channel);
            }
        }
    }
    ProjectionGridOptions sparse_options;
    sparse_options.grid_resolution = 4;
    sparse_options.image_resolution = 8;
    ProjectedGrid sparse_dense;
    require(pixal3d::project_grid_features(
                sparse_map.data(), sparse_height, sparse_width, sparse_channels,
                sparse_options, ProjectionCamera::front(1.1f, 2.3f, 0.9f),
                sparse_dense, &error),
            error);
    const std::vector<std::int32_t> sparse_coords = {
        0, 3, 1, 2,
        0, 0, 0, 0,
        0, 3, 1, 2,
        0, 2, 3, 1,
    };
    std::vector<float> sparse_features;
    std::vector<std::uint8_t> sparse_valid;
    require(pixal3d::project_grid_features_at_coords(
                sparse_map.data(), sparse_height, sparse_width, sparse_channels,
                sparse_options, ProjectionCamera::front(1.1f, 2.3f, 0.9f),
                sparse_coords, sparse_features, sparse_valid, &error),
            error);
    require(sparse_features.size() == sparse_coords.size() / 4 * sparse_channels,
            "sparse projection feature shape");
    require(sparse_valid.size() == sparse_coords.size() / 4,
            "sparse projection validity shape");
    for (std::size_t point = 0; point < sparse_coords.size() / 4; ++point) {
        const std::size_t coord = point * 4;
        const int x = sparse_coords[coord + 1];
        const int y = sparse_coords[coord + 2];
        const int z = sparse_coords[coord + 3];
        const std::size_t dense_point =
            (static_cast<std::size_t>(x) * sparse_options.grid_resolution +
             static_cast<std::size_t>(y)) * sparse_options.grid_resolution +
            static_cast<std::size_t>(z);
        require(sparse_valid[point] == sparse_dense.valid[dense_point],
                "sparse projection validity matches dense path");
        for (int channel = 0; channel < sparse_channels; ++channel) {
            require_close(
                sparse_features[point * sparse_channels + channel],
                sparse_dense.features[dense_point * sparse_channels + channel],
                1e-6f, "sparse projection matches dense path");
        }
    }

    // A 2x2 one-channel map sampled at the canonical image center exercises
    // the image-resolution-to-feature-map scale used by align_corners=False.
    // Use a one-point grid so the projected point is exactly the canonical
    // center.
    const std::vector<float> map = {0.0f, 2.0f, 4.0f, 6.0f};
    options.grid_resolution = 1;
    ProjectedGrid center;
    require(pixal3d::project_grid_features(
                map.data(), 2, 2, 1, options,
                ProjectionCamera::front(1.0f, 2.0f), center, &error),
            error);
    require(center.points() == 1, "one-point grid");
    require_close(center.features[0], 3.75f, 1e-5f,
                  "center align_corners=False interpolation");

    pixal3d::SparseTensorF32 gathered;
    require(pixal3d::projected_grid_to_sparse_f32(
                center, {0, 0, 0, 0}, gathered, &error), error);
    require(gathered.batch_size == 1 && gathered.channels == 1 &&
                gathered.points() == 1,
            "projected grid sparse bridge shape");
    require_close(gathered.feats[0], center.features[0], 1e-6f,
                  "projected grid sparse bridge value");

    // Multi-view conditioning must average views elementwise while preserving
    // the channel/grid contract.
    const std::vector<float> map_a(2 * 2, 2.0f);
    const std::vector<float> map_b(2 * 2, 6.0f);
    ProjectionView view_a{map_a.data(), 2, 2, 1,
                          ProjectionCamera::front(1.0f, 2.0f)};
    ProjectionView view_b{map_b.data(), 2, 2, 1,
                          ProjectionCamera::front(1.0f, 2.0f)};
    ProjectedGrid average;
    require(pixal3d::project_grid_features_average(
                {view_a, view_b}, options, average, &error),
            error);
    require(average.points() == 1, "multi-view point count");
    require_close(average.features[0], 4.0f, 1e-6f,
                  "multi-view elementwise average");

    const std::array<float, 16> identity = {{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    }};
    const std::array<float, 16> translated = {{
        1.0f, 0.0f, 0.0f, 0.5f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    }};
    std::vector<std::array<float, 16>> relative;
    require(pixal3d::compute_relative_calc_matrices(
                {identity, translated}, {2.0f, 2.0f}, relative, &error),
            error);
    require(relative.size() == 2, "relative camera count");
    const std::array<float, 16> expected_front = {{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f, -2.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    }};
    for (std::size_t i = 0; i < expected_front.size(); ++i) {
        require_close(relative[0][i], expected_front[i], 1e-6f,
                      "main view snaps to canonical front matrix");
    }
    require_close(relative[1][3], 0.5f, 1e-6f,
                  "relative view preserves translation");

    ProjectionGridOptions bad_options;
    bad_options.grid_resolution = 0;
    require(!pixal3d::project_grid_features(
                map.data(), 2, 2, 1, bad_options,
                ProjectionCamera::front(1.0f, 2.0f), projected, &error),
            "invalid grid resolution is rejected");

    std::cout << "projection tests: PASS\n";
    return EXIT_SUCCESS;
}
