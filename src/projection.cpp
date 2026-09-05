#include "pixal3d/projection.h"
#include "pixal3d/sparse.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace pixal3d {
namespace {

constexpr float kPi = 3.14159265358979323846f;

void set_error(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
}

std::array<float, 16> front_matrix(float distance) {
    // Same row-major matrix as ProjGrid.front_view_transform_matrix.
    return {{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f, -distance,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    }};
}

std::array<float, 16> multiply4x4(const std::array<float, 16> & lhs,
                                  const std::array<float, 16> & rhs) {
    std::array<float, 16> result{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            float value = 0.0f;
            for (int k = 0; k < 4; ++k) {
                value += lhs[static_cast<std::size_t>(row * 4 + k)] *
                         rhs[static_cast<std::size_t>(k * 4 + col)];
            }
            result[static_cast<std::size_t>(row * 4 + col)] = value;
        }
    }
    return result;
}

bool invert4x4(const std::array<float, 16> & input,
               std::array<float, 16> & output) {
    float a[4][8] = {};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            a[r][c] = input[static_cast<std::size_t>(r * 4 + c)];
        }
        a[r][r + 4] = 1.0f;
    }

    for (int col = 0; col < 4; ++col) {
        int pivot = col;
        float pivot_abs = std::fabs(a[pivot][col]);
        for (int row = col + 1; row < 4; ++row) {
            const float candidate = std::fabs(a[row][col]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if (!(pivot_abs > 1e-8f) || !std::isfinite(pivot_abs)) {
            return false;
        }
        if (pivot != col) {
            for (int c = 0; c < 8; ++c) {
                std::swap(a[pivot][c], a[col][c]);
            }
        }

        const float scale = a[col][col];
        for (int c = 0; c < 8; ++c) {
            a[col][c] /= scale;
        }
        for (int row = 0; row < 4; ++row) {
            if (row == col) {
                continue;
            }
            const float factor = a[row][col];
            if (factor == 0.0f) {
                continue;
            }
            for (int c = 0; c < 8; ++c) {
                a[row][c] -= factor * a[col][c];
            }
        }
    }

    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            output[static_cast<std::size_t>(r * 4 + c)] = a[r][c + 4];
        }
    }
    return true;
}

struct ProjectedPoint {
    float x_pixel = 0.0f;
    float y_pixel = 0.0f;
    float depth = 0.0f;
    bool valid = false;
};

ProjectedPoint project_point(float x,
                             float y,
                             float z,
                             const ProjectionCamera & camera,
                             const ProjectionGridOptions & options,
                             const std::array<float, 16> & world_to_camera) {
    const float x_cam = world_to_camera[0] * x + world_to_camera[1] * y +
                        world_to_camera[2] * z + world_to_camera[3];
    const float y_cam = world_to_camera[4] * x + world_to_camera[5] * y +
                        world_to_camera[6] * z + world_to_camera[7];
    const float z_cam = world_to_camera[8] * x + world_to_camera[9] * y +
                        world_to_camera[10] * z + world_to_camera[11];

    const float depth = -z_cam;
    const float focal_length = 16.0f / std::tan(camera.camera_angle_x * 0.5f);
    const float focal_length_pixels =
        focal_length * static_cast<float>(options.image_resolution) / 32.0f;
    const float denominator = -z_cam + 1e-8f;
    const float x_ndc = focal_length_pixels * x_cam / denominator;
    const float y_ndc = focal_length_pixels * y_cam / denominator;
    const float half = static_cast<float>(options.image_resolution) * 0.5f;

    ProjectedPoint result;
    result.x_pixel = x_ndc + half;
    result.y_pixel = -y_ndc + half;
    result.depth = depth;
    result.valid = result.x_pixel >= 0.0f &&
                   result.x_pixel < static_cast<float>(options.image_resolution) &&
                   result.y_pixel >= 0.0f &&
                   result.y_pixel < static_cast<float>(options.image_resolution) &&
                   depth > 0.0f;
    return result;
}

float sample_bilinear_border(const float * fmap,
                             int height,
                             int width,
                             int channels,
                             float x_pixel,
                             float y_pixel,
                             int channel,
                             int image_resolution) {
    // Python first maps image-space pixels to NDC using
    // (pixel + 0.5) / image_resolution * 2 - 1.  grid_sample with
    // align_corners=False maps that NDC value to this feature-map coordinate.
    float x = (x_pixel + 0.5f) * static_cast<float>(width) /
                  static_cast<float>(image_resolution) -
              0.5f;
    float y = (y_pixel + 0.5f) * static_cast<float>(height) /
                  static_cast<float>(image_resolution) -
              0.5f;

    // padding_mode="border": clamp before taking the four neighbours.
    x = std::max(0.0f, std::min(x, static_cast<float>(width - 1)));
    y = std::max(0.0f, std::min(y, static_cast<float>(height - 1)));

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const float wx = x - static_cast<float>(x0);
    const float wy = y - static_cast<float>(y0);

    const auto at = [&](int iy, int ix) {
        const std::size_t offset =
            (static_cast<std::size_t>(iy) * static_cast<std::size_t>(width) +
             static_cast<std::size_t>(ix)) * static_cast<std::size_t>(channels) +
            static_cast<std::size_t>(channel);
        return fmap[offset];
    };

    const float top = at(y0, x0) * (1.0f - wx) + at(y0, x1) * wx;
    const float bottom = at(y1, x0) * (1.0f - wx) + at(y1, x1) * wx;
    return top * (1.0f - wy) + bottom * wy;
}

bool validate_options(const ProjectionGridOptions & options,
                      const ProjectionCamera & camera,
                      std::string * error) {
    if (options.grid_resolution <= 0) {
        set_error(error, "grid_resolution must be positive");
        return false;
    }
    if (options.image_resolution <= 0) {
        set_error(error, "image_resolution must be positive");
        return false;
    }
    if (!(camera.camera_angle_x > 0.0f && camera.camera_angle_x < kPi) ||
        !std::isfinite(camera.camera_angle_x)) {
        set_error(error, "camera_angle_x must be finite and in (0, pi)");
        return false;
    }
    if (!(camera.mesh_scale > 0.0f) || !std::isfinite(camera.mesh_scale)) {
        set_error(error, "mesh_scale must be finite and positive");
        return false;
    }
    if (!camera.has_transform &&
        (!(camera.distance > 0.0f) || !std::isfinite(camera.distance))) {
        set_error(error, "distance must be finite and positive");
        return false;
    }
    return true;
}

bool prepare_projection_view(const ProjectionView & view,
                             const ProjectionGridOptions & options,
                             std::array<float, 16> & world_to_camera,
                             std::string * error) {
    if (!view.feature_map) {
        set_error(error, "feature_map is null");
        return false;
    }
    if (view.height <= 0 || view.width <= 0 || view.channels <= 0) {
        set_error(error, "feature map dimensions must be positive");
        return false;
    }
    if (!validate_options(options, view.camera, error)) return false;

    const std::array<float, 16> c2w = view.camera.has_transform
                                          ? view.camera.transform_matrix
                                          : front_matrix(view.camera.distance);
    if (!invert4x4(c2w, world_to_camera)) {
        set_error(error, "camera transform matrix is singular");
        return false;
    }
    return true;
}

bool project_one(const ProjectionView & view,
                 const ProjectionGridOptions & options,
                 ProjectedGrid & output,
                 std::string * error) {
    std::array<float, 16> world_to_camera{};
    if (!prepare_projection_view(view, options, world_to_camera, error)) return false;

    const std::size_t r = static_cast<std::size_t>(options.grid_resolution);
    const std::size_t point_count = r * r * r;
    output.grid_resolution = options.grid_resolution;
    output.channels = view.channels;
    output.features.assign(point_count * static_cast<std::size_t>(view.channels), 0.0f);
    output.valid.assign(point_count, 0);

    std::size_t point_index = 0;
    for (int ix = 0; ix < options.grid_resolution; ++ix) {
        const float base_x = options.grid_resolution == 1
                                 ? 0.0f
                                 : -1.0f + 2.0f * static_cast<float>(ix) /
                                               static_cast<float>(options.grid_resolution - 1);
        for (int iy = 0; iy < options.grid_resolution; ++iy) {
            const float base_y = options.grid_resolution == 1
                                     ? 0.0f
                                     : -1.0f + 2.0f * static_cast<float>(iy) /
                                                   static_cast<float>(options.grid_resolution - 1);
            for (int iz = 0; iz < options.grid_resolution; ++iz, ++point_index) {
                const float base_z = options.grid_resolution == 1
                                         ? 0.0f
                                         : -1.0f + 2.0f * static_cast<float>(iz) /
                                                       static_cast<float>(options.grid_resolution - 1);

                // torch.matmul(points, rotation_matrix.T):
                // (x, y, z) -> (x, -z, y), then scale by mesh_scale / 2.
                const float scale = 1.0f / (view.camera.mesh_scale * 2.0f);
                const float x = base_x * scale;
                const float y = -base_z * scale;
                const float z = base_y * scale;
                const ProjectedPoint p = project_point(
                    x, y, z, view.camera, options, world_to_camera);
                output.valid[point_index] = p.valid ? 1 : 0;

                float * dst = output.features.data() +
                              point_index * static_cast<std::size_t>(view.channels);
                for (int c = 0; c < view.channels; ++c) {
                    dst[c] = sample_bilinear_border(
                        view.feature_map, view.height, view.width, view.channels,
                        p.x_pixel, p.y_pixel, c, options.image_resolution);
                }
            }
        }
    }
    return true;
}

} // namespace

ProjectionCamera ProjectionCamera::front(float camera_angle_x,
                                          float distance,
                                          float mesh_scale) {
    ProjectionCamera camera;
    camera.camera_angle_x = camera_angle_x;
    camera.distance = distance;
    camera.mesh_scale = mesh_scale;
    camera.has_transform = false;
    return camera;
}

std::size_t ProjectedGrid::points() const {
    return grid_resolution > 0
               ? static_cast<std::size_t>(grid_resolution) *
                     static_cast<std::size_t>(grid_resolution) *
                     static_cast<std::size_t>(grid_resolution)
               : 0;
}

bool ProjectedGrid::empty() const {
    return features.empty();
}

bool project_grid_features(const float * feature_map,
                           int height,
                           int width,
                           int channels,
                           const ProjectionGridOptions & options,
                           const ProjectionCamera & camera,
                           ProjectedGrid & output,
                           std::string * error) {
    ProjectionView view;
    view.feature_map = feature_map;
    view.height = height;
    view.width = width;
    view.channels = channels;
    view.camera = camera;
    return project_one(view, options, output, error);
}

bool project_grid_features_at_coords(
    const float * feature_map,
    int height,
    int width,
    int channels,
    const ProjectionGridOptions & options,
    const ProjectionCamera & camera,
    const std::vector<std::int32_t> & coords,
    std::vector<float> & features,
    std::vector<std::uint8_t> & valid,
    std::string * error) {
    features.clear();
    valid.clear();
    ProjectionView view;
    view.feature_map = feature_map;
    view.height = height;
    view.width = width;
    view.channels = channels;
    view.camera = camera;
    std::array<float, 16> world_to_camera{};
    if (!prepare_projection_view(view, options, world_to_camera, error)) return false;
    if (coords.size() % 4 != 0) {
        set_error(error, "sparse projection coordinates must contain [batch,x,y,z] rows");
        return false;
    }

    const std::size_t points = coords.size() / 4;
    if (points > std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(channels)) {
        set_error(error, "sparse projection feature size overflows size_t");
        return false;
    }
    features.resize(points * static_cast<std::size_t>(channels));
    valid.resize(points, 0);
    for (std::size_t point = 0; point < points; ++point) {
        const std::int32_t batch = coords[point * 4 + 0];
        const std::int32_t ix = coords[point * 4 + 1];
        const std::int32_t iy = coords[point * 4 + 2];
        const std::int32_t iz = coords[point * 4 + 3];
        if (batch != 0 || ix < 0 || iy < 0 || iz < 0 ||
            ix >= options.grid_resolution || iy >= options.grid_resolution ||
            iz >= options.grid_resolution) {
            set_error(error, "sparse projection coordinate is outside the requested grid");
            features.clear();
            valid.clear();
            return false;
        }
        const float base_x = options.grid_resolution == 1
                                 ? 0.0f
                                 : -1.0f + 2.0f * static_cast<float>(ix) /
                                               static_cast<float>(options.grid_resolution - 1);
        const float base_y = options.grid_resolution == 1
                                 ? 0.0f
                                 : -1.0f + 2.0f * static_cast<float>(iy) /
                                               static_cast<float>(options.grid_resolution - 1);
        const float base_z = options.grid_resolution == 1
                                 ? 0.0f
                                 : -1.0f + 2.0f * static_cast<float>(iz) /
                                               static_cast<float>(options.grid_resolution - 1);
        const float scale = 1.0f / (camera.mesh_scale * 2.0f);
        const ProjectedPoint projected = project_point(
            base_x * scale, -base_z * scale, base_y * scale,
            camera, options, world_to_camera);
        valid[point] = projected.valid ? 1 : 0;
        float * dst = features.data() + point * static_cast<std::size_t>(channels);
        for (int channel = 0; channel < channels; ++channel) {
            dst[channel] = sample_bilinear_border(
                feature_map, height, width, channels,
                projected.x_pixel, projected.y_pixel, channel,
                options.image_resolution);
        }
    }
    return true;
}

bool project_grid_features_average(const std::vector<ProjectionView> & views,
                                   const ProjectionGridOptions & options,
                                   ProjectedGrid & output,
                                   std::string * error) {
    if (views.empty()) {
        set_error(error, "at least one projection view is required");
        return false;
    }

    ProjectedGrid accumulated;
    int channels = -1;
    for (const ProjectionView & view : views) {
        if (channels >= 0 && view.channels != channels) {
            set_error(error, "all projection views must have the same channel count");
            return false;
        }
        ProjectedGrid projected;
        if (!project_one(view, options, projected, error)) {
            return false;
        }
        channels = projected.channels;
        if (accumulated.empty()) {
            accumulated.grid_resolution = projected.grid_resolution;
            accumulated.channels = projected.channels;
            accumulated.features.assign(projected.features.size(), 0.0f);
            accumulated.valid.assign(projected.valid.size(), 0);
        } else if (accumulated.features.size() != projected.features.size()) {
            set_error(error, "projection views must produce the same grid shape");
            return false;
        }
        for (std::size_t i = 0; i < accumulated.features.size(); ++i) {
            accumulated.features[i] += projected.features[i];
        }
        for (std::size_t i = 0; i < accumulated.valid.size(); ++i) {
            accumulated.valid[i] = static_cast<std::uint8_t>(
                accumulated.valid[i] || projected.valid[i]);
        }
    }

    const float inv_views = 1.0f / static_cast<float>(views.size());
    for (float & value : accumulated.features) {
        value *= inv_views;
    }
    output = std::move(accumulated);
    return true;
}

bool projected_grid_to_sparse_f32(const ProjectedGrid & grid,
                                  const std::vector<std::int32_t> & coords,
                                  SparseTensorF32 & output,
                                  std::string * error) {
    output = SparseTensorF32{};
    if (grid.grid_resolution <= 0 || grid.channels <= 0 ||
        grid.features.size() != grid.points() * static_cast<std::size_t>(grid.channels) ||
        coords.size() % 4 != 0) {
        set_error(error, "invalid projected grid or sparse coordinate list");
        return false;
    }
    const int resolution = grid.grid_resolution;
    const std::size_t points = coords.size() / 4;
    output.batch_size = 1;
    output.channels = grid.channels;
    output.spatial_x = resolution;
    output.spatial_y = resolution;
    output.spatial_z = resolution;
    output.coords = coords;
    output.feats.resize(points * static_cast<std::size_t>(grid.channels));
    for (std::size_t point = 0; point < points; ++point) {
        const std::size_t base = point * 4;
        const std::int32_t batch = coords[base + 0];
        const std::int32_t x = coords[base + 1];
        const std::int32_t y = coords[base + 2];
        const std::int32_t z = coords[base + 3];
        if (batch != 0 || x < 0 || x >= resolution || y < 0 || y >= resolution ||
            z < 0 || z >= resolution) {
            output = SparseTensorF32{};
            set_error(error, "projected sparse coordinate is outside the dense grid");
            return false;
        }
        const std::size_t grid_index =
            (static_cast<std::size_t>(x) * static_cast<std::size_t>(resolution) +
             static_cast<std::size_t>(y)) * static_cast<std::size_t>(resolution) +
            static_cast<std::size_t>(z);
        const float * source = grid.features.data() +
                              grid_index * static_cast<std::size_t>(grid.channels);
        float * destination = output.feats.data() +
                              point * static_cast<std::size_t>(grid.channels);
        std::copy(source, source + grid.channels, destination);
    }
    return output.valid(error);
}

bool compute_relative_calc_matrices(
    const std::vector<std::array<float, 16>> & camera_to_world,
    const std::vector<float> & distances,
    std::vector<std::array<float, 16>> & relative,
    std::string * error) {
    if (camera_to_world.empty()) {
        set_error(error, "at least one camera transform is required");
        return false;
    }
    if (camera_to_world.size() != distances.size()) {
        set_error(error, "camera transform and distance counts must match");
        return false;
    }
    if (!(distances[0] > 0.0f) || !std::isfinite(distances[0])) {
        set_error(error, "main-view distance must be finite and positive");
        return false;
    }

    std::array<float, 16> inverse_main{};
    if (!invert4x4(camera_to_world[0], inverse_main)) {
        set_error(error, "main-view camera transform matrix is singular");
        return false;
    }

    const std::array<float, 16> canonical = front_matrix(distances[0]);
    relative.resize(camera_to_world.size());
    for (std::size_t i = 0; i < camera_to_world.size(); ++i) {
        const std::array<float, 16> view_relative =
            multiply4x4(inverse_main, camera_to_world[i]);
        relative[i] = multiply4x4(canonical, view_relative);
    }
    return true;
}

} // namespace pixal3d
