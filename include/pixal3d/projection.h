#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pixal3d {

struct SparseTensorF32;

// Camera parameters used by the Python ProjGrid implementation.  The matrix
// is a row-major c2w matrix.  When has_transform is false, the canonical
// front-view matrix is constructed from distance.
struct ProjectionCamera {
    float camera_angle_x = 0.2f; // horizontal FOV, radians
    float distance = 2.0f;
    float mesh_scale = 1.0f;
    bool has_transform = false;
    std::array<float, 16> transform_matrix{};

    static ProjectionCamera front(float camera_angle_x,
                                  float distance,
                                  float mesh_scale = 1.0f);
};

struct ProjectionGridOptions {
    int grid_resolution = 16;
    int image_resolution = 518;
};

// A feature map is a single HxWxC, row-major float32 image feature tensor.
// The output is KxC, row-major, where K = grid_resolution^3 and the point
// order is identical to torch.meshgrid(..., indexing="ij").reshape(-1, 3).
struct ProjectedGrid {
    int grid_resolution = 0;
    int channels = 0;
    std::vector<float> features;
    std::vector<std::uint8_t> valid;

    std::size_t points() const;
    bool empty() const;
};

struct ProjectionView {
    const float * feature_map = nullptr;
    int height = 0;
    int width = 0;
    int channels = 0;
    ProjectionCamera camera;
};

// Project one image feature map into a camera-aligned 3D grid and sample it
// with the same bilinear/border rules as torch.nn.functional.grid_sample:
// mode=bilinear, align_corners=False, padding_mode=border.
bool project_grid_features(const float * feature_map,
                           int height,
                           int width,
                           int channels,
                           const ProjectionGridOptions & options,
                           const ProjectionCamera & camera,
                           ProjectedGrid & output,
                           std::string * error = nullptr);

// Project only the supplied sparse [batch,x,y,z] coordinates.  This uses the
// same camera, align_corners=false, and border-sampling contract as
// project_grid_features(), but avoids materializing a grid_resolution^3 x C
// dense tensor.  The output feature rows and validity flags follow coords.
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
    std::string * error = nullptr);

// Project all views into the same grid and average their sampled features.
// This mirrors ProjGridMV: every view contributes equally, including samples
// outside the image (which use border padding).  The valid mask is set when
// at least one view projects the voxel in front of and inside the image.
bool project_grid_features_average(const std::vector<ProjectionView> & views,
                                   const ProjectionGridOptions & options,
                                   ProjectedGrid & output,
                                   std::string * error = nullptr);

// Gather a projected dense grid at sparse [batch,x,y,z] coordinates.  The
// projected grid represents one image/object, so this bridge currently
// accepts one batch (batch index 0) and preserves the caller's coordinate
// order.  Projection validity is diagnostic only; the Python reference also
// consumes border-sampled values for every grid cell.
bool projected_grid_to_sparse_f32(const ProjectedGrid & grid,
                                  const std::vector<std::int32_t> & coords,
                                  SparseTensorF32 & output,
                                  std::string * error = nullptr);

// Convert a set of per-view c2w matrices into the relative matrices consumed
// by ProjGridMV.  The first view is snapped to the canonical front view:
// calc_mat[i] = F(distance[0]) * inverse(c2w[0]) * c2w[i].
bool compute_relative_calc_matrices(
    const std::vector<std::array<float, 16>> & camera_to_world,
    const std::vector<float> & distances,
    std::vector<std::array<float, 16>> & relative,
    std::string * error = nullptr);

} // namespace pixal3d
