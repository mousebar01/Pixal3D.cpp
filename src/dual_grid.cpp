#include "pixal3d/dual_grid.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pixal3d {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

struct VoxelCoord {
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::int32_t z = 0;

    bool operator==(const VoxelCoord & other) const noexcept {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelCoordHash {
    std::size_t operator()(const VoxelCoord & value) const noexcept {
        // The same inexpensive multiplicative hash family is used by the
        // reference O-Voxel implementation.  Coordinates are validated
        // before insertion, so signed-to-unsigned conversion is well-defined.
        constexpr std::size_t p1 = 73856093u;
        constexpr std::size_t p2 = 19349663u;
        constexpr std::size_t p3 = 83492791u;
        return static_cast<std::size_t>(value.x) * p1 ^
               static_cast<std::size_t>(value.y) * p2 ^
               static_cast<std::size_t>(value.z) * p3;
    }
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

Vec3 subtract(const Vec3 & a, const Vec3 & b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 cross(const Vec3 & a, const Vec3 & b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

float dot(const Vec3 & a, const Vec3 & b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 vertex_at(const std::vector<float> & vertices, std::size_t index) {
    return {vertices[index * 3 + 0], vertices[index * 3 + 1],
            vertices[index * 3 + 2]};
}

void append_triangle(std::vector<std::int32_t> & faces,
                     std::size_t a, std::size_t b, std::size_t c) {
    faces.push_back(static_cast<std::int32_t>(a));
    faces.push_back(static_cast<std::int32_t>(b));
    faces.push_back(static_cast<std::int32_t>(c));
}

} // namespace

bool flexible_dual_grid_to_mesh_f32(
    const std::vector<std::int32_t> & coords_xyz,
    const std::vector<float> & dual_vertices_xyz,
    const std::vector<std::uint8_t> & intersected_xyz,
    const std::vector<float> * split_weights,
    const std::array<float, 3> & aabb_min,
    const std::array<float, 3> & aabb_max,
    const std::array<std::int32_t, 3> & grid_size,
    DualGridMeshF32 & output,
    std::string * error) {
    output = DualGridMeshF32{};
    if (coords_xyz.empty() && dual_vertices_xyz.empty() && intersected_xyz.empty() &&
        (!split_weights || split_weights->empty())) {
        return true;
    }
    if (coords_xyz.empty() || coords_xyz.size() % 3 != 0) {
        set_error(error, "Flexible Dual Grid coordinates must contain xyz triples");
        return false;
    }
    const std::size_t count = coords_xyz.size() / 3;
    if (dual_vertices_xyz.size() != count * 3 || intersected_xyz.size() != count * 3 ||
        (split_weights && split_weights->size() != count)) {
        set_error(error, "Flexible Dual Grid arrays have inconsistent voxel counts");
        return false;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (grid_size[axis] <= 0 || !(aabb_max[axis] > aabb_min[axis]) ||
            !std::isfinite(aabb_min[axis]) || !std::isfinite(aabb_max[axis])) {
            set_error(error, "Flexible Dual Grid bounds must be finite and positive");
            return false;
        }
    }
    if (count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        set_error(error, "Flexible Dual Grid has too many voxels for int32 faces");
        return false;
    }

    std::unordered_map<VoxelCoord, std::size_t, VoxelCoordHash> lookup;
    lookup.reserve(count * 2 + 1);
    for (std::size_t index = 0; index < count; ++index) {
        const VoxelCoord coord{coords_xyz[index * 3 + 0], coords_xyz[index * 3 + 1],
                               coords_xyz[index * 3 + 2]};
        if (coord.x < 0 || coord.y < 0 || coord.z < 0 ||
            coord.x >= grid_size[0] || coord.y >= grid_size[1] || coord.z >= grid_size[2]) {
            set_error(error, "Flexible Dual Grid coordinate is outside grid_size");
            return false;
        }
        if (!lookup.emplace(coord, index).second) {
            set_error(error, "Flexible Dual Grid coordinates must be unique");
            return false;
        }
        for (int component = 0; component < 3; ++component) {
            if (!std::isfinite(dual_vertices_xyz[index * 3 + component])) {
                set_error(error, "Flexible Dual Grid dual vertex is non-finite");
                return false;
            }
        }
        if (split_weights && !std::isfinite((*split_weights)[index])) {
            set_error(error, "Flexible Dual Grid split weight is non-finite");
            return false;
        }
    }

    const std::array<float, 3> voxel_size = {
        (aabb_max[0] - aabb_min[0]) / static_cast<float>(grid_size[0]),
        (aabb_max[1] - aabb_min[1]) / static_cast<float>(grid_size[1]),
        (aabb_max[2] - aabb_min[2]) / static_cast<float>(grid_size[2]),
    };
    output.vertices.resize(count * 3);
    for (std::size_t index = 0; index < count; ++index) {
        for (int axis = 0; axis < 3; ++axis) {
            const float coordinate = static_cast<float>(coords_xyz[index * 3 + axis]);
            const float offset = dual_vertices_xyz[index * 3 + axis];
            output.vertices[index * 3 + axis] =
                (coordinate + offset) * voxel_size[axis] + aabb_min[axis];
        }
    }

    // O-Voxel's edge-neighbor order.  Each row describes the four voxels of
    // one dual-grid quad around an intersected x/y/z primal edge.
    constexpr std::int32_t offsets[3][4][3] = {
        {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}},
        {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}},
        {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}},
    };
    std::vector<std::array<std::size_t, 4>> quads;
    quads.reserve(count * 3);
    for (std::size_t index = 0; index < count; ++index) {
        const VoxelCoord base{coords_xyz[index * 3 + 0], coords_xyz[index * 3 + 1],
                              coords_xyz[index * 3 + 2]};
        for (int axis = 0; axis < 3; ++axis) {
            if (intersected_xyz[index * 3 + axis] == 0) continue;
            std::array<std::size_t, 4> quad{};
            bool complete = true;
            for (int corner = 0; corner < 4; ++corner) {
                const VoxelCoord neighbor{
                    base.x + offsets[axis][corner][0],
                    base.y + offsets[axis][corner][1],
                    base.z + offsets[axis][corner][2]};
                const auto found = lookup.find(neighbor);
                if (found == lookup.end()) {
                    complete = false;
                    break;
                }
                quad[corner] = found->second;
            }
            if (complete) quads.push_back(quad);
        }
    }

    output.faces.reserve(quads.size() * 6);
    for (const std::array<std::size_t, 4> & quad : quads) {
        bool use_split_1 = false;
        if (split_weights) {
            const float weight_02 = (*split_weights)[quad[0]] * (*split_weights)[quad[2]];
            const float weight_13 = (*split_weights)[quad[1]] * (*split_weights)[quad[3]];
            use_split_1 = weight_02 > weight_13;
        } else {
            // Match the intended min-angle selection in O-Voxel: compare the
            // absolute dot product of the two triangle normals for each
            // diagonal.  This branch is not used by Pixal3D inference, which
            // supplies the decoder's positive split weights.
            const Vec3 a = vertex_at(output.vertices, quad[0]);
            const Vec3 b = vertex_at(output.vertices, quad[1]);
            const Vec3 c = vertex_at(output.vertices, quad[2]);
            const Vec3 d = vertex_at(output.vertices, quad[3]);
            const float align_1 = std::fabs(dot(cross(subtract(b, a), subtract(c, a)),
                                             cross(subtract(c, a), subtract(d, a))));
            const float align_2 = std::fabs(dot(cross(subtract(b, a), subtract(d, a)),
                                             cross(subtract(d, a), subtract(c, a))));
            use_split_1 = align_1 > align_2;
        }
        if (use_split_1) {
            append_triangle(output.faces, quad[0], quad[1], quad[2]);
            append_triangle(output.faces, quad[0], quad[2], quad[3]);
        } else {
            append_triangle(output.faces, quad[0], quad[1], quad[3]);
            append_triangle(output.faces, quad[3], quad[1], quad[2]);
        }
    }
    return true;
}

bool flexi_dual_grid_decode_mesh_f32(
    const SparseTensorF32 & decoded,
    int resolution,
    float voxel_margin,
    std::vector<DualGridMeshF32> & outputs,
    std::string * error) {
    outputs.clear();
    if (!decoded.valid(error) || decoded.channels != 7 || resolution <= 0 ||
        !std::isfinite(voxel_margin) || voxel_margin < 0.0f) {
        set_error(error, "invalid FlexiDualGrid decoder output or resolution");
        return false;
    }
    outputs.resize(static_cast<std::size_t>(decoded.batch_size));
    const std::array<float, 3> aabb_min = {-0.5f, -0.5f, -0.5f};
    const std::array<float, 3> aabb_max = {0.5f, 0.5f, 0.5f};
    const std::array<std::int32_t, 3> grid_size = {resolution, resolution, resolution};
    for (int batch = 0; batch < decoded.batch_size; ++batch) {
        std::vector<std::int32_t> coords;
        std::vector<float> dual_vertices;
        std::vector<std::uint8_t> intersected;
        std::vector<float> split_weights;
        for (std::size_t point = 0; point < decoded.points(); ++point) {
            if (decoded.coords[point * 4] != batch) continue;
            coords.insert(coords.end(), {decoded.coords[point * 4 + 1],
                                         decoded.coords[point * 4 + 2],
                                         decoded.coords[point * 4 + 3]});
            const float * feature = decoded.feats.data() + point * 7;
            for (int axis = 0; axis < 3; ++axis) {
                const float sigmoid = feature[axis] >= 0.0f
                    ? 1.0f / (1.0f + std::exp(-feature[axis]))
                    : std::exp(feature[axis]) / (1.0f + std::exp(feature[axis]));
                dual_vertices.push_back((1.0f + 2.0f * voxel_margin) * sigmoid - voxel_margin);
                intersected.push_back(feature[axis + 3] > 0.0f ? 1 : 0);
            }
            split_weights.push_back(feature[6] >= 0.0f
                ? feature[6] + std::log1p(std::exp(-feature[6]))
                : std::log1p(std::exp(feature[6])));
        }
        if (!flexible_dual_grid_to_mesh_f32(
                coords, dual_vertices, intersected, &split_weights,
                aabb_min, aabb_max, grid_size, outputs[static_cast<std::size_t>(batch)], error)) {
            outputs.clear();
            return false;
        }
    }
    return true;
}

} // namespace pixal3d
