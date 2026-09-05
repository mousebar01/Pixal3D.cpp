#include "pixal3d/dual_grid.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void emit(const char * name, const std::vector<float> & values) {
    std::cout << name << " " << values.size();
    for (float value : values) std::cout << " " << value;
    std::cout << "\n";
}

void emit(const char * name, const std::vector<std::int32_t> & values) {
    std::cout << name << " " << values.size();
    for (std::int32_t value : values) std::cout << " " << value;
    std::cout << "\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    std::vector<std::int32_t> coords;
    std::vector<float> dual_vertices;
    std::vector<std::uint8_t> intersected;
    std::vector<float> split_weights;
    for (std::int32_t x = 0; x < 2; ++x) {
        for (std::int32_t y = 0; y < 2; ++y) {
            for (std::int32_t z = 0; z < 2; ++z) {
                coords.insert(coords.end(), {x, y, z});
                const float base = 0.11f + 0.037f * static_cast<float>(coords.size() / 3);
                dual_vertices.insert(dual_vertices.end(), {base, base + 0.07f, base + 0.13f});
                intersected.insert(intersected.end(), {1, 1, 1});
                split_weights.push_back(0.2f + 0.17f * static_cast<float>(coords.size() / 3));
            }
        }
    }
    const std::array<float, 3> aabb_min = {-1.0f, -1.0f, -1.0f};
    const std::array<float, 3> aabb_max = {1.0f, 1.0f, 1.0f};
    const std::array<std::int32_t, 3> grid_size = {2, 2, 2};
    pixal3d::DualGridMeshF32 weighted;
    pixal3d::DualGridMeshF32 automatic;
    std::string error;
    if (!pixal3d::flexible_dual_grid_to_mesh_f32(
            coords, dual_vertices, intersected, &split_weights,
            aabb_min, aabb_max, grid_size, weighted, &error) ||
        !pixal3d::flexible_dual_grid_to_mesh_f32(
            coords, dual_vertices, intersected, nullptr,
            aabb_min, aabb_max, grid_size, automatic, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    pixal3d::SparseTensorF32 decoded;
    decoded.batch_size = 1;
    decoded.channels = 7;
    decoded.spatial_x = decoded.spatial_y = decoded.spatial_z = 2;
    decoded.coords.resize(coords.size() / 3 * 4);
    decoded.feats.resize(coords.size() / 3 * 7);
    for (std::size_t point = 0; point < coords.size() / 3; ++point) {
        decoded.coords[point * 4 + 0] = 0;
        decoded.coords[point * 4 + 1] = coords[point * 3 + 0];
        decoded.coords[point * 4 + 2] = coords[point * 3 + 1];
        decoded.coords[point * 4 + 3] = coords[point * 3 + 2];
        decoded.feats[point * 7 + 0] = -0.31f + 0.09f * static_cast<float>(point);
        decoded.feats[point * 7 + 1] = 0.17f - 0.06f * static_cast<float>(point);
        decoded.feats[point * 7 + 2] = -0.05f + 0.04f * static_cast<float>(point);
        decoded.feats[point * 7 + 3] = 1.0f;
        decoded.feats[point * 7 + 4] = 1.0f;
        decoded.feats[point * 7 + 5] = 1.0f;
        decoded.feats[point * 7 + 6] = -0.4f + 0.13f * static_cast<float>(point);
    }
    std::vector<pixal3d::DualGridMeshF32> decoded_meshes;
    if (!pixal3d::flexi_dual_grid_decode_mesh_f32(decoded, 2, 0.5f,
                                                   decoded_meshes, &error) ||
        decoded_meshes.size() != 1) {
        std::cerr << error << "\n";
        return 1;
    }
    emit("dual_grid_vertices", weighted.vertices);
    emit("dual_grid_weighted_faces", weighted.faces);
    emit("dual_grid_auto_faces", automatic.faces);
    emit("flexi_decode_vertices", decoded_meshes[0].vertices);
    emit("flexi_decode_faces", decoded_meshes[0].faces);
    return 0;
}
