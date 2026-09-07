#include "pixal3d/mesh_topology.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

bool check(bool condition, const char * message) {
    if (condition) return true;
    std::fprintf(stderr, "mesh topology test: %s\n", message);
    return false;
}

pixal3d::DualGridMeshF32 open_square(float side) {
    pixal3d::DualGridMeshF32 mesh;
    mesh.vertices = {0.0f, 0.0f, 0.0f,
                     side, 0.0f, 0.0f,
                     side, side, 0.0f,
                     0.0f, side, 0.0f};
    mesh.faces = {0, 1, 2, 0, 2, 3};
    return mesh;
}

std::map<std::pair<std::int32_t, std::int32_t>, int> edge_counts(
    const pixal3d::DualGridMeshF32 & mesh) {
    std::map<std::pair<std::int32_t, std::int32_t>, int> counts;
    for (std::size_t index = 0; index < mesh.faces.size(); index += 3) {
        for (int side = 0; side < 3; ++side) {
            std::int32_t a = mesh.faces[index + static_cast<std::size_t>(side)];
            std::int32_t b = mesh.faces[index + static_cast<std::size_t>((side + 1) % 3)];
            if (b < a) std::swap(a, b);
            ++counts[{a, b}];
        }
    }
    return counts;
}

int nonmanifold_edge_count(const pixal3d::DualGridMeshF32 & mesh) {
    int count = 0;
    for (const auto & entry : edge_counts(mesh)) {
        if (entry.second > 2) ++count;
    }
    return count;
}

} // namespace

int main() {
    std::string error;
    auto mesh = open_square(0.007f);
    if (!check(pixal3d::fill_mesh_holes_f32(mesh, 0.03f, &error),
               "small square fill failed")) return 1;
    if (!check(mesh.vertices.size() == 15, "small square vertex count mismatch") ||
        !check(mesh.faces.size() == 18, "small square face count mismatch") ||
        !check(std::fabs(mesh.vertices[12] - 0.0035f) < 1.0e-6f,
               "small square center x mismatch") ||
        !check(std::fabs(mesh.vertices[13] - 0.0035f) < 1.0e-6f,
               "small square center y mismatch") ||
        !check(std::fabs(mesh.vertices[14]) < 1.0e-6f,
               "small square center z mismatch")) return 1;
    for (const auto & entry : edge_counts(mesh)) {
        if (!check(entry.second == 2, "filled square has a boundary edge")) return 1;
    }

    auto over_limit = open_square(0.0075f);
    if (!check(pixal3d::fill_mesh_holes_f32(over_limit, 0.03f, &error),
               "over-limit square returned failure") ||
        !check(over_limit.vertices.size() == 12,
               "over-limit square was modified") ||
        !check(over_limit.faces.size() == 6,
               "over-limit square face count changed")) return 1;

    pixal3d::DualGridMeshF32 tetrahedron;
    tetrahedron.vertices = {0.0f, 0.0f, 0.0f,
                            1.0f, 0.0f, 0.0f,
                            0.0f, 1.0f, 0.0f,
                            0.0f, 0.0f, 1.0f};
    tetrahedron.faces = {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3};
    const auto tetra_faces = tetrahedron.faces;
    if (!check(pixal3d::fill_mesh_holes_f32(tetrahedron, 0.03f, &error),
               "closed tetrahedron returned failure") ||
        !check(tetrahedron.faces == tetra_faces,
               "closed tetrahedron was modified")) return 1;

    pixal3d::DualGridMeshF32 nonmanifold;
    nonmanifold.vertices = {0.0f, 0.0f, 0.0f,
                            1.0f, 0.0f, 0.0f,
                            0.0f, 1.0f, 0.0f,
                            0.0f, 0.0f, 1.0f,
                            1.0f, 1.0f, 0.0f};
    nonmanifold.faces = {0, 1, 2, 1, 0, 3, 0, 1, 4};
    const auto nonmanifold_vertices = nonmanifold.vertices;
    const auto nonmanifold_faces = nonmanifold.faces;
    if (!check(pixal3d::fill_mesh_holes_f32(nonmanifold, 0.03f, &error),
               "non-manifold mesh returned failure") ||
        !check(nonmanifold.vertices == nonmanifold_vertices,
               "non-manifold vertices changed") ||
        !check(nonmanifold.faces == nonmanifold_faces,
               "non-manifold faces changed")) return 1;

    // An unsafe boundary branch must not suppress an unrelated small closed
    // boundary loop elsewhere in the same mesh.
    pixal3d::DualGridMeshF32 mixed = nonmanifold;
    const std::int32_t base = 5;
    mixed.vertices.insert(mixed.vertices.end(), {
        2.0f, 0.0f, 0.0f,
        2.007f, 0.0f, 0.0f,
        2.007f, 0.007f, 0.0f,
        2.0f, 0.007f, 0.0f});
    mixed.faces.insert(mixed.faces.end(), {
        base + 0, base + 1, base + 2,
        base + 0, base + 2, base + 3});
    if (!check(pixal3d::fill_mesh_holes_f32(mixed, 0.03f, &error),
               "mixed topology returned failure") ||
        !check(mixed.vertices.size() == nonmanifold_vertices.size() + 15,
               "mixed topology did not fill the independent loop") ||
        !check(mixed.faces.size() == nonmanifold_faces.size() + 18,
               "mixed topology added unexpected faces")) return 1;

    pixal3d::MeshTopologyRepairReport repair_report;
    const auto repair_input_vertices = nonmanifold.vertices;
    const auto repair_input_faces = nonmanifold.faces;
    if (!check(pixal3d::repair_non_manifold_edges_f32(
                   nonmanifold, &repair_report, &error),
               "non-manifold repair failed") ||
        !check(nonmanifold.faces.size() == repair_input_faces.size(),
               "non-manifold repair removed faces") ||
        !check(nonmanifold_edge_count(nonmanifold) == 0,
               "non-manifold repair left an over-shared edge") ||
        !check(repair_report.nonmanifold_edges_before == 1 &&
                   repair_report.nonmanifold_edges_after == 0 &&
                   repair_report.face_count == 3 &&
                   repair_report.split_vertices > 0,
               "non-manifold repair report mismatch") ||
        !check(nonmanifold.vertices.size() > repair_input_vertices.size(),
               "non-manifold repair did not split a vertex")) return 1;
    for (std::size_t index = repair_input_faces.size(); index < nonmanifold.faces.size(); ++index) {
        if (!check(false, "non-manifold repair changed face buffer size")) return 1;
    }

    auto deterministic_repair = pixal3d::DualGridMeshF32{};
    deterministic_repair.vertices = repair_input_vertices;
    deterministic_repair.faces = repair_input_faces;
    pixal3d::MeshTopologyRepairReport second_report;
    if (!check(pixal3d::repair_non_manifold_edges_f32(
                   deterministic_repair, &second_report, &error),
               "second non-manifold repair failed") ||
        !check(deterministic_repair.vertices == nonmanifold.vertices &&
                   deterministic_repair.faces == nonmanifold.faces,
               "non-manifold repair is not deterministic")) return 1;
    return 0;
}
