#pragma once

#include "pixal3d/dual_grid.h"

#include <cstddef>
#include <string>

namespace pixal3d {

// Fill small, independently traversable manifold boundary loops in a raw
// Dual Grid mesh.  This mirrors the observable contract of the Python CuMesh
// postprocess, but its CPU implementation is not guaranteed to be bit-for-bit
// identical to CuMesh. Invalid packed arrays are rejected; branching or
// non-manifold boundary regions are preserved while unrelated safe loops may
// still be filled.
bool fill_mesh_holes_f32(
    DualGridMeshF32 & mesh,
    float max_hole_perimeter = 3.0e-2f,
    std::string * error = nullptr);

// Summary of a deterministic non-manifold edge repair. Vertex splitting
// preserves every valid face and duplicates only the affected vertex positions.
struct MeshTopologyRepairReport {
    std::size_t input_vertices = 0;
    std::size_t output_vertices = 0;
    std::size_t face_count = 0;
    std::size_t nonmanifold_edges_before = 0;
    std::size_t nonmanifold_edges_after = 0;
    std::size_t split_vertices = 0;
    std::size_t affected_faces = 0;
};

// Split vertex fans around edges with more than two incident faces. This is a
// CPU export-stage helper corresponding to CuMesh repair_non_manifold_edges();
// it does not delete faces, fill holes, simplify, remesh, or unify winding.
bool repair_non_manifold_edges_f32(
    DualGridMeshF32 & mesh,
    MeshTopologyRepairReport * report = nullptr,
    std::string * error = nullptr);

} // namespace pixal3d
