#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "pixal3d/sparse.h"

namespace pixal3d {

// CPU result of the O-Voxel Flexible Dual Grid extraction.  Vertices are
// packed xyz triples and faces are packed triangle index triples.
struct DualGridMeshF32 {
    std::vector<float> vertices;
    std::vector<std::int32_t> faces;
};

// Reconnect a sparse Flexible Dual Grid into a triangle mesh.  coords_xyz has
// one integer xyz triple per voxel; dual_vertices_xyz has one float xyz triple
// per voxel; intersected_xyz has three non-zero flags per voxel (x/y/z edge
// families); split_weights has one scalar per voxel or is null for geometric
// split selection.  The voxel order must be shared by all three inputs.
//
// This is the inference-time, non-training path corresponding to
// o_voxel.convert.flexible_dual_grid_to_mesh(..., train=False).  It uses a
// CPU hash map and emits all input dual vertices, including vertices that do
// not participate in a complete quad.
bool flexible_dual_grid_to_mesh_f32(
    const std::vector<std::int32_t> & coords_xyz,
    const std::vector<float> & dual_vertices_xyz,
    const std::vector<std::uint8_t> & intersected_xyz,
    const std::vector<float> * split_weights,
    const std::array<float, 3> & aabb_min,
    const std::array<float, 3> & aabb_max,
    const std::array<std::int32_t, 3> & grid_size,
    DualGridMeshF32 & output,
    std::string * error = nullptr);

// Convert the seven-channel inference output of FlexiDualGridVaeDecoder into
// one mesh per batch.  Channels 0..2 are sigmoid dual-vertex logits, 3..5 are
// intersected-edge logits, and channel 6 is the positive quad split weight.
// The decoder coordinates are [batch,x,y,z] on a cubic grid.
bool flexi_dual_grid_decode_mesh_f32(
    const SparseTensorF32 & decoded,
    int resolution,
    float voxel_margin,
    std::vector<DualGridMeshF32> & outputs,
    std::string * error = nullptr);

} // namespace pixal3d
