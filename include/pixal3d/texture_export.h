#pragma once

#include "pixal3d/dual_grid.h"
#include "pixal3d/sparse.h"

#include <string>

namespace pixal3d {

// Native approximate PBR export.  The texture atlas is baked from the
// six-channel sparse texture decoder output using the canonical decoder AABB.
// GLB positions and normals use the final Python textured-GLB frame
// (x, y, z) -> (-x, +y, -z).
struct Pixal3DGlbOptions {
    int texture_size = 1024;
};

// Write one mesh and its decoded Pixal3D PBR voxel attributes as a glTF 2.0
// binary asset.  The texture channel contract is fixed to RGB base color,
// metallic, roughness, and alpha in channels 0..5 respectively.  UV unwrap
// and baking are deterministic CPU approximations of the Python postprocess;
// model inference and the public mesh representation are not modified.
bool write_pixal3d_glb(
    const DualGridMeshF32 & mesh,
    const SparseTensorF32 & texture_decoded,
    int resolution,
    const Pixal3DGlbOptions & options,
    const std::string & path,
    std::string * error = nullptr);

} // namespace pixal3d
