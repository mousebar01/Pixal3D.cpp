#pragma once

#include "pixal3d/dual_grid.h"

#include <string>

namespace pixal3d {

// Fill small, manifold boundary loops in a raw Dual Grid mesh.  This mirrors
// the observable contract of the Python CuMesh postprocess, but its CPU
// implementation is not guaranteed to be bit-for-bit identical to CuMesh.
// Invalid packed arrays are rejected; non-manifold topology is left unchanged
// because CuMesh also has no fillable boundary loop in that case.
bool fill_mesh_holes_f32(
    DualGridMeshF32 & mesh,
    float max_hole_perimeter = 3.0e-2f,
    std::string * error = nullptr);

} // namespace pixal3d
