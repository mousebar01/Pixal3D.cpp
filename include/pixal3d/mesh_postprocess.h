// Mesh postprocess chain for the textured GLB export, adapted from
// pwilkin/trellis.cpp 2516c48b (MIT): weld hairline cracks, unify winding,
// drop floating fragments, Taubin smoothing, fan hole filling, CuMesh-port
// QEM decimation, then xatlas unwrap with the trilinear voxel-PBR bake.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "pixal3d/tri_bvh.h"

namespace pixal3d {

struct BakedMesh {
    std::vector<float>   verts;   // [Vo*3] atlas vertices (positions, decoder space)
    std::vector<float>   uv;      // [Vo*2] normalized [0,1] in GLB convention
    std::vector<int32_t> faces;   // [Fo*3]
    std::vector<uint8_t> base;    // [T*T*4] RGBA base color
    std::vector<uint8_t> mr;      // [T*T*4] RGBA glTF metallic-roughness (G=rough, B=metal)
    int T = 0;
    bool ok() const { return T > 0 && !faces.empty(); }
};

// Sparse per-voxel PBR field at grid resolution `res` (mesh space [-0.5,0.5]^3;
// voxel i covers [i/res-0.5, (i+1)/res-0.5)). feats layout [N*6] in [0,1]:
// base RGB, metallic, roughness, alpha. Matches the reference bake, which
// trilinearly samples this volume per texel (texturing.py grid_sample_3d)
// instead of interpolating decimation-averaged per-vertex colors. When `snap`
// is set (a BVH over the pre-decimation mesh), texels whose position falls
// off the voxel shell are first snapped to the closest surface point - the
// reference's cuBVH unsigned_distance correction.
struct VoxelPbr {
    const std::vector<std::array<int, 3>>* coords = nullptr;
    const std::vector<float>* feats = nullptr;
    int res = 0;
    const TriBvh* snap = nullptr;
    bool ok() const { return coords && feats && res > 0 && !coords->empty(); }
};

// Faithful CPU port of CuMesh's QEM edge-collapse simplifier (refs/CuMesh/src/simplify.cu):
// Garland-Heckbert quadrics + a skinny-triangle shape penalty + flip rejection + boundary
// weighting, driven by the reference threshold ladder. Produces the reference's adaptive,
// low-sliver triangulation from a dense dual-contour mesh, unlike the meshopt/FQMS path.
// This explicit entry point is used by the CUDA parity fixture; production callers should
// use decimate_qem(), which prefers the GPU path in CUDA builds and logs any CPU fallback.
void decimate_qem_cpu(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                      int target_faces, std::vector<float>& ov, std::vector<int32_t>& of);

// GPU QEM prototype. Available only in PIXAL3D_ENABLE_CUDA builds. It preserves the
// input mesh on failure and returns false so the caller can select the CPU reference.
// When provided, failure_reason receives a human-readable CUDA/HIP/device reason.
#ifdef PIXAL3D_HAVE_GPU_DECIMATE
bool decimate_qem_gpu(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                      int target_faces, std::vector<float>& ov, std::vector<int32_t>& of,
                      std::string * failure_reason = nullptr);
#endif

// Preferred QEM entry point: CUDA when the custom postprocess kernel is built, otherwise CPU.
void decimate_qem(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                  int target_faces, std::vector<float>& ov, std::vector<int32_t>& of);

// Drop connected components (shared-vertex face adjacency) whose face count is below
// frac*(largest component's). Removes decode floaters + spurious ground fragments.
// In place; returns the number of components dropped.
int drop_small_components(std::vector<float>& verts, std::vector<int32_t>& faces, float frac = 0.02f);

// Taubin (lambda/mu) shrink-free Laplacian smoothing. Strips the ~1-voxel stair-step noise
// of the dual-contour surface so the quadric simplifier stays curvature-adaptive instead of
// emitting a uniform-dense sliver mesh. Boundary verts pinned. In place.
void taubin_smooth(std::vector<float>& verts, const std::vector<int32_t>& faces,
                   int iters = 5, float lambda = 0.5f, float mu = -0.53f);

// In-place cleanup of a welded surface mesh (faces only): drop degenerate/duplicate faces
// and unify face orientations by BFS over manifold-edge adjacency. The dual grid mesh is
// heavily inconsistently wound, which stalls meshopt-style collapse and shatters xatlas
// charts into single-texel fragments.
void clean_mesh(int V, std::vector<int32_t>& faces);

// Fan-fill boundary loops of at most max_loop edges in place; returns the number of holes filled.
int fill_small_holes(std::vector<int32_t>& faces, int max_loop = 64);

// Merge vertices within `step` of each other in place (optionally remapping per-vertex RGB);
// returns the number of duplicates removed. The dual-grid decoder emits epsilon-different
// positions for corners shared across cells, so the raw mesh is full of hairline cracks that
// read as borders: they produce pinhole boundary loops and block edge-collapse simplification.
int weld_vertices(std::vector<float>& verts, std::vector<int32_t>& faces,
                  std::vector<float>* colors3 = nullptr, float step = 1.0f / 8192.0f);

// verts [V*3], faces [F*3], pbr6 [V*6] per-vertex (base3, metallic, roughness, alpha) in [0,1].
// Unwraps with xatlas, shades texels from `vox` (trilinear volume sampling) when provided,
// else from interpolated per-vertex PBR; inpaints seams (Telea) and dilates.
BakedMesh uv_bake(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                  const std::vector<float>& pbr6, int texsize, const VoxelPbr* vox = nullptr);

} // namespace pixal3d
