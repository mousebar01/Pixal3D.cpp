#pragma once

#include "pixal3d/projection.h"
#include "pixal3d/sparse.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pixal3d {

// One row-major HxWxC feature map exported by an external image encoder.
// image_resolution is the square image-space size used by ProjGrid; it is
// intentionally independent of the map's H/W (DINO patch maps are smaller
// than the input image, while NAF maps may be larger).
struct Pixal3DFeatureMapF32 {
    int image_resolution = 0;
    int height = 0;
    int width = 0;
    int channels = 0;
    std::vector<float> features;

    std::size_t elements() const noexcept { return features.size(); }
    bool empty() const noexcept { return features.empty(); }
    bool valid(std::string * error = nullptr) const;
};

// Conditions for one of the four projection stages.  The global rows are
// normally the DINO CLS/register tokens.  dino_map is required; naf_map is
// optional and, when present, is projected independently then concatenated to
// dino_map exactly as DinoV3ProjFeatureExtractor does.
struct Pixal3DConditionStageF32 {
    std::string name;
    VarLenTensorF32 global;
    Pixal3DFeatureMapF32 dino_map;
    Pixal3DFeatureMapF32 naf_map;

    bool has_naf() const noexcept { return !naf_map.empty(); }
    bool valid(std::string * error = nullptr) const;
};

// One view of a multi-view condition stage.  camera.transform_matrix is the
// absolute c2w pose supplied by the caller; the projection helper converts the
// view set to the reference ProjGridMV relative matrices before sampling.
struct Pixal3DConditionViewF32 {
    ProjectionCamera camera;
    VarLenTensorF32 global;
    Pixal3DFeatureMapF32 dino_map;
    Pixal3DFeatureMapF32 naf_map;

    bool has_naf() const noexcept { return !naf_map.empty(); }
    bool valid(std::string * error = nullptr) const;
};

// A multi-view stage using the reference "average" fusion mode.  Each view
// keeps its own DINO/NAF maps and global tokens; projection rows and global
// tokens are averaged elementwise, yielding the same shapes expected by the
// existing denoisers.  The attention fusion mode is intentionally not encoded
// here because the current ggml flow blocks consume one projected row per
// voxel.
struct Pixal3DConditionStageMVF32 {
    std::string name;
    std::vector<Pixal3DConditionViewF32> views;

    bool valid(std::string * error = nullptr) const;
};

struct Pixal3DMultiViewConditionBundleF32 {
    std::uint32_t format_version = 0;
    std::vector<Pixal3DConditionStageMVF32> stages;

    const Pixal3DConditionStageMVF32 * find(const std::string & name) const noexcept;
    Pixal3DConditionStageMVF32 * find(const std::string & name) noexcept;
    bool valid(std::string * error = nullptr) const;
};

// A compact, external-condition file.  Its on-disk magic is P3DCOND\0 and
// its payload is little-endian F32.  It stores 2D feature maps rather than
// dense 3D projections, avoiding multi-gigabyte grid duplication and allowing
// the C++ side to handle 1024/1536 token-budget grid overrides.
struct Pixal3DConditionBundleF32 {
    std::uint32_t format_version = 0;
    std::vector<Pixal3DConditionStageF32> stages;

    const Pixal3DConditionStageF32 * find(const std::string & name) const noexcept;
    Pixal3DConditionStageF32 * find(const std::string & name) noexcept;
    bool valid(std::string * error = nullptr) const;
};

// Load a P3DCOND bundle without depending on the Python/DINO runtime.  The
// bundle stores one batch/object and canonical stage names: ss, shape_512,
// shape_1024, and tex_1024.  A producer may include a subset for stage-level
// testing, but the production cascade requires all four.
bool load_pixal3d_condition_bundle(const std::string & path,
                                   Pixal3DConditionBundleF32 & output,
                                   std::string * error = nullptr);

// Atomically write a validated P3DCOND bundle.  The byte layout is identical
// to scripts/export_pixal3d_condition_bundle.py, so a bundle produced from
// native DINO/NAF output can be consumed by the existing CLI and Python tools.
bool save_pixal3d_condition_bundle(const std::string & path,
                                   const Pixal3DConditionBundleF32 & bundle,
                                   std::string * error = nullptr);

// Project one stage's maps at the requested grid resolution and gather rows
// for exact sparse coordinates.  The returned global condition is copied
// without modification; projection rows are dino (and optional NAF)
// concatenated in feature-channel order.
bool project_condition_stage_f32(
    const Pixal3DConditionStageF32 & stage,
    const ProjectionCamera & camera,
    int grid_resolution,
    const std::vector<std::int32_t> & coords,
    VarLenTensorF32 & global,
    SparseTensorF32 & projection,
    std::string * error = nullptr);

// Fuse and project a multi-view stage using ProjGridMV's average semantics.
// The first view is the main view.  All views must provide absolute c2w
// transforms, matching the Python extractor's compute_relative_calc_mat()
// contract.  Border-sampled values are averaged even when a view is outside
// its image; validity is diagnostic only, as in the single-view bridge.
bool project_condition_stage_multiview_average_f32(
    const Pixal3DConditionStageMVF32 & stage,
    int grid_resolution,
    const std::vector<std::int32_t> & coords,
    VarLenTensorF32 & global,
    SparseTensorF32 & projection,
    std::string * error = nullptr);

// Load a multi-view condition bundle.  The on-disk magic is P3DMVCON and the
// payload is little-endian F32.  It is intentionally a separate format from
// P3DCOND so existing single-view files remain byte-for-byte stable.
bool load_pixal3d_multiview_condition_bundle(
    const std::string & path,
    Pixal3DMultiViewConditionBundleF32 & output,
    std::string * error = nullptr);

// Atomically write a validated P3DMVCON bundle using the same little-endian
// layout as scripts/export_pixal3d_multiview_condition_bundle.py.
bool save_pixal3d_multiview_condition_bundle(
    const std::string & path,
    const Pixal3DMultiViewConditionBundleF32 & bundle,
    std::string * error = nullptr);

} // namespace pixal3d
