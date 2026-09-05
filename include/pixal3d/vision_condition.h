#pragma once

#include "pixal3d/dino_vit.h"
#include "pixal3d/image.h"
#include "pixal3d/naf.h"
#include "pixal3d/pipeline.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pixal3d {

// Assemble one native DINO/NAF encoder result into the same stage contract
// used by P3DCOND files.  dino.patch_map is the low-resolution map; when naf
// is present, its HxWxC output becomes the optional high-resolution map.  Both
// maps use the original square image resolution for projection coordinates,
// even when NAF's target map is smaller (for example the 1024px stage uses a
// 512px NAF target in the reference pipeline).
bool make_pixal3d_condition_stage_f32(
    const DinoV3FeaturesF32 & dino,
    const NafOutputF32 * naf,
    const std::string & name,
    int image_resolution,
    Pixal3DConditionStageF32 & output,
    std::string * error = nullptr);

// Run one native DINO/NAF image-condition stage from a square CHW image in
// [0,1].  The function applies the ImageNet normalization required by DINO,
// keeps the unnormalized image as NAF's guide, and assembles the same stage
// contract as make_pixal3d_condition_stage_f32().  naf_output_resolution == 0
// disables NAF; otherwise the native NAF model must be loaded and the output
// map is square with that resolution.
bool encode_pixal3d_condition_stage_f32(
    const DinoV3Model & dino_model,
    const NafModel * naf_model,
    const float * image_chw_01,
    int image_height,
    int image_width,
    int image_resolution,
    int naf_output_resolution,
    const std::string & name,
    Pixal3DConditionStageF32 & output,
    std::string * error = nullptr);

// Description of one image buffer used by the bundle helper.  The buffer is
// borrowed for the duration of the call and must contain exactly
// 3*image_resolution*image_resolution CHW F32 values in [0,1].
struct Pixal3DVisionStageInputF32 {
    std::string name;
    int image_resolution = 0;
    int naf_output_resolution = 0;
    const float * image_chw_01 = nullptr;
    std::size_t image_elements = 0;
};

// Stage image sizes used by the released single-view pipeline.  The source
// image is resized independently for each entry because the 512 and 1024
// DINO calls are distinct model inputs.  NAF resolution zero disables the
// high-resolution guide for that stage (the SS stage has no NAF branch).
struct Pixal3DImageConditionBundleConfig {
    int ss_resolution = 512;
    int shape_512_resolution = 512;
    int shape_1024_resolution = 1024;
    int tex_1024_resolution = 1024;
    int ss_naf_resolution = 0;
    int shape_512_naf_resolution = 512;
    int shape_1024_naf_resolution = 512;
    int tex_1024_naf_resolution = 1024;
};

// Resize one decoded RGB image into the four configured stage inputs and run
// the native DINO/NAF encoders.  This returns a complete four-stage bundle
// ready for save_pixal3d_condition_bundle() or the in-memory cascade API.
bool encode_pixal3d_condition_bundle_from_image_f32(
    const DinoV3Model & dino_model,
    const NafModel * naf_model,
    const Pixal3DImageF32 & image,
    const Pixal3DImageConditionBundleConfig & config,
    Pixal3DConditionBundleF32 & output,
    std::string * error = nullptr);

// Encode a set of stages into a validated in-memory P3DCOND bundle.  This
// keeps stage-specific image sizes and NAF targets explicit (512/512,
// 1024/512, and 1024/1024 in the released pipeline) while allowing callers to
// save the result with save_pixal3d_condition_bundle() or pass it directly to
// run_pixal3d_from_condition_stages().
bool encode_pixal3d_condition_bundle_f32(
    const DinoV3Model & dino_model,
    const NafModel * naf_model,
    const std::vector<Pixal3DVisionStageInputF32> & inputs,
    Pixal3DConditionBundleF32 & output,
    std::string * error = nullptr);

// Assemble a multi-view stage from already encoded per-view DINO/NAF output.
// The optional NAF vector is either empty (no NAF for any view) or contains one
// pointer per view; all entries must be non-null so view channel contracts stay
// identical for ProjGridMV average fusion.
bool make_pixal3d_multiview_condition_stage_f32(
    const std::vector<DinoV3FeaturesF32> & dino_views,
    const std::vector<const NafOutputF32 *> & naf_views,
    const std::vector<ProjectionCamera> & cameras,
    const std::string & name,
    int image_resolution,
    Pixal3DConditionStageMVF32 & output,
    std::string * error = nullptr);

// Project one assembled native-vision stage at the exact sparse coordinates
// requested by run_pixal3d_cascade_f32().  The output is ready to return from
// Pixal3DConditionBuilderF32, including concatenated DINO+NAF rows.
bool make_pixal3d_image_condition_f32(
    const Pixal3DConditionStageF32 & stage,
    const ProjectionCamera & camera,
    int grid_resolution,
    const std::vector<std::int32_t> & coords,
    Pixal3DImageConditionF32 & output,
    std::string * error = nullptr);

} // namespace pixal3d
