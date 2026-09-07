#pragma once

#include "pixal3d/condition.h"
#include "pixal3d/pipeline.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pixal3d {

// Options for running the production seven-model cascade from an externally
// generated P3DCOND file.  The default values mirror weights/Pixal3D's
// pipeline.json (1536 cascade, 12 Euler steps).  max_model_bytes is a hard
// pre-load resident F32 weight estimate; zero disables that guard.
struct Pixal3DInferenceConfig {
    Pixal3DCascadeConfig cascade;
    ProjectionCamera camera = ProjectionCamera::front(0.8575560450553894f, 2.0f);
    std::uint64_t seed = 42;
    std::size_t max_model_bytes = 0;
};

Pixal3DInferenceConfig default_pixal3d_inference_config();

// Estimate the resident host bytes required when the selected GGUF tensors
// are converted to F32 by the loaders.  This parses metadata only and does
// not allocate or read the multi-gigabyte payloads.
bool estimate_pixal3d_model_bytes(const std::string & shared_pack,
                                  const std::string & flow_pack,
                                  std::size_t & bytes,
                                  std::string * error = nullptr);

// Load the three decoders, three base SLat/texture flow components, and
// sparse-structure flow from the two grouped packs, then run the cascade
// using global + projected conditions from condition_path.
bool run_pixal3d_from_condition_bundle(
    const std::string & shared_pack,
    const std::string & flow_pack,
    const std::string & condition_path,
    const Pixal3DInferenceConfig & config,
    Pixal3DCascadeOutputF32 & output,
    std::string * error = nullptr);

// Run the same cascade from already assembled native vision stages.  This is
// the in-memory counterpart of run_pixal3d_from_condition_bundle(): callers
// can run DinoV3Model/NafModel, use make_pixal3d_condition_stage_f32(), and
// pass the four HWC maps directly without writing a temporary P3DCOND file.
bool run_pixal3d_from_condition_stages(
    const std::string & shared_pack,
    const std::string & flow_pack,
    const std::vector<Pixal3DConditionStageF32> & stages,
    const Pixal3DInferenceConfig & config,
    Pixal3DCascadeOutputF32 & output,
    std::string * error = nullptr);

// Multi-view counterpart of run_pixal3d_from_condition_bundle().  The
// P3DMVCON bundle is fused with ProjGridMV's reference average policy before
// entering the same seven-model cascade; the supplied flow pack may be the
// converted pixal3d-mv-flow bundle.
bool run_pixal3d_from_multiview_condition_bundle(
    const std::string & shared_pack,
    const std::string & flow_pack,
    const std::string & condition_path,
    const Pixal3DInferenceConfig & config,
    Pixal3DCascadeOutputF32 & output,
    std::string * error = nullptr);

// In-memory counterpart of run_pixal3d_from_multiview_condition_bundle().
// The supplied native per-view stages are fused with the reference average
// policy and then passed through the same multi-view flow cascade.
bool run_pixal3d_from_multiview_condition_stages(
    const std::string & shared_pack,
    const std::string & flow_pack,
    const std::vector<Pixal3DConditionStageMVF32> & stages,
    const Pixal3DInferenceConfig & config,
    Pixal3DCascadeOutputF32 & output,
    std::string * error = nullptr);

// Write one decoded mesh as an ordinary 1-indexed Wavefront OBJ.  Vertices
// are emitted in the Python reference export frame (x, y, z) -> (-x, -z, -y);
// DualGridMeshF32 remains in the canonical decoder mesh frame.  Texture voxel
// attributes remain available in Pixal3DCascadeOutputF32::texture_decoded for
// a future material/voxel sidecar writer.
bool write_pixal3d_obj(const DualGridMeshF32 & mesh,
                       const std::string & path,
                       std::string * error = nullptr);

} // namespace pixal3d
