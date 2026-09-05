#pragma once

#include "pixal3d/ss_decoder.h"
#include "pixal3d/ss_flow.h"
#include "pixal3d/slat_decoder.h"
#include "pixal3d/slat_flow.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace pixal3d {

// Image features produced by an external image-condition encoder.  Keeping
// this boundary explicit lets the ggml port use the native DINOv3/NAF bridge
// or precomputed F32 features without changing the denoiser and decoder
// stages.
struct Pixal3DImageConditionF32 {
    VarLenTensorF32 global;
    SparseTensorF32 projection;

    bool has_projection() const noexcept {
        return projection.batch_size > 0 && projection.channels > 0;
    }
};

struct SparseStructureStageConfig {
    FlowEulerSamplerConfig sampler;
    float occupancy_threshold = 0.0f;
    int target_resolution = 0;
};

struct SparseStructureStageOutputF32 {
    FlowEulerSampleF32 flow;
    std::vector<std::int32_t> coords;
};

// The normalization used by the shape/texture SLat samplers.  The Python
// pipeline denormalizes every sampled latent as latent * std + mean before it
// is consumed by the decoder or the next cascade stage.
struct SLatNormalizationF32 {
    std::vector<float> mean;
    std::vector<float> std;
};

struct SLatStageOutputF32 {
    FlowEulerSampleF32 flow;
    SparseTensorF32 latent;
};

enum class Pixal3DCascadeStage {
    sparse_structure,
    shape_slat_low,
    shape_slat_high,
    texture_slat,
};

// The vision side is deliberately a callback.  A producer may use native
// DINOv3/NAF, a Python helper, or a precomputed file, but every stage must
// return the global tokens and sparse projection rows for the exact requested
// coordinates.
using Pixal3DConditionBuilderF32 = std::function<bool(
    Pixal3DCascadeStage stage,
    const std::vector<std::int32_t> & coords,
    int grid_resolution,
    Pixal3DImageConditionF32 & output,
    std::string * error)>;

// Noise is also supplied by the caller so seed handling and memory placement
// stay outside the model graph.  For texture_slat, channels is the number of
// fresh noise channels; the shape latent is passed separately as concat_cond.
using Pixal3DNoiseBuilderF32 = std::function<bool(
    Pixal3DCascadeStage stage,
    const std::vector<std::int32_t> & coords,
    int channels,
    int grid_resolution,
    SparseTensorF32 & output,
    std::string * error)>;

struct Pixal3DCascadeConfig {
    SparseStructureStageConfig sparse_structure;
    FlowEulerSamplerConfig shape_sampler;
    FlowEulerSamplerConfig texture_sampler;
    SLatNormalizationF32 shape_normalization;
    SLatNormalizationF32 texture_normalization;
    int requested_resolution = 1024;
    std::size_t max_num_tokens = 49152;
    // Optional development-only cap applied after SS occupancy extraction.
    // Zero preserves the production behavior and keeps every active point.
    std::size_t max_structure_points = 0;
    int decoder_upsample_times = 4;
    float voxel_margin = 0.5f;
};

struct Pixal3DCascadeOutputF32 {
    int resolution = 0;
    SparseStructureStageOutputF32 sparse_structure;
    SLatStageOutputF32 shape_slat_low;
    SparseTensorF32 shape_upsampled;
    std::vector<std::int32_t> high_coords;
    SLatStageOutputF32 shape_slat_high;
    SLatStageOutputF32 texture_slat;
    SparseTensorF32 shape_decoded;
    SparseTensorF32 texture_decoded;
    std::vector<SparseTensorF32> shape_subdivisions;
    std::vector<DualGridMeshF32> meshes;
};

// Run the complete external-condition Pixal3D cascade.  The callbacks are
// invoked in stage order and must construct tensors with the supplied exact
// coordinates.  This function mirrors the Python pipeline's SS -> shape LR ->
// decoder coordinate upsample/quantization -> shape HR -> texture -> decode
// order, including SLat normalization, texture concat_cond, guide
// subdivisions, and Flexible Dual Grid mesh extraction.
bool run_pixal3d_cascade_f32(
    SSFlowModel & ss_flow,
    SSDecoderModel & ss_decoder,
    SLatFlowModel & shape_flow_low,
    SLatFlowModel & shape_flow_high,
    SLatFlowModel & texture_flow,
    SLatDecoderModel & shape_decoder,
    SLatDecoderModel & texture_decoder,
    const Pixal3DCascadeConfig & config,
    const Pixal3DNoiseBuilderF32 & noise_builder,
    const Pixal3DConditionBuilderF32 & condition_builder,
    Pixal3DCascadeOutputF32 & output,
    std::string * error = nullptr);

// Bind one SLat flow sampler to the reference latent denormalization step.
// Noise and image conditions remain explicit inputs so callers can control
// RNG, projected features, and multi-view batches outside this API.
bool run_slat_stage_f32(
    SLatFlowModel & flow_model,
    const SparseTensorF32 & noise,
    const Pixal3DImageConditionF32 & condition,
    const FlowEulerSamplerConfig & sampler_config,
    const SLatNormalizationF32 & normalization,
    SLatStageOutputF32 & output,
    std::string * error = nullptr,
    const SparseTensorF32 * concat_condition = nullptr);

// Match Pixal3DImageTo3DPipeline.sample_shape_slat_cascade(): map decoder
// coordinates at low-resolution grid units to a high-resolution SLat grid,
// lexicographically unique them, and lower the requested resolution by 128
// until max_num_tokens is met (the reference never lowers below 1024).
bool quantize_slat_coords_f32(
    const SparseTensorF32 & upsampled_coords,
    int low_resolution,
    int requested_resolution,
    std::size_t max_num_tokens,
    std::vector<std::int32_t> & coords,
    int & actual_resolution,
    std::string * error = nullptr);

// Run the complete stage-1 contract: dense SS-flow Euler sampling followed by
// SS decoder occupancy thresholding/max-pooling.  The input noise is a full
// regular grid and the condition tensors are expected to be generated by an
// image feature encoder; this function deliberately does not hide or invent
// that encoder.
bool run_sparse_structure_stage_f32(
    SSFlowModel & flow_model,
    SSDecoderModel & decoder_model,
    const SparseTensorF32 & noise,
    const Pixal3DImageConditionF32 & condition,
    const SparseStructureStageConfig & config,
    SparseStructureStageOutputF32 & output,
    std::string * error = nullptr);

} // namespace pixal3d
