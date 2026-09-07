#pragma once

#include "pixal3d/sparse.h"
#include "pixal3d/dual_grid.h"

#include <string>
#include <vector>

namespace pixal3d {

enum class SLatDecoderBlockKind {
    convnext,
    channel_to_spatial,
};

// One sparse decoder block.  All linear weights use [out,in] and all
// convolution weights use compact-v1 [kernel_volume,out,in] layout.
struct SLatDecoderBlockWeightsF32 {
    SLatDecoderBlockKind kind = SLatDecoderBlockKind::convnext;
    int channels = 0;
    int out_channels = 0;
    const float * norm_weight = nullptr;
    const float * norm_bias = nullptr;
    const float * norm1_weight = nullptr;
    const float * norm1_bias = nullptr;
    const float * conv1_weight = nullptr;
    const float * conv1_bias = nullptr;
    const float * norm2_weight = nullptr;
    const float * norm2_bias = nullptr;
    const float * conv2_weight = nullptr;
    const float * conv2_bias = nullptr;
    const float * mlp0_weight = nullptr;
    const float * mlp0_bias = nullptr;
    const float * mlp2_weight = nullptr;
    const float * mlp2_bias = nullptr;
    const float * to_subdiv_weight = nullptr;
    const float * to_subdiv_bias = nullptr;
    int mlp_hidden = 0;
};

// The final decoder normalization mirrors Python's F.layer_norm call without
// an explicit eps argument; intermediate block normalizations use norm_eps.
constexpr float k_slat_decoder_final_layer_norm_eps = 1.0e-5f;

struct SLatDecoderConfig {
    int latent_channels = 0;
    int out_channels = 0;
    float norm_eps = 1e-6f;
    bool pred_subdiv = false;
    std::vector<int> model_channels;
    std::vector<int> num_blocks;
};

struct SLatDecoderWeightsF32 {
    const float * from_latent_weight = nullptr;
    const float * from_latent_bias = nullptr;
    const float * output_weight = nullptr;
    const float * output_bias = nullptr;
    std::vector<SLatDecoderBlockWeightsF32> blocks;
};

// Evaluate one SparseConvNeXtBlock3d from Pixal3D's sparse decoder.
bool sparse_convnext_block_f32(const SparseTensorF32 & input,
                               const SLatDecoderBlockWeightsF32 & weights,
                               float norm_eps,
                               SparseTensorF32 & output,
                               std::string * error = nullptr);

// Evaluate one SparseResBlockC2S3d.  A non-null subdivision selects active
// children using values > 0; when pred_subdiv is true, the block predicts and
// returns its own subdivision logits in predicted_subdiv.
bool sparse_resblock_c2s_f32(
    const SparseTensorF32 & input,
    const SLatDecoderBlockWeightsF32 & weights,
    float norm_eps,
    bool pred_subdiv,
    const SparseTensorF32 * subdivision,
    SparseTensorF32 & output,
    SparseTensorF32 * predicted_subdiv = nullptr,
    std::string * error = nullptr);

// Evaluate a complete sparse SLat decoder stack.  Blocks are flattened in
// level order: num_blocks[level] ConvNeXt blocks followed by one C2S block
// for each non-final level.  guide_subdivisions is optional for a decoder
// with pred_subdiv=false and has one entry per upsample level.
bool slat_decoder_forward_f32(
    const SparseTensorF32 & input,
    const SLatDecoderConfig & config,
    const SLatDecoderWeightsF32 & weights,
    const std::vector<SparseTensorF32> * guide_subdivisions,
    SparseTensorF32 & output,
    std::vector<SparseTensorF32> * predicted_subdivisions = nullptr,
    std::string * error = nullptr);

// Run only the coordinate-producing prefix of a pred_subdiv shape decoder.
// upsample_times=0 returns the coordinates after the latent projection;
// upsample_times=N returns the coordinates after N channel-to-spatial levels.
// Feature channels are retained in output solely to keep the result a valid
// SparseTensorF32 for subsequent coordinate quantization.
bool slat_decoder_upsample_coords_f32(
    const SparseTensorF32 & input,
    const SLatDecoderConfig & config,
    const SLatDecoderWeightsF32 & weights,
    int upsample_times,
    SparseTensorF32 & output,
    std::string * error = nullptr);

struct SLatDecoderHParams {
    std::string component;
    std::string model_class;
    int resolution = 0;
    int latent_channels = 0;
    int out_channels = 0;
    float norm_eps = 1e-6f;
    bool pred_subdiv = false;
    std::vector<int> model_channels;
    std::vector<int> num_blocks;
};

// F32 loader for one logical shape_decoder or texture_decoder component in a
// grouped GGUF pack. F16/BF16 payloads are converted once at load time so the
// validated CPU kernels and the optional ggml GPU sparse-convolution path can
// consume stable float pointers. CUDA builds select the GPU path by default;
// PIXAL3D_SLAT_DECODER_BACKEND=cpu forces the reference path.
class SLatDecoderModel {
public:
    SLatDecoderModel() = default;
    ~SLatDecoderModel();

    SLatDecoderModel(const SLatDecoderModel &) = delete;
    SLatDecoderModel & operator=(const SLatDecoderModel &) = delete;

    bool load(const std::string & path,
              const std::string & component,
              bool load_tensors = true,
              std::string * error = nullptr);
    void close() noexcept;

    bool is_loaded() const noexcept;
    bool has_data() const noexcept;
    const SLatDecoderHParams & hparams() const noexcept;
    int tensor_count() const noexcept;
    bool has_tensor(const std::string & name) const noexcept;

    // Decode a sparse latent.  Shape decoders (pred_subdiv=true) return the
    // subdivision logits needed by the texture decoder; texture decoders
    // require one guide subdivision per upsample level.
    bool decode(const SparseTensorF32 & input,
                const std::vector<SparseTensorF32> * guide_subdivisions,
                SparseTensorF32 & output,
                std::vector<SparseTensorF32> * predicted_subdivisions = nullptr,
                std::string * error = nullptr);

    // Shape-decoder convenience path: decode seven FlexiDualGrid channels and
    // reconnect them into one mesh per batch.  A non-positive resolution uses
    // the GGUF metadata; voxel_margin follows the Python decoder default 0.5.
    bool decode_shape_mesh(
        const SparseTensorF32 & input,
        const std::vector<SparseTensorF32> * guide_subdivisions,
        std::vector<DualGridMeshF32> & meshes,
        std::vector<SparseTensorF32> * predicted_subdivisions = nullptr,
        int resolution_override = 0,
        float voxel_margin = 0.5f,
        std::string * error = nullptr);

    // Return the sparse coordinates after the requested number of shape
    // decoder upsample levels without evaluating the final occupancy head.
    bool upsample_coords(const SparseTensorF32 & input,
                         int upsample_times,
                         SparseTensorF32 & output,
                         std::string * error = nullptr);

private:
    struct Impl;
    Impl * impl_ = nullptr;
};

} // namespace pixal3d
