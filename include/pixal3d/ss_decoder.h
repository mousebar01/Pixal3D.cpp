#pragma once

#include "pixal3d/sparse.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pixal3d {

struct SSDecoderHParams {
    int resolution = 16;
    int out_channels = 0;
    int latent_channels = 0;
    int num_res_blocks = 0;
    int num_res_blocks_middle = 0;
    std::vector<int> channels;
    float norm_eps = 1e-5f;
    std::string norm_type = "layer";

    int upscale() const noexcept {
        int value = 1;
        for (std::size_t i = 1; i < channels.size(); ++i) value *= 2;
        return value;
    }

    int output_resolution() const noexcept {
        return resolution * upscale();
    }
};

// Convert one channel-major occupancy-logit grid into [batch,x,y,z]
// coordinates.  The reference pipeline first thresholds at zero and, when a
// lower target resolution is requested, applies binary max-pooling with a
// matching integer stride.  Passing target_resolution <= 0 keeps the decoder
// output resolution.
bool ss_occupancy_to_coords_f32(const float * occupancy_logits,
                                int out_channels,
                                int output_resolution,
                                float threshold,
                                int target_resolution,
                                std::vector<std::int32_t> & coords,
                                std::string * error = nullptr);

// Stage-1 dense sparse-structure decoder.  The implementation follows the
// trellis2cpp Conv3d/ChannelLayerNorm/pixel-shuffle graph while using the
// ss_decoder.* compact-v1 namespace in the shared Pixal3D pack.  CUDA builds
// use an F32 im2col + matmul Conv3d recipe when supported; CPU keeps the
// native Conv3d fallback.
class SSDecoderModel {
public:
    SSDecoderModel() = default;
    ~SSDecoderModel();

    SSDecoderModel(const SSDecoderModel &) = delete;
    SSDecoderModel & operator=(const SSDecoderModel &) = delete;

    bool load(const std::string & path,
              bool load_tensors = true,
              std::string * error = nullptr);
    void close() noexcept;

    bool is_loaded() const noexcept;
    bool has_data() const noexcept;
    const SSDecoderHParams & hparams() const noexcept;
    const std::string & backend_name() const noexcept;
    int tensor_count() const noexcept;
    bool has_tensor(const std::string & name) const noexcept;

    // Decode channel-major latent [latent_channels, resolution^3] into
    // channel-major occupancy logits [out_channels, output_resolution^3].
    bool decode(const float * latent,
                float * output,
                std::string * error = nullptr);

    // Decode and immediately apply the reference occupancy threshold and
    // optional integer max-pool, returning one batch of [x,y,z] grid points
    // in [batch,x,y,z] coordinate form.
    bool decode_coords(const float * latent,
                       float threshold,
                       int target_resolution,
                       std::vector<std::int32_t> & coords,
                       std::string * error = nullptr);

    // Convenience bridge for the stage-1 sampler: accepts a complete
    // regular-grid SparseTensorF32, converts its row-major features to the
    // decoder's channel-major layout, and returns thresholded coordinates.
    bool decode_sparse_coords(const SparseTensorF32 & latent,
                              float threshold,
                              int target_resolution,
                              std::vector<std::int32_t> & coords,
                              std::string * error = nullptr);

private:
    struct Impl;
    Impl * impl_ = nullptr;
};

} // namespace pixal3d
