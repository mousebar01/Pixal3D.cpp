#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pixal3d {

// Variable-length token rows used for global image conditioning.  offsets has
// batch_size + 1 entries and feats is row-major [token, channel].
struct VarLenTensorF32 {
    int batch_size = 0;
    int channels = 0;
    std::vector<std::size_t> offsets;
    std::vector<float> feats;

    std::size_t tokens() const noexcept {
        return channels > 0 ? feats.size() / static_cast<std::size_t>(channels) : 0;
    }
    bool valid(std::string * error = nullptr) const;
};

// CPU reference representation used by the SLat stages.  Coordinates are
// contiguous rows of [batch, x, y, z], and features are contiguous rows of
// [active_point, channel].  Coordinate-preserving operations retain row order;
// reduction operations document when they intentionally sort or coalesce
// rows.  This matches Pixal3D's SparseTensor convention.
struct SparseTensorF32 {
    int batch_size = 0;
    int channels = 0;
    int spatial_x = 0;
    int spatial_y = 0;
    int spatial_z = 0;
    std::vector<std::int32_t> coords;
    std::vector<float> feats;

    std::size_t points() const noexcept { return coords.size() / 4; }
    bool valid(std::string * error = nullptr) const;
};

// Apply a PyTorch/ggml linear weight with source layout [out, in].
bool sparse_linear(const SparseTensorF32 & input,
                  const float * weight,
                  const float * bias,
                  int out_channels,
                  SparseTensorF32 & output,
                  std::string * error = nullptr);

// Apply the same [out,in] linear contract to variable-length token rows.
bool varlen_linear(const VarLenTensorF32 & input,
                   const float * weight,
                   const float * bias,
                   int out_channels,
                   VarLenTensorF32 & output,
                   std::string * error = nullptr);

// Normalize each active point over its feature channels.  Passing both
// gamma and beta applies elementwise affine parameters; passing both null
// selects the non-affine path used by the sparse transformer blocks.
bool sparse_layer_norm(const SparseTensorF32 & input,
                       float epsilon,
                       const float * gamma,
                       const float * beta,
                       SparseTensorF32 & output,
                       std::string * error = nullptr);

// Average-pool active points into a coarser integer grid.  Coordinates are
// divided by factor, duplicate cells are reduced by mean, and output rows are
// sorted lexicographically by [batch,x,y,z] like torch.unique in the Python
// reference.  The output spatial shape uses ceil(input_shape / factor).
bool sparse_downsample_mean(const SparseTensorF32 & input,
                            int factor,
                            SparseTensorF32 & output,
                            std::string * error = nullptr);

// Rearrange packed child voxels from channels into spatial coordinates.  The
// input channel count must be divisible by factor^3; child index i uses
// little-endian x/y/z bits, matching Pixal3D's SparseChannel2Spatial.  When
// subdivision is non-null it must have the same coordinates and factor^3
// channels; values greater than zero select the active children.
bool sparse_channel_to_spatial(const SparseTensorF32 & input,
                               int factor,
                               const SparseTensorF32 * subdivision,
                               SparseTensorF32 & output,
                               std::string * error = nullptr);

// Batch-isolated scaled dot-product attention.  Query/key features are
// flattened [point, head, head_dim], value features are
// [point, head, value_dim], and output keeps query coordinates with
// [point, head, value_dim] features.  Q/K/V may have different active point
// counts per batch; rows must remain contiguous by batch.
bool sparse_scaled_dot_product_attention(const SparseTensorF32 & query,
                                         const SparseTensorF32 & key,
                                         const SparseTensorF32 & value,
                                         int num_heads,
                                         int head_dim,
                                         float scale,
                                         SparseTensorF32 & output,
                                         std::string * error = nullptr);

// Cross-attention variant with sparse queries and variable-length dense
// key/value tokens.  key/value offsets must match, while query active rows
// retain their coordinate order in the output.
bool sparse_cross_attention(const SparseTensorF32 & query,
                            const VarLenTensorF32 & key,
                            const VarLenTensorF32 & value,
                            int num_heads,
                            int head_dim,
                            float scale,
                            SparseTensorF32 & output,
                            std::string * error = nullptr);

// Apply the reference 3D rotary position embedding to [point, head, dim]
// features.  The first head_dim/2 complex pairs consume x, y, z coordinate
// frequencies in that order; any remaining pairs are unrotated.
bool sparse_rotary_position_embedding(const SparseTensorF32 & input,
                                      int num_heads,
                                      int head_dim,
                                      float frequency_min,
                                      float frequency_base,
                                      SparseTensorF32 & output,
                                      std::string * error = nullptr);

// Apply the reference SparseMultiHeadRMSNorm to each [head, head_dim] slice.
// Gamma is flattened in head-major order and the normalized values are
// multiplied by sqrt(head_dim), matching torch.nn.functional.normalize.
bool sparse_multihead_rms_norm(const SparseTensorF32 & input,
                               int num_heads,
                               int head_dim,
                               const float * gamma,
                               SparseTensorF32 & output,
                               std::string * error = nullptr);

// Apply the same per-head RMS normalization to variable-length token rows.
// Gamma is flattened in head-major order and the output preserves offsets.
bool varlen_multihead_rms_norm(const VarLenTensorF32 & input,
                               int num_heads,
                               int head_dim,
                               const float * gamma,
                               VarLenTensorF32 & output,
                               std::string * error = nullptr);

// Apply a 3x3x3 submanifold convolution.  The compact-v1 SLat weight layout
// is [kernel, out, in], with kernel = kd * 9 + kh * 3 + kw and offsets
// kd/kh/kw in {-1, 0, +1}.  Missing neighbors contribute zero; outputs keep
// exactly the input coordinates and ordering.  This is the CPU fallback for
// the flex_gemm sparse_submanifold_conv3d operation used by the reference.
bool sparse_submanifold_conv3d(const SparseTensorF32 & input,
                               const float * weight,
                               const float * bias,
                               int out_channels,
                               SparseTensorF32 & output,
                               std::string * error = nullptr);

} // namespace pixal3d
