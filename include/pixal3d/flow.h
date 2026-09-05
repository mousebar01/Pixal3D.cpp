#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace pixal3d {

// Build the float32 sinusoidal timestep embedding used by the Python
// TimestepEmbedder.  The output is row-major [batch, dim].
//
// This is kept as a small, weight-free primitive so its numerical contract
// can be checked before it is embedded in a full ggml graph.  Timesteps may
// be fractional, as they are during flow-matching integration.
bool timestep_embedding(const float * timesteps,
                        std::size_t batch,
                        int dim,
                        int max_period,
                        std::vector<float> & output,
                        std::string * error = nullptr);

// Evaluate the two Linear layers and SiLU used by TimestepEmbedder.  Weights
// use the PyTorch layout [out_features, in_features], and all buffers are
// row-major.  The result is float32 [batch, hidden_size].
bool timestep_embed_mlp(const float * timesteps,
                        std::size_t batch,
                        int hidden_size,
                        int frequency_embedding_size,
                        const float * linear0_weight,
                        const float * linear0_bias,
                        const float * linear2_weight,
                        const float * linear2_bias,
                        std::vector<float> & output,
                        std::string * error = nullptr);

// Layer-normalize row-major [batch, rows, channels] data in float32.  When
// gamma/beta are null this is the elementwise_affine=False path used by the
// first and third norms in ModulatedTransformerCrossBlock.  The reduction is
// over the final channels dimension and uses PyTorch's population variance.
bool layer_norm_rows(const float * input,
                     std::size_t batch,
                     std::size_t rows,
                     int channels,
                     float epsilon,
                     const float * gamma,
                     const float * beta,
                     std::vector<float> & output,
                     std::string * error = nullptr);

// Apply the AdaLN affine modulation used by the transformer blocks.  The
// shift_scale buffer is [batch, 2 * channels], with shift first and scale
// second, and input/output are [batch, rows, channels].
bool adaln_modulate_rows(const float * normalized,
                         std::size_t batch,
                         std::size_t rows,
                         int channels,
                         const float * shift_scale,
                         std::vector<float> & output,
                         std::string * error = nullptr);

} // namespace pixal3d
