#include "pixal3d/flow.h"

#include <cmath>
#include <cstddef>
#include <limits>

namespace pixal3d {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
}

bool valid_embedding_args(const float * timesteps,
                          std::size_t batch,
                          int dim,
                          int max_period,
                          std::string * error) {
    if (batch > 0 && !timesteps) {
        set_error(error, "timesteps is null");
        return false;
    }
    if (dim < 2) {
        set_error(error, "embedding dimension must be at least 2");
        return false;
    }
    if (max_period <= 0) {
        set_error(error, "max_period must be positive");
        return false;
    }
    return true;
}

bool valid_mlp_args(const float * timesteps,
                    std::size_t batch,
                    int hidden_size,
                    int frequency_embedding_size,
                    const float * linear0_weight,
                    const float * linear0_bias,
                    const float * linear2_weight,
                    const float * linear2_bias,
                    std::string * error) {
    if (!valid_embedding_args(timesteps, batch, frequency_embedding_size, 10000,
                               error)) {
        return false;
    }
    if (hidden_size <= 0) {
        set_error(error, "hidden_size must be positive");
        return false;
    }
    if (batch > 0 && (!linear0_weight || !linear0_bias || !linear2_weight ||
                      !linear2_bias)) {
        set_error(error, "timestep MLP weights/biases are null");
        return false;
    }
    return true;
}

bool valid_rows_args(const float * input,
                     std::size_t batch,
                     std::size_t rows,
                     int channels,
                     const float * gamma,
                     const float * beta,
                     std::string * error) {
    if (batch > 0 && rows > 0 && channels > 0 && !input) {
        set_error(error, "row input is null");
        return false;
    }
    if (channels <= 0) {
        set_error(error, "channels must be positive");
        return false;
    }
    if ((gamma == nullptr) != (beta == nullptr)) {
        set_error(error, "gamma and beta must both be null or non-null");
        return false;
    }
    return true;
}

float silu(float value) {
    // This is the same x * sigmoid(x) definition used by torch.nn.SiLU.
    // Keep the intermediate values in float32 to match the reference dtype.
    const float sigmoid = 1.0f / (1.0f + std::exp(-value));
    return value * sigmoid;
}

} // namespace

bool timestep_embedding(const float * timesteps,
                        std::size_t batch,
                        int dim,
                        int max_period,
                        std::vector<float> & output,
                        std::string * error) {
    output.clear();
    if (!valid_embedding_args(timesteps, batch, dim, max_period, error)) {
        return false;
    }

    const int half = dim / 2;
    const float log_period = static_cast<float>(std::log(
        static_cast<double>(max_period)));
    output.resize(batch * static_cast<std::size_t>(dim));
    for (std::size_t row = 0; row < batch; ++row) {
        const float timestep = timesteps[row];
        float * destination = output.data() + row * static_cast<std::size_t>(dim);
        for (int index = 0; index < half; ++index) {
            const float fraction = static_cast<float>(index) /
                                   static_cast<float>(half);
            const float frequency = std::exp(-log_period * fraction);
            const float argument = timestep * frequency;
            destination[index] = std::cos(argument);
            destination[half + index] = std::sin(argument);
        }
        if (dim % 2 != 0) {
            destination[dim - 1] = 0.0f;
        }
    }
    return true;
}

bool timestep_embed_mlp(const float * timesteps,
                        std::size_t batch,
                        int hidden_size,
                        int frequency_embedding_size,
                        const float * linear0_weight,
                        const float * linear0_bias,
                        const float * linear2_weight,
                        const float * linear2_bias,
                        std::vector<float> & output,
                        std::string * error) {
    output.clear();
    if (!valid_mlp_args(timesteps, batch, hidden_size, frequency_embedding_size,
                        linear0_weight, linear0_bias, linear2_weight,
                        linear2_bias, error)) {
        return false;
    }

    std::vector<float> frequency;
    if (!timestep_embedding(timesteps, batch, frequency_embedding_size, 10000,
                            frequency, error)) {
        return false;
    }

    std::vector<float> activated(batch * static_cast<std::size_t>(hidden_size));
    for (std::size_t row = 0; row < batch; ++row) {
        for (int out = 0; out < hidden_size; ++out) {
            float value = linear0_bias[out];
            for (int in = 0; in < frequency_embedding_size; ++in) {
                value += linear0_weight[static_cast<std::size_t>(out) *
                                            static_cast<std::size_t>(frequency_embedding_size) +
                                        static_cast<std::size_t>(in)] *
                          frequency[row * static_cast<std::size_t>(frequency_embedding_size) +
                                    static_cast<std::size_t>(in)];
            }
            activated[row * static_cast<std::size_t>(hidden_size) +
                      static_cast<std::size_t>(out)] = silu(value);
        }
    }

    output.resize(batch * static_cast<std::size_t>(hidden_size));
    for (std::size_t row = 0; row < batch; ++row) {
        for (int out = 0; out < hidden_size; ++out) {
            float value = linear2_bias[out];
            for (int in = 0; in < hidden_size; ++in) {
                value += linear2_weight[static_cast<std::size_t>(out) *
                                            static_cast<std::size_t>(hidden_size) +
                                        static_cast<std::size_t>(in)] *
                          activated[row * static_cast<std::size_t>(hidden_size) +
                                    static_cast<std::size_t>(in)];
            }
            output[row * static_cast<std::size_t>(hidden_size) +
                   static_cast<std::size_t>(out)] = value;
        }
    }
    return true;
}

bool layer_norm_rows(const float * input,
                     std::size_t batch,
                     std::size_t rows,
                     int channels,
                     float epsilon,
                     const float * gamma,
                     const float * beta,
                     std::vector<float> & output,
                     std::string * error) {
    output.clear();
    if (!valid_rows_args(input, batch, rows, channels, gamma, beta, error)) {
        return false;
    }
    if (!(epsilon > 0.0f) || !std::isfinite(epsilon)) {
        set_error(error, "epsilon must be finite and positive");
        return false;
    }
    const std::size_t channel_count = static_cast<std::size_t>(channels);
    const std::size_t row_count = batch * rows;
    output.resize(row_count * channel_count);
    for (std::size_t row = 0; row < row_count; ++row) {
        const float * source = input + row * channel_count;
        float * destination = output.data() + row * channel_count;
        float mean = 0.0f;
        for (int channel = 0; channel < channels; ++channel) {
            mean += source[channel];
        }
        mean /= static_cast<float>(channels);
        float variance = 0.0f;
        for (int channel = 0; channel < channels; ++channel) {
            const float delta = source[channel] - mean;
            variance += delta * delta;
        }
        variance /= static_cast<float>(channels);
        const float inverse_std = 1.0f / std::sqrt(variance + epsilon);
        for (int channel = 0; channel < channels; ++channel) {
            float value = (source[channel] - mean) * inverse_std;
            if (gamma) {
                value = value * gamma[channel] + beta[channel];
            }
            destination[channel] = value;
        }
    }
    return true;
}

bool adaln_modulate_rows(const float * normalized,
                         std::size_t batch,
                         std::size_t rows,
                         int channels,
                         const float * shift_scale,
                         std::vector<float> & output,
                         std::string * error) {
    output.clear();
    if (!valid_rows_args(normalized, batch, rows, channels, nullptr, nullptr,
                         error)) {
        return false;
    }
    if (batch > 0 && !shift_scale) {
        set_error(error, "shift_scale is null");
        return false;
    }
    const std::size_t channel_count = static_cast<std::size_t>(channels);
    output.resize(batch * rows * channel_count);
    for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
        const float * shift = shift_scale + batch_index * 2 * channel_count;
        const float * scale = shift + channel_count;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t offset =
                (batch_index * rows + row) * channel_count;
            for (int channel = 0; channel < channels; ++channel) {
                output[offset + static_cast<std::size_t>(channel)] =
                    normalized[offset + static_cast<std::size_t>(channel)] *
                        (1.0f + scale[channel]) + shift[channel];
            }
        }
    }
    return true;
}

} // namespace pixal3d
