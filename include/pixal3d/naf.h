#pragma once

#include "pixal3d/condition.h"

#include <string>
#include <vector>

namespace pixal3d {

struct NafHParams {
    int dim = 256;
    int in_channels = 3;
    int heads_attn = 4;
    int heads_rope = 4;
    int kernel_size = 9;
    int img_layers = 2;
    int num_groups = 8;
    float rope_base = 100.0f;
};

// NAF output in row-major HxWxC layout, ready to be placed in a
// Pixal3DFeatureMapF32 as the high-resolution half of the 2048-channel
// projection condition.
struct NafOutputF32 {
    int height = 0;
    int width = 0;
    int channels = 0;
    std::vector<float> features;

    bool valid(std::string * error = nullptr) const;
};

// Native NAF implementation matching comfy/image_encoders/naf.py.  The model
// uses ggml's validated GGUF boundary for weights; CUDA builds offload the
// reflect-padded convolution stacks, while adaptive pooling and neighborhood
// attention remain explicit reference operations for comparison with PyTorch.
class NafModel {
public:
    NafModel() = default;
    ~NafModel();

    NafModel(const NafModel &) = delete;
    NafModel & operator=(const NafModel &) = delete;

    bool load(const std::string & path,
              bool load_tensors = true,
              std::string * error = nullptr);
    void close() noexcept;

    bool is_loaded() const noexcept;
    bool has_data() const noexcept;
    const NafHParams & hparams() const noexcept;
    int tensor_count() const noexcept;
    bool has_tensor(const std::string & name) const noexcept;

    // image is CHW float32 in [0,1], low_resolution is an HxWxC feature map
    // (normally DINO patch features), and output_size is the requested NAF
    // high-resolution map size.  The low-resolution map's channel count may
    // be any positive multiple of heads_attn.
    bool upsample(const float * image,
                  int image_height,
                  int image_width,
                  const float * low_resolution,
                  int low_height,
                  int low_width,
                  int low_channels,
                  int output_height,
                  int output_width,
                  NafOutputF32 & output,
                  std::string * error = nullptr) const;

private:
    struct Impl;
    Impl * impl_ = nullptr;
};

} // namespace pixal3d
