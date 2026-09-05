#pragma once

#include "pixal3d/condition.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pixal3d {

// Hyperparameters serialized under pixal3d.dino.* in the standalone DINOv3
// GGUF component.  This is the ViT-L/16 configuration used by Pixal3D, but
// the loader keeps the dimensions explicit so tiny fixtures can exercise the
// complete graph without the 1.2 GiB production checkpoint.
struct DinoV3HParams {
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int patch_size = 0;
    int num_channels = 0;
    int num_register_tokens = 0;
    float layer_norm_eps = 1e-5f;
    float rope_theta = 100.0f;
    bool use_gated_mlp = false;
    bool query_bias = true;
    bool key_bias = false;
    bool value_bias = true;
    bool proj_bias = true;
    bool mlp_bias = true;

    int head_dim() const noexcept {
        return num_attention_heads > 0 ? hidden_size / num_attention_heads : 0;
    }
};

// Output of the exact Pixal3D DINOv3 feature contract.  global contains CLS
// plus register tokens; patch_map contains only patch tokens in row-major
// HxWxC order.  Both are after the reference affine-free final LayerNorm.
struct DinoV3FeaturesF32 {
    int image_height = 0;
    int image_width = 0;
    int patch_height = 0;
    int patch_width = 0;
    int channels = 0;
    int num_register_tokens = 0;
    VarLenTensorF32 global;
    Pixal3DFeatureMapF32 patch_map;

    bool valid(std::string * error = nullptr) const;
};

// Native ggml DINOv3 ViT inference.  Input pixels are CHW float32 already
// normalized with ImageNet mean/std, matching the Python image-condition
// extractors.  The graph uses patch projection, prefix-only 2D RoPE, exact
// GELU-erf, pre-norm residual blocks, and affine-free output normalization.
class DinoV3Model {
public:
    DinoV3Model() = default;
    ~DinoV3Model();

    DinoV3Model(const DinoV3Model &) = delete;
    DinoV3Model & operator=(const DinoV3Model &) = delete;

    bool load(const std::string & path,
              bool load_tensors = true,
              std::string * error = nullptr);
    void close() noexcept;

    bool is_loaded() const noexcept;
    bool has_data() const noexcept;
    const DinoV3HParams & hparams() const noexcept;
    const std::string & backend_name() const noexcept;
    int tensor_count() const noexcept;
    bool has_tensor(const std::string & name) const noexcept;

    bool encode(const float * pixels,
                int height,
                int width,
                DinoV3FeaturesF32 & output,
                std::string * error = nullptr) const;

private:
    struct Impl;
    Impl * impl_ = nullptr;
};

} // namespace pixal3d
