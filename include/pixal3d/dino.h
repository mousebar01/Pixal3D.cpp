#pragma once

#include "pixal3d/sparse.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pixal3d {

// DINOv3 conditioning exported by the trellis2cpp-compatible DINOCOND
// format.  The canonical payload is [1, 1029, 1024] and is affine-free
// LayerNorm output from the final ViT layer, stored row-major as F32.
struct DinoConditionF32 {
    std::uint32_t format_version = 0;
    std::vector<std::int64_t> shape;
    VarLenTensorF32 global;

    std::size_t count() const noexcept { return global.feats.size(); }
    bool empty() const noexcept { return global.feats.empty(); }
};

struct DinoFingerprintF32 {
    float minimum = 0.0f;
    float maximum = 0.0f;
    double mean = 0.0;
    double sum = 0.0;
    double l2 = 0.0;
    std::size_t count = 0;
};

// Load a DINOCOND file.  Only little-endian float32 payloads are accepted;
// both canonical [1,tokens,channels] and unbatched [tokens,channels] shapes
// are normalized to a one-batch VarLenTensorF32.
bool load_dino_condition(const std::string & path,
                         DinoConditionF32 & output,
                         std::string * error = nullptr);

DinoFingerprintF32 fingerprint_dino_condition(const DinoConditionF32 & condition);

} // namespace pixal3d
