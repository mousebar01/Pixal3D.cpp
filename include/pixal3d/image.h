#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace pixal3d {

// A decoded RGB image in row-major CHW float32 layout.  Values are normalized
// to [0,1] at the file boundary, matching the image-condition encoders.
struct Pixal3DImageF32 {
    int height = 0;
    int width = 0;
    int channels = 0;
    std::vector<float> pixels;

    std::size_t elements() const noexcept { return pixels.size(); }
    bool valid(std::string * error = nullptr) const;
};

// Decode a PNG/JPEG image when the corresponding optional CMake dependency is
// available, or a PGM/PPM (P2/P3/P5/P6) image without external dependencies.
// PNG/JPEG alpha is intentionally dropped like PIL.Image.convert("RGB").
bool load_pixal3d_image_f32(const std::string & path,
                            Pixal3DImageF32 & output,
                            std::string * error = nullptr);

// Resize an RGB CHW image with a separable Lanczos-3 filter and half-pixel
// centers, compatible with the PIL Image.Resampling.LANCZOS path used by
// Pixal3D's list-of-images condition extractor. The native boundary keeps F32
// values instead of applying PIL's final uint8 quantization.
bool resize_pixal3d_image_f32(const Pixal3DImageF32 & input,
                              int output_height,
                              int output_width,
                              Pixal3DImageF32 & output,
                              std::string * error = nullptr);

// Decode and directly resize an image to a square encoder resolution.
bool load_and_resize_pixal3d_image_f32(const std::string & path,
                                       int image_resolution,
                                       Pixal3DImageF32 & output,
                                       std::string * error = nullptr);

} // namespace pixal3d
