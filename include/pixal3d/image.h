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
    // Optional alpha plane in row-major layout.  RGB/JPEG/PNM inputs may leave
    // this empty; PNG inputs preserve it so the Python image preprocessor can
    // reproduce foreground cropping and background compositing.
    std::vector<float> alpha;

    std::size_t elements() const noexcept { return pixels.size(); }
    bool valid(std::string * error = nullptr) const;
};

// Decode a PNG/JPEG image when the corresponding optional CMake dependency is
// available, or a PGM/PPM (P2/P3/P5/P6) image without external dependencies.
// PNG alpha is preserved in Pixal3DImageF32::alpha; JPEG and PNM inputs are
// treated as opaque RGB images.
bool load_pixal3d_image_f32(const std::string & path,
                            Pixal3DImageF32 & output,
                            std::string * error = nullptr);

// Resize an RGB CHW image with Pillow's two-pass Lanczos-3 RGB semantics,
// including normalized coefficients and uint8 quantization at each pass. The
// result is returned as normalized F32 CHW for the image-condition encoders.
bool resize_pixal3d_image_f32(const Pixal3DImageF32 & input,
                              int output_height,
                              int output_width,
                              Pixal3DImageF32 & output,
                              std::string * error = nullptr);

// Match the reference single-image preprocessing for an image with an alpha
// foreground: cap the longest side, find the alpha>0.8 foreground box, expand
// it to a centered square with 10% margin, crop with black out-of-bounds
// padding, and composite the result onto black. RGB-only inputs are resized to
// the capped size without segmentation because no rembg model is available in
// this native path.
bool preprocess_pixal3d_image_f32(const Pixal3DImageF32 & input,
                                  Pixal3DImageF32 & output,
                                  std::string * error = nullptr);

// Decode and directly resize an image to a square encoder resolution.
bool load_and_resize_pixal3d_image_f32(const std::string & path,
                                       int image_resolution,
                                       Pixal3DImageF32 & output,
                                       std::string * error = nullptr);

} // namespace pixal3d
