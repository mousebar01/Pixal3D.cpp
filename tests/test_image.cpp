#include "pixal3d/image.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

namespace {

bool close_enough(float actual, float expected, float tolerance = 1.0e-6f) {
    return std::fabs(actual - expected) <= tolerance;
}

bool write_ppm(const std::string & path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    file << "P6\r\n2 2\r\n255\r\n";
    const std::uint8_t pixels[] = {
        255, 0, 0,     0, 255, 0,
        0, 0, 255,     255, 255, 255,
    };
    file.write(reinterpret_cast<const char *>(pixels), sizeof(pixels));
    return static_cast<bool>(file);
}

} // namespace

int main() {
    const std::string path = "pixal3d_image_test.ppm";
    if (!write_ppm(path)) {
        std::cerr << "failed to write test image\n";
        return 1;
    }

    pixal3d::Pixal3DImageF32 image;
    std::string error;
    if (!pixal3d::load_pixal3d_image_f32(path, image, &error)) {
        std::remove(path.c_str());
        std::cerr << error << "\n";
        return 1;
    }
    if (image.height != 2 || image.width != 2 || image.channels != 3 ||
        image.pixels.size() != 12 ||
        !close_enough(image.pixels[0], 1.0f) ||
        !close_enough(image.pixels[1], 0.0f) ||
        !close_enough(image.pixels[4], 0.0f) ||
        !close_enough(image.pixels[5], 1.0f) ||
        !close_enough(image.pixels[8], 0.0f) ||
        !close_enough(image.pixels[9], 0.0f) ||
        !close_enough(image.pixels[10], 1.0f) ||
        !close_enough(image.pixels[11], 1.0f)) {
        std::remove(path.c_str());
        std::cerr << "decoded CHW image does not match PPM payload\n";
        return 1;
    }

    pixal3d::Pixal3DImageF32 resized;
    if (!pixal3d::resize_pixal3d_image_f32(image, 1, 1, resized, &error)) {
        std::remove(path.c_str());
        std::cerr << error << "\n";
        return 1;
    }
    // Pillow's RGB Lanczos path quantizes the two-pass result to uint8.
    const float expected = 128.0f / 255.0f;
    if (resized.height != 1 || resized.width != 1 || resized.channels != 3 ||
        resized.pixels.size() != 3 ||
        !close_enough(resized.pixels[0], expected) ||
        !close_enough(resized.pixels[1], expected) ||
        !close_enough(resized.pixels[2], expected)) {
        std::remove(path.c_str());
        std::cerr << "Lanczos resize does not preserve the expected average\n";
        return 1;
    }

    pixal3d::Pixal3DImageF32 gray;
    const std::string gray_path = "pixal3d_image_test.pgm";
    {
        std::ofstream file(gray_path);
        file << "P2\n# comment\n2 1\n100\n0 100\n";
    }
    if (!pixal3d::load_pixal3d_image_f32(gray_path, gray, &error) ||
        gray.pixels.size() != 6 || !close_enough(gray.pixels[0], 0.0f) ||
        !close_enough(gray.pixels[1], 1.0f) || !close_enough(gray.pixels[2], 0.0f) ||
        !close_enough(gray.pixels[3], 1.0f)) {
        std::remove(path.c_str());
        std::remove(gray_path.c_str());
        std::cerr << (error.empty() ? "decoded PGM image mismatch" : error) << "\n";
        return 1;
    }

    pixal3d::Pixal3DImageF32 rgba;
    rgba.height = 6;
    rgba.width = 4;
    rgba.channels = 3;
    const std::size_t rgba_plane = static_cast<std::size_t>(rgba.height) * rgba.width;
    rgba.pixels.assign(rgba_plane * 3u, 1.0f);
    rgba.alpha.assign(rgba_plane, 0.0f);
    for (int y = 1; y <= 4; ++y) {
        for (int x = 1; x <= 2; ++x) {
            rgba.alpha[static_cast<std::size_t>(y) * rgba.width + x] = 1.0f;
        }
    }
    pixal3d::Pixal3DImageF32 preprocessed;
    if (!pixal3d::preprocess_pixal3d_image_f32(rgba, preprocessed, &error) ||
        preprocessed.width != 3 || preprocessed.height != 3 ||
        !preprocessed.alpha.empty()) {
        std::remove(path.c_str());
        std::remove(gray_path.c_str());
        std::cerr << (error.empty() ? "alpha preprocessing shape mismatch" : error) << "\n";
        return 1;
    }
    const std::size_t preprocessed_plane = 9;
    if (!close_enough(preprocessed.pixels[preprocessed_plane], 0.0f) ||
        !close_enough(preprocessed.pixels[1 * preprocessed_plane + 4], 1.0f)) {
        std::remove(path.c_str());
        std::remove(gray_path.c_str());
        std::cerr << "alpha preprocessing did not composite onto black\n";
        return 1;
    }

    // Pillow resizes RGBA through premultiplied RGBa bytes.  Keep this
    // boundary behavior explicit so transparent-edge colors do not leak into
    // the vision condition.
    pixal3d::Pixal3DImageF32 alpha_resize;
    alpha_resize.height = 1;
    alpha_resize.width = 2;
    alpha_resize.channels = 3;
    alpha_resize.pixels = {1.0f, 0.0f,
                           0.0f, 0.0f,
                           0.0f, 0.0f};
    alpha_resize.alpha = {128.0f / 255.0f, 1.0f};
    pixal3d::Pixal3DImageF32 alpha_resized;
    if (!pixal3d::resize_pixal3d_image_f32(
            alpha_resize, 1, 1, alpha_resized, &error) ||
        alpha_resized.alpha.size() != 1 ||
        !close_enough(alpha_resized.alpha[0], 192.0f / 255.0f) ||
        !close_enough(alpha_resized.pixels[0], 85.0f / 255.0f) ||
        !close_enough(alpha_resized.pixels[1], 0.0f) ||
        !close_enough(alpha_resized.pixels[2], 0.0f)) {
        std::remove(path.c_str());
        std::remove(gray_path.c_str());
        std::cerr << "RGBA resize did not match premultiplied Pillow semantics\n";
        return 1;
    }

    std::remove(path.c_str());
    std::remove(gray_path.c_str());
    return 0;
}
