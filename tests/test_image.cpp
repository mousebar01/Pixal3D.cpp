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
    // Half-pixel Lanczos resize of the four corners is their arithmetic mean.
    const float expected = 0.5f;
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

    std::remove(path.c_str());
    std::remove(gray_path.c_str());
    return 0;
}
