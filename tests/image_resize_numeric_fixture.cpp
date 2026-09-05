#include "pixal3d/image.h"

#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: image_resize_numeric_fixture <image> <size>\n";
        return 2;
    }
    int size = 0;
    try {
        std::size_t consumed = 0;
        size = std::stoi(argv[2], &consumed);
        if (consumed != std::string(argv[2]).size() || size <= 0) return 2;
    } catch (...) {
        return 2;
    }
    pixal3d::Pixal3DImageF32 input;
    pixal3d::Pixal3DImageF32 output;
    std::string error;
    if (!pixal3d::load_pixal3d_image_f32(argv[1], input, &error) ||
        !pixal3d::resize_pixal3d_image_f32(input, size, size, output, &error)) {
        std::cerr << error << '\n';
        return 1;
    }
    std::cout << output.height << ' ' << output.width << ' ' << output.channels << '\n'
              << std::setprecision(9);
    for (float value : output.pixels) std::cout << value << ' ';
    std::cout << '\n';
    return 0;
}
