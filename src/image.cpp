#include "pixal3d/image.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#ifdef PIXAL3D_HAVE_JPEG
#include <jpeglib.h>
#include <setjmp.h>
#endif

#ifdef PIXAL3D_HAVE_PNG
#include <png.h>
#endif

namespace pixal3d {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

bool checked_product(std::size_t left, std::size_t right, std::size_t & output) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        return false;
    }
    output = left * right;
    return true;
}

float sinc(float value) {
    constexpr float kPi = 3.14159265358979323846f;
    if (std::fabs(value) < 1.0e-7f) return 1.0f;
    const float x = kPi * value;
    return std::sin(x) / x;
}

float lanczos(float distance, float scale) {
    const float filter_scale = std::max(1.0f, scale);
    const float x = distance / filter_scale;
    if (std::fabs(x) >= 3.0f) return 0.0f;
    return sinc(x) * sinc(x / 3.0f) / filter_scale;
}

bool read_token(std::ifstream & file, std::string & token,
                bool leave_delimiter = false) {
    token.clear();
    int value = 0;
    while ((value = file.get()) != EOF) {
        if (std::isspace(static_cast<unsigned char>(value))) continue;
        if (value == '#') {
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        token.push_back(static_cast<char>(value));
        break;
    }
    if (token.empty()) return false;
    while ((value = file.get()) != EOF &&
           !std::isspace(static_cast<unsigned char>(value))) {
        token.push_back(static_cast<char>(value));
    }
    if (value != EOF && leave_delimiter) file.unget();
    return true;
}

bool parse_positive(const std::string & token, int & value) {
    try {
        std::size_t consumed = 0;
        const long parsed = std::stol(token, &consumed, 10);
        if (consumed != token.size() || parsed <= 0 ||
            parsed > std::numeric_limits<int>::max()) return false;
        value = static_cast<int>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_maxval(const std::string & token, int & value) {
    try {
        std::size_t consumed = 0;
        const long parsed = std::stol(token, &consumed, 10);
        if (consumed != token.size() || parsed <= 0 || parsed > 65535) return false;
        value = static_cast<int>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_sample(const std::string & token, int maxval, int & value) {
    try {
        std::size_t consumed = 0;
        const long parsed = std::stol(token, &consumed, 10);
        if (consumed != token.size() || parsed < 0 || parsed > maxval) return false;
        value = static_cast<int>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool finish_rgb8(int height, int width, const std::vector<std::uint8_t> & rgb,
                 Pixal3DImageF32 & output, std::string * error) {
    std::size_t elements = 0;
    if (!checked_product(static_cast<std::size_t>(height),
                         static_cast<std::size_t>(width), elements) ||
        !checked_product(elements, 3u, elements) || rgb.size() != elements) {
        set_error(error, "decoded image dimensions overflow or payload is invalid");
        return false;
    }
    output.height = height;
    output.width = width;
    output.channels = 3;
    output.pixels.resize(elements);
    const std::size_t plane = static_cast<std::size_t>(height) * width;
    for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        output.pixels[pixel] = static_cast<float>(rgb[pixel * 3 + 0]) / 255.0f;
        output.pixels[plane + pixel] = static_cast<float>(rgb[pixel * 3 + 1]) / 255.0f;
        output.pixels[2 * plane + pixel] = static_cast<float>(rgb[pixel * 3 + 2]) / 255.0f;
    }
    return output.valid(error);
}

bool load_pnm(const std::string & path, Pixal3DImageF32 & output,
              std::string * error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        set_error(error, "cannot open image: " + path);
        return false;
    }
    std::string magic;
    std::string width_token;
    std::string height_token;
    std::string maxval_token;
    if (!read_token(file, magic) || !read_token(file, width_token) ||
        !read_token(file, height_token) || !read_token(file, maxval_token, true) ||
        (magic != "P2" && magic != "P3" && magic != "P5" && magic != "P6")) {
        set_error(error, "unsupported or truncated PNM header: " + path);
        return false;
    }
    int width = 0;
    int height = 0;
    int maxval = 0;
    if (!parse_positive(width_token, width) || !parse_positive(height_token, height) ||
        !parse_maxval(maxval_token, maxval)) {
        set_error(error, "invalid PNM dimensions or maxval: " + path);
        return false;
    }
    const bool ascii = magic == "P2" || magic == "P3";
    const int source_channels = magic == "P2" || magic == "P5" ? 1 : 3;
    std::size_t sample_count = 0;
    if (!checked_product(static_cast<std::size_t>(height),
                         static_cast<std::size_t>(width), sample_count) ||
        !checked_product(sample_count, static_cast<std::size_t>(source_channels),
                         sample_count)) {
        set_error(error, "PNM dimensions overflow: " + path);
        return false;
    }
    std::vector<std::uint8_t> rgb;
    std::size_t rgb_count = 0;
    if (!checked_product(static_cast<std::size_t>(height),
                         static_cast<std::size_t>(width), rgb_count) ||
        !checked_product(rgb_count, 3u, rgb_count)) {
        set_error(error, "PNM RGB dimensions overflow: " + path);
        return false;
    }
    rgb.resize(rgb_count);
    if (ascii) {
        std::string token;
        for (std::size_t sample = 0; sample < sample_count; ++sample) {
            int parsed = 0;
            if (!read_token(file, token) || !parse_sample(token, maxval, parsed)) {
                set_error(error, "truncated or invalid ASCII PNM payload: " + path);
                return false;
            }
            const std::uint8_t value = static_cast<std::uint8_t>(
                std::lround(static_cast<double>(parsed) * 255.0 / maxval));
            const std::size_t pixel = sample / static_cast<std::size_t>(source_channels);
            if (source_channels == 1) {
                rgb[pixel * 3 + 0] = value;
                rgb[pixel * 3 + 1] = value;
                rgb[pixel * 3 + 2] = value;
            } else {
                rgb[pixel * 3 + sample % 3] = value;
            }
        }
    } else {
        const int delimiter = file.get();
        if (delimiter == EOF || !std::isspace(static_cast<unsigned char>(delimiter))) {
            set_error(error, "missing binary PNM payload delimiter: " + path);
            return false;
        }
        if (delimiter == '\r' && file.peek() == '\n') file.get();
        const std::size_t bytes_per_sample = maxval < 256 ? 1u : 2u;
        std::size_t payload_bytes = 0;
        if (!checked_product(sample_count, bytes_per_sample, payload_bytes) ||
            payload_bytes > static_cast<std::size_t>(
                std::numeric_limits<std::streamsize>::max())) {
            set_error(error, "PNM payload size overflows stream limits: " + path);
            return false;
        }
        std::vector<std::uint8_t> payload(payload_bytes);
        file.read(reinterpret_cast<char *>(payload.data()),
                  static_cast<std::streamsize>(payload.size()));
        if (!file) {
            set_error(error, "truncated binary PNM payload: " + path);
            return false;
        }
        for (std::size_t sample = 0; sample < sample_count; ++sample) {
            int parsed = 0;
            if (bytes_per_sample == 1) {
                parsed = payload[sample];
            } else {
                parsed = (static_cast<int>(payload[sample * 2]) << 8) |
                         static_cast<int>(payload[sample * 2 + 1]);
            }
            const std::uint8_t value = static_cast<std::uint8_t>(
                std::lround(static_cast<double>(parsed) * 255.0 / maxval));
            const std::size_t pixel = sample / static_cast<std::size_t>(source_channels);
            if (source_channels == 1) {
                rgb[pixel * 3 + 0] = value;
                rgb[pixel * 3 + 1] = value;
                rgb[pixel * 3 + 2] = value;
            } else {
                rgb[pixel * 3 + sample % 3] = value;
            }
        }
    }
    return finish_rgb8(height, width, rgb, output, error);
}

#ifdef PIXAL3D_HAVE_PNG
bool load_png(const std::string & path, Pixal3DImageF32 & output,
              std::string * error) {
    FILE * file = std::fopen(path.c_str(), "rb");
    if (!file) {
        set_error(error, "cannot open PNG image: " + path);
        return false;
    }
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png ? png_create_info_struct(png) : nullptr;
    if (!png || !info) {
        if (png) png_destroy_read_struct(&png, nullptr, nullptr);
        std::fclose(file);
        set_error(error, "cannot initialize PNG decoder");
        return false;
    }
    std::vector<std::uint8_t> bytes;
    std::vector<png_bytep> rows;
    bool failed = false;
    if (setjmp(png_jmpbuf(png)) != 0) {
        failed = true;
    }
    if (!failed) {
        png_init_io(png, file);
        png_read_info(png, info);
        const png_uint_32 width = png_get_image_width(png, info);
        const png_uint_32 height = png_get_image_height(png, info);
        int bit_depth = png_get_bit_depth(png, info);
        int color_type = png_get_color_type(png, info);
        if (width == 0 || height == 0 || width > static_cast<png_uint_32>(
                std::numeric_limits<int>::max()) || height > static_cast<png_uint_32>(
                std::numeric_limits<int>::max())) {
            failed = true;
        } else {
            if (bit_depth == 16) png_set_strip_16(png);
            if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
            if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
            if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
            if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
                png_set_gray_to_rgb(png);
            }
            if (color_type == PNG_COLOR_TYPE_RGB_ALPHA ||
                color_type == PNG_COLOR_TYPE_GRAY_ALPHA ||
                png_get_valid(png, info, PNG_INFO_tRNS)) {
                png_set_strip_alpha(png);
            }
            png_read_update_info(png, info);
            if (png_get_channels(png, info) != 3) {
                failed = true;
            } else {
                const png_size_t rowbytes = png_get_rowbytes(png, info);
                std::size_t total = 0;
                if (rowbytes > static_cast<png_size_t>(std::numeric_limits<std::size_t>::max()) ||
                    !checked_product(static_cast<std::size_t>(rowbytes),
                                     static_cast<std::size_t>(height), total)) {
                    failed = true;
                } else {
                    bytes.resize(total);
                    rows.resize(static_cast<std::size_t>(height));
                    for (std::size_t row = 0; row < rows.size(); ++row) {
                        rows[row] = bytes.data() + row * static_cast<std::size_t>(rowbytes);
                    }
                    png_read_image(png, rows.data());
                    png_read_end(png, nullptr);
                    output.height = static_cast<int>(height);
                    output.width = static_cast<int>(width);
                    output.channels = 3;
                    const std::size_t plane = static_cast<std::size_t>(height) * width;
                    output.pixels.resize(plane * 3u);
                    for (std::size_t y = 0; y < static_cast<std::size_t>(height); ++y) {
                        for (std::size_t x = 0; x < static_cast<std::size_t>(width); ++x) {
                            const std::size_t pixel = y * width + x;
                            const std::uint8_t * source = rows[y] + x * 3u;
                            output.pixels[pixel] = source[0] / 255.0f;
                            output.pixels[plane + pixel] = source[1] / 255.0f;
                            output.pixels[2 * plane + pixel] = source[2] / 255.0f;
                        }
                    }
                }
            }
        }
    }
    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(file);
    if (failed) {
        output = Pixal3DImageF32{};
        set_error(error, "failed to decode PNG image: " + path);
        return false;
    }
    return output.valid(error);
}
#endif

#ifdef PIXAL3D_HAVE_JPEG
struct JpegErrorManager {
    jpeg_error_mgr base;
    jmp_buf jump;
    char message[JMSG_LENGTH_MAX]{};
};

void jpeg_error_exit(j_common_ptr common) {
    auto * manager = reinterpret_cast<JpegErrorManager *>(common->err);
    (*common->err->format_message)(common, manager->message);
    longjmp(manager->jump, 1);
}

bool load_jpeg(const std::string & path, Pixal3DImageF32 & output,
               std::string * error) {
    FILE * file = std::fopen(path.c_str(), "rb");
    if (!file) {
        set_error(error, "cannot open JPEG image: " + path);
        return false;
    }
    jpeg_decompress_struct cinfo{};
    JpegErrorManager manager{};
    cinfo.err = jpeg_std_error(&manager.base);
    manager.base.error_exit = jpeg_error_exit;
    bool created = false;
    bool failed = false;
    int decoded_width = 0;
    int decoded_height = 0;
    if (setjmp(manager.jump) != 0) {
        failed = true;
    }
    std::vector<std::uint8_t> rgb;
    if (!failed) {
        jpeg_create_decompress(&cinfo);
        created = true;
        jpeg_stdio_src(&cinfo, file);
        if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
            failed = true;
        } else {
            cinfo.out_color_space = JCS_RGB;
            jpeg_start_decompress(&cinfo);
            decoded_width = static_cast<int>(cinfo.output_width);
            decoded_height = static_cast<int>(cinfo.output_height);
            if (decoded_width <= 0 || decoded_height <= 0 || cinfo.output_components != 3) {
                failed = true;
            } else {
                std::size_t elements = 0;
                if (!checked_product(static_cast<std::size_t>(decoded_width),
                                     static_cast<std::size_t>(decoded_height), elements) ||
                    !checked_product(elements, 3u, elements)) {
                    failed = true;
                } else {
                    rgb.resize(elements);
                    while (cinfo.output_scanline < cinfo.output_height) {
                        JSAMPLE * row = rgb.data() +
                            static_cast<std::size_t>(cinfo.output_scanline) * decoded_width * 3u;
                        JSAMPROW rows[1] = {row};
                        jpeg_read_scanlines(&cinfo, rows, 1);
                    }
                }
            }
            jpeg_finish_decompress(&cinfo);
        }
    }
    if (created) jpeg_destroy_decompress(&cinfo);
    std::fclose(file);
    if (failed) {
        output = Pixal3DImageF32{};
        const std::string detail = manager.message[0] ? std::string(": ") + manager.message : "";
        set_error(error, "failed to decode JPEG image" + detail + ": " + path);
        return false;
    }
    return finish_rgb8(decoded_height, decoded_width, rgb, output, error);
}
#endif

} // namespace

bool Pixal3DImageF32::valid(std::string * error) const {
    if (height <= 0 || width <= 0 || channels != 3) {
        set_error(error, "image must be a positive RGB image");
        return false;
    }
    std::size_t expected = 0;
    if (!checked_product(static_cast<std::size_t>(height),
                         static_cast<std::size_t>(width), expected) ||
        !checked_product(expected, 3u, expected) || pixels.size() != expected) {
        set_error(error, "image payload size does not match dimensions");
        return false;
    }
    for (float value : pixels) {
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
            set_error(error, "image pixels must be finite values in [0,1]");
            return false;
        }
    }
    return true;
}

bool load_pixal3d_image_f32(const std::string & path,
                            Pixal3DImageF32 & output,
                            std::string * error) {
    output = Pixal3DImageF32{};
    if (path.empty()) {
        set_error(error, "image path is empty");
        return false;
    }
    std::ifstream signature_file(path, std::ios::binary);
    if (!signature_file) {
        set_error(error, "cannot open image: " + path);
        return false;
    }
    std::uint8_t signature[8]{};
    signature_file.read(reinterpret_cast<char *>(signature), sizeof(signature));
    const std::streamsize count = signature_file.gcount();
    signature_file.close();
    const bool is_png = count >= 8 && signature[0] == 0x89 && signature[1] == 0x50 &&
                        signature[2] == 0x4e && signature[3] == 0x47 &&
                        signature[4] == 0x0d && signature[5] == 0x0a &&
                        signature[6] == 0x1a && signature[7] == 0x0a;
    const bool is_jpeg = count >= 2 && signature[0] == 0xff && signature[1] == 0xd8;
    const bool is_pnm = count >= 2 && signature[0] == 'P' &&
                        (signature[1] == '2' || signature[1] == '3' ||
                         signature[1] == '5' || signature[1] == '6');
    if (is_pnm) return load_pnm(path, output, error);
    if (is_png) {
#ifdef PIXAL3D_HAVE_PNG
        return load_png(path, output, error);
#else
        set_error(error, "PNG support is not enabled in this build");
        return false;
#endif
    }
    if (is_jpeg) {
#ifdef PIXAL3D_HAVE_JPEG
        return load_jpeg(path, output, error);
#else
        set_error(error, "JPEG support is not enabled in this build");
        return false;
#endif
    }
    set_error(error, "unsupported image format (expected PNG, JPEG, or PNM): " + path);
    return false;
}

bool resize_pixal3d_image_f32(const Pixal3DImageF32 & input,
                              int output_height,
                              int output_width,
                              Pixal3DImageF32 & output,
                              std::string * error) {
    output = Pixal3DImageF32{};
    if (!input.valid(error) || output_height <= 0 || output_width <= 0) {
        if (!error || error->empty()) set_error(error, "invalid image resize dimensions");
        return false;
    }
    std::size_t output_elements = 0;
    if (!checked_product(static_cast<std::size_t>(output_height),
                         static_cast<std::size_t>(output_width), output_elements) ||
        !checked_product(output_elements, 3u, output_elements)) {
        set_error(error, "resized image dimensions overflow size_t");
        return false;
    }
    output.height = output_height;
    output.width = output_width;
    output.channels = 3;
    output.pixels.resize(output_elements);
    const std::size_t input_plane = static_cast<std::size_t>(input.height) * input.width;
    const std::size_t output_plane = static_cast<std::size_t>(output_height) * output_width;
    const float scale_y = static_cast<float>(input.height) / output_height;
    const float scale_x = static_cast<float>(input.width) / output_width;
    for (int channel = 0; channel < 3; ++channel) {
        for (int oy = 0; oy < output_height; ++oy) {
            const float source_y = (static_cast<float>(oy) + 0.5f) * scale_y - 0.5f;
            const float support_y = 3.0f * std::max(1.0f, scale_y);
            const int first_y = static_cast<int>(std::ceil(source_y - support_y));
            const int last_y = static_cast<int>(std::floor(source_y + support_y));
            for (int ox = 0; ox < output_width; ++ox) {
                const float source_x = (static_cast<float>(ox) + 0.5f) * scale_x - 0.5f;
                const float support_x = 3.0f * std::max(1.0f, scale_x);
                const int first_x = static_cast<int>(std::ceil(source_x - support_x));
                const int last_x = static_cast<int>(std::floor(source_x + support_x));
                float value = 0.0f;
                float weight_sum = 0.0f;
                for (int sy = first_y; sy <= last_y; ++sy) {
                    if (sy < 0 || sy >= input.height) continue;
                    const float wy = lanczos(static_cast<float>(sy) - source_y, scale_y);
                    for (int sx = first_x; sx <= last_x; ++sx) {
                        if (sx < 0 || sx >= input.width) continue;
                        const float weight = wy *
                            lanczos(static_cast<float>(sx) - source_x, scale_x);
                        value += input.pixels[static_cast<std::size_t>(channel) * input_plane +
                                              static_cast<std::size_t>(sy) * input.width + sx] * weight;
                        weight_sum += weight;
                    }
                }
                if (std::fabs(weight_sum) > 1.0e-12f) value /= weight_sum;
                output.pixels[static_cast<std::size_t>(channel) * output_plane +
                              static_cast<std::size_t>(oy) * output_width + ox] =
                    std::max(0.0f, std::min(1.0f, value));
            }
        }
    }
    return output.valid(error);
}

bool load_and_resize_pixal3d_image_f32(const std::string & path,
                                       int image_resolution,
                                       Pixal3DImageF32 & output,
                                       std::string * error) {
    output = Pixal3DImageF32{};
    if (image_resolution <= 0) {
        set_error(error, "image resolution must be positive");
        return false;
    }
    Pixal3DImageF32 decoded;
    if (!load_pixal3d_image_f32(path, decoded, error)) return false;
    return resize_pixal3d_image_f32(decoded, image_resolution, image_resolution,
                                    output, error);
}

} // namespace pixal3d
