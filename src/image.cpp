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
#include <utility>
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

double sinc(double value) {
    constexpr double kPi = 3.14159265358979323846264338327950288;
    if (value == 0.0) return 1.0;
    const double x = value * kPi;
    return std::sin(x) / x;
}

double pillow_lanczos_filter(double value) {
    if (value >= -3.0 && value < 3.0) {
        return sinc(value) * sinc(value / 3.0);
    }
    return 0.0;
}

constexpr int kPillowResizePrecisionBits = 22;

bool precompute_pillow_resize_coefficients(
    int input_size,
    int output_size,
    std::vector<int> & bounds,
    std::vector<std::int32_t> & coefficients,
    int & kernel_size) {
    if (input_size <= 0 || output_size <= 0) return false;
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(1.0, scale);
    const double support = 3.0 * filter_scale;
    kernel_size = static_cast<int>(std::ceil(support)) * 2 + 1;
    bounds.resize(static_cast<std::size_t>(output_size) * 2);
    coefficients.assign(static_cast<std::size_t>(output_size) * kernel_size, 0);
    const double inverse_filter_scale = 1.0 / filter_scale;
    const int coefficient_scale = 1 << kPillowResizePrecisionBits;
    for (int output = 0; output < output_size; ++output) {
        const double center = (static_cast<double>(output) + 0.5) * scale;
        int first = static_cast<int>(center - support + 0.5);
        if (first < 0) first = 0;
        int last = static_cast<int>(center + support + 0.5);
        if (last > input_size) last = input_size;
        const int count = std::max(0, last - first);
        bounds[static_cast<std::size_t>(output) * 2] = first;
        bounds[static_cast<std::size_t>(output) * 2 + 1] = count;

        double weight_sum = 0.0;
        for (int index = 0; index < count; ++index) {
            const double value = (static_cast<double>(index + first) - center + 0.5) *
                                 inverse_filter_scale;
            const double weight = pillow_lanczos_filter(value);
            coefficients[static_cast<std::size_t>(output) * kernel_size + index] =
                static_cast<std::int32_t>(weight * coefficient_scale);
            weight_sum += weight;
        }
        if (weight_sum != 0.0) {
            for (int index = 0; index < count; ++index) {
                const double value = (static_cast<double>(index + first) - center + 0.5) *
                                     inverse_filter_scale;
                const double normalized = pillow_lanczos_filter(value) / weight_sum;
                const double scaled = normalized * coefficient_scale;
                coefficients[static_cast<std::size_t>(output) * kernel_size + index] =
                    static_cast<std::int32_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5);
            }
        }
    }
    return true;
}

std::uint8_t clip_pillow_resize_value(std::int64_t accumulated) {
    const std::int64_t value = accumulated >> kPillowResizePrecisionBits;
    if (value <= 0) return 0;
    if (value >= 255) return 255;
    return static_cast<std::uint8_t>(value);
}

bool resize_rgb_u8(const float * input_pixels,
                   int input_height,
                   int input_width,
                   int output_height,
                   int output_width,
                   std::vector<std::uint8_t> & output,
                   std::string * error) {
    if (!input_pixels || input_height <= 0 || input_width <= 0 ||
        output_height <= 0 || output_width <= 0) {
        set_error(error, "invalid RGB resize dimensions or input");
        return false;
    }
    std::size_t input_plane = 0;
    std::size_t output_elements = 0;
    if (!checked_product(static_cast<std::size_t>(input_height),
                         static_cast<std::size_t>(input_width), input_plane) ||
        !checked_product(static_cast<std::size_t>(output_height),
                         static_cast<std::size_t>(output_width), output_elements) ||
        !checked_product(output_elements, 3u, output_elements)) {
        set_error(error, "RGB resize dimensions overflow size_t");
        return false;
    }

    std::vector<int> x_bounds;
    std::vector<int> y_bounds;
    std::vector<std::int32_t> x_coefficients;
    std::vector<std::int32_t> y_coefficients;
    int x_kernel_size = 0;
    int y_kernel_size = 0;
    if (!precompute_pillow_resize_coefficients(
            input_width, output_width, x_bounds, x_coefficients, x_kernel_size) ||
        !precompute_pillow_resize_coefficients(
            input_height, output_height, y_bounds, y_coefficients, y_kernel_size)) {
        set_error(error, "failed to prepare Pillow-compatible resize coefficients");
        return false;
    }

    // Pillow resizes an RGB-like image in two passes as 8-bit data.  Keep that
    // boundary contract here, then let callers expose the result as F32.
    std::vector<std::uint8_t> horizontal(
        static_cast<std::size_t>(input_height) * output_width * 3u);
    output.resize(output_elements);
    const std::int64_t rounding =
        static_cast<std::int64_t>(1) << (kPillowResizePrecisionBits - 1);
    for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            const int first = x_bounds[static_cast<std::size_t>(x) * 2];
            const int count = x_bounds[static_cast<std::size_t>(x) * 2 + 1];
            for (int channel = 0; channel < 3; ++channel) {
                std::int64_t accumulated = rounding;
                for (int index = 0; index < count; ++index) {
                    const std::size_t source =
                        static_cast<std::size_t>(channel) * input_plane +
                        static_cast<std::size_t>(y) * input_width + first + index;
                    accumulated += static_cast<std::int64_t>(
                        std::lround(std::max(0.0f, std::min(1.0f, input_pixels[source])) * 255.0f)) *
                        x_coefficients[static_cast<std::size_t>(x) * x_kernel_size + index];
                }
                horizontal[(static_cast<std::size_t>(y) * output_width + x) * 3u +
                           static_cast<std::size_t>(channel)] =
                    clip_pillow_resize_value(accumulated);
            }
        }
    }
    for (int y = 0; y < output_height; ++y) {
        const int first = y_bounds[static_cast<std::size_t>(y) * 2];
        const int count = y_bounds[static_cast<std::size_t>(y) * 2 + 1];
        for (int x = 0; x < output_width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                std::int64_t accumulated = rounding;
                for (int index = 0; index < count; ++index) {
                    const std::size_t source =
                        (static_cast<std::size_t>(first + index) * output_width + x) * 3u +
                        static_cast<std::size_t>(channel);
                    accumulated += static_cast<std::int64_t>(horizontal[source]) *
                        y_coefficients[static_cast<std::size_t>(y) * y_kernel_size + index];
                }
                output[static_cast<std::size_t>(channel) *
                           static_cast<std::size_t>(output_height) * output_width +
                       static_cast<std::size_t>(y) * output_width + x] =
                    clip_pillow_resize_value(accumulated);
            }
        }
    }
    return true;
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
                 const std::vector<std::uint8_t> * alpha,
                 Pixal3DImageF32 & output, std::string * error) {
    std::size_t elements = 0;
    if (!checked_product(static_cast<std::size_t>(height),
                         static_cast<std::size_t>(width), elements) ||
        !checked_product(elements, 3u, elements) || rgb.size() != elements ||
        (alpha && alpha->size() != static_cast<std::size_t>(height) *
                                  static_cast<std::size_t>(width))) {
        set_error(error, "decoded image dimensions overflow or payload is invalid");
        return false;
    }
    output.height = height;
    output.width = width;
    output.channels = 3;
    output.pixels.resize(elements);
    const std::size_t plane = static_cast<std::size_t>(height) * width;
    if (alpha) {
        output.alpha.resize(plane);
    } else {
        output.alpha.clear();
    }
    for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        output.pixels[pixel] = static_cast<float>(rgb[pixel * 3 + 0]) / 255.0f;
        output.pixels[plane + pixel] = static_cast<float>(rgb[pixel * 3 + 1]) / 255.0f;
        output.pixels[2 * plane + pixel] = static_cast<float>(rgb[pixel * 3 + 2]) / 255.0f;
        if (alpha) output.alpha[pixel] = static_cast<float>((*alpha)[pixel]) / 255.0f;
    }
    return output.valid(error);
}

bool finish_rgb8(int height, int width, const std::vector<std::uint8_t> & rgb,
                 Pixal3DImageF32 & output, std::string * error) {
    return finish_rgb8(height, width, rgb, nullptr, output, error);
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
            const bool source_has_alpha =
                color_type == PNG_COLOR_TYPE_RGB_ALPHA ||
                color_type == PNG_COLOR_TYPE_GRAY_ALPHA ||
                png_get_valid(png, info, PNG_INFO_tRNS) != 0;
            if (!source_has_alpha) {
                png_set_add_alpha(png, 0xff, PNG_FILLER_AFTER);
            }
            png_read_update_info(png, info);
            if (png_get_channels(png, info) != 4) {
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
                    output.alpha.resize(plane);
                    for (std::size_t y = 0; y < static_cast<std::size_t>(height); ++y) {
                        for (std::size_t x = 0; x < static_cast<std::size_t>(width); ++x) {
                            const std::size_t pixel = y * width + x;
                            const std::uint8_t * source = rows[y] + x * 4u;
                            output.pixels[pixel] = source[0] / 255.0f;
                            output.pixels[plane + pixel] = source[1] / 255.0f;
                            output.pixels[2 * plane + pixel] = source[2] / 255.0f;
                            output.alpha[pixel] = source[3] / 255.0f;
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
    if (!alpha.empty() && alpha.size() != static_cast<std::size_t>(height) *
                                   static_cast<std::size_t>(width)) {
        set_error(error, "image alpha payload size does not match dimensions");
        return false;
    }
    for (float value : pixels) {
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
            set_error(error, "image pixels must be finite values in [0,1]");
            return false;
        }
    }
    for (float value : alpha) {
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
            set_error(error, "image alpha must contain finite values in [0,1]");
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

    const std::size_t input_plane = static_cast<std::size_t>(input.height) * input.width;
    const std::size_t output_plane = static_cast<std::size_t>(output_height) * output_width;
    std::vector<float> source_pixels = input.pixels;
    if (!input.alpha.empty()) {
        // Pillow resizes RGBA through its premultiplied RGBa representation.
        // Convert the source bytes to premultiplied 8-bit RGB before filtering.
        source_pixels.assign(input_plane * 3u, 0.0f);
        for (std::size_t pixel = 0; pixel < input_plane; ++pixel) {
            const int alpha = static_cast<int>(std::lround(
                std::max(0.0f, std::min(1.0f, input.alpha[pixel])) * 255.0f));
            for (int channel = 0; channel < 3; ++channel) {
                const int rgb = static_cast<int>(std::lround(
                    std::max(0.0f, std::min(1.0f,
                        input.pixels[static_cast<std::size_t>(channel) * input_plane + pixel])) *
                    255.0f));
                const int premultiplied = (rgb * alpha) / 255;
                source_pixels[static_cast<std::size_t>(channel) * input_plane + pixel] =
                    static_cast<float>(premultiplied) / 255.0f;
            }
        }
    }

    std::vector<std::uint8_t> resized_rgb;
    if (!resize_rgb_u8(source_pixels.data(), input.height, input.width,
                       output_height, output_width, resized_rgb, error)) {
        return false;
    }

    output.height = output_height;
    output.width = output_width;
    output.channels = 3;
    output.pixels.resize(output_plane * 3u);
    if (input.alpha.empty()) {
        for (std::size_t index = 0; index < resized_rgb.size(); ++index) {
            output.pixels[index] = static_cast<float>(resized_rgb[index]) / 255.0f;
        }
        return output.valid(error);
    }

    std::vector<float> alpha_pixels(input_plane * 3u);
    for (int channel = 0; channel < 3; ++channel) {
        std::copy(input.alpha.begin(), input.alpha.end(),
                  alpha_pixels.begin() + static_cast<std::size_t>(channel) * input_plane);
    }
    std::vector<std::uint8_t> resized_alpha;
    if (!resize_rgb_u8(alpha_pixels.data(), input.height, input.width,
                       output_height, output_width, resized_alpha, error)) {
        output = Pixal3DImageF32{};
        return false;
    }
    output.alpha.resize(output_plane);
    for (std::size_t pixel = 0; pixel < output_plane; ++pixel) {
        const int alpha = resized_alpha[pixel];
        output.alpha[pixel] = static_cast<float>(alpha) / 255.0f;
        for (int channel = 0; channel < 3; ++channel) {
            const std::size_t index = static_cast<std::size_t>(channel) * output_plane + pixel;
            const int premultiplied = resized_rgb[index];
            const int straight = alpha == 0 ? 0 : std::min(255, (premultiplied * 255) / alpha);
            output.pixels[index] = static_cast<float>(straight) / 255.0f;
        }
    }
    return output.valid(error);
}

bool preprocess_pixal3d_image_f32(const Pixal3DImageF32 & input,
                                  Pixal3DImageF32 & output,
                                  std::string * error) {
    output = Pixal3DImageF32{};
    if (!input.valid(error)) return false;

    const int max_size = std::max(input.width, input.height);
    Pixal3DImageF32 capped = input;
    if (max_size > 1024) {
        const double scale = 1024.0 / static_cast<double>(max_size);
        const int width = std::max(1, static_cast<int>(input.width * scale));
        const int height = std::max(1, static_cast<int>(input.height * scale));
        if (!resize_pixal3d_image_f32(input, height, width, capped, error)) return false;
    }

    bool has_partial_alpha = false;
    if (!capped.alpha.empty()) {
        for (float value : capped.alpha) {
            if (value < 1.0f - 1.0e-6f) {
                has_partial_alpha = true;
                break;
            }
        }
    }
    if (!has_partial_alpha) {
        output = std::move(capped);
        output.alpha.clear();
        return output.valid(error);
    }

    const int width = capped.width;
    const int height = capped.height;
    int min_x = width;
    int min_y = height;
    int max_x = -1;
    int max_y = -1;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float alpha = capped.alpha[static_cast<std::size_t>(y) * width + x];
            if (alpha > 0.8f) {
                min_x = std::min(min_x, x);
                min_y = std::min(min_y, y);
                max_x = std::max(max_x, x);
                max_y = std::max(max_y, y);
            }
        }
    }
    if (max_x < min_x || max_y < min_y) {
        set_error(error, "alpha image contains no foreground pixels above 0.8");
        return false;
    }

    const int foreground_width = max_x - min_x;
    const int foreground_height = max_y - min_y;
    const int square = std::max(1, static_cast<int>(
        static_cast<double>(std::max(foreground_width, foreground_height)) * 1.1));
    const double center_x = (static_cast<double>(min_x) + max_x) * 0.5;
    const double center_y = (static_cast<double>(min_y) + max_y) * 0.5;
    // PIL rounds fractional crop coordinates to the nearest integer, with
    // ties-to-even semantics, before padding outside the source image.
    const int left = static_cast<int>(std::nearbyint(center_x - square * 0.5));
    const int top = static_cast<int>(std::nearbyint(center_y - square * 0.5));

    output.height = square;
    output.width = square;
    output.channels = 3;
    const std::size_t output_plane = static_cast<std::size_t>(square) * square;
    output.pixels.assign(output_plane * 3u, 0.0f);
    for (int y = 0; y < square; ++y) {
        for (int x = 0; x < square; ++x) {
            const int source_x = left + x;
            const int source_y = top + y;
            if (source_x < 0 || source_x >= width || source_y < 0 || source_y >= height) {
                continue;
            }
            const std::size_t source_pixel = static_cast<std::size_t>(source_y) * width + source_x;
            const float alpha = capped.alpha[source_pixel];
            const std::size_t destination_pixel = static_cast<std::size_t>(y) * square + x;
            for (int channel = 0; channel < 3; ++channel) {
                const float composited = capped.pixels[
                    static_cast<std::size_t>(channel) * static_cast<std::size_t>(height) * width +
                    source_pixel] * alpha;
                // Python creates an RGB uint8 image after compositing, before
                // each stage performs its own Pillow resize.  Truncate here
                // to preserve that image-boundary quantization.
                const float clipped = std::max(0.0f, std::min(1.0f, composited));
                const auto quantized = static_cast<std::uint8_t>(clipped * 255.0f);
                output.pixels[static_cast<std::size_t>(channel) * output_plane + destination_pixel] =
                    static_cast<float>(quantized) / 255.0f;
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
    Pixal3DImageF32 preprocessed;
    if (!preprocess_pixal3d_image_f32(decoded, preprocessed, error)) return false;
    return resize_pixal3d_image_f32(preprocessed, image_resolution, image_resolution,
                                    output, error);
}

} // namespace pixal3d
