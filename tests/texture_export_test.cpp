#include "pixal3d/texture_export.h"

#if defined(PIXAL3D_HAVE_PNG)
#include <png.h>
#include <zlib.h>
#endif

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

#if defined(PIXAL3D_HAVE_PNG)
struct PngSource {
    const std::uint8_t * data = nullptr;
    std::size_t size = 0;
    std::size_t offset = 0;
};

void read_png_source(png_structp png, png_bytep out, png_size_t length) {
    PngSource * source = static_cast<PngSource *>(png_get_io_ptr(png));
    if (source->offset + length > source->size) png_error(png, "truncated PNG");
    std::memcpy(out, source->data + source->offset, length);
    source->offset += length;
}

// Minimal RGBA8 decode of one in-memory PNG through libpng.
bool decode_png_rgba(const std::uint8_t * data, std::size_t size,
                     std::vector<std::uint8_t> & rgba, int & width, int & height) {
    if (size < 8 || std::memcmp(data, "\x89PNG\r\n\x1a\n", 8) != 0) return false;
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) return false;
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        return false;
    }
    PngSource source{data, size, 0};
    rgba.clear();
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        return false;
    }
    png_set_read_fn(png, &source, read_png_source);
    png_read_info(png, info);
    width = static_cast<int>(png_get_image_width(png, info));
    height = static_cast<int>(png_get_image_height(png, info));
    if (png_get_bit_depth(png, info) != 8 ||
        png_get_color_type(png, info) != PNG_COLOR_TYPE_RGBA) {
        png_destroy_read_struct(&png, &info, nullptr);
        return false;
    }
    rgba.resize(static_cast<std::size_t>(width) * height * 4);
    std::vector<png_bytep> rows(height);
    for (int row = 0; row < height; ++row) {
        rows[row] = rgba.data() + static_cast<std::size_t>(row) * width * 4;
    }
    png_read_rows(png, rows.data(), nullptr, height);
    png_read_end(png, info);
    png_destroy_read_struct(&png, &info, nullptr);
    return true;
}
#endif

} // namespace

int main() {
    pixal3d::DualGridMeshF32 mesh;
    // The bake grid is locked to the cascade resolution (1024).  The export
    // chain (weld/clean/orient/taubin smoothing/QEM) is designed for
    // production-density meshes, so the fixture is a densely sampled sphere
    // rather than a handful of faces: smoothing barely moves a uniform
    // spherical sampling and the bounds stay stable.  Fill the volume block
    // around it; a sparse surface field whose trilinear neighborhoods are
    // empty must stay rejected rather than bake black.
    constexpr float kMeshScale = 0.016f;
    constexpr int kStacks = 16;
    constexpr int kSlices = 32;
    for (int stack = 0; stack <= kStacks; ++stack) {
        const double theta = 3.14159265358979323846 * stack / kStacks;
        for (int slice = 0; slice < kSlices; ++slice) {
            const double phi = 6.28318530717958647692 * slice / kSlices;
            mesh.vertices.push_back(static_cast<float>(
                0.25 * kMeshScale * std::sin(theta) * std::cos(phi)));
            mesh.vertices.push_back(static_cast<float>(
                0.25 * kMeshScale * std::sin(theta) * std::sin(phi)));
            mesh.vertices.push_back(static_cast<float>(
                0.25 * kMeshScale * std::cos(theta)));
        }
    }
    auto vertex_at = [kSlices](int stack, int slice) {
        return stack * kSlices + (slice % kSlices);
    };
    for (int stack = 0; stack < kStacks; ++stack) {
        for (int slice = 0; slice < kSlices; ++slice) {
            const int a = vertex_at(stack, slice);
            const int b = vertex_at(stack, slice + 1);
            const int c = vertex_at(stack + 1, slice);
            const int d = vertex_at(stack + 1, slice + 1);
            mesh.faces.insert(mesh.faces.end(), {a, c, b});
            mesh.faces.insert(mesh.faces.end(), {b, c, d});
        }
    }
    const auto original_vertices = mesh.vertices;
    const auto original_faces = mesh.faces;

    pixal3d::SparseTensorF32 texture;
    texture.batch_size = 1;
    texture.channels = 6;
    texture.spatial_x = texture.spatial_y = texture.spatial_z = 1024;
    // One occupied cell block around the whole mesh so every sample position
    // finds trilinear neighbors (grid 508..516 plus one cell of margin).
    constexpr std::int32_t kVoxelLow = 506;
    constexpr std::int32_t kVoxelHigh = 517;
    for (std::int32_t x = kVoxelLow; x <= kVoxelHigh; ++x) {
        for (std::int32_t y = kVoxelLow; y <= kVoxelHigh; ++y) {
            for (std::int32_t z = kVoxelLow; z <= kVoxelHigh; ++z) {
                texture.coords.insert(texture.coords.end(), {0, x, y, z});
                texture.feats.insert(texture.feats.end(), {
                    0.2f + 0.3f * static_cast<float>((x - kVoxelLow) /
                                                     (kVoxelHigh - kVoxelLow)),
                    0.3f, 0.4f, 0.1f, 0.8f, 1.0f});
            }
        }
    }
    const auto original_coords = texture.coords;
    const auto original_feats = texture.feats;

    pixal3d::Pixal3DGlbOptions options;
    options.texture_size = 8;
    const std::string path = "/tmp/pixal3d-texture-export-test.glb";
    std::string error;
    if (!pixal3d::write_pixal3d_glb(
            mesh, texture, 1024, options, path, &error)) {
        std::cerr << "write_pixal3d_glb failed: " << error << "\n";
        return 1;
    }
    if (mesh.vertices != original_vertices || mesh.faces != original_faces ||
        texture.coords != original_coords || texture.feats != original_feats) {
        std::cerr << "GLB export mutated its inputs\n";
        return 1;
    }
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "GLB output file was not created\n";
        return 1;
    }
    std::vector<std::uint8_t> bytes;
    bytes.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (bytes.size() <= 20 || bytes[0] != 'g' || bytes[1] != 'l' ||
        bytes[2] != 'T' || bytes[3] != 'F' || bytes[4] != 2 ||
        bytes[5] != 0 || bytes[6] != 0 || bytes[7] != 0) {
        std::cerr << "invalid GLB header\n";
        return 1;
    }
    const std::uint32_t json_length = static_cast<std::uint32_t>(bytes[12]) |
        (static_cast<std::uint32_t>(bytes[13]) << 8) |
        (static_cast<std::uint32_t>(bytes[14]) << 16) |
        (static_cast<std::uint32_t>(bytes[15]) << 24);
    if (20u + json_length > bytes.size()) {
        std::cerr << "invalid GLB JSON chunk length\n";
        return 1;
    }
    const std::string json(bytes.begin() + 20, bytes.begin() + 20 + json_length);
    if (json.find("Pixal3D PBR material") == std::string::npos ||
        json.find("baseColorTexture") == std::string::npos ||
        json.find("metallicRoughnessTexture") == std::string::npos ||
        json.find("doubleSided") == std::string::npos ||
        json.find("OPAQUE") == std::string::npos) {
        std::cerr << "GLB JSON is missing PBR material fields\n";
        return 1;
    }
    // The remesh chain (weld/clean/taubin smoothing/QEM) moves vertices a
    // little, so the Python textured-GLB frame is asserted with a tolerance:
    // the bounds must stay symmetric about the origin with the H-frame
    // orientation (|component| within 25% of the exact 0.004 extent).
    std::size_t min_at = json.find("\"min\":[");
    std::size_t max_at = json.find("\"max\":[");
    if (min_at == std::string::npos || max_at == std::string::npos) {
        std::cerr << "GLB POSITION bounds are missing\n";
        return 1;
    }
    double minimum_bounds[3] = {};
    double maximum_bounds[3] = {};
    if (std::sscanf(json.c_str() + min_at + 7, "%lf,%lf,%lf", &minimum_bounds[0],
                    &minimum_bounds[1], &minimum_bounds[2]) != 3 ||
        std::sscanf(json.c_str() + max_at + 7, "%lf,%lf,%lf", &maximum_bounds[0],
                    &maximum_bounds[1], &maximum_bounds[2]) != 3) {
        std::cerr << "GLB POSITION bounds do not parse\n";
        return 1;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (minimum_bounds[axis] > -0.003 || minimum_bounds[axis] < -0.005 ||
            maximum_bounds[axis] < 0.003 || maximum_bounds[axis] > 0.005) {
            std::cerr << "GLB POSITION bounds do not use the Python textured-GLB frame\n";
            return 1;
        }
    }
    const std::string png_signature("\x89PNG\r\n\x1a\n", 8);
    const std::size_t binary_begin = 20u + json_length + 8u;
    if (binary_begin + 12u > bytes.size()) {
        std::cerr << "GLB binary chunk is truncated\n";
        return 1;
    }
    const std::string binary(bytes.begin() + binary_begin, bytes.end());
    if (binary.find(png_signature) == std::string::npos) {
        std::cerr << "GLB binary chunk is missing embedded PNG data\n";
        return 1;
    }
    // The remesh chain reorders and welds vertices, so the vertex order in
    // the GLB is not the decoder mesh's order; the H-frame orientation is
    // covered by the symmetric bounds assertion above.

    // Decode the baked albedo atlas and require it to preserve the decoded
    // field's brightness.  Sparse trilinear sampling that skips empty
    // neighbors without renormalizing would scale colors by the covered
    // fraction (about 0.1 here) and fail this check.
    const std::size_t png_begin = binary.find(png_signature);
    std::vector<std::uint8_t> rgba;
    int png_width = 0, png_height = 0;
    if (png_begin == std::string::npos ||
        !decode_png_rgba(reinterpret_cast<const std::uint8_t *>(binary.data()) + png_begin,
                         binary.size() - png_begin, rgba, png_width, png_height)) {
        std::cerr << "cannot decode the baked albedo PNG\n";
        return 1;
    }
    double covered_alpha = 0.0;
    double covered_red = 0.0;
    std::size_t covered = 0;
    for (int pixel = 0; pixel < png_width * png_height; ++pixel) {
        const std::uint8_t alpha = rgba[pixel * 4 + 3];
        if (alpha < 8) continue;
        covered += 1;
        covered_alpha += alpha;
        covered_red += rgba[pixel * 4 + 0];
    }
    if (covered == 0) {
        std::cerr << "baked albedo has no covered texels\n";
        return 1;
    }
    if (covered_alpha / covered < 250.0) {
        std::cerr << "baked albedo is not opaque over covered texels\n";
        return 1;
    }
    const double red_mean = covered_red / covered / 255.0;
    if (red_mean < 0.15) {
        std::cerr << "baked albedo lost brightness (red mean " << red_mean
                  << "); sparse volume sampling is not renormalized\n";
        return 1;
    }
    if (std::getenv("PIXAL3D_KEEP_TEXTURE_EXPORT_TEST") == nullptr) {
        std::remove(path.c_str());
    }

    options.texture_size = 0;
    if (pixal3d::write_pixal3d_glb(
            mesh, texture, 1024, options, path, &error) ||
        error.find("texture size") == std::string::npos) {
        std::cerr << "invalid texture size was not rejected\n";
        return 1;
    }
    return 0;
}
