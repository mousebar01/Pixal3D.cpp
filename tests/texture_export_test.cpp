#include "pixal3d/texture_export.h"

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

int main() {
    pixal3d::DualGridMeshF32 mesh;
    mesh.vertices = {
        -0.25f, -0.25f, -0.25f,
         0.25f, -0.25f, -0.25f,
         0.0f,   0.25f,  -0.25f,
         0.0f,   0.0f,    0.25f,
    };
    mesh.faces = {0, 1, 2, 0, 3, 1, 1, 3, 2, 2, 3, 0};
    const auto original_vertices = mesh.vertices;
    const auto original_faces = mesh.faces;

    pixal3d::SparseTensorF32 texture;
    texture.batch_size = 1;
    texture.channels = 6;
    texture.spatial_x = texture.spatial_y = texture.spatial_z = 2;
    texture.coords = {
        0, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 1, 1, 0,
        0, 0, 0, 1,
        0, 1, 0, 1,
        0, 0, 1, 1,
        0, 1, 1, 1,
    };
    texture.feats.reserve(texture.points() * 6);
    for (std::size_t point = 0; point < texture.points(); ++point) {
        texture.feats.insert(texture.feats.end(), {
            0.2f + 0.05f * static_cast<float>(point),
            0.3f, 0.4f, 0.1f, 0.8f, 1.0f});
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
    if (json.find("\"min\":[-0.25,-0.25,-0.25]") == std::string::npos ||
        json.find("\"max\":[0.25,0.25,0.25]") == std::string::npos) {
        std::cerr << "GLB POSITION bounds do not use the Python textured-GLB frame\n";
        return 1;
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
    float first_position[3] = {};
    std::memcpy(first_position, bytes.data() + binary_begin, sizeof(first_position));
    if (std::fabs(first_position[0] - 0.25f) > 1e-6f ||
        std::fabs(first_position[1] + 0.25f) > 1e-6f ||
        std::fabs(first_position[2] - 0.25f) > 1e-6f) {
        std::cerr << "GLB first position does not use H=(-x,+y,-z)\n";
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
