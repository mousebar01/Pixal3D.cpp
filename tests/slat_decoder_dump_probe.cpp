#include "pixal3d/inference.h"
#include "pixal3d/slat_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct SparseDump {
    std::int32_t batch_size = 0;
    std::int32_t channels = 0;
    std::int32_t spatial_x = 0;
    std::int32_t spatial_y = 0;
    std::int32_t spatial_z = 0;
    std::vector<std::int32_t> coords;
    std::vector<float> feats;
};

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

std::uint64_t fnv1a(const void * data, std::size_t size) {
    const auto * bytes = static_cast<const std::uint8_t *>(data);
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool read_sparse_dump(const std::string & path, SparseDump & output,
                      std::string * error) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        set_error(error, "cannot open sparse dump: " + path);
        return false;
    }
    std::uint32_t version = 0;
    std::int32_t shape[5] = {};
    std::uint64_t points = 0;
    std::uint64_t feature_count = 0;
    stream.read(reinterpret_cast<char *>(&version), sizeof(version));
    stream.read(reinterpret_cast<char *>(shape), sizeof(shape));
    stream.read(reinterpret_cast<char *>(&points), sizeof(points));
    stream.read(reinterpret_cast<char *>(&feature_count), sizeof(feature_count));
    if (!stream || version != 1 || points > std::numeric_limits<std::size_t>::max() ||
        feature_count > std::numeric_limits<std::size_t>::max()) {
        set_error(error, "invalid sparse dump header: " + path);
        return false;
    }
    const std::size_t point_count = static_cast<std::size_t>(points);
    const std::size_t values = static_cast<std::size_t>(feature_count);
    if (shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 || shape[3] <= 0 ||
        shape[4] <= 0 || point_count > std::numeric_limits<std::size_t>::max() / 4 ||
        point_count * 4 > std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t) ||
        values > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        values != point_count * static_cast<std::size_t>(shape[1])) {
        set_error(error, "sparse dump dimensions do not match payload: " + path);
        return false;
    }
    output.batch_size = shape[0];
    output.channels = shape[1];
    output.spatial_x = shape[2];
    output.spatial_y = shape[3];
    output.spatial_z = shape[4];
    output.coords.resize(point_count * 4);
    output.feats.resize(values);
    stream.read(reinterpret_cast<char *>(output.coords.data()),
                static_cast<std::streamsize>(output.coords.size() * sizeof(std::int32_t)));
    stream.read(reinterpret_cast<char *>(output.feats.data()),
                static_cast<std::streamsize>(output.feats.size() * sizeof(float)));
    if (!stream) {
        set_error(error, "truncated sparse dump: " + path);
        return false;
    }
    return true;
}

bool write_sparse_dump(const std::filesystem::path & path,
                       const pixal3d::SparseTensorF32 & tensor,
                       std::string * error) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        set_error(error, "cannot open sparse probe output: " + path.string());
        return false;
    }
    const std::uint32_t version = 1;
    const std::int32_t shape[5] = {
        tensor.batch_size, tensor.channels, tensor.spatial_x,
        tensor.spatial_y, tensor.spatial_z,
    };
    const std::uint64_t points = tensor.points();
    const std::uint64_t feature_count = tensor.feats.size();
    stream.write(reinterpret_cast<const char *>(&version), sizeof(version));
    stream.write(reinterpret_cast<const char *>(shape), sizeof(shape));
    stream.write(reinterpret_cast<const char *>(&points), sizeof(points));
    stream.write(reinterpret_cast<const char *>(&feature_count), sizeof(feature_count));
    stream.write(reinterpret_cast<const char *>(tensor.coords.data()),
                 static_cast<std::streamsize>(tensor.coords.size() * sizeof(std::int32_t)));
    stream.write(reinterpret_cast<const char *>(tensor.feats.data()),
                 static_cast<std::streamsize>(tensor.feats.size() * sizeof(float)));
    if (!stream) {
        set_error(error, "failed writing sparse probe output: " + path.string());
        return false;
    }
    return true;
}

bool make_input(const SparseDump & source, pixal3d::SparseTensorF32 & output,
                std::string * error) {
    output = pixal3d::SparseTensorF32{};
    output.batch_size = source.batch_size;
    output.channels = source.channels;
    output.spatial_x = source.spatial_x;
    output.spatial_y = source.spatial_y;
    output.spatial_z = source.spatial_z;
    output.coords = source.coords;
    output.feats = source.feats;
    return output.valid(error);
}

void print_stats(const char * label, const pixal3d::SparseTensorF32 & tensor) {
    std::size_t positive = 0;
    std::size_t near_zero = 0;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    bool finite = true;
    for (float value : tensor.feats) {
        finite = finite && std::isfinite(value);
        if (value > 0.0f) ++positive;
        if (std::fabs(value) <= 1.0e-4f) ++near_zero;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    std::cout << label << " points=" << tensor.points()
              << " channels=" << tensor.channels
              << " finite=" << (finite ? 1 : 0)
              << " positive=" << positive
              << " near_zero=" << near_zero
              << " min=" << std::setprecision(9) << minimum
              << " max=" << maximum
              << " coords=0x" << std::hex
              << fnv1a(tensor.coords.data(), tensor.coords.size() * sizeof(std::int32_t))
              << " feats=0x"
              << fnv1a(tensor.feats.data(), tensor.feats.size() * sizeof(float))
              << std::dec << "\n";
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::cerr << "usage: " << argv[0]
                  << " <shared.gguf> <shape_slat_high.bin> <output-dir>\n";
        return 2;
    }
    std::string error;
    SparseDump source;
    if (!read_sparse_dump(argv[2], source, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    pixal3d::SparseTensorF32 input;
    if (!make_input(source, input, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    pixal3d::SLatDecoderModel model;
    if (!model.load(argv[1], "shape_decoder", true, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    pixal3d::SparseTensorF32 output;
    std::vector<pixal3d::SparseTensorF32> subdivisions;
    if (!model.decode(input, nullptr, output, &subdivisions, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    const std::filesystem::path directory(argv[3]);
    std::error_code filesystem_error;
    std::filesystem::create_directories(directory, filesystem_error);
    if (filesystem_error) {
        std::cerr << "failed to create output directory: "
                  << filesystem_error.message() << "\n";
        return 1;
    }
    if (!write_sparse_dump(directory / "shape_decoded.bin", output, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    for (std::size_t level = 0; level < subdivisions.size(); ++level) {
        if (!write_sparse_dump(directory / ("shape_subdiv_" + std::to_string(level) + ".bin"),
                               subdivisions[level], &error)) {
            std::cerr << error << "\n";
            return 1;
        }
    }
    std::vector<pixal3d::DualGridMeshF32> meshes;
    if (!pixal3d::flexi_dual_grid_decode_mesh_f32(
            output, 1024, 0.5f, meshes, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    if (!meshes.empty()) {
        std::ofstream mesh_file(directory / "mesh_stats.txt", std::ios::trunc);
        mesh_file << "vertices " << meshes.front().vertices.size() / 3 << "\n"
                  << "triangles " << meshes.front().faces.size() / 3 << "\n";
        if (!pixal3d::write_pixal3d_obj(meshes.front(),
                                        (directory / "mesh.obj").string(), &error)) {
            std::cerr << error << "\n";
            return 1;
        }
    }
    print_stats("input", input);
    for (std::size_t level = 0; level < subdivisions.size(); ++level) {
        print_stats(("subdiv_" + std::to_string(level)).c_str(), subdivisions[level]);
    }
    print_stats("output", output);
    return 0;
}
