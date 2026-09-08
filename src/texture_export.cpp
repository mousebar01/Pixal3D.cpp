#include "pixal3d/texture_export.h"

#include <meshoptimizer.h>
#include <xatlas.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(PIXAL3D_HAVE_PNG)
#include <png.h>
#endif

namespace pixal3d {
namespace {

constexpr float kPi = 3.14159265358979323846f;

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

Vec3 operator+(const Vec3 & a, const Vec3 & b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3 & a, const Vec3 & b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(const Vec3 & a, float scale) {
    return {a.x * scale, a.y * scale, a.z * scale};
}

float dot(const Vec3 & a, const Vec3 & b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3 & a, const Vec3 & b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

Vec3 normalize(const Vec3 & value) {
    const float length = std::sqrt(dot(value, value));
    if (!(length > 1.0e-20f) || !std::isfinite(length)) return {0.0f, 0.0f, 1.0f};
    return value * (1.0f / length);
}

std::uint64_t voxel_key(std::int32_t x, std::int32_t y, std::int32_t z) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) << 42) |
           (static_cast<std::uint64_t>(static_cast<std::uint32_t>(y)) << 21) |
           static_cast<std::uint64_t>(static_cast<std::uint32_t>(z));
}

struct VoxelSample {
    std::array<float, 6> value{};
};

struct TextureVolume {
    int resolution = 0;
    std::unordered_map<std::uint64_t, VoxelSample> values;
};

bool validate_mesh(const DualGridMeshF32 & mesh, std::string * error) {
    if (mesh.vertices.empty() || mesh.vertices.size() % 3 != 0 ||
        mesh.faces.empty() || mesh.faces.size() % 3 != 0) {
        set_error(error, "GLB export requires non-empty packed mesh vertices and triangles");
        return false;
    }
    const std::size_t vertex_count = mesh.vertices.size() / 3;
    for (float value : mesh.vertices) {
        if (!std::isfinite(value)) {
            set_error(error, "GLB export mesh contains a non-finite vertex");
            return false;
        }
    }
    for (std::int32_t index : mesh.faces) {
        if (index < 0 || static_cast<std::size_t>(index) >= vertex_count) {
            set_error(error, "GLB export mesh face index is outside the vertex buffer");
            return false;
        }
    }
    return true;
}

bool make_texture_volume(const SparseTensorF32 & input, int resolution,
                         TextureVolume & volume, std::string * error) {
    if (!input.valid(error)) return false;
    if (input.channels != 6) {
        set_error(error, "GLB export requires six texture channels (base_color, metallic, roughness, alpha)");
        return false;
    }
    volume = TextureVolume{};
    volume.resolution = resolution;
    volume.values.reserve(input.points() * 2 + 1);
    for (std::size_t point = 0; point < input.points(); ++point) {
        const std::size_t coord_base = point * 4;
        const std::int32_t batch = input.coords[coord_base + 0];
        if (batch != 0) {
            set_error(error, "GLB export currently supports the first texture batch only");
            return false;
        }
        const std::int32_t x = input.coords[coord_base + 1];
        const std::int32_t y = input.coords[coord_base + 2];
        const std::int32_t z = input.coords[coord_base + 3];
        if (x < 0 || y < 0 || z < 0 || x >= resolution || y >= resolution || z >= resolution) {
            set_error(error, "texture voxel coordinate is outside the cascade resolution");
            return false;
        }
        VoxelSample sample;
        for (std::size_t channel = 0; channel < sample.value.size(); ++channel) {
            const float value = input.feats[point * 6 + channel];
            if (!std::isfinite(value)) {
                set_error(error, "texture voxel attributes contain a non-finite value");
                return false;
            }
            sample.value[channel] = value;
        }
        volume.values[voxel_key(x, y, z)] = sample;
    }
    return true;
}

std::array<float, 6> sample_volume(const TextureVolume & volume, const Vec3 & position,
                                   float & weight_sum) {
    std::array<float, 6> result{};
    weight_sum = 0.0f;
    const float scale = static_cast<float>(volume.resolution);
    const float gx = (position.x + 0.5f) * scale;
    const float gy = (position.y + 0.5f) * scale;
    const float gz = (position.z + 0.5f) * scale;
    if (!std::isfinite(gx) || !std::isfinite(gy) || !std::isfinite(gz)) return result;
    const int x0 = static_cast<int>(std::floor(gx));
    const int y0 = static_cast<int>(std::floor(gy));
    const int z0 = static_cast<int>(std::floor(gz));
    const float tx = gx - static_cast<float>(x0);
    const float ty = gy - static_cast<float>(y0);
    const float tz = gz - static_cast<float>(z0);
    for (int dz = 0; dz <= 1; ++dz) {
        for (int dy = 0; dy <= 1; ++dy) {
            for (int dx = 0; dx <= 1; ++dx) {
                const int x = x0 + dx;
                const int y = y0 + dy;
                const int z = z0 + dz;
                if (x < 0 || y < 0 || z < 0 || x >= volume.resolution ||
                    y >= volume.resolution || z >= volume.resolution) continue;
                const auto found = volume.values.find(voxel_key(x, y, z));
                if (found == volume.values.end()) continue;
                const float wx = dx ? tx : 1.0f - tx;
                const float wy = dy ? ty : 1.0f - ty;
                const float wz = dz ? tz : 1.0f - tz;
                const float weight = wx * wy * wz;
                // The decoded field only exists on the surface shell, so most
                // trilinear neighbors are absent.  Accumulate the weight so the
                // caller can renormalize; scaling by the covered fraction would
                // darken every sample and braid coverage steps into the atlas.
                weight_sum += weight;
                for (std::size_t channel = 0; channel < result.size(); ++channel) {
                    result[channel] += found->second.value[channel] * weight;
                }
            }
        }
    }
    return result;
}

std::uint8_t to_byte(float value) {
    if (!std::isfinite(value)) value = 0.0f;
    value = std::clamp(value, 0.0f, 1.0f);
    return static_cast<std::uint8_t>(std::lround(value * 255.0f));
}

#if defined(PIXAL3D_HAVE_PNG)
bool encode_png(const std::vector<std::uint8_t> & pixels, int width, int height, int channels,
               std::vector<std::uint8_t> & encoded, std::string * error) {
    encoded.clear();
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        set_error(error, "cannot create libpng writer");
        return false;
    }
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, nullptr);
        set_error(error, "cannot create libpng image info");
        return false;
    }
    struct Sink {
        std::vector<std::uint8_t> * bytes = nullptr;
    } sink{&encoded};
    if (setjmp(png_jmpbuf(png)) != 0) {
        png_destroy_write_struct(&png, &info);
        set_error(error, "libpng failed while encoding texture image");
        return false;
    }
    png_set_write_fn(
        png, &sink,
        [](png_structp png_ptr, png_bytep data, png_size_t length) {
            auto * target = static_cast<Sink *>(png_get_io_ptr(png_ptr));
            target->bytes->insert(target->bytes->end(), data, data + length);
        },
        [](png_structp) {});
    const int color_type = channels == 4 ? PNG_COLOR_TYPE_RGBA : PNG_COLOR_TYPE_RGB;
    png_set_IHDR(png, info, width, height, 8, color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    const std::size_t row_bytes = static_cast<std::size_t>(width) *
                                  static_cast<std::size_t>(channels);
    for (int row = 0; row < height; ++row) {
        png_write_row(png, const_cast<png_bytep>(
            pixels.data() + static_cast<std::size_t>(row) * row_bytes));
    }
    png_write_end(png, info);
    png_destroy_write_struct(&png, &info);
    return true;
}
#else
bool encode_png(const std::vector<std::uint8_t> &, int, int, int,
                std::vector<std::uint8_t> &, std::string * error) {
    set_error(error, "GLB texture export requires libpng; reconfigure with PNG support");
    return false;
}
#endif

struct TextureBake {
    int size = 0;
    std::vector<std::uint8_t> base_color;
    std::vector<std::uint8_t> metallic_roughness;
    std::vector<std::uint8_t> valid;
};

bool barycentric(const Vec2 & p, const Vec2 & a, const Vec2 & b, const Vec2 & c,
                 float & w0, float & w1, float & w2) {
    const float denominator = (b.y - c.y) * (a.x - c.x) +
                              (c.x - b.x) * (a.y - c.y);
    if (std::fabs(denominator) < 1.0e-12f) return false;
    w0 = ((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y)) / denominator;
    w1 = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) / denominator;
    w2 = 1.0f - w0 - w1;
    return w0 >= -1.0e-5f && w1 >= -1.0e-5f && w2 >= -1.0e-5f;
}

void write_baked_pixel(TextureBake & bake, int x, int y, const std::array<float, 6> & value) {
    if (x < 0 || y < 0 || x >= bake.size || y >= bake.size) return;
    const std::size_t pixel = static_cast<std::size_t>(y * bake.size + x);
    if (bake.valid[pixel]) return;
    bake.valid[pixel] = 1;
    bake.base_color[pixel * 4 + 0] = to_byte(value[0]);
    bake.base_color[pixel * 4 + 1] = to_byte(value[1]);
    bake.base_color[pixel * 4 + 2] = to_byte(value[2]);
    bake.base_color[pixel * 4 + 3] = to_byte(value[5]);
    bake.metallic_roughness[pixel * 3 + 0] = 0;
    bake.metallic_roughness[pixel * 3 + 1] = to_byte(value[4]);
    bake.metallic_roughness[pixel * 3 + 2] = to_byte(value[3]);
}

void fill_baked_gaps(TextureBake & bake) {
    const std::size_t pixel_count = bake.valid.size();
    std::vector<std::int32_t> source(pixel_count, -1);
    std::queue<std::size_t> pending;
    for (std::size_t index = 0; index < pixel_count; ++index) {
        if (bake.valid[index]) {
            source[index] = static_cast<std::int32_t>(index);
            pending.push(index);
        }
    }
    const int width = bake.size;
    const int height = bake.size;
    while (!pending.empty()) {
        const std::size_t current = pending.front();
        pending.pop();
        const int x = static_cast<int>(current % static_cast<std::size_t>(width));
        const int y = static_cast<int>(current / static_cast<std::size_t>(width));
        const std::array<std::pair<int, int>, 4> neighbors = {
            std::make_pair(x - 1, y), std::make_pair(x + 1, y),
            std::make_pair(x, y - 1), std::make_pair(x, y + 1)};
        for (const auto & neighbor : neighbors) {
            if (neighbor.first < 0 || neighbor.second < 0 ||
                neighbor.first >= width || neighbor.second >= height) continue;
            const std::size_t next = static_cast<std::size_t>(neighbor.second * width + neighbor.first);
            if (source[next] >= 0) continue;
            source[next] = source[current];
            pending.push(next);
        }
    }
    for (std::size_t index = 0; index < pixel_count; ++index) {
        if (bake.valid[index] || source[index] < 0) continue;
        const std::size_t from = static_cast<std::size_t>(source[index]);
        std::copy_n(bake.base_color.data() + from * 4, 4,
                    bake.base_color.data() + index * 4);
        std::copy_n(bake.metallic_roughness.data() + from * 3, 3,
                    bake.metallic_roughness.data() + index * 3);
        bake.valid[index] = 1;
    }
}

// Chart-unwrapped export mesh.  Chart seams duplicate vertices, so the
// exported vertex set is generally larger than the decoder mesh's; positions
// are identical because xatlas only splits and welds, never displaces.
struct ExportMesh {
    std::vector<Vec3> positions;
    std::vector<Vec2> uvs;
    std::vector<std::uint32_t> indices;
};

// Quadric simplification to the reference decimation target.  Mirrors the
// reference to_glb standard branch, which always decimates before UV
// unwrapping; running xatlas on the raw multi-million-face dual grid mesh is
// impractically slow.  Indices reference the input vertex buffer unchanged.
bool simplify_export_mesh(const DualGridMeshF32 & mesh, std::size_t target,
                          std::vector<std::uint32_t> & indices,
                          std::string * error) {
    indices.assign(mesh.faces.size(), 0);
    for (std::size_t index = 0; index < mesh.faces.size(); ++index) {
        indices[index] = static_cast<std::uint32_t>(mesh.faces[index]);
    }
    if (target != 0 && mesh.faces.size() / 3 > target) {
        std::vector<std::uint32_t> destination(mesh.faces.size());
        float result_error = 0.0f;
        const std::size_t simplified = meshopt_simplify(
            destination.data(), indices.data(), indices.size(), mesh.vertices.data(),
            mesh.vertices.size() / 3, sizeof(float) * 3, target * 3, 1.0e-2f, 0,
            &result_error);
        if (simplified < 3 || simplified % 3 != 0) {
            set_error(error, "mesh simplification produced no usable faces");
            return false;
        }
        indices.assign(destination.begin(), destination.begin() + simplified);
    }

    // xatlas ignores faces at or below its FLT_EPSILON area threshold, and
    // every ignored face acts as a cut that keeps splitting charts into
    // single-texel fragments.  Simplification leaves collapsed faces behind,
    // so prune them exactly like the reference remove_degenerate_faces pass.
    const std::size_t before_prune = indices.size() / 3;
    std::vector<std::uint32_t> pruned;
    pruned.reserve(indices.size());
    for (std::size_t face = 0; face < before_prune; ++face) {
        const std::uint32_t a = indices[face * 3 + 0];
        const std::uint32_t b = indices[face * 3 + 1];
        const std::uint32_t c = indices[face * 3 + 2];
        if (a == b || b == c || a == c) continue;
        const float ux = mesh.vertices[std::size_t(b) * 3 + 0] - mesh.vertices[std::size_t(a) * 3 + 0];
        const float uy = mesh.vertices[std::size_t(b) * 3 + 1] - mesh.vertices[std::size_t(a) * 3 + 1];
        const float uz = mesh.vertices[std::size_t(b) * 3 + 2] - mesh.vertices[std::size_t(a) * 3 + 2];
        const float vx = mesh.vertices[std::size_t(c) * 3 + 0] - mesh.vertices[std::size_t(a) * 3 + 0];
        const float vy = mesh.vertices[std::size_t(c) * 3 + 1] - mesh.vertices[std::size_t(a) * 3 + 1];
        const float vz = mesh.vertices[std::size_t(c) * 3 + 2] - mesh.vertices[std::size_t(a) * 3 + 2];
        const float cx = uy * vz - uz * vy;
        const float cy = uz * vx - ux * vz;
        const float cz = ux * vy - uy * vx;
        const float area = 0.5f * std::sqrt(cx * cx + cy * cy + cz * cz);
        if (area <= std::numeric_limits<float>::epsilon()) continue;
        pruned.push_back(a);
        pruned.push_back(b);
        pruned.push_back(c);
    }
    indices.swap(pruned);
    if (indices.size() < 3) {
        set_error(error, "mesh simplification pruned every face");
        return false;
    }
    return true;
}

// Parameterize the mesh with xatlas, the same backend the reference
// postprocess wraps.  The dual grid export mesh is one multi-million-face
// connected component, and xatlas chart growing stalls on a mesh that large
// (progress sits at a few percent for hours).  The reference postprocess
// therefore splits the surface first and parameterizes each cluster as an
// independent xatlas mesh, which finishes in under a minute; faces are
// bucketed by centroid into a uniform spatial grid the same way.  All charts
// still pack into a single atlas, so the GLB keeps one texture.  Positions
// and faces come from the canonical decoder mesh; the UVs land in [0, 1]
// after packing.
bool unwrap_charts(const DualGridMeshF32 & mesh,
                   const std::vector<std::uint32_t> & indices, int texture_size,
                   ExportMesh & out, std::string * error) {
    const std::size_t vertex_count = mesh.vertices.size() / 3;
    const std::size_t face_count = indices.size() / 3;
    Vec3 minimum{mesh.vertices[0], mesh.vertices[1], mesh.vertices[2]};
    Vec3 maximum = minimum;
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        const Vec3 position{mesh.vertices[vertex * 3 + 0],
                            mesh.vertices[vertex * 3 + 1],
                            mesh.vertices[vertex * 3 + 2]};
        minimum = {std::min(minimum.x, position.x), std::min(minimum.y, position.y),
                   std::min(minimum.z, position.z)};
        maximum = {std::max(maximum.x, position.x), std::max(maximum.y, position.y),
                   std::max(maximum.z, position.z)};
    }
    const float extent = std::max({maximum.x - minimum.x, maximum.y - minimum.y,
                                   maximum.z - minimum.z});
    if (!std::isfinite(extent) || extent <= 0.0f) {
        set_error(error, "export mesh has a degenerate bounding extent");
        return false;
    }

    // One bucket per grid cell keeps xatlas input meshes in the low thousands
    // of faces, which it charts in milliseconds.  Bucket keys are ordered so
    // the atlas layout is deterministic.
    const std::size_t grid = face_count < 1000 ? 1 : 24;
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> buckets;
    buckets.reserve(face_count / 8 + 1);
    const float minimum_component[3] = {minimum.x, minimum.y, minimum.z};
    for (std::size_t face = 0; face < face_count; ++face) {
        float centroid[3] = {0.0f, 0.0f, 0.0f};
        for (int corner = 0; corner < 3; ++corner) {
            const std::uint32_t vertex = indices[face * 3 + corner];
            for (int axis = 0; axis < 3; ++axis) {
                centroid[axis] += mesh.vertices[std::size_t(vertex) * 3 + axis];
            }
        }
        std::uint64_t key = 0;
        for (int axis = 0; axis < 3; ++axis) {
            centroid[axis] /= 3.0f;
            const float normalized =
                (centroid[axis] - minimum_component[axis]) / extent;
            const auto cell = std::size_t(std::clamp(normalized, 0.0f, 0.9999f) *
                                          static_cast<float>(grid));
            key |= std::uint64_t(cell) << (20 * (2 - axis));
        }
        buckets[key].push_back(static_cast<std::uint32_t>(face));
    }
    std::vector<std::uint64_t> bucket_keys;
    bucket_keys.reserve(buckets.size());
    for (const auto & bucket : buckets) bucket_keys.push_back(bucket.first);
    std::sort(bucket_keys.begin(), bucket_keys.end());

    // Local vertex remap per bucket, so output vertices can be traced back to
    // the canonical decoder mesh.
    std::vector<std::vector<std::uint32_t>> bucket_vertices(bucket_keys.size());
    std::vector<std::vector<float>> bucket_positions(bucket_keys.size());
    std::vector<std::vector<std::uint32_t>> bucket_indices(bucket_keys.size());
    for (std::size_t bucket = 0; bucket < bucket_keys.size(); ++bucket) {
        std::unordered_map<std::uint32_t, std::uint32_t> remap;
        for (std::uint32_t face : buckets[bucket_keys[bucket]]) {
            for (int corner = 0; corner < 3; ++corner) {
                const std::uint32_t vertex = indices[face * 3 + corner];
                const auto inserted = remap.emplace(vertex,
                                                    std::uint32_t(bucket_vertices[bucket].size()));
                if (inserted.second) {
                    bucket_vertices[bucket].push_back(vertex);
                    for (int axis = 0; axis < 3; ++axis) {
                        bucket_positions[bucket].push_back(
                            mesh.vertices[std::size_t(vertex) * 3 + axis]);
                    }
                }
                bucket_indices[bucket].push_back(inserted.first->second);
            }
        }
    }

    xatlas::Atlas * atlas = nullptr;
    float texels_per_unit = 0.19f * texture_size / extent;
    for (int attempt = 0; attempt < 3 && !atlas; ++attempt) {
        // PackCharts cannot be re-run in place, so an over-dense packing is
        // retried from scratch at half the texel density.
        atlas = xatlas::Create();
        for (std::size_t bucket = 0; bucket < bucket_keys.size(); ++bucket) {
            xatlas::MeshDecl decl;
            decl.vertexCount = static_cast<std::uint32_t>(bucket_positions[bucket].size() / 3);
            decl.vertexPositionData = bucket_positions[bucket].data();
            decl.vertexPositionStride = sizeof(float) * 3;
            decl.indexCount = static_cast<std::uint32_t>(bucket_indices[bucket].size());
            decl.indexData = bucket_indices[bucket].data();
            decl.indexFormat = xatlas::IndexFormat::UInt32;
            const xatlas::AddMeshError added = xatlas::AddMesh(atlas, decl);
            if (added != xatlas::AddMeshError::Success) {
                xatlas::Destroy(atlas);
                set_error(error, std::string("xatlas rejected export mesh cluster ") +
                                     std::to_string(bucket) + ": " +
                                     xatlas::StringForEnum(added));
                return false;
            }
        }
        xatlas::ComputeCharts(atlas, xatlas::ChartOptions());
        xatlas::PackOptions pack_options;
        pack_options.resolution = texture_size;
        pack_options.texelsPerUnit = texels_per_unit;
        xatlas::PackCharts(atlas, pack_options);
        if (atlas->atlasCount > 1) {
            xatlas::Destroy(atlas);
            atlas = nullptr;
            texels_per_unit *= 0.5f;
        }
    }
    if (atlas == nullptr) {
        set_error(error, "xatlas could not pack the unwrapped charts into a " +
                             std::to_string(texture_size) + "x" +
                             std::to_string(texture_size) + " atlas");
        return false;
    }
    if (atlas->meshCount == 0 || atlas->width == 0 || atlas->height == 0) {
        xatlas::Destroy(atlas);
        set_error(error, "xatlas produced an empty UV atlas");
        return false;
    }
    for (std::uint32_t bucket = 0; bucket < atlas->meshCount; ++bucket) {
        const xatlas::Mesh & output = atlas->meshes[bucket];
        if (output.indexCount == 0 || output.vertexCount == 0) continue;
        const std::vector<std::uint32_t> & local_to_global =
            bucket_vertices[bucket];
        const std::uint32_t vertex_base = static_cast<std::uint32_t>(out.positions.size());
        out.positions.resize(out.positions.size() + output.vertexCount);
        out.uvs.resize(out.uvs.size() + output.vertexCount);
        for (std::uint32_t vertex = 0; vertex < output.vertexCount; ++vertex) {
            const xatlas::Vertex & source = output.vertexArray[vertex];
            const std::uint32_t global = local_to_global[source.xref];
            out.positions[vertex_base + vertex] = {
                mesh.vertices[std::size_t(global) * 3 + 0],
                mesh.vertices[std::size_t(global) * 3 + 1],
                mesh.vertices[std::size_t(global) * 3 + 2]};
            out.uvs[vertex_base + vertex] = {
                source.uv[0] / static_cast<float>(atlas->width),
                source.uv[1] / static_cast<float>(atlas->height)};
        }
        out.indices.reserve(out.indices.size() + output.indexCount);
        for (std::uint32_t index = 0; index < output.indexCount; ++index) {
            out.indices.push_back(vertex_base + output.indexArray[index]);
        }
    }
    const std::uint32_t atlas_width = atlas->width;
    const std::uint32_t atlas_height = atlas->height;
    xatlas::Destroy(atlas);
    if (out.indices.size() < 3 || out.positions.empty()) {
        set_error(error, "xatlas produced no unwrapped faces");
        return false;
    }
    if (atlas_width == 0 || atlas_height == 0) {
        set_error(error, "xatlas produced a zero-sized atlas");
        return false;
    }
    return true;
}

bool bake_textures(const ExportMesh & mesh, const TextureVolume & volume,
                   int texture_size, TextureBake & bake, std::string * error) {
    bake = TextureBake{};
    bake.size = texture_size;
    const std::size_t pixels = static_cast<std::size_t>(texture_size) *
                               static_cast<std::size_t>(texture_size);
    bake.base_color.assign(pixels * 4, 0);
    bake.metallic_roughness.assign(pixels * 3, 0);
    bake.valid.assign(pixels, 0);
    for (std::size_t face = 0; face < mesh.indices.size() / 3; ++face) {
        const std::size_t ia = static_cast<std::size_t>(mesh.indices[face * 3 + 0]);
        const std::size_t ib = static_cast<std::size_t>(mesh.indices[face * 3 + 1]);
        const std::size_t ic = static_cast<std::size_t>(mesh.indices[face * 3 + 2]);
        std::array<Vec2, 3> uv = {mesh.uvs[ia], mesh.uvs[ib], mesh.uvs[ic]};
        const float minimum_u = std::min({uv[0].x, uv[1].x, uv[2].x});
        const float maximum_u = std::max({uv[0].x, uv[1].x, uv[2].x});
        if (maximum_u - minimum_u > 0.5f) {
            for (Vec2 & value : uv) if (value.x < 0.5f) value.x += 1.0f;
        }
        for (int shift = -1; shift <= 1; ++shift) {
            std::array<Vec2, 3> shifted = uv;
            for (Vec2 & value : shifted) value.x -= static_cast<float>(shift);
            const float min_x = std::max(0.0f, std::min({shifted[0].x, shifted[1].x, shifted[2].x}));
            const float max_x = std::min(1.0f, std::max({shifted[0].x, shifted[1].x, shifted[2].x}));
            const float min_y = std::max(0.0f, std::min({shifted[0].y, shifted[1].y, shifted[2].y}));
            const float max_y = std::min(1.0f, std::max({shifted[0].y, shifted[1].y, shifted[2].y}));
            if (min_x > max_x || min_y > max_y) continue;
            const int x0 = std::max(0, static_cast<int>(std::floor(min_x * texture_size)) - 1);
            const int x1 = std::min(texture_size - 1, static_cast<int>(std::ceil(max_x * texture_size)) + 1);
            const int y0 = std::max(0, static_cast<int>(std::floor(min_y * texture_size)) - 1);
            const int y1 = std::min(texture_size - 1, static_cast<int>(std::ceil(max_y * texture_size)) + 1);
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    const Vec2 sample_uv{
                        (static_cast<float>(x) + 0.5f) / static_cast<float>(texture_size),
                        (static_cast<float>(y) + 0.5f) / static_cast<float>(texture_size)};
                    float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f;
                    if (!barycentric(sample_uv, shifted[0], shifted[1], shifted[2], w0, w1, w2)) continue;
                    const Vec3 position = mesh.positions[ia] * w0 + mesh.positions[ib] * w1 +
                                          mesh.positions[ic] * w2;
                    float weight_sum = 0.0f;
                    const std::array<float, 6> value =
                        sample_volume(volume, position, weight_sum);
                    if (weight_sum <= 0.0f) continue;
                    std::array<float, 6> normalized = value;
                    for (float & channel : normalized) channel /= weight_sum;
                    write_baked_pixel(bake, x, y, normalized);
                }
            }
        }
    }
    if (std::none_of(bake.valid.begin(), bake.valid.end(), [](std::uint8_t value) { return value != 0; })) {
        set_error(error, "GLB texture bake produced no covered texels");
        return false;
    }
    fill_baked_gaps(bake);
    return true;
}

void append_u32(std::vector<std::uint8_t> & output, std::uint32_t value) {
    output.push_back(static_cast<std::uint8_t>(value & 0xffu));
    output.push_back(static_cast<std::uint8_t>((value >> 8) & 0xffu));
    output.push_back(static_cast<std::uint8_t>((value >> 16) & 0xffu));
    output.push_back(static_cast<std::uint8_t>((value >> 24) & 0xffu));
}

void append_f32(std::vector<std::uint8_t> & output, float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "unexpected float size");
    std::memcpy(&bits, &value, sizeof(bits));
    append_u32(output, bits);
}

void align4(std::vector<std::uint8_t> & output, std::uint8_t value = 0) {
    while (output.size() % 4 != 0) output.push_back(value);
}

std::string json_escape(const std::string & value) {
    std::string result;
    for (char character : value) {
        if (character == '"' || character == '\\') {
            result.push_back('\\');
            result.push_back(character);
        } else if (character == '\n') {
            result += "\\n";
        } else {
            result.push_back(character);
        }
    }
    return result;
}

bool write_glb(const std::string & path, const std::string & json,
               const std::vector<std::uint8_t> & binary, std::string * error) {
    std::vector<std::uint8_t> json_chunk(json.begin(), json.end());
    while (json_chunk.size() % 4 != 0) json_chunk.push_back(' ');
    std::vector<std::uint8_t> bin_chunk = binary;
    while (bin_chunk.size() % 4 != 0) bin_chunk.push_back(0);
    const std::uint64_t total_size = 12ull + 8ull + json_chunk.size() + 8ull + bin_chunk.size();
    if (total_size > std::numeric_limits<std::uint32_t>::max()) {
        set_error(error, "GLB output exceeds the 32-bit container size limit");
        return false;
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        set_error(error, "cannot open GLB output: " + path);
        return false;
    }
    std::vector<std::uint8_t> header;
    append_u32(header, 0x46546c67u);
    append_u32(header, 2u);
    append_u32(header, static_cast<std::uint32_t>(total_size));
    append_u32(header, static_cast<std::uint32_t>(json_chunk.size()));
    append_u32(header, 0x4e4f534au);
    header.insert(header.end(), json_chunk.begin(), json_chunk.end());
    append_u32(header, static_cast<std::uint32_t>(bin_chunk.size()));
    append_u32(header, 0x004e4942u);
    header.insert(header.end(), bin_chunk.begin(), bin_chunk.end());
    file.write(reinterpret_cast<const char *>(header.data()), static_cast<std::streamsize>(header.size()));
    if (!file) {
        set_error(error, "failed while writing GLB output: " + path);
        return false;
    }
    return true;
}

} // namespace

bool write_pixal3d_glb(const DualGridMeshF32 & mesh,
                       const SparseTensorF32 & texture_decoded,
                       int resolution,
                       const Pixal3DGlbOptions & options,
                       const std::string & path,
                       std::string * error) {
    if (!validate_mesh(mesh, error)) return false;
    if (resolution != 1024) {
        set_error(error, "GLB export requires the supported 1024 cascade resolution");
        return false;
    }
    if (options.texture_size <= 0 || options.texture_size > 4096) {
        set_error(error, "GLB texture size must be in the range 1..4096");
        return false;
    }
    TextureVolume volume;
    if (!make_texture_volume(texture_decoded, resolution, volume, error)) return false;
    const auto export_start = std::chrono::steady_clock::now();
    const auto phase_log = [&export_start](const std::string & label) {
        std::cerr << "pixal3d: export " << label << " took "
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - export_start).count()
                  << " s" << std::endl;
    };
    phase_log("texture volume (" + std::to_string(texture_decoded.points()) +
              " points)");
    std::vector<std::uint32_t> simplified_indices;
    if (!simplify_export_mesh(mesh, options.simplify_target, simplified_indices,
                              error)) return false;
    phase_log("simplify (" + std::to_string(mesh.faces.size() / 3) + " -> " +
              std::to_string(simplified_indices.size() / 3) + " faces)");
    ExportMesh mesh_export;
    if (!unwrap_charts(mesh, simplified_indices, options.texture_size,
                       mesh_export, error)) return false;
    phase_log("xatlas unwrap (" + std::to_string(mesh_export.positions.size()) +
              " vertices, " + std::to_string(mesh_export.indices.size() / 3) +
              " faces)");
    TextureBake bake;
    if (!bake_textures(mesh_export, volume, options.texture_size, bake, error)) return false;
    phase_log("bake (" + std::to_string(bake.size) + "x" +
              std::to_string(bake.size) + ")");
    std::vector<std::uint8_t> base_png;
    std::vector<std::uint8_t> mr_png;
    if (!encode_png(bake.base_color, bake.size, bake.size, 4, base_png, error) ||
        !encode_png(bake.metallic_roughness, bake.size, bake.size, 3, mr_png, error)) return false;

    const std::size_t vertex_count = mesh_export.positions.size();
    const std::size_t index_count = mesh_export.indices.size();
    std::vector<Vec3> normals(vertex_count, {0.0f, 0.0f, 0.0f});
    for (std::size_t face = 0; face < index_count; face += 3) {
        const std::size_t ia = static_cast<std::size_t>(mesh_export.indices[face + 0]);
        const std::size_t ib = static_cast<std::size_t>(mesh_export.indices[face + 1]);
        const std::size_t ic = static_cast<std::size_t>(mesh_export.indices[face + 2]);
        const Vec3 normal = cross(mesh_export.positions[ib] - mesh_export.positions[ia],
                                  mesh_export.positions[ic] - mesh_export.positions[ia]);
        normals[ia] = normals[ia] + normal;
        normals[ib] = normals[ib] + normal;
        normals[ic] = normals[ic] + normal;
    }
    for (Vec3 & normal : normals) normal = normalize(normal);

    std::vector<std::uint8_t> binary;
    const auto append_blob = [&binary](const void * data, std::size_t bytes) {
        align4(binary);
        const std::size_t offset = binary.size();
        const auto * source = static_cast<const std::uint8_t *>(data);
        binary.insert(binary.end(), source, source + bytes);
        return offset;
    };
    std::vector<std::uint8_t> position_bytes;
    std::vector<std::uint8_t> normal_bytes;
    std::vector<std::uint8_t> uv_bytes;
    std::vector<std::uint8_t> index_bytes;
    position_bytes.reserve(vertex_count * 12);
    normal_bytes.reserve(vertex_count * 12);
    uv_bytes.reserve(vertex_count * 8);
    index_bytes.reserve(index_count * 4);
    // Match the final coordinate frame of the Python textured GLB path.
    // o_voxel.postprocess.to_glb() first maps (x, y, z) -> (x, z, -y),
    // then Pixal3D applies (-x, -z, -y), yielding H=(-x, +y, -z).
    const Vec3 first_exported{-mesh_export.positions.front().x,
                              mesh_export.positions.front().y,
                              -mesh_export.positions.front().z};
    Vec3 minimum = first_exported;
    Vec3 maximum = first_exported;
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        const Vec3 p = mesh_export.positions[vertex];
        const Vec3 exported{-p.x, p.y, -p.z};
        append_f32(position_bytes, exported.x);
        append_f32(position_bytes, exported.y);
        append_f32(position_bytes, exported.z);
        const Vec3 n = normals[vertex];
        const Vec3 exported_normal{-n.x, n.y, -n.z};
        append_f32(normal_bytes, exported_normal.x);
        append_f32(normal_bytes, exported_normal.y);
        append_f32(normal_bytes, exported_normal.z);
        append_f32(uv_bytes, mesh_export.uvs[vertex].x);
        append_f32(uv_bytes, mesh_export.uvs[vertex].y);
        minimum.x = std::min(minimum.x, exported.x);
        minimum.y = std::min(minimum.y, exported.y);
        minimum.z = std::min(minimum.z, exported.z);
        maximum.x = std::max(maximum.x, exported.x);
        maximum.y = std::max(maximum.y, exported.y);
        maximum.z = std::max(maximum.z, exported.z);
    }
    for (std::uint32_t index : mesh_export.indices) append_u32(index_bytes, index);
    const std::size_t position_offset = append_blob(position_bytes.data(), position_bytes.size());
    const std::size_t normal_offset = append_blob(normal_bytes.data(), normal_bytes.size());
    const std::size_t uv_offset = append_blob(uv_bytes.data(), uv_bytes.size());
    const std::size_t index_offset = append_blob(index_bytes.data(), index_bytes.size());
    const std::size_t base_png_offset = append_blob(base_png.data(), base_png.size());
    const std::size_t mr_png_offset = append_blob(mr_png.data(), mr_png.size());
    const std::size_t base_png_length = base_png.size();
    const std::size_t mr_png_length = mr_png.size();

    std::ostringstream json;
    json << "{"
         << "\"asset\":{\"version\":\"2.0\",\"generator\":\"Pixal3D.cpp native approximate PBR exporter\"},"
         << "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],"
         << "\"nodes\":[{\"mesh\":0}],"
         << "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"NORMAL\":1,\"TEXCOORD_0\":2},\"indices\":3,\"material\":0}]}],"
         << "\"materials\":[{\"name\":\"Pixal3D PBR material\",\"doubleSided\":true,\"alphaMode\":\"OPAQUE\",\"pbrMetallicRoughness\":{\"baseColorTexture\":{\"index\":0},\"metallicRoughnessTexture\":{\"index\":1},\"metallicFactor\":1.0,\"roughnessFactor\":1.0,\"baseColorFactor\":[1,1,1,1]}}],"
         << "\"textures\":[{\"sampler\":0,\"source\":0},{\"sampler\":0,\"source\":1}],"
         << "\"samplers\":[{\"magFilter\":9729,\"minFilter\":9987,\"wrapS\":10497,\"wrapT\":10497}],"
         << "\"images\":[{\"bufferView\":4,\"mimeType\":\"image/png\"},{\"bufferView\":5,\"mimeType\":\"image/png\"}],"
         << "\"accessors\":["
         << "{\"bufferView\":0,\"componentType\":5126,\"count\":" << vertex_count << ",\"type\":\"VEC3\",\"min\":["
         << minimum.x << "," << minimum.y << "," << minimum.z << "],\"max\":["
         << maximum.x << "," << maximum.y << "," << maximum.z << "]},"
         << "{\"bufferView\":1,\"componentType\":5126,\"count\":" << vertex_count << ",\"type\":\"VEC3\"},"
         << "{\"bufferView\":2,\"componentType\":5126,\"count\":" << vertex_count << ",\"type\":\"VEC2\"},"
         << "{\"bufferView\":3,\"componentType\":5125,\"count\":" << index_count << ",\"type\":\"SCALAR\"}],"
         << "\"bufferViews\":["
         << "{\"buffer\":0,\"byteOffset\":" << position_offset << ",\"byteLength\":" << position_bytes.size() << ",\"target\":34962},"
         << "{\"buffer\":0,\"byteOffset\":" << normal_offset << ",\"byteLength\":" << normal_bytes.size() << ",\"target\":34962},"
         << "{\"buffer\":0,\"byteOffset\":" << uv_offset << ",\"byteLength\":" << uv_bytes.size() << ",\"target\":34962},"
         << "{\"buffer\":0,\"byteOffset\":" << index_offset << ",\"byteLength\":" << index_bytes.size() << ",\"target\":34963},"
         << "{\"buffer\":0,\"byteOffset\":" << base_png_offset << ",\"byteLength\":" << base_png_length << "},"
         << "{\"buffer\":0,\"byteOffset\":" << mr_png_offset << ",\"byteLength\":" << mr_png_length << "}],"
         << "\"buffers\":[{\"byteLength\":" << ((binary.size() + 3u) & ~std::size_t(3u)) << "}]"
         << "}";
    return write_glb(path, json.str(), binary, error);
}

} // namespace pixal3d
