#include "pixal3d/texture_export.h"

#include "pixal3d/mesh_postprocess.h"
#include "pixal3d/tri_bvh.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(PIXAL3D_HAVE_PNG)
#include <png.h>
#endif

namespace pixal3d {
namespace {

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

    // Sparse PBR volume for the bake: strip the batch index from the decoded
    // coordinates.  Feats are [0,1] with (base RGB, metallic, roughness,
    // alpha), matching the reference bake input layout.
    const std::size_t point_count = texture_decoded.points();
    std::vector<std::array<int, 3>> voxel_coords(point_count);
    for (std::size_t point = 0; point < point_count; ++point) {
        const std::int32_t batch = texture_decoded.coords[point * 4 + 0];
        if (batch != 0) {
            set_error(error, "GLB export currently supports the first texture batch only");
            return false;
        }
        voxel_coords[point] = {texture_decoded.coords[point * 4 + 1],
                               texture_decoded.coords[point * 4 + 2],
                               texture_decoded.coords[point * 4 + 3]};
    }
    VoxelPbr vox;
    vox.coords = &voxel_coords;
    vox.feats = &texture_decoded.feats;
    vox.res = resolution;

    // Mesh postprocess chain (mesh_postprocess.cpp, adapted from
    // pwilkin/trellis.cpp): weld hairline cracks, unify winding, drop
    // floating fragments, Taubin-smooth the voxel stair-step noise, then
    // CuMesh-port QEM decimation to the reference target and hole filling.
    // The dual grid mesh ships inconsistently wound, cracked, and too sliver
    // heavy for a plain meshopt pass; this chain is the same one trellis.cpp
    // validated end to end against the reference postprocess.
    std::vector<float> verts(mesh.vertices.begin(), mesh.vertices.end());
    std::vector<std::int32_t> faces(mesh.faces.begin(), mesh.faces.end());
    const auto export_start = std::chrono::steady_clock::now();
    const auto phase_log = [&export_start](const std::string & label) {
        std::cerr << "pixal3d: export " << label << " took "
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - export_start).count()
                  << " s" << std::endl;
    };
    weld_vertices(verts, faces, nullptr, 1.0f / 8192.0f);
    clean_mesh(static_cast<int>(verts.size() / 3), faces);
    drop_small_components(verts, faces, 0.02f);
    taubin_smooth(verts, faces, 5, 0.5f, -0.53f);
    phase_log("weld+clean+smooth (" + std::to_string(faces.size() / 3) + " faces)");

    // Snap BVH over the pre-decimation surface, the equivalent of the
    // reference cuBVH off-shell correction for texels between voxels.
    TriBvh snap_bvh = TriBvh::build(
        verts.data(), static_cast<std::int64_t>(verts.size() / 3),
        faces.data(), static_cast<std::int64_t>(faces.size() / 3));
    vox.snap = &snap_bvh;

    std::vector<float> decimated_verts;
    std::vector<std::int32_t> decimated_faces;
    decimate_qem(verts, static_cast<int>(verts.size() / 3), faces,
                          static_cast<int>(faces.size() / 3),
                          static_cast<int>(options.simplify_target),
                          decimated_verts, decimated_faces);
    fill_small_holes(decimated_faces, 64);
    phase_log("qem decimate (" + std::to_string(faces.size() / 3) + " -> " +
              std::to_string(decimated_faces.size() / 3) + " faces)");

    const BakedMesh baked = uv_bake(
        decimated_verts, static_cast<int>(decimated_verts.size() / 3),
        decimated_faces, static_cast<int>(decimated_faces.size() / 3),
        {}, options.texture_size, &vox);
    phase_log("uv bake (" + std::to_string(baked.T) + "x" +
              std::to_string(baked.T) + ")");
    if (!baked.ok()) {
        set_error(error, "texture bake produced no usable mesh");
        return false;
    }

    std::vector<std::uint8_t> base_png;
    std::vector<std::uint8_t> mr_png;
    if (!encode_png(baked.base, baked.T, baked.T, 4, base_png, error) ||
        !encode_png(baked.mr, baked.T, baked.T, 4, mr_png, error)) return false;

    const std::size_t vertex_count = baked.verts.size() / 3;
    const std::size_t index_count = baked.faces.size();
    std::vector<Vec3> positions(vertex_count);
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        positions[vertex] = {baked.verts[vertex * 3 + 0], baked.verts[vertex * 3 + 1],
                             baked.verts[vertex * 3 + 2]};
    }
    std::vector<Vec3> normals(vertex_count, {0.0f, 0.0f, 0.0f});
    for (std::size_t face = 0; face < index_count; face += 3) {
        const std::size_t ia = static_cast<std::size_t>(baked.faces[face + 0]);
        const std::size_t ib = static_cast<std::size_t>(baked.faces[face + 1]);
        const std::size_t ic = static_cast<std::size_t>(baked.faces[face + 2]);
        const Vec3 normal = cross(positions[ib] - positions[ia],
                                  positions[ic] - positions[ia]);
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
    // then Pixal3D applies (-x, -z, -y), yielding H=(-x, +y, -z).  The
    // uv_bake emits GLB-convention UVs directly (verified by render
    // comparison); applying the reference's V-flip here mirrors the texture.
    const Vec3 first_exported{-positions.front().x, positions.front().y,
                              -positions.front().z};
    Vec3 minimum = first_exported;
    Vec3 maximum = first_exported;
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        const Vec3 p = positions[vertex];
        const Vec3 exported{-p.x, p.y, -p.z};
        append_f32(position_bytes, exported.x);
        append_f32(position_bytes, exported.y);
        append_f32(position_bytes, exported.z);
        const Vec3 n = normals[vertex];
        const Vec3 exported_normal{-n.x, n.y, -n.z};
        append_f32(normal_bytes, exported_normal.x);
        append_f32(normal_bytes, exported_normal.y);
        append_f32(normal_bytes, exported_normal.z);
        append_f32(uv_bytes, baked.uv[vertex * 2 + 0]);
        append_f32(uv_bytes, baked.uv[vertex * 2 + 1]);
        minimum.x = std::min(minimum.x, exported.x);
        minimum.y = std::min(minimum.y, exported.y);
        minimum.z = std::min(minimum.z, exported.z);
        maximum.x = std::max(maximum.x, exported.x);
        maximum.y = std::max(maximum.y, exported.y);
        maximum.z = std::max(maximum.z, exported.z);
    }
    for (std::int32_t index : baked.faces) append_u32(index_bytes, static_cast<std::uint32_t>(index));
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
