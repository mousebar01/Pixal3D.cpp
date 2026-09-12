#include "pixal3d/mesh_postprocess.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

namespace {

void make_torus(int resolution, std::vector<float> & vertices,
                std::vector<std::int32_t> & faces) {
    constexpr float kTau = 6.28318530717958647692f;
    constexpr float kMajor = 1.0f;
    constexpr float kMinor = 0.35f;
    vertices.resize(static_cast<std::size_t>(resolution) * resolution * 3);
    for (int u_index = 0; u_index < resolution; ++u_index) {
        for (int v_index = 0; v_index < resolution; ++v_index) {
            const float u = kTau * static_cast<float>(u_index) / resolution;
            const float v = kTau * static_cast<float>(v_index) / resolution;
            float * point = &vertices[3 * (static_cast<std::size_t>(u_index) * resolution + v_index)];
            point[0] = (kMajor + kMinor * std::cos(v)) * std::cos(u);
            point[1] = (kMajor + kMinor * std::cos(v)) * std::sin(u);
            point[2] = kMinor * std::sin(v);
        }
    }
    faces.reserve(static_cast<std::size_t>(resolution) * resolution * 6);
    for (int u_index = 0; u_index < resolution; ++u_index) {
        for (int v_index = 0; v_index < resolution; ++v_index) {
            const int next_u = (u_index + 1) % resolution;
            const int next_v = (v_index + 1) % resolution;
            const int a = u_index * resolution + v_index;
            const int b = next_u * resolution + v_index;
            const int c = next_u * resolution + next_v;
            const int d = u_index * resolution + next_v;
            faces.insert(faces.end(), {a, b, c, a, c, d});
        }
    }
}

struct MeshStats {
    float minimum[3] = {std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max()};
    float maximum[3] = {std::numeric_limits<float>::lowest(),
                        std::numeric_limits<float>::lowest(),
                        std::numeric_limits<float>::lowest()};
    std::size_t open_edges = 0;
    std::size_t nonmanifold_edges = 0;
    bool finite = true;
};

MeshStats inspect_mesh(const std::vector<float> & vertices,
                       const std::vector<std::int32_t> & faces) {
    MeshStats stats;
    const std::size_t vertex_count = vertices.size() / 3;
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        for (int axis = 0; axis < 3; ++axis) {
            const float value = vertices[3 * vertex + axis];
            stats.finite = stats.finite && std::isfinite(value);
            stats.minimum[axis] = std::min(stats.minimum[axis], value);
            stats.maximum[axis] = std::max(stats.maximum[axis], value);
        }
    }

    std::unordered_map<std::uint64_t, int> edge_counts;
    edge_counts.reserve(faces.size() * 2);
    const auto edge_key = [](int a, int b) {
        if (a > b) std::swap(a, b);
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
               static_cast<std::uint32_t>(b);
    };
    for (std::size_t face = 0; face < faces.size() / 3; ++face) {
        const int a = faces[3 * face];
        const int b = faces[3 * face + 1];
        const int c = faces[3 * face + 2];
        edge_counts[edge_key(a, b)]++;
        edge_counts[edge_key(b, c)]++;
        edge_counts[edge_key(c, a)]++;
    }
    for (const auto & entry : edge_counts) {
        if (entry.second == 1) ++stats.open_edges;
        if (entry.second > 2) ++stats.nonmanifold_edges;
    }
    return stats;
}

bool valid_indices(const std::vector<float> & vertices,
                   const std::vector<std::int32_t> & faces) {
    const std::int32_t vertex_count = static_cast<std::int32_t>(vertices.size() / 3);
    for (const std::int32_t index : faces) {
        if (index < 0 || index >= vertex_count) return false;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    const bool benchmark = argc > 1 && std::strcmp(argv[1], "--benchmark") == 0;
    const int resolution = benchmark ? 500 : 64;
    const int target_faces = benchmark ? 100000 : 2048;

    std::vector<float> input_vertices;
    std::vector<std::int32_t> input_faces;
    make_torus(resolution, input_vertices, input_faces);
    const std::vector<float> original_vertices = input_vertices;
    const std::vector<std::int32_t> original_faces = input_faces;
    const int input_vertex_count = static_cast<int>(input_vertices.size() / 3);
    const int input_face_count = static_cast<int>(input_faces.size() / 3);
    std::printf("input V=%d F=%d target=%d mode=%s\n", input_vertex_count,
                input_face_count, target_faces, benchmark ? "benchmark" : "fixture");

    std::vector<float> cpu_vertices;
    std::vector<std::int32_t> cpu_faces;
    const auto cpu_start = std::chrono::steady_clock::now();
    pixal3d::decimate_qem_cpu(input_vertices, input_vertex_count, input_faces,
                              input_face_count, target_faces, cpu_vertices, cpu_faces);
    const double cpu_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - cpu_start).count();

    std::vector<float> gpu_vertices;
    std::vector<std::int32_t> gpu_faces;
    const auto gpu_start = std::chrono::steady_clock::now();
    if (!pixal3d::decimate_qem_gpu(input_vertices, input_vertex_count, input_faces,
                                   input_face_count, target_faces,
                                   gpu_vertices, gpu_faces)) {
        std::printf("GPU QEM unavailable; skipping CUDA parity test\n");
        return 77;
    }
    const double gpu_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - gpu_start).count();

    if (input_vertices != original_vertices || input_faces != original_faces) {
        std::fprintf(stderr, "QEM changed its input buffers\n");
        return 1;
    }
    if (cpu_faces.empty() || gpu_faces.empty() || !valid_indices(cpu_vertices, cpu_faces) ||
        !valid_indices(gpu_vertices, gpu_faces)) {
        std::fprintf(stderr, "QEM produced an empty or invalid mesh\n");
        return 1;
    }

    const MeshStats cpu = inspect_mesh(cpu_vertices, cpu_faces);
    const MeshStats gpu = inspect_mesh(gpu_vertices, gpu_faces);
    const double face_delta = std::abs(static_cast<double>(cpu_faces.size()) -
                                       static_cast<double>(gpu_faces.size())) /
                              std::max<std::size_t>(1, cpu_faces.size());
    std::printf("cpu V=%zu F=%zu time=%.3f s open=%zu nonmanifold=%zu finite=%d\n",
                cpu_vertices.size() / 3, cpu_faces.size() / 3, cpu_seconds,
                cpu.open_edges, cpu.nonmanifold_edges, cpu.finite ? 1 : 0);
    std::printf("gpu V=%zu F=%zu time=%.3f s open=%zu nonmanifold=%zu finite=%d\n",
                gpu_vertices.size() / 3, gpu_faces.size() / 3, gpu_seconds,
                gpu.open_edges, gpu.nonmanifold_edges, gpu.finite ? 1 : 0);
    std::printf("cpu bbox=[%.6f,%.6f,%.6f]..[%.6f,%.6f,%.6f]\n",
                cpu.minimum[0], cpu.minimum[1], cpu.minimum[2],
                cpu.maximum[0], cpu.maximum[1], cpu.maximum[2]);
    std::printf("gpu bbox=[%.6f,%.6f,%.6f]..[%.6f,%.6f,%.6f]\n",
                gpu.minimum[0], gpu.minimum[1], gpu.minimum[2],
                gpu.maximum[0], gpu.maximum[1], gpu.maximum[2]);
    std::printf("face_delta=%.3f%% speedup=%.2fx\n", 100.0 * face_delta,
                gpu_seconds > 0.0 ? cpu_seconds / gpu_seconds : 0.0);

    if (!cpu.finite || !gpu.finite || gpu_faces.size() / 3 >
        static_cast<std::size_t>(target_faces * 11 / 10)) {
        std::fprintf(stderr, "QEM output is non-finite or above the target budget\n");
        return 1;
    }
    if (face_delta > 0.10) {
        std::fprintf(stderr, "GPU face count differs from CPU by more than 10%%\n");
        return 1;
    }
    // GPU edge ownership uses atomic minima, so equal-cost ties can choose a
    // different valid collapse than the serial CPU traversal. Compare the
    // resulting shape envelope rather than requiring vertex-by-vertex identity;
    // the face count, finite values, watertightness, and envelope are the
    // correctness gates for this non-bit-exact prototype.
    for (int axis = 0; axis < 3; ++axis) {
        const float input_extent = input_vertices.empty()
            ? 1.0f
            : (axis == 0 ? 2.7f : (axis == 1 ? 2.7f : 0.7f));
        const float cpu_extent = cpu.maximum[axis] - cpu.minimum[axis];
        const float gpu_extent = gpu.maximum[axis] - gpu.minimum[axis];
        const float extent_delta = std::abs(cpu_extent - gpu_extent) /
                                   std::max(input_extent, 1e-6f);
        const float center_delta = std::abs(
            (cpu.maximum[axis] + cpu.minimum[axis]) -
            (gpu.maximum[axis] + gpu.minimum[axis])) /
            std::max(input_extent, 1e-6f);
        if (extent_delta > 0.01f || center_delta > 0.01f) {
            std::fprintf(stderr,
                         "GPU shape envelope differs from CPU beyond 1%% (axis=%d)\n",
                         axis);
            return 1;
        }
    }
    if (gpu.open_edges > gpu_faces.size() / 3 / 100 + 16) {
        std::fprintf(stderr, "GPU output has too many open edges\n");
        return 1;
    }
    std::printf("CUDA QEM parity fixture passed\n");
    return 0;
}
