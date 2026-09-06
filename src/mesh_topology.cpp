#include "pixal3d/mesh_topology.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace pixal3d {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vec3 operator-(const Vec3 & left, const Vec3 & right) {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

Vec3 operator+(const Vec3 & left, const Vec3 & right) {
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

Vec3 operator/(const Vec3 & value, double divisor) {
    return {value.x / divisor, value.y / divisor, value.z / divisor};
}

Vec3 cross(const Vec3 & left, const Vec3 & right) {
    return {left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x};
}

double dot(const Vec3 & left, const Vec3 & right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

double norm_squared(const Vec3 & value) {
    return dot(value, value);
}

Vec3 vertex_at(const DualGridMeshF32 & mesh, std::int32_t index) {
    const std::size_t offset = static_cast<std::size_t>(index) * 3;
    return {mesh.vertices[offset + 0], mesh.vertices[offset + 1],
            mesh.vertices[offset + 2]};
}

struct Edge {
    std::int32_t a = 0;
    std::int32_t b = 0;

    bool operator<(const Edge & other) const noexcept {
        return a < other.a || (a == other.a && b < other.b);
    }

    bool operator==(const Edge & other) const noexcept {
        return a == other.a && b == other.b;
    }
};

Edge make_edge(std::int32_t a, std::int32_t b) {
    if (b < a) std::swap(a, b);
    return {a, b};
}

struct EdgeRecord {
    Edge edge;
    std::int32_t directed_a = 0;
    std::int32_t directed_b = 0;
    std::int32_t third = 0;
};

bool edge_record_less(const EdgeRecord & left, const EdgeRecord & right) {
    return left.edge < right.edge;
}

bool validate_mesh(const DualGridMeshF32 & mesh, std::string * error) {
    if (mesh.vertices.size() % 3 != 0 || mesh.faces.size() % 3 != 0) {
        set_error(error, "mesh topology arrays must contain xyz/index triples");
        return false;
    }
    const std::size_t vertex_count = mesh.vertices.size() / 3;
    if (vertex_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        set_error(error, "mesh has too many vertices for int32 indices");
        return false;
    }
    for (float value : mesh.vertices) {
        if (!std::isfinite(value)) {
            set_error(error, "mesh contains a non-finite vertex");
            return false;
        }
    }
    for (std::size_t face = 0; face < mesh.faces.size() / 3; ++face) {
        const std::int32_t a = mesh.faces[face * 3 + 0];
        const std::int32_t b = mesh.faces[face * 3 + 1];
        const std::int32_t c = mesh.faces[face * 3 + 2];
        if (a < 0 || b < 0 || c < 0 ||
            static_cast<std::size_t>(a) >= vertex_count ||
            static_cast<std::size_t>(b) >= vertex_count ||
            static_cast<std::size_t>(c) >= vertex_count ||
            a == b || b == c || a == c) {
            set_error(error, "mesh contains an invalid or degenerate face");
            return false;
        }
    }
    return true;
}

std::size_t find_edge(const std::vector<EdgeRecord> & edges, const Edge & edge) {
    const auto found = std::lower_bound(
        edges.begin(), edges.end(), EdgeRecord{edge, 0, 0, 0}, edge_record_less);
    if (found == edges.end() || !(found->edge == edge)) return edges.size();
    return static_cast<std::size_t>(found - edges.begin());
}

} // namespace

bool fill_mesh_holes_f32(DualGridMeshF32 & mesh,
                         float max_hole_perimeter,
                         std::string * error) {
    if (!std::isfinite(max_hole_perimeter) || max_hole_perimeter < 0.0f) {
        set_error(error, "mesh hole perimeter limit must be finite and non-negative");
        return false;
    }
    if (!validate_mesh(mesh, error)) return false;
    if (mesh.faces.empty() || mesh.vertices.empty()) return true;

    const std::size_t face_count = mesh.faces.size() / 3;
    if (face_count > std::numeric_limits<std::size_t>::max() / 3) {
        set_error(error, "mesh edge buffer size overflows size_t");
        return false;
    }
    std::vector<EdgeRecord> edges;
    edges.reserve(face_count * 3);
    for (std::size_t face = 0; face < face_count; ++face) {
        const std::int32_t vertices[3] = {
            mesh.faces[face * 3 + 0], mesh.faces[face * 3 + 1],
            mesh.faces[face * 3 + 2]};
        for (int side = 0; side < 3; ++side) {
            const std::int32_t a = vertices[side];
            const std::int32_t b = vertices[(side + 1) % 3];
            edges.push_back({make_edge(a, b), a, b, vertices[(side + 2) % 3]});
        }
    }
    std::sort(edges.begin(), edges.end(), edge_record_less);

    std::vector<EdgeRecord> boundary_edges;
    boundary_edges.reserve(edges.size());
    for (std::size_t index = 0; index < edges.size();) {
        std::size_t end = index + 1;
        while (end < edges.size() && edges[end].edge == edges[index].edge) ++end;
        const std::size_t count = end - index;
        if (count > 2) {
            // CuMesh exposes no fillable manifold boundary loop for this
            // topology; preserve the raw mesh rather than inventing faces.
            return true;
        }
        if (count == 1) boundary_edges.push_back(edges[index]);
        index = end;
    }
    if (boundary_edges.empty()) return true;

    const std::size_t vertex_count = mesh.vertices.size() / 3;
    std::vector<std::int32_t> neighbor_a(vertex_count, -1);
    std::vector<std::int32_t> neighbor_b(vertex_count, -1);
    for (const EdgeRecord & record : boundary_edges) {
        const std::size_t a = static_cast<std::size_t>(record.edge.a);
        const std::size_t b = static_cast<std::size_t>(record.edge.b);
        if (neighbor_a[a] < 0) neighbor_a[a] = record.edge.b;
        else if (neighbor_b[a] < 0) neighbor_b[a] = record.edge.b;
        else return true;
        if (neighbor_a[b] < 0) neighbor_a[b] = record.edge.a;
        else if (neighbor_b[b] < 0) neighbor_b[b] = record.edge.a;
        else return true;
    }
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        if (neighbor_a[vertex] >= 0 && neighbor_b[vertex] < 0) return true;
    }

    std::vector<std::uint8_t> visited(boundary_edges.size(), 0);
    std::vector<std::array<std::int32_t, 3>> additions;
    std::size_t extra_vertices = 0;
    for (std::size_t seed_index = 0; seed_index < boundary_edges.size(); ++seed_index) {
        if (visited[seed_index]) continue;
        const EdgeRecord & seed = boundary_edges[seed_index];
        const std::int32_t start = seed.directed_a;
        std::int32_t previous = seed.directed_a;
        std::int32_t current = seed.directed_b;
        std::vector<std::int32_t> loop{start};
        visited[seed_index] = 1;
        while (current != start) {
            loop.push_back(current);
            const std::size_t current_index = static_cast<std::size_t>(current);
            const std::int32_t first_neighbor = neighbor_a[current_index];
            const std::int32_t second_neighbor = neighbor_b[current_index];
            const std::int32_t next = first_neighbor == previous
                ? second_neighbor : first_neighbor;
            const std::size_t edge_index = find_edge(boundary_edges, make_edge(current, next));
            if (edge_index == boundary_edges.size()) {
                set_error(error, "mesh boundary edge lookup failed");
                return false;
            }
            if (visited[edge_index]) {
                set_error(error, "mesh boundary loop traversal failed");
                return false;
            }
            visited[edge_index] = 1;
            previous = current;
            current = next;
            if (loop.size() > boundary_edges.size()) {
                set_error(error, "mesh boundary loop traversal failed");
                return false;
            }
        }
        if (loop.size() < 3) continue;

        double perimeter = 0.0;
        Vec3 center{};
        for (std::size_t index = 0; index < loop.size(); ++index) {
            const Vec3 a = vertex_at(mesh, loop[index]);
            const Vec3 b = vertex_at(mesh, loop[(index + 1) % loop.size()]);
            perimeter += std::sqrt(norm_squared(b - a));
            center = center + a;
        }
        if (!std::isfinite(perimeter) || !(perimeter < static_cast<double>(max_hole_perimeter))) {
            continue;
        }
        center = center / static_cast<double>(loop.size());
        if (!std::isfinite(center.x) || !std::isfinite(center.y) ||
            !std::isfinite(center.z)) continue;

        const std::size_t center_index = vertex_count + extra_vertices;
        if (center_index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            set_error(error, "mesh hole fill vertex index overflows int32");
            return false;
        }
        std::vector<std::array<std::int32_t, 3>> loop_faces;
        loop_faces.reserve(loop.size());
        bool valid_loop = true;
        for (std::size_t index = 0; index < loop.size(); ++index) {
            const std::int32_t a = loop[index];
            const std::int32_t b = loop[(index + 1) % loop.size()];
            const std::size_t edge_index = find_edge(boundary_edges, make_edge(a, b));
            if (edge_index == boundary_edges.size()) {
                valid_loop = false;
                break;
            }
            const EdgeRecord & boundary = boundary_edges[edge_index];
            // Close the hole with the reverse of the lone incident face's
            // directed boundary edge, as CuMesh does.
            std::array<std::int32_t, 3> triangle = {
                boundary.directed_b, boundary.directed_a,
                static_cast<std::int32_t>(center_index)};
            const Vec3 cap_normal = cross(
                vertex_at(mesh, triangle[1]) - vertex_at(mesh, triangle[0]),
                center - vertex_at(mesh, triangle[0]));
            if (norm_squared(cap_normal) <= 1.0e-30) {
                valid_loop = false;
                break;
            }
            loop_faces.push_back(triangle);
        }
        if (!valid_loop) continue;

        mesh.vertices.push_back(static_cast<float>(center.x));
        mesh.vertices.push_back(static_cast<float>(center.y));
        mesh.vertices.push_back(static_cast<float>(center.z));
        additions.insert(additions.end(), loop_faces.begin(), loop_faces.end());
        ++extra_vertices;
    }

    if (!additions.empty()) {
        mesh.faces.reserve(mesh.faces.size() + additions.size() * 3);
        for (const auto & triangle : additions) {
            mesh.faces.insert(mesh.faces.end(), triangle.begin(), triangle.end());
        }
    }
    return true;
}

} // namespace pixal3d
