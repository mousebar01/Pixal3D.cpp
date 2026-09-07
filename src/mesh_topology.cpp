#include "pixal3d/mesh_topology.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

struct RepairEdgeIncident {
    Edge edge;
    std::size_t face = 0;
    std::uint8_t corner_a = 0;
    std::uint8_t corner_b = 0;
};

bool repair_edge_less(const RepairEdgeIncident & left,
                      const RepairEdgeIncident & right) {
    if (left.edge < right.edge) return true;
    if (right.edge < left.edge) return false;
    return left.face < right.face;
}

class CornerDisjointSet {
public:
    explicit CornerDisjointSet(std::size_t count)
        : parent_(count), rank_(count, 0) {
        for (std::size_t index = 0; index < count; ++index) {
            parent_[index] = static_cast<std::uint32_t>(index);
        }
    }

    std::uint32_t find(std::uint32_t value) {
        std::uint32_t root = value;
        while (parent_[root] != root) root = parent_[root];
        while (parent_[value] != value) {
            const std::uint32_t next = parent_[value];
            parent_[value] = root;
            value = next;
        }
        return root;
    }

    void unite(std::uint32_t left, std::uint32_t right) {
        left = find(left);
        right = find(right);
        if (left == right) return;
        if (rank_[left] < rank_[right]) std::swap(left, right);
        parent_[right] = left;
        if (rank_[left] == rank_[right]) ++rank_[left];
    }

private:
    std::vector<std::uint32_t> parent_;
    std::vector<std::uint8_t> rank_;
};

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

std::size_t count_nonmanifold_edges(const DualGridMeshF32 & mesh) {
    const std::size_t face_count = mesh.faces.size() / 3;
    std::vector<Edge> edges;
    edges.reserve(face_count * 3);
    for (std::size_t face = 0; face < face_count; ++face) {
        const std::int32_t a = mesh.faces[face * 3 + 0];
        const std::int32_t b = mesh.faces[face * 3 + 1];
        const std::int32_t c = mesh.faces[face * 3 + 2];
        edges.push_back(make_edge(a, b));
        edges.push_back(make_edge(b, c));
        edges.push_back(make_edge(c, a));
    }
    std::sort(edges.begin(), edges.end());
    std::size_t count = 0;
    for (std::size_t begin = 0; begin < edges.size();) {
        std::size_t end = begin + 1;
        while (end < edges.size() && edges[end] == edges[begin]) ++end;
        if (end - begin > 2) ++count;
        begin = end;
    }
    return count;
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
        else {
            // A branching boundary vertex belongs to no fillable manifold
            // loop. Keep its existing two neighbors and skip only its branch.
            neighbor_b[a] = -2;
        }
        if (neighbor_a[b] < 0) neighbor_a[b] = record.edge.a;
        else if (neighbor_b[b] < 0) neighbor_b[b] = record.edge.a;
        else {
            neighbor_b[b] = -2;
        }
    }
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        if (neighbor_b[vertex] == -2) {
            neighbor_a[vertex] = -2;
        }
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
        bool valid_loop = start >= 0 && current >= 0 &&
                          neighbor_a[static_cast<std::size_t>(start)] >= 0 &&
                          neighbor_b[static_cast<std::size_t>(start)] >= 0 &&
                          neighbor_a[static_cast<std::size_t>(current)] >= 0 &&
                          neighbor_b[static_cast<std::size_t>(current)] >= 0;
        while (valid_loop && current != start) {
            loop.push_back(current);
            const std::size_t current_index = static_cast<std::size_t>(current);
            const std::int32_t first_neighbor = neighbor_a[current_index];
            const std::int32_t second_neighbor = neighbor_b[current_index];
            if (first_neighbor < 0 || second_neighbor < 0 ||
                (first_neighbor != previous && second_neighbor != previous)) {
                valid_loop = false;
                break;
            }
            const std::int32_t next = first_neighbor == previous
                ? second_neighbor : first_neighbor;
            if (next < 0) {
                valid_loop = false;
                break;
            }
            const std::size_t edge_index = find_edge(boundary_edges, make_edge(current, next));
            if (edge_index == boundary_edges.size() || visited[edge_index]) {
                valid_loop = false;
                break;
            }
            visited[edge_index] = 1;
            previous = current;
            current = next;
            if (loop.size() > boundary_edges.size()) {
                valid_loop = false;
                break;
            }
        }
        if (!valid_loop || loop.size() < 3 || current != start) continue;

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
        bool valid_faces = true;
        for (std::size_t index = 0; index < loop.size(); ++index) {
            const std::int32_t a = loop[index];
            const std::int32_t b = loop[(index + 1) % loop.size()];
            const std::size_t edge_index = find_edge(boundary_edges, make_edge(a, b));
            if (edge_index == boundary_edges.size()) {
                valid_faces = false;
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
                valid_faces = false;
                break;
            }
            loop_faces.push_back(triangle);
        }
        if (!valid_faces) continue;

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

bool repair_non_manifold_edges_f32(
    DualGridMeshF32 & mesh,
    MeshTopologyRepairReport * report,
    std::string * error) {
    MeshTopologyRepairReport local_report;
    if (report) *report = local_report;
    if (!validate_mesh(mesh, error)) return false;

    const std::size_t vertex_count = mesh.vertices.size() / 3;
    const std::size_t face_count = mesh.faces.size() / 3;
    local_report.input_vertices = vertex_count;
    local_report.output_vertices = vertex_count;
    local_report.face_count = face_count;
    local_report.nonmanifold_edges_before = count_nonmanifold_edges(mesh);
    local_report.nonmanifold_edges_after = local_report.nonmanifold_edges_before;
    if (local_report.nonmanifold_edges_before == 0 || face_count == 0) {
        if (report) *report = local_report;
        return true;
    }
    if (face_count > std::numeric_limits<std::size_t>::max() / 3) {
        set_error(error, "mesh corner count overflows size_t");
        return false;
    }

    const std::size_t corner_count = face_count * 3;
    std::vector<RepairEdgeIncident> edge_incidents;
    edge_incidents.reserve(corner_count);
    for (std::size_t face = 0; face < face_count; ++face) {
        for (std::uint8_t corner = 0; corner < 3; ++corner) {
            const std::uint8_t next = static_cast<std::uint8_t>((corner + 1) % 3);
            const std::int32_t a = mesh.faces[face * 3 + corner];
            const std::int32_t b = mesh.faces[face * 3 + next];
            edge_incidents.push_back({make_edge(a, b), face, corner, next});
        }
    }
    std::sort(edge_incidents.begin(), edge_incidents.end(), repair_edge_less);

    CornerDisjointSet disjoint_set(corner_count);
    auto corner_for_vertex = [&](const RepairEdgeIncident & incident,
                                 std::int32_t vertex) -> std::uint32_t {
        const std::size_t base = incident.face * 3;
        if (mesh.faces[base + incident.corner_a] == vertex) {
            return static_cast<std::uint32_t>(base + incident.corner_a);
        }
        return static_cast<std::uint32_t>(base + incident.corner_b);
    };
    for (std::size_t begin = 0; begin < edge_incidents.size();) {
        std::size_t end = begin + 1;
        while (end < edge_incidents.size() &&
               edge_incidents[end].edge == edge_incidents[begin].edge) {
            ++end;
        }
        if (end - begin == 2) {
            const RepairEdgeIncident & left = edge_incidents[begin];
            const RepairEdgeIncident & right = edge_incidents[begin + 1];
            disjoint_set.unite(
                corner_for_vertex(left, left.edge.a),
                corner_for_vertex(right, right.edge.a));
            disjoint_set.unite(
                corner_for_vertex(left, left.edge.b),
                corner_for_vertex(right, right.edge.b));
        }
        begin = end;
    }

    std::unordered_map<std::uint64_t, std::int32_t> fan_vertices;
    fan_vertices.reserve(corner_count);
    std::unordered_set<std::int32_t> original_fan_seen;
    original_fan_seen.reserve(vertex_count);
    std::vector<std::int32_t> corner_vertices(corner_count, -1);
    std::vector<float> repaired_vertices = mesh.vertices;
    for (std::size_t face = 0; face < face_count; ++face) {
        for (std::size_t corner = 0; corner < 3; ++corner) {
            const std::size_t corner_index = face * 3 + corner;
            const std::int32_t original = mesh.faces[corner_index];
            const std::uint32_t root = disjoint_set.find(
                static_cast<std::uint32_t>(corner_index));
            const std::uint64_t key =
                (static_cast<std::uint64_t>(static_cast<std::uint32_t>(original)) << 32) |
                static_cast<std::uint64_t>(root);
            const auto found = fan_vertices.find(key);
            if (found != fan_vertices.end()) {
                corner_vertices[corner_index] = found->second;
                continue;
            }

            std::int32_t replacement = original;
            if (!original_fan_seen.emplace(original).second) {
                replacement = static_cast<std::int32_t>(repaired_vertices.size() / 3);
                const std::size_t source = static_cast<std::size_t>(original) * 3;
                repaired_vertices.insert(repaired_vertices.end(), {
                    mesh.vertices[source + 0], mesh.vertices[source + 1],
                    mesh.vertices[source + 2]});
            }
            fan_vertices.emplace(key, replacement);
            corner_vertices[corner_index] = replacement;
        }
    }

    std::vector<std::int32_t> repaired_faces = mesh.faces;
    for (std::size_t index = 0; index < corner_count; ++index) {
        repaired_faces[index] = corner_vertices[index];
    }
    DualGridMeshF32 candidate;
    candidate.vertices = std::move(repaired_vertices);
    candidate.faces = std::move(repaired_faces);
    if (!validate_mesh(candidate, error)) return false;
    local_report.output_vertices = candidate.vertices.size() / 3;
    local_report.split_vertices = local_report.output_vertices - vertex_count;
    local_report.affected_faces = 0;
    for (std::size_t face = 0; face < face_count; ++face) {
        if (candidate.faces[face * 3 + 0] != mesh.faces[face * 3 + 0] ||
            candidate.faces[face * 3 + 1] != mesh.faces[face * 3 + 1] ||
            candidate.faces[face * 3 + 2] != mesh.faces[face * 3 + 2]) {
            ++local_report.affected_faces;
        }
    }
    local_report.nonmanifold_edges_after = count_nonmanifold_edges(candidate);
    if (local_report.nonmanifold_edges_after != 0) {
        if (report) *report = local_report;
        set_error(error, "non-manifold edge repair left " +
                           std::to_string(local_report.nonmanifold_edges_after) +
                           " unresolved edges");
        return false;
    }
    mesh = std::move(candidate);
    if (report) *report = local_report;
    return true;
}

} // namespace pixal3d
