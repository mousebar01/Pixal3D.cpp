#include "pixal3d/condition.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

void put_u32(std::ofstream & file, std::uint32_t value) {
    const char bytes[4] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
        static_cast<char>((value >> 16) & 0xffu),
        static_cast<char>((value >> 24) & 0xffu),
    };
    file.write(bytes, sizeof(bytes));
}

void put_f32(std::ofstream & file, float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    put_u32(file, bits);
}

void put_array(std::ofstream & file, const std::vector<float> & values) {
    for (float value : values) put_f32(file, value);
}

std::array<float, 16> front_transform(float distance) {
    return {{1.0f, 0.0f, 0.0f, 0.0f,
             0.0f, 0.0f, -1.0f, -distance,
             0.0f, 1.0f, 0.0f, 0.0f,
             0.0f, 0.0f, 0.0f, 1.0f}};
}

std::array<float, 16> translated_transform(float distance, float x) {
    auto result = front_transform(distance);
    result[3] = x;
    return result;
}

void write_camera(std::ofstream & file, const pixal3d::ProjectionCamera & camera) {
    put_f32(file, camera.camera_angle_x);
    put_f32(file, camera.distance);
    put_f32(file, camera.mesh_scale);
    put_u32(file, camera.has_transform ? 1 : 0);
    for (float value : camera.transform_matrix) put_f32(file, value);
}

void write_view(std::ofstream & file, const pixal3d::ProjectionCamera & camera,
                const std::vector<float> & global,
                const std::vector<float> & dino,
                const std::vector<float> & naf) {
    write_camera(file, camera);
    put_u32(file, 2); // tokens
    put_u32(file, 2); // global channels
    put_u32(file, 2); // dino + NAF
    put_array(file, global);
    put_u32(file, 0); // DINO role
    put_u32(file, 4); // image resolution
    put_u32(file, 2); // height
    put_u32(file, 2); // width
    put_u32(file, 1); // channels
    put_array(file, dino);
    put_u32(file, 1); // NAF role
    put_u32(file, 4);
    put_u32(file, 2);
    put_u32(file, 2);
    put_u32(file, 1);
    put_array(file, naf);
}

bool write_bundle(const std::string & path) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) return false;
    file.write("P3DMVCON", 8);
    put_u32(file, 1); // version
    put_u32(file, 1); // stage count
    put_u32(file, 2); // name length
    file.write("ss", 2);
    put_u32(file, 2); // view count
    pixal3d::ProjectionCamera first;
    first.camera_angle_x = 0.9f;
    first.distance = 2.0f;
    first.mesh_scale = 1.0f;
    first.has_transform = true;
    first.transform_matrix = front_transform(2.0f);
    pixal3d::ProjectionCamera second = first;
    second.transform_matrix = translated_transform(2.0f, 0.25f);
    write_view(file, first, {-1.0f, 1.0f, 0.0f, 2.0f},
               {0.0f, 1.0f, 2.0f, 3.0f}, {10.0f, 11.0f, 12.0f, 13.0f});
    write_view(file, second, {1.0f, 3.0f, 2.0f, 4.0f},
               {4.0f, 5.0f, 6.0f, 7.0f}, {14.0f, 15.0f, 16.0f, 17.0f});
    return static_cast<bool>(file);
}

bool close_enough(float left, float right) {
    return std::fabs(left - right) < 2e-6f;
}

} // namespace

int main() {
    const std::string path = "/tmp/pixal3d-multiview-condition-test.p3dmvcon";
    const std::string roundtrip_path = "/tmp/pixal3d-multiview-condition-roundtrip.p3dmvcon";
    if (!write_bundle(path)) return 1;

    pixal3d::Pixal3DMultiViewConditionBundleF32 bundle;
    std::string error;
    if (!pixal3d::load_pixal3d_multiview_condition_bundle(path, bundle, &error) ||
        bundle.format_version != 1 || bundle.stages.size() != 1) return 1;
    const auto * stage = bundle.find("ss");
    if (!stage || stage->views.size() != 2 || !stage->views[0].has_naf()) return 1;
    if (!pixal3d::save_pixal3d_multiview_condition_bundle(roundtrip_path, bundle, &error)) {
        std::remove(path.c_str());
        return 1;
    }
    pixal3d::Pixal3DMultiViewConditionBundleF32 roundtrip;
    if (!pixal3d::load_pixal3d_multiview_condition_bundle(
            roundtrip_path, roundtrip, &error) || roundtrip.format_version != 1 ||
        roundtrip.stages.size() != 1 || roundtrip.stages[0].views.size() != 2) {
        std::remove(path.c_str());
        std::remove(roundtrip_path.c_str());
        return 1;
    }
    for (std::size_t index = 0; index < stage->views.size(); ++index) {
        const auto & expected = stage->views[index];
        const auto & actual = roundtrip.stages[0].views[index];
        if (!close_enough(actual.camera.camera_angle_x, expected.camera.camera_angle_x) ||
            !close_enough(actual.camera.distance, expected.camera.distance) ||
            !close_enough(actual.camera.mesh_scale, expected.camera.mesh_scale) ||
            actual.camera.has_transform != expected.camera.has_transform ||
            actual.camera.transform_matrix != expected.camera.transform_matrix ||
            actual.global.offsets != expected.global.offsets ||
            actual.global.feats != expected.global.feats ||
            actual.dino_map.image_resolution != expected.dino_map.image_resolution ||
            actual.dino_map.height != expected.dino_map.height ||
            actual.dino_map.width != expected.dino_map.width ||
            actual.dino_map.channels != expected.dino_map.channels ||
            actual.dino_map.features != expected.dino_map.features ||
            actual.naf_map.features != expected.naf_map.features) {
            std::remove(path.c_str());
            std::remove(roundtrip_path.c_str());
            return 1;
        }
    }

    const std::vector<std::int32_t> coords = {
        0, 0, 0, 0,
        0, 1, 1, 1,
    };
    pixal3d::VarLenTensorF32 global;
    pixal3d::SparseTensorF32 projection;
    if (!pixal3d::project_condition_stage_multiview_average_f32(
            *stage, 2, coords, global, projection, &error)) return 1;
    if (global.channels != 2 || global.tokens() != 2 || projection.channels != 2 ||
        projection.coords != coords) return 1;
    if (!close_enough(global.feats[0], 0.0f) ||
        !close_enough(global.feats[1], 2.0f) ||
        !close_enough(global.feats[2], 1.0f) ||
        !close_enough(global.feats[3], 3.0f)) return 1;

    std::vector<std::array<float, 16>> transforms;
    std::vector<float> distances;
    for (const auto & view : stage->views) {
        transforms.push_back(view.camera.transform_matrix);
        distances.push_back(view.camera.distance);
    }
    std::vector<std::array<float, 16>> relative;
    if (!pixal3d::compute_relative_calc_matrices(
            transforms, distances, relative, &error)) return 1;
    std::vector<float> first_dino;
    std::vector<std::uint8_t> valid;
    auto first_camera = stage->views[0].camera;
    first_camera.transform_matrix = relative[0];
    if (!pixal3d::project_grid_features_at_coords(
            stage->views[0].dino_map.features.data(), 2, 2, 1,
            pixal3d::ProjectionGridOptions{2, 4}, first_camera, coords,
            first_dino, valid, &error)) return 1;
    std::vector<float> second_dino;
    auto second_camera = stage->views[1].camera;
    second_camera.transform_matrix = relative[1];
    if (!pixal3d::project_grid_features_at_coords(
            stage->views[1].dino_map.features.data(), 2, 2, 1,
            pixal3d::ProjectionGridOptions{2, 4}, second_camera, coords,
            second_dino, valid, &error)) return 1;
    for (std::size_t point = 0; point < 2; ++point) {
        const float expected = 0.5f * (first_dino[point] + second_dino[point]);
        if (!close_enough(projection.feats[point * 2], expected)) return 1;
    }

    {
        std::ofstream append(path, std::ios::binary | std::ios::app);
        const char trailing = 1;
        append.write(&trailing, 1);
    }
    if (pixal3d::load_pixal3d_multiview_condition_bundle(path, bundle, &error)) {
        std::remove(path.c_str());
        std::remove(roundtrip_path.c_str());
        return 1;
    }
    std::remove(path.c_str());
    std::remove(roundtrip_path.c_str());
    return 0;
}
