#include "pixal3d/condition.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void emit(const char * name, const std::vector<float> & values) {
    std::cout << name << " " << values.size();
    for (float value : values) std::cout << " " << value;
    std::cout << "\n";
}

std::array<float, 16> front_transform(float distance) {
    return {{1.0f, 0.0f, 0.0f, 0.0f,
             0.0f, 0.0f, -1.0f, -distance,
             0.0f, 1.0f, 0.0f, 0.0f,
             0.0f, 0.0f, 0.0f, 1.0f}};
}

} // namespace

int main() {
    pixal3d::Pixal3DConditionStageMVF32 stage;
    stage.name = "ss";
    for (int view_index = 0; view_index < 2; ++view_index) {
        pixal3d::Pixal3DConditionViewF32 view;
        view.camera.camera_angle_x = 0.9f;
        view.camera.distance = 2.0f;
        view.camera.mesh_scale = 1.0f;
        view.camera.has_transform = true;
        view.camera.transform_matrix = front_transform(2.0f);
        if (view_index == 1) view.camera.transform_matrix[3] = 0.25f;
        view.global.batch_size = 1;
        view.global.channels = 2;
        view.global.offsets = {0, 2};
        view.global.feats = {
            static_cast<float>(view_index), 1.0f + static_cast<float>(view_index),
            2.0f + static_cast<float>(view_index), 3.0f + static_cast<float>(view_index)};
        view.dino_map.image_resolution = 4;
        view.dino_map.height = 2;
        view.dino_map.width = 2;
        view.dino_map.channels = 1;
        view.dino_map.features = {
            0.0f + 4.0f * view_index, 1.0f + 4.0f * view_index,
            2.0f + 4.0f * view_index, 3.0f + 4.0f * view_index};
        view.naf_map.image_resolution = 4;
        view.naf_map.height = 2;
        view.naf_map.width = 2;
        view.naf_map.channels = 1;
        view.naf_map.features = {
            10.0f + 4.0f * view_index, 11.0f + 4.0f * view_index,
            12.0f + 4.0f * view_index, 13.0f + 4.0f * view_index};
        stage.views.push_back(std::move(view));
    }

    const std::vector<std::int32_t> coords = {
        0, 0, 0, 0,
        0, 1, 1, 1,
    };
    pixal3d::VarLenTensorF32 global;
    pixal3d::SparseTensorF32 projection;
    std::string error;
    if (!pixal3d::project_condition_stage_multiview_average_f32(
            stage, 2, coords, global, projection, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    emit("mv_global", global.feats);
    emit("mv_projection", projection.feats);
    return 0;
}
