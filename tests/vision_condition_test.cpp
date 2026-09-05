#include "pixal3d/vision_condition.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool close_enough(float lhs, float rhs) {
    return std::fabs(lhs - rhs) <= 1e-6f;
}

} // namespace

int main() {
    pixal3d::DinoV3FeaturesF32 dino;
    dino.image_height = 8;
    dino.image_width = 8;
    dino.patch_height = 2;
    dino.patch_width = 2;
    dino.channels = 2;
    dino.num_register_tokens = 1;
    dino.global.batch_size = 1;
    dino.global.channels = 2;
    dino.global.offsets = {0, 2};
    dino.global.feats = {1.0f, 2.0f, 3.0f, 4.0f};
    dino.patch_map.image_resolution = 8;
    dino.patch_map.height = 2;
    dino.patch_map.width = 2;
    dino.patch_map.channels = 2;
    dino.patch_map.features = {0.0f, 1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f, 7.0f};
    pixal3d::NafOutputF32 naf;
    naf.height = 4;
    naf.width = 4;
    naf.channels = 2;
    naf.features.resize(naf.height * naf.width * naf.channels);
    for (std::size_t i = 0; i < naf.features.size(); ++i) {
        naf.features[i] = 10.0f + static_cast<float>(i);
    }

    pixal3d::Pixal3DConditionStageF32 stage;
    std::string error;
    if (!pixal3d::make_pixal3d_condition_stage_f32(
            dino, &naf, "shape_512", 8, stage, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    if (!stage.valid(&error) || !stage.has_naf() ||
        stage.naf_map.image_resolution != 8 || stage.naf_map.channels != 2 ||
        !close_enough(stage.naf_map.features[7], 17.0f)) {
        std::cerr << "native vision stage assembly mismatch\n";
        return 1;
    }
    pixal3d::Pixal3DConditionStageF32 no_naf_stage;
    if (!pixal3d::make_pixal3d_condition_stage_f32(
            dino, nullptr, "ss", 8, no_naf_stage, &error) ||
        !no_naf_stage.valid(&error) || no_naf_stage.has_naf()) {
        std::cerr << "native DINO-only stage assembly mismatch\n";
        return 1;
    }

    const auto camera = pixal3d::ProjectionCamera::front(0.2f, 2.0f, 1.0f);
    const std::vector<std::int32_t> coords = {0, 0, 0, 0, 0, 1, 1, 1};
    pixal3d::Pixal3DImageConditionF32 condition;
    if (!pixal3d::make_pixal3d_image_condition_f32(
            stage, camera, 2, coords, condition, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    if (!condition.global.valid(&error) || !condition.projection.valid(&error) ||
        condition.global.channels != 2 || condition.global.tokens() != 2 ||
        condition.projection.points() != 2 || condition.projection.channels != 4) {
        std::cerr << "native vision projection shape mismatch\n";
        return 1;
    }

    pixal3d::DinoV3FeaturesF32 dino_second = dino;
    for (float & value : dino_second.global.feats) value += 0.5f;
    for (float & value : dino_second.patch_map.features) value += 0.5f;
    pixal3d::NafOutputF32 naf_second = naf;
    for (float & value : naf_second.features) value += 1.0f;
    pixal3d::ProjectionCamera camera_first = camera;
    camera_first.has_transform = true;
    camera_first.transform_matrix = {{1.0f, 0.0f, 0.0f, 0.0f,
                                      0.0f, 1.0f, 0.0f, 0.0f,
                                      0.0f, 0.0f, 1.0f, 0.0f,
                                      0.0f, 0.0f, 0.0f, 1.0f}};
    pixal3d::ProjectionCamera camera_second = camera_first;
    camera_second.transform_matrix[3] = 0.1f;
    pixal3d::Pixal3DConditionStageMVF32 multiview_stage;
    const std::vector<pixal3d::DinoV3FeaturesF32> dino_views = {dino, dino_second};
    const std::vector<const pixal3d::NafOutputF32 *> naf_views = {&naf, &naf_second};
    const std::vector<pixal3d::ProjectionCamera> cameras = {camera_first, camera_second};
    if (!pixal3d::make_pixal3d_multiview_condition_stage_f32(
            dino_views, naf_views, cameras, "shape_512", 8, multiview_stage, &error) ||
        !multiview_stage.valid(&error) || multiview_stage.views.size() != 2 ||
        !multiview_stage.views[0].has_naf() || !multiview_stage.views[1].has_naf()) {
        std::cerr << "native multi-view stage assembly mismatch\n";
        return 1;
    }
    pixal3d::VarLenTensorF32 multiview_global;
    pixal3d::SparseTensorF32 multiview_projection;
    if (!pixal3d::project_condition_stage_multiview_average_f32(
            multiview_stage, 2, coords, multiview_global, multiview_projection, &error) ||
        multiview_global.feats.size() != dino.global.feats.size() ||
        !close_enough(multiview_global.feats[0], 1.25f) ||
        multiview_projection.channels != 4 || multiview_projection.points() != 2) {
        std::cerr << "native multi-view projection mismatch\n";
        return 1;
    }
    pixal3d::VarLenTensorF32 reference_global;
    pixal3d::SparseTensorF32 reference_projection;
    if (!pixal3d::project_condition_stage_f32(
            stage, camera, 2, coords, reference_global, reference_projection, &error) ||
        condition.global.feats != reference_global.feats ||
        condition.projection.coords != reference_projection.coords ||
        condition.projection.feats != reference_projection.feats) {
        std::cerr << "native vision projection adapter mismatch\n";
        return 1;
    }
    return 0;
}
