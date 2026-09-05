#include "pixal3d/condition.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
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

void put_f32(std::ofstream & file, const std::vector<float> & values) {
    file.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
}

bool close_enough(float left, float right) {
    return std::fabs(left - right) < 1e-6f;
}

bool write_bundle(const std::string & path) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) return false;
    file.write("P3DCOND\0", 8);
    put_u32(file, 1); // version
    put_u32(file, 1); // one stage
    put_u32(file, 2); // stage name length
    file.write("ss", 2);
    put_u32(file, 2); // global tokens
    put_u32(file, 3); // global channels
    put_u32(file, 2); // dino + NAF maps
    put_f32(file, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f});

    put_u32(file, 0); // dino role
    put_u32(file, 4); // image resolution
    put_u32(file, 2); // height
    put_u32(file, 2); // width
    put_u32(file, 2); // channels
    put_f32(file, {0.0f, 1.0f, 2.0f, 3.0f,
                   4.0f, 5.0f, 6.0f, 7.0f});

    put_u32(file, 1); // NAF role
    put_u32(file, 4); // image resolution
    put_u32(file, 2); // height
    put_u32(file, 2); // width
    put_u32(file, 1); // channels
    put_f32(file, {10.0f, 11.0f, 12.0f, 13.0f});
    return static_cast<bool>(file);
}

} // namespace

int main() {
    const std::string path = "/tmp/pixal3d-condition-test.p3dcond";
    const std::string roundtrip_path = "/tmp/pixal3d-condition-roundtrip.p3dcond";
    if (!write_bundle(path)) return 1;

    pixal3d::Pixal3DConditionBundleF32 bundle;
    std::string error;
    if (!pixal3d::load_pixal3d_condition_bundle(path, bundle, &error) ||
        bundle.format_version != 1 || bundle.stages.size() != 1) return 1;
    const auto * stage = bundle.find("ss");
    if (!stage || stage->global.tokens() != 2 || stage->global.channels != 3 ||
        stage->dino_map.channels != 2 || !stage->has_naf()) return 1;

    const std::vector<std::int32_t> coords = {0, 0, 0, 0, 0, 1, 1, 1};
    pixal3d::VarLenTensorF32 global;
    pixal3d::SparseTensorF32 projection;
    const auto camera = pixal3d::ProjectionCamera::front(0.9f, 2.0f);
    if (!pixal3d::project_condition_stage_f32(
            *stage, camera, 2, coords, global, projection, &error)) return 1;
    if (global.feats != stage->global.feats || projection.points() != 2 ||
        projection.channels != 3 || projection.coords != coords) return 1;

    pixal3d::ProjectionGridOptions options;
    options.grid_resolution = 2;
    options.image_resolution = 4;
    std::vector<float> dino;
    std::vector<float> naf;
    std::vector<std::uint8_t> valid;
    if (!pixal3d::project_grid_features_at_coords(
            stage->dino_map.features.data(), 2, 2, 2, options, camera, coords,
            dino, valid, &error) ||
        !pixal3d::project_grid_features_at_coords(
            stage->naf_map.features.data(), 2, 2, 1, options, camera, coords,
            naf, valid, &error)) return 1;
    for (std::size_t point = 0; point < 2; ++point) {
        for (int channel = 0; channel < 2; ++channel) {
            if (!close_enough(projection.feats[point * 3 + channel],
                              dino[point * 2 + channel])) return 1;
        }
        if (!close_enough(projection.feats[point * 3 + 2], naf[point])) return 1;
    }
    if (!pixal3d::save_pixal3d_condition_bundle(roundtrip_path, bundle, &error)) {
        std::remove(path.c_str());
        return 1;
    }
    pixal3d::Pixal3DConditionBundleF32 roundtrip;
    if (!pixal3d::load_pixal3d_condition_bundle(roundtrip_path, roundtrip, &error) ||
        roundtrip.format_version != bundle.format_version ||
        roundtrip.stages.size() != bundle.stages.size()) {
        std::remove(path.c_str());
        std::remove(roundtrip_path.c_str());
        return 1;
    }
    const auto * roundtrip_stage = roundtrip.find("ss");
    if (!roundtrip_stage || roundtrip_stage->global.batch_size != stage->global.batch_size ||
        roundtrip_stage->global.channels != stage->global.channels ||
        roundtrip_stage->global.offsets != stage->global.offsets ||
        roundtrip_stage->global.feats != stage->global.feats ||
        roundtrip_stage->dino_map.image_resolution != stage->dino_map.image_resolution ||
        roundtrip_stage->dino_map.height != stage->dino_map.height ||
        roundtrip_stage->dino_map.width != stage->dino_map.width ||
        roundtrip_stage->dino_map.channels != stage->dino_map.channels ||
        roundtrip_stage->dino_map.features != stage->dino_map.features ||
        roundtrip_stage->naf_map.image_resolution != stage->naf_map.image_resolution ||
        roundtrip_stage->naf_map.height != stage->naf_map.height ||
        roundtrip_stage->naf_map.width != stage->naf_map.width ||
        roundtrip_stage->naf_map.channels != stage->naf_map.channels ||
        roundtrip_stage->naf_map.features != stage->naf_map.features) {
        std::remove(path.c_str());
        std::remove(roundtrip_path.c_str());
        return 1;
    }
    {
        std::ofstream append(path, std::ios::binary | std::ios::app);
        const char trailing = 1;
        append.write(&trailing, 1);
    }
    if (pixal3d::load_pixal3d_condition_bundle(path, bundle, &error)) {
        std::remove(path.c_str());
        std::remove(roundtrip_path.c_str());
        return 1;
    }
    std::remove(path.c_str());
    std::remove(roundtrip_path.c_str());
    return 0;
}
