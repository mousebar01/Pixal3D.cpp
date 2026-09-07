#include "pixal3d/vision_condition.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace pixal3d {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

} // namespace

bool make_pixal3d_condition_stage_f32(
    const DinoV3FeaturesF32 & dino,
    const NafOutputF32 * naf,
    const std::string & name,
    int image_resolution,
    Pixal3DConditionStageF32 & output,
    std::string * error) {
    output = Pixal3DConditionStageF32{};
    if (!dino.valid(error)) return false;
    if (name.empty() || image_resolution <= 0 ||
        dino.image_height != image_resolution || dino.image_width != image_resolution ||
        dino.patch_map.image_resolution != image_resolution) {
        set_error(error, "native vision stage image resolution does not match DINO output");
        return false;
    }
    if (naf && !naf->valid(error)) return false;
    if (naf && naf->channels != dino.channels) {
        set_error(error, "NAF output channels must match DINO channels");
        return false;
    }

    output.name = name;
    output.global = dino.global;
    output.dino_map = dino.patch_map;
    output.dino_map.image_resolution = image_resolution;
    if (naf) {
        output.naf_map.image_resolution = image_resolution;
        output.naf_map.height = naf->height;
        output.naf_map.width = naf->width;
        output.naf_map.channels = naf->channels;
        output.naf_map.features = naf->features;
    }
    return output.valid(error);
}

bool encode_pixal3d_condition_stage_f32(
    const DinoV3Model & dino_model,
    const NafModel * naf_model,
    const float * image_chw_01,
    int image_height,
    int image_width,
    int image_resolution,
    int naf_output_resolution,
    const std::string & name,
    Pixal3DConditionStageF32 & output,
    std::string * error) {
    output = Pixal3DConditionStageF32{};
    if (!image_chw_01 || image_height <= 0 || image_width <= 0 ||
        image_resolution <= 0 || image_height != image_resolution ||
        image_width != image_resolution || naf_output_resolution < 0 ||
        name.empty()) {
        set_error(error, "native vision encoder expects a square image at image_resolution");
        return false;
    }
    const auto & dino_hp = dino_model.hparams();
    if (dino_hp.num_channels != 3) {
        set_error(error, "native vision encoder currently expects a 3-channel DINO model");
        return false;
    }
    if (naf_output_resolution > 0 && !naf_model) {
        set_error(error, "NAF output resolution was requested without a NAF model");
        return false;
    }
    std::size_t image_count = static_cast<std::size_t>(image_height);
    if (static_cast<std::size_t>(image_width) != 0 &&
        image_count > std::numeric_limits<std::size_t>::max() /
                          static_cast<std::size_t>(image_width)) {
        set_error(error, "native vision image size overflows size_t");
        return false;
    }
    image_count *= static_cast<std::size_t>(image_width);
    if (image_count > std::numeric_limits<std::size_t>::max() / 3u) {
        set_error(error, "native vision image channel size overflows size_t");
        return false;
    }
    image_count *= 3u;
    std::vector<float> normalized(image_count);
    constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
    const std::size_t plane = static_cast<std::size_t>(image_height) * image_width;
    for (int channel = 0; channel < 3; ++channel) {
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            const std::size_t index = static_cast<std::size_t>(channel) * plane + pixel;
            const float value = image_chw_01[index];
            if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
                set_error(error, "native vision image must contain finite [0,1] values");
                return false;
            }
            normalized[index] = (value - kMean[channel]) / kStd[channel];
        }
    }

    DinoV3FeaturesF32 dino;
    if (!dino_model.encode(normalized.data(), image_height, image_width, dino, error)) {
        return false;
    }
    NafOutputF32 naf;
    const NafOutputF32 * naf_ptr = nullptr;
    if (naf_output_resolution > 0) {
        const auto & naf_hp = naf_model->hparams();
        if (naf_hp.in_channels != 3) {
            set_error(error, "native vision encoder currently expects a 3-channel NAF model");
            return false;
        }
        if (!naf_model->upsample(
                image_chw_01, image_height, image_width,
                dino.patch_map.features.data(), dino.patch_map.height,
                dino.patch_map.width, dino.patch_map.channels,
                naf_output_resolution, naf_output_resolution, naf, error)) {
            return false;
        }
        naf_ptr = &naf;
    }
    return make_pixal3d_condition_stage_f32(
        dino, naf_ptr, name, image_resolution, output, error);
}

bool encode_pixal3d_condition_bundle_from_image_f32(
    const DinoV3Model & dino_model,
    const NafModel * naf_model,
    const Pixal3DImageF32 & image,
    const Pixal3DImageConditionBundleConfig & config,
    Pixal3DConditionBundleF32 & output,
    std::string * error) {
    output = Pixal3DConditionBundleF32{};
    if (!image.valid(error)) return false;
    Pixal3DImageF32 preprocessed_image;
    if (!preprocess_pixal3d_image_f32(image, preprocessed_image, error)) return false;
    const struct StageSpec {
        const char * name;
        int image_resolution;
        int naf_resolution;
    } specs[] = {
        {"ss", config.ss_resolution, config.ss_naf_resolution},
        {"shape_512", config.shape_512_resolution,
         config.shape_512_naf_resolution},
        {"shape_1024", config.shape_1024_resolution,
         config.shape_1024_naf_resolution},
        {"tex_1024", config.tex_1024_resolution,
         config.tex_1024_naf_resolution},
    };
    std::vector<Pixal3DImageF32> stage_images;
    std::vector<Pixal3DVisionStageInputF32> inputs;
    stage_images.reserve(4);
    inputs.reserve(4);
    for (const StageSpec & spec : specs) {
        if (spec.image_resolution <= 0 || spec.naf_resolution < 0) {
            set_error(error, std::string("invalid image-condition resolution for stage: ") +
                              spec.name);
            return false;
        }
        if (spec.naf_resolution > 0 && !naf_model) {
            set_error(error, std::string("stage requests NAF without a loaded NAF model: ") +
                              spec.name);
            return false;
        }
        stage_images.emplace_back();
        if (!resize_pixal3d_image_f32(
                preprocessed_image, spec.image_resolution, spec.image_resolution,
                stage_images.back(), error)) {
            return false;
        }
        Pixal3DVisionStageInputF32 input;
        input.name = spec.name;
        input.image_resolution = spec.image_resolution;
        input.naf_output_resolution = spec.naf_resolution;
        input.image_chw_01 = stage_images.back().pixels.data();
        input.image_elements = stage_images.back().pixels.size();
        inputs.push_back(std::move(input));
    }
    return encode_pixal3d_condition_bundle_f32(
        dino_model, naf_model, inputs, output, error);
}

bool encode_pixal3d_condition_bundle_f32(
    const DinoV3Model & dino_model,
    const NafModel * naf_model,
    const std::vector<Pixal3DVisionStageInputF32> & inputs,
    Pixal3DConditionBundleF32 & output,
    std::string * error) {
    output = Pixal3DConditionBundleF32{};
    if (inputs.empty() || inputs.size() > 4) {
        set_error(error, "native vision condition bundle needs between one and four stages");
        return false;
    }
    output.format_version = 1;
    output.stages.reserve(inputs.size());
    for (const Pixal3DVisionStageInputF32 & input : inputs) {
        if (input.name != "ss" && input.name != "shape_512" &&
            input.name != "shape_1024" && input.name != "tex_1024") {
            set_error(error, "unsupported native vision condition stage: " + input.name);
            output = Pixal3DConditionBundleF32{};
            return false;
        }
        if (input.image_resolution <= 0 || input.naf_output_resolution < 0) {
            set_error(error, "native vision condition stage has invalid resolution: " +
                              input.name);
            output = Pixal3DConditionBundleF32{};
            return false;
        }
        std::size_t pixels = static_cast<std::size_t>(input.image_resolution);
        if (pixels > std::numeric_limits<std::size_t>::max() / pixels) {
            set_error(error, "native vision condition image size overflows size_t: " +
                              input.name);
            output = Pixal3DConditionBundleF32{};
            return false;
        }
        pixels *= static_cast<std::size_t>(input.image_resolution);
        if (pixels > std::numeric_limits<std::size_t>::max() / 3u ||
            input.image_elements != pixels * 3u) {
            set_error(error, "native vision condition image payload size mismatch: " +
                              input.name);
            output = Pixal3DConditionBundleF32{};
            return false;
        }
        Pixal3DConditionStageF32 stage;
        if (!encode_pixal3d_condition_stage_f32(
                dino_model, naf_model, input.image_chw_01,
                input.image_resolution, input.image_resolution,
                input.image_resolution, input.naf_output_resolution,
                input.name, stage, error)) {
            output = Pixal3DConditionBundleF32{};
            return false;
        }
        output.stages.push_back(std::move(stage));
    }
    if (!output.valid(error)) {
        output = Pixal3DConditionBundleF32{};
        return false;
    }
    return true;
}

bool make_pixal3d_multiview_condition_stage_f32(
    const std::vector<DinoV3FeaturesF32> & dino_views,
    const std::vector<const NafOutputF32 *> & naf_views,
    const std::vector<ProjectionCamera> & cameras,
    const std::string & name,
    int image_resolution,
    Pixal3DConditionStageMVF32 & output,
    std::string * error) {
    output = Pixal3DConditionStageMVF32{};
    if (dino_views.empty() || dino_views.size() > 64 || cameras.size() != dino_views.size() ||
        (image_resolution <= 0) || name.empty()) {
        set_error(error, "native multi-view vision inputs have inconsistent sizes");
        return false;
    }
    const bool use_naf = !naf_views.empty();
    if (use_naf && naf_views.size() != dino_views.size()) {
        set_error(error, "native multi-view NAF inputs must match the DINO view count");
        return false;
    }
    output.name = name;
    output.views.reserve(dino_views.size());
    for (std::size_t index = 0; index < dino_views.size(); ++index) {
        if (use_naf && !naf_views[index]) {
            set_error(error, "native multi-view NAF inputs cannot contain null views");
            output = Pixal3DConditionStageMVF32{};
            return false;
        }
        Pixal3DConditionStageF32 single;
        if (!make_pixal3d_condition_stage_f32(
                dino_views[index], use_naf ? naf_views[index] : nullptr,
                name, image_resolution, single, error)) {
            output = Pixal3DConditionStageMVF32{};
            return false;
        }
        Pixal3DConditionViewF32 view;
        view.camera = cameras[index];
        view.global = std::move(single.global);
        view.dino_map = std::move(single.dino_map);
        view.naf_map = std::move(single.naf_map);
        if (!view.valid(error)) {
            output = Pixal3DConditionStageMVF32{};
            return false;
        }
        output.views.push_back(std::move(view));
    }
    if (!output.valid(error)) {
        output = Pixal3DConditionStageMVF32{};
        return false;
    }
    return true;
}

bool make_pixal3d_image_condition_f32(
    const Pixal3DConditionStageF32 & stage,
    const ProjectionCamera & camera,
    int grid_resolution,
    const std::vector<std::int32_t> & coords,
    Pixal3DImageConditionF32 & output,
    std::string * error) {
    output = Pixal3DImageConditionF32{};
    VarLenTensorF32 global;
    SparseTensorF32 projection;
    if (!project_condition_stage_f32(stage, camera, grid_resolution, coords,
                                     global, projection, error)) {
        return false;
    }
    output.global = std::move(global);
    output.projection = std::move(projection);
    if (!output.global.valid(error) || !output.projection.valid(error)) {
        output = Pixal3DImageConditionF32{};
        return false;
    }
    return true;
}

} // namespace pixal3d
