#include "pixal3d/scaffold.h"
#include "pixal3d/condition.h"
#include "pixal3d/dino.h"
#include "pixal3d/dino_vit.h"
#include "pixal3d/inference.h"
#include "pixal3d/image.h"
#include "pixal3d/naf.h"
#include "pixal3d/slat_decoder.h"
#include "pixal3d/slat_flow.h"
#include "pixal3d/ss_decoder.h"
#include "pixal3d/ss_flow.h"
#include "pixal3d/vision_condition.h"

#include <cstdint>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

namespace {

void print_usage(const char * program, std::ostream & out) {
    out << "Pixal3D C++/ggml migration scaffold\n\n"
        << "Usage:\n"
        << "  " << program << " --help\n"
        << "  " << program << " --version\n"
        << "  " << program << " inspect-pack <pixal3d-pack.gguf>\n\n"

        << "  " << program << " estimate-model <shared.gguf> <flow.gguf>\n\n"

        << "  " << program << " run-cascade <shared.gguf> <flow.gguf>"
        << " <conditions.p3dcond> <output.obj> [options]\n\n"

        << "  " << program << " run-image <shared.gguf> <flow.gguf> <dino.gguf>"
        << " <naf.gguf> <input-image> <output.obj> [options]\n\n"
        << "      run-image options include --vision-resolution N (smoke test)\n\n"
        << "  " << program << " run-cascade-mv <shared.gguf> <mv-flow.gguf>"
        << " <conditions.p3dmvcon> <output.obj> [options]\n\n"
        << "  " << program << " inspect-dinodata <condition.dinodata>\n\n"
        << "  " << program << " inspect-dino-vit <dino.gguf>\n\n"

        << "  " << program << " inspect-naf <naf.gguf>\n\n"
        << "  " << program << " inspect-condition <conditions.p3dcond>\n\n"
        << "  " << program << " inspect-mv-condition <conditions.p3dmvcon>\n\n"

        << "  " << program << " encode-condition-stage <dino.gguf> <naf.gguf|->"
        << " <input-image> <stage> <output.p3dcond> [options]\n\n"
        << "  " << program << " encode-condition-bundle <dino.gguf> <naf.gguf>"
        << " <input-image> <output.p3dcond> [options]\n\n"
        << "  " << program << " inspect-ss-flow <pixal3d-pack.gguf>\n\n"
        << "  " << program << " inspect-ss-decoder <pixal3d-pack.gguf>\n\n"
        << "  " << program << " inspect-slat-decoder <pixal3d-pack.gguf>"
        << " <shape_decoder|texture_decoder>\n\n"
        << "  " << program << " inspect-slat-flow <pixal3d-pack.gguf>"
        << " <shape_flow_512|shape_flow_1024|texture_flow_1024>\n\n"
        << "The inspect command validates the Pixal3D/ggml pack boundary.\n";
}

bool parse_u64(const char * text, std::uint64_t & value) {
    try {
        std::size_t consumed = 0;
        const std::string input(text);
        value = std::stoull(input, &consumed, 10);
        return consumed == input.size();
    } catch (...) {
        return false;
    }
}

bool parse_int(const char * text, int & value) {
    try {
        std::size_t consumed = 0;
        const std::string input(text);
        const long parsed = std::stol(input, &consumed, 10);
        if (consumed != input.size() || parsed < std::numeric_limits<int>::min() ||
            parsed > std::numeric_limits<int>::max()) return false;
        value = static_cast<int>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_float(const char * text, float & value) {
    try {
        std::size_t consumed = 0;
        const std::string input(text);
        value = std::stof(input, &consumed);
        return consumed == input.size() && std::isfinite(value);
    } catch (...) {
        return false;
    }
}

bool parse_cascade_options(int argc, char ** argv, int first,
                           pixal3d::Pixal3DInferenceConfig & config,
                           std::string * error,
                           pixal3d::Pixal3DImageConditionBundleConfig * image_config = nullptr) {
    for (int index = first; index < argc;) {
        const std::string option = argv[index++];
        if (option == "--vision-resolution" && index < argc && image_config) {
            int resolution = 0;
            if (!parse_int(argv[index++], resolution) || resolution < 16 ||
                resolution % 16 != 0) {
                if (error) *error = "--vision-resolution must be a multiple of 16 >= 16";
                return false;
            }
            // Development-only override: keep every stage at one small DINO
            // input size so an end-to-end smoke test does not allocate the
            // released 512/1024px vision graphs.  The production defaults in
            // Pixal3DImageConditionBundleConfig remain unchanged.
            image_config->ss_resolution = resolution;
            image_config->shape_512_resolution = resolution;
            image_config->shape_1024_resolution = resolution;
            image_config->tex_1024_resolution = resolution;
            image_config->ss_naf_resolution = 0;
            image_config->shape_512_naf_resolution = resolution;
            image_config->shape_1024_naf_resolution = resolution;
            image_config->tex_1024_naf_resolution = resolution;
        } else if (option == "--seed" && index < argc) {
            if (!parse_u64(argv[index++], config.seed)) {
                if (error) *error = "invalid --seed";
                return false;
            }
        } else if (option == "--resolution" && index < argc) {
            if (!parse_int(argv[index++], config.cascade.requested_resolution) ||
                config.cascade.requested_resolution < 1024 ||
                config.cascade.requested_resolution % 16 != 0) {
                if (error) *error = "--resolution must be a multiple of 16 >= 1024";
                return false;
            }
        } else if (option == "--max-tokens" && index < argc) {
            std::uint64_t tokens = 0;
            if (!parse_u64(argv[index++], tokens) ||
                tokens > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
                if (error) *error = "invalid --max-tokens";
                return false;
            }
            config.cascade.max_num_tokens = static_cast<std::size_t>(tokens);
        } else if (option == "--steps" && index < argc) {
            int steps = 0;
            if (!parse_int(argv[index++], steps) || steps <= 0) {
                if (error) *error = "--steps must be positive";
                return false;
            }
            config.cascade.sparse_structure.sampler.steps = steps;
            config.cascade.shape_sampler.steps = steps;
            config.cascade.texture_sampler.steps = steps;
        } else if (option == "--occupancy-threshold" && index < argc) {
            if (!parse_float(argv[index++],
                             config.cascade.sparse_structure.occupancy_threshold)) {
                if (error) *error = "invalid --occupancy-threshold";
                return false;
            }
        } else if (option == "--max-structure-points" && index < argc) {
            std::uint64_t points = 0;
            if (!parse_u64(argv[index++], points) || points == 0 ||
                points > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
                if (error) *error = "--max-structure-points must be positive";
                return false;
            }
            config.cascade.max_structure_points = static_cast<std::size_t>(points);
        } else if (option == "--fov" && index < argc) {
            if (!parse_float(argv[index++], config.camera.camera_angle_x) ||
                !(config.camera.camera_angle_x > 0.0f &&
                  config.camera.camera_angle_x < 3.1415927f)) {
                if (error) *error = "--fov must be in (0, pi) radians";
                return false;
            }
        } else if (option == "--distance" && index < argc) {
            if (!parse_float(argv[index++], config.camera.distance) ||
                !(config.camera.distance > 0.0f)) {
                if (error) *error = "--distance must be positive";
                return false;
            }
        } else if (option == "--mesh-scale" && index < argc) {
            if (!parse_float(argv[index++], config.camera.mesh_scale) ||
                !(config.camera.mesh_scale > 0.0f)) {
                if (error) *error = "--mesh-scale must be positive";
                return false;
            }
        } else if (option == "--max-model-gib" && index < argc) {
            float gib = 0.0f;
            if (!parse_float(argv[index++], gib) || !(gib > 0.0f) ||
                static_cast<double>(gib) * 1024.0 * 1024.0 * 1024.0 >
                    static_cast<double>(std::numeric_limits<std::size_t>::max())) {
                if (error) *error = "--max-model-gib must be positive";
                return false;
            }
            config.max_model_bytes = static_cast<std::size_t>(
                static_cast<double>(gib) * 1024.0 * 1024.0 * 1024.0);
        } else {
            if (error) *error = "unknown or incomplete run option: " + option;
            return false;
        }
    }
    config.camera = pixal3d::ProjectionCamera::front(
        config.camera.camera_angle_x, config.camera.distance,
        config.camera.mesh_scale);
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || std::string(argv[1]) == "--help" ||
        std::string(argv[1]) == "-h") {
        print_usage(argv[0], std::cout);
        return argc < 2 ? 2 : 0;
    }

    const std::string command = argv[1];
    if (command == "--version" || command == "-V") {
        std::cout << "pixal3d " << pixal3d::version() << "\n";
        return 0;
    }
    if (command == "encode-condition-bundle") {
        if (argc < 6) {
            print_usage(argv[0], std::cerr);
            return 2;
        }
        pixal3d::Pixal3DImageConditionBundleConfig bundle_config;
        for (int index = 6; index < argc;) {
            const std::string option = argv[index++];
            int * destination = nullptr;
            if (option == "--ss-resolution") {
                destination = &bundle_config.ss_resolution;
            } else if (option == "--shape-512-resolution") {
                destination = &bundle_config.shape_512_resolution;
            } else if (option == "--shape-1024-resolution") {
                destination = &bundle_config.shape_1024_resolution;
            } else if (option == "--tex-1024-resolution") {
                destination = &bundle_config.tex_1024_resolution;
            } else if (option == "--ss-naf-resolution") {
                destination = &bundle_config.ss_naf_resolution;
            } else if (option == "--shape-512-naf-resolution") {
                destination = &bundle_config.shape_512_naf_resolution;
            } else if (option == "--shape-1024-naf-resolution") {
                destination = &bundle_config.shape_1024_naf_resolution;
            } else if (option == "--tex-1024-naf-resolution") {
                destination = &bundle_config.tex_1024_naf_resolution;
            } else {
                std::cerr << "error: unknown or incomplete encode-condition-bundle option: "
                          << option << "\n";
                return 2;
            }
            if (index >= argc || !parse_int(argv[index++], *destination) ||
                *destination < 0 ||
                (option.find("naf") == std::string::npos && *destination == 0)) {
                std::cerr << "error: " << option << " must be "
                          << (option.find("naf") == std::string::npos
                                  ? "positive" : "non-negative") << "\n";
                return 2;
            }
        }
        if (std::string(argv[3]) == "-") {
            std::cerr << "error: encode-condition-bundle requires a NAF GGUF pack\n";
            return 2;
        }
        pixal3d::DinoV3Model dino_model;
        pixal3d::NafModel naf_model;
        std::string error;
        if (!dino_model.load(argv[2], true, &error) ||
            !naf_model.load(argv[3], true, &error)) {
            std::cerr << "error: " << error << "\n";
            return 1;
        }
        std::cout << "dino_backend    : " << dino_model.backend_name() << "\n";
        pixal3d::Pixal3DImageF32 image;
        if (!pixal3d::load_pixal3d_image_f32(argv[4], image, &error)) {
            std::cerr << "error: " << error << "\n";
            return 1;
        }
        pixal3d::Pixal3DConditionBundleF32 bundle;
        if (!pixal3d::encode_pixal3d_condition_bundle_from_image_f32(
                dino_model, &naf_model, image, bundle_config, bundle, &error) ||
            !pixal3d::save_pixal3d_condition_bundle(argv[5], bundle, &error)) {
            std::cerr << "error: " << error << "\n";
            return 1;
        }
        std::cout << "kind            : condition-bundle\n"
                  << "stages          : " << bundle.stages.size() << "\n"
                  << "source_image    : " << image.width << "x" << image.height << " RGB F32\n"
                  << "output          : " << argv[5] << "\n";
        return 0;
    }
    if (command == "encode-condition-stage") {
        if (argc < 7) {
            print_usage(argv[0], std::cerr);
            return 2;
        }
        const std::string dino_path = argv[2];
        const std::string naf_path = argv[3];
        const std::string image_path = argv[4];
        const std::string stage_name = argv[5];
        int image_resolution = 0;
        int naf_resolution = 0;
        if (stage_name == "ss") {
            image_resolution = 512;
        } else if (stage_name == "shape_512") {
            image_resolution = 512;
            naf_resolution = 512;
        } else if (stage_name == "shape_1024") {
            image_resolution = 1024;
            naf_resolution = 512;
        } else if (stage_name == "tex_1024") {
            image_resolution = 1024;
            naf_resolution = 1024;
        } else {
            std::cerr << "error: stage must be ss, shape_512, shape_1024, or tex_1024\n";
            return 2;
        }
        for (int index = 7; index < argc;) {
            const std::string option = argv[index++];
            if (option == "--resolution" && index < argc) {
                if (!parse_int(argv[index++], image_resolution) || image_resolution <= 0) {
                    std::cerr << "error: --resolution must be positive\n";
                    return 2;
                }
            } else if (option == "--naf-resolution" && index < argc) {
                if (!parse_int(argv[index++], naf_resolution) || naf_resolution < 0) {
                    std::cerr << "error: --naf-resolution must be non-negative\n";
                    return 2;
                }
            } else {
                std::cerr << "error: unknown or incomplete encode-condition-stage option: "
                          << option << "\n";
                return 2;
            }
        }
        if (naf_resolution > 0 && naf_path == "-") {
            std::cerr << "error: this stage requests NAF but naf.gguf is '-': "
                      << "provide a NAF pack\n";
            return 2;
        }
        pixal3d::DinoV3Model dino_model;
        pixal3d::NafModel naf_model;
        std::string error;
        if (!dino_model.load(dino_path, true, &error)) {
            std::cerr << "error: " << error << "\n";
            return 1;
        }
        std::cout << "dino_backend    : " << dino_model.backend_name() << "\n";
        const int patch_size = dino_model.hparams().patch_size;
        if (patch_size <= 0 || image_resolution % patch_size != 0) {
            std::cerr << "error: --resolution must be divisible by the DINO patch size ("
                      << patch_size << ")\n";
            return 2;
        }
        const pixal3d::NafModel * naf_ptr = nullptr;
        if (naf_resolution > 0) {
            if (!naf_model.load(naf_path, true, &error)) {
                std::cerr << "error: " << error << "\n";
                return 1;
            }
            naf_ptr = &naf_model;
        }
        pixal3d::Pixal3DImageF32 image;
        if (!pixal3d::load_and_resize_pixal3d_image_f32(
                image_path, image_resolution, image, &error)) {
            std::cerr << "error: " << error << "\n";
            return 1;
        }
        pixal3d::Pixal3DVisionStageInputF32 input;
        input.name = stage_name;
        input.image_resolution = image_resolution;
        input.naf_output_resolution = naf_resolution;
        input.image_chw_01 = image.pixels.data();
        input.image_elements = image.pixels.size();
        pixal3d::Pixal3DConditionBundleF32 bundle;
        if (!pixal3d::encode_pixal3d_condition_bundle_f32(
                dino_model, naf_ptr, {input}, bundle, &error)) {
            std::cerr << "error: " << error << "\n";
            return 1;
        }
        if (!pixal3d::save_pixal3d_condition_bundle(argv[6], bundle, &error)) {
            std::cerr << "error: " << error << "\n";
            return 1;
        }
        const auto & encoded = bundle.stages.front();
        std::cout << "kind            : condition-bundle\n"
                  << "stage           : " << encoded.name << "\n"
                  << "image           : " << image.width << "x" << image.height << " RGB F32\n"
                  << "global          : " << encoded.global.tokens() << "x"
                  << encoded.global.channels << "\n"
                  << "dino_map        : " << encoded.dino_map.height << "x"
                  << encoded.dino_map.width << "x" << encoded.dino_map.channels << "\n"
                  << "naf_map         : " << (encoded.has_naf() ? "enabled" : "disabled") << "\n"
                  << "output          : " << argv[6] << "\n";
        return 0;
    }
    if (command == "estimate-model") {
        if (argc != 4) {
            print_usage(argv[0], std::cerr);
            return 2;
        }
        std::size_t bytes = 0;
        std::string error;
        if (!pixal3d::estimate_pixal3d_model_bytes(argv[2], argv[3], bytes, &error)) {
            std::cerr << "error: " << error << "\n";
            return 1;
        }
        std::cout << "pixal3d " << pixal3d::version() << "\n"
                  << "kind            : model-memory-estimate\n"
                  << "shared_pack     : " << argv[2] << "\n"
                  << "flow_pack       : " << argv[3] << "\n"
                  << "host_f32_bytes  : " << bytes << "\n"
                  << "host_f32_gib    : " << std::fixed << std::setprecision(3)
                  << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) << "\n";
        return 0;
    }
    if (command == "run-image") {
        if (argc < 8) {
            print_usage(argv[0], std::cerr);
            return 2;
        }
        pixal3d::Pixal3DInferenceConfig config =
            pixal3d::default_pixal3d_inference_config();
        pixal3d::Pixal3DImageConditionBundleConfig vision_config;
        std::string error;
        if (!parse_cascade_options(argc, argv, 8, config, &error, &vision_config)) {
            std::cerr << "error: " << error << "\n";
            return 2;
        }
        // Check the large cascade packs before loading the vision encoders or
        // constructing their image features.  This keeps --max-model-gib a
        // genuine preflight guard for the single-process image path.
        std::size_t estimated_bytes = 0;
        if (!pixal3d::estimate_pixal3d_model_bytes(argv[2], argv[3],
                                                  estimated_bytes, &error)) {
            std::cerr << "error: " << error << "\n";
            return 1;
        }
        if (config.max_model_bytes > 0 && estimated_bytes > config.max_model_bytes) {
            std::cerr << "error: Pixal3D model F32 memory estimate exceeds configured limit: "
                      << estimated_bytes << " > " << config.max_model_bytes << " bytes\n";
            return 1;
        }
        std::cout << "model_host_f32_gib " << std::fixed << std::setprecision(3)
                  << (static_cast<double>(estimated_bytes) /
                      (1024.0 * 1024.0 * 1024.0)) << "\n";

        pixal3d::DinoV3Model dino_model;
        pixal3d::NafModel naf_model;
        std::cerr << "pixal3d: loading DINOv3 tensors" << std::endl;
        if (!dino_model.load(argv[4], true, &error) ||
            !naf_model.load(argv[5], true, &error)) {
            std::cerr << "error: " << error << "\n";
            return 1;
        }
        std::cerr << "pixal3d: vision tensors loaded" << std::endl;
        std::cout << "dino_backend    : " << dino_model.backend_name() << "\n";
        pixal3d::Pixal3DImageF32 image;
        if (!pixal3d::load_pixal3d_image_f32(argv[6], image, &error)) {
            std::cerr << "error: " << error << "\n";
            return 1;
        }
        pixal3d::Pixal3DConditionBundleF32 bundle;
        std::cerr << "pixal3d: encoding image conditions" << std::endl;
        if (!pixal3d::encode_pixal3d_condition_bundle_from_image_f32(
                dino_model, &naf_model, image,
                vision_config, bundle, &error)) {
                std::cerr << "error: " << error << "\n";
            return 1;
        }
        std::cerr << "pixal3d: image conditions encoded" << std::endl;
        std::cout << "condition_stages " << bundle.stages.size() << "\n";
        pixal3d::Pixal3DCascadeOutputF32 output;
        std::cerr << "pixal3d: running cascade" << std::endl;
        if (!pixal3d::run_pixal3d_from_condition_stages(
                argv[2], argv[3], bundle.stages, config, output, &error)) {
            std::cerr << "error: " << error << "\n";
            return 1;
        }
        if (output.meshes.empty()) {
            std::cerr << "error: cascade produced no mesh\n";
            return 1;
        }
        if (!pixal3d::write_pixal3d_obj(output.meshes.front(), argv[7], &error)) {
            std::cerr << "error: " << error << "\n";
            return 1;
        }
        const auto & mesh = output.meshes.front();
        std::cout << "resolution      : " << output.resolution << "\n"
                  << "vertices        : " << mesh.vertices.size() / 3 << "\n"
                  << "triangles       : " << mesh.faces.size() / 3 << "\n"
                  << "output          : " << argv[7] << "\n";
        return 0;
    }
    if (command == "run-cascade" || command == "run-cascade-mv") {
        const bool multi_view = command == "run-cascade-mv";
        if (argc < 6) {
            print_usage(argv[0], std::cerr);
            return 2;
        }
        pixal3d::Pixal3DInferenceConfig config =
            pixal3d::default_pixal3d_inference_config();
        for (int index = 6; index < argc;) {
            const std::string option = argv[index++];
            if (option == "--seed" && index < argc) {
                if (!parse_u64(argv[index++], config.seed)) {
                    std::cerr << "error: invalid --seed\n";
                    return 2;
                }
            } else if (option == "--resolution" && index < argc) {
                if (!parse_int(argv[index++], config.cascade.requested_resolution) ||
                    config.cascade.requested_resolution < 1024 ||
                    config.cascade.requested_resolution % 16 != 0) {
                    std::cerr << "error: --resolution must be a multiple of 16 >= 1024\n";
                    return 2;
                }
            } else if (option == "--max-tokens" && index < argc) {
                std::uint64_t tokens = 0;
                if (!parse_u64(argv[index++], tokens) ||
                    tokens > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
                    std::cerr << "error: invalid --max-tokens\n";
                    return 2;
                }
                config.cascade.max_num_tokens = static_cast<std::size_t>(tokens);
            } else if (option == "--steps" && index < argc) {
                int steps = 0;
                if (!parse_int(argv[index++], steps) || steps <= 0) {
                    std::cerr << "error: --steps must be positive\n";
                    return 2;
                }
                config.cascade.sparse_structure.sampler.steps = steps;
                config.cascade.shape_sampler.steps = steps;
                config.cascade.texture_sampler.steps = steps;
            } else if (option == "--occupancy-threshold" && index < argc) {
                if (!parse_float(argv[index++],
                                 config.cascade.sparse_structure.occupancy_threshold)) {
                    std::cerr << "error: invalid --occupancy-threshold\n";
                    return 2;
                }
            } else if (option == "--max-structure-points" && index < argc) {
                std::uint64_t points = 0;
                if (!parse_u64(argv[index++], points) || points == 0 ||
                    points > static_cast<std::uint64_t>(
                        std::numeric_limits<std::size_t>::max())) {
                    std::cerr << "error: --max-structure-points must be positive\n";
                    return 2;
                }
                config.cascade.max_structure_points = static_cast<std::size_t>(points);
            } else if (option == "--fov" && index < argc) {
                if (!parse_float(argv[index++], config.camera.camera_angle_x) ||
                    !(config.camera.camera_angle_x > 0.0f &&
                      config.camera.camera_angle_x < 3.1415927f)) {
                    std::cerr << "error: --fov must be in (0, pi) radians\n";
                    return 2;
                }
            } else if (option == "--distance" && index < argc) {
                if (!parse_float(argv[index++], config.camera.distance) ||
                    !(config.camera.distance > 0.0f)) {
                    std::cerr << "error: --distance must be positive\n";
                    return 2;
                }
            } else if (option == "--mesh-scale" && index < argc) {
                if (!parse_float(argv[index++], config.camera.mesh_scale) ||
                    !(config.camera.mesh_scale > 0.0f)) {
                    std::cerr << "error: --mesh-scale must be positive\n";
                    return 2;
                }
            } else if (option == "--max-model-gib" && index < argc) {
                float gib = 0.0f;
                if (!parse_float(argv[index++], gib) || !(gib > 0.0f) ||
                    static_cast<double>(gib) * 1024.0 * 1024.0 * 1024.0 >
                        static_cast<double>(std::numeric_limits<std::size_t>::max())) {
                    std::cerr << "error: --max-model-gib must be positive\n";
                    return 2;
                }
                config.max_model_bytes = static_cast<std::size_t>(
                    static_cast<double>(gib) * 1024.0 * 1024.0 * 1024.0);
            } else {
                std::cerr << "error: unknown or incomplete run-cascade option: "
                          << option << "\n";
                return 2;
            }
        }
        // Rebuild the front-view camera after parsing individual scalar fields.
        config.camera = pixal3d::ProjectionCamera::front(
            config.camera.camera_angle_x, config.camera.distance,
            config.camera.mesh_scale);
        std::size_t estimated_bytes = 0;
        std::string error;
        if (!pixal3d::estimate_pixal3d_model_bytes(argv[2], argv[3],
                                                  estimated_bytes, &error)) {
            std::cerr << "error: " << error << "\n";
            return 1;
        }
        std::cout << "model_host_f32_gib " << std::fixed << std::setprecision(3)
                  << (static_cast<double>(estimated_bytes) /
                      (1024.0 * 1024.0 * 1024.0)) << "\n";
        pixal3d::Pixal3DCascadeOutputF32 output;
        const bool ok = multi_view
            ? pixal3d::run_pixal3d_from_multiview_condition_bundle(
                  argv[2], argv[3], argv[4], config, output, &error)
            : pixal3d::run_pixal3d_from_condition_bundle(
                  argv[2], argv[3], argv[4], config, output, &error);
        if (!ok) {
            std::cerr << "error: " << error << "\n";
            return 1;
        }
        if (output.meshes.empty()) {
            std::cerr << "error: cascade produced no mesh\n";
            return 1;
        }
        if (!pixal3d::write_pixal3d_obj(output.meshes.front(), argv[5], &error)) {
            std::cerr << "error: " << error << "\n";
            return 1;
        }
        const auto & mesh = output.meshes.front();
        std::cout << "resolution      : " << output.resolution << "\n"
                  << "vertices        : " << mesh.vertices.size() / 3 << "\n"
                  << "triangles       : " << mesh.faces.size() / 3 << "\n"
                  << "output          : " << argv[5] << "\n";
        return 0;
    }
    if (command != "inspect-pack" || argc != 3) {
        if (command == "inspect-dinodata" && argc == 3) {
            pixal3d::DinoConditionF32 condition;
            std::string error;
            if (!pixal3d::load_dino_condition(argv[2], condition, &error)) {
                std::cerr << "error: " << error << "\n";
                return 1;
            }
            const pixal3d::DinoFingerprintF32 fp =
                pixal3d::fingerprint_dino_condition(condition);
            std::cout << "pixal3d " << pixal3d::version() << "\n"
                      << "kind            : dino-condition\n"
                      << "file            : " << argv[2] << "\n"
                      << "format_version  : " << condition.format_version << "\n"
                      << "shape           :";
            for (std::int64_t dimension : condition.shape) {
                std::cout << " " << dimension;
            }
            std::cout << "\n"
                      << "tokens/channels : " << condition.global.tokens() << "/"
                      << condition.global.channels << "\n"
                      << "fingerprint     : min=" << fp.minimum
                      << " max=" << fp.maximum << " mean=" << fp.mean
                      << " sum=" << fp.sum << " l2=" << fp.l2
                      << " count=" << fp.count << "\n";
            return 0;
        }
        if (command == "inspect-dino-vit" && argc == 3) {
            pixal3d::DinoV3Model model;
            std::string error;
            if (!model.load(argv[2], false, &error)) {
                std::cerr << "error: " << error << "\n";
                return 1;
            }
            const auto & hp = model.hparams();
            std::cout << "pixal3d " << pixal3d::version() << "\n"
                      << "kind            : dino-vit\n"
                      << "file            : " << argv[2] << "\n"
                      << "hidden/ffn       : " << hp.hidden_size << "/"
                      << hp.intermediate_size << "\n"
                      << "layers/heads     : " << hp.num_hidden_layers << "/"
                      << hp.num_attention_heads << "\n"
                      << "patch/registers : " << hp.patch_size << "/"
                      << hp.num_register_tokens << "\n"
                      << "tensor_count     : " << model.tensor_count() << "\n";
            return 0;
        }
        if (command == "inspect-naf" && argc == 3) {
            pixal3d::NafModel model;
            std::string error;
            if (!model.load(argv[2], false, &error)) {
                std::cerr << "error: " << error << "\n";
                return 1;
            }
            const auto & hp = model.hparams();
            std::cout << "pixal3d " << pixal3d::version() << "\n"
                      << "kind            : naf\n"
                      << "file            : " << argv[2] << "\n"
                      << "dim/in_channels : " << hp.dim << "/" << hp.in_channels << "\n"
                      << "attn/rope heads  : " << hp.heads_attn << "/" << hp.heads_rope << "\n"
                      << "kernel/layers    : " << hp.kernel_size << "/" << hp.img_layers << "\n"
                      << "tensor_count     : " << model.tensor_count() << "\n";
            return 0;
        }
        if (command == "inspect-condition" && argc == 3) {
            pixal3d::Pixal3DConditionBundleF32 bundle;
            std::string error;
            if (!pixal3d::load_pixal3d_condition_bundle(argv[2], bundle, &error)) {
                std::cerr << "error: " << error << "\n";
                return 1;
            }
            std::cout << "pixal3d " << pixal3d::version() << "\n"
                      << "kind            : condition-bundle\n"
                      << "file            : " << argv[2] << "\n"
                      << "format_version  : " << bundle.format_version << "\n"
                      << "stages          : " << bundle.stages.size() << "\n";
            for (const auto & stage : bundle.stages) {
                std::cout << "  stage         : " << stage.name << "\n"
                          << "    global      : " << stage.global.tokens() << "x"
                          << stage.global.channels << "\n"
                          << "    dino_map    : " << stage.dino_map.height << "x"
                          << stage.dino_map.width << "x" << stage.dino_map.channels
                          << " @" << stage.dino_map.image_resolution << "\n";
                if (stage.has_naf()) {
                    std::cout << "    naf_map     : " << stage.naf_map.height << "x"
                              << stage.naf_map.width << "x" << stage.naf_map.channels
                              << " @" << stage.naf_map.image_resolution << "\n";
                }
            }
            return 0;
        }
        if (command == "inspect-mv-condition" && argc == 3) {
            pixal3d::Pixal3DMultiViewConditionBundleF32 bundle;
            std::string error;
            if (!pixal3d::load_pixal3d_multiview_condition_bundle(
                    argv[2], bundle, &error)) {
                std::cerr << "error: " << error << "\n";
                return 1;
            }
            std::cout << "pixal3d " << pixal3d::version() << "\n"
                      << "kind            : multiview-condition-bundle\n"
                      << "file            : " << argv[2] << "\n"
                      << "format_version  : " << bundle.format_version << "\n"
                      << "stages          : " << bundle.stages.size() << "\n";
            for (const auto & stage : bundle.stages) {
                std::cout << "  stage         : " << stage.name << "\n"
                          << "    views       : " << stage.views.size() << "\n";
                if (!stage.views.empty()) {
                    const auto & view = stage.views.front();
                    std::cout << "    global      : " << view.global.tokens() << "x"
                              << view.global.channels << "\n"
                              << "    dino_map    : " << view.dino_map.height << "x"
                              << view.dino_map.width << "x" << view.dino_map.channels
                              << " @" << view.dino_map.image_resolution << "\n";
                    if (view.has_naf()) {
                        std::cout << "    naf_map     : " << view.naf_map.height << "x"
                                  << view.naf_map.width << "x" << view.naf_map.channels
                                  << " @" << view.naf_map.image_resolution << "\n";
                    }
                }
            }
            return 0;
        }
        if (command == "inspect-ss-flow" && argc == 3) {
            pixal3d::SSFlowModel model;
            std::string error;
            if (!model.load(argv[2], false, &error)) {
                std::cerr << "error: " << error << "\n";
                return 1;
            }
            const pixal3d::SSFlowHParams & hp = model.hparams();
            std::cout << "pixal3d " << pixal3d::version() << "\n"
                      << "kind            : ss-flow\n"
                      << "file            : " << argv[2] << "\n"
                      << "resolution      : " << (hp.resolution > 0
                                                    ? std::to_string(hp.resolution)
                                                    : "unspecified") << "\n"
                      << "channels        : " << hp.in_channels << " -> "
                      << hp.out_channels << "\n"
                      << "model_channels  : " << hp.model_channels << "\n"
                      << "conditioning    : " << hp.cond_channels << "\n"
                      << "blocks/heads    : " << hp.num_blocks << "/" << hp.num_heads << "\n"
                      << "attention       : " << hp.image_attn_mode << "\n"
                      << "tensors         : " << model.tensor_count() << "\n";
            return 0;
        }
        if (command == "inspect-ss-decoder" && argc == 3) {
            pixal3d::SSDecoderModel model;
            std::string error;
            if (!model.load(argv[2], false, &error)) {
                std::cerr << "error: " << error << "\n";
                return 1;
            }
            const pixal3d::SSDecoderHParams & hp = model.hparams();
            std::cout << "pixal3d " << pixal3d::version() << "\n"
                      << "kind            : ss-decoder\n"
                      << "file            : " << argv[2] << "\n"
                      << "resolution      : " << hp.resolution << " -> "
                      << hp.output_resolution() << "\n"
                      << "channels        : " << hp.latent_channels << " -> "
                      << hp.out_channels << "\n"
                      << "levels          : " << hp.channels.size() << "\n"
                      << "res_blocks      : " << hp.num_res_blocks << " + "
                      << hp.num_res_blocks_middle << " middle\n"
                      << "tensors         : " << model.tensor_count() << "\n";
            return 0;
        }
        if (command == "inspect-slat-decoder" && argc == 4) {
            pixal3d::SLatDecoderModel model;
            std::string error;
            if (!model.load(argv[2], argv[3], false, &error)) {
                std::cerr << "error: " << error << "\n";
                return 1;
            }
            const pixal3d::SLatDecoderHParams & hp = model.hparams();
            std::cout << "pixal3d " << pixal3d::version() << "\n"
                      << "kind            : slat-decoder\n"
                      << "component       : " << hp.component << "\n"
                      << "model_class     : " << hp.model_class << "\n"
                      << "resolution      : " << (hp.resolution > 0
                                                    ? std::to_string(hp.resolution)
                                                    : "unspecified") << "\n"
                      << "channels        : " << hp.latent_channels << " -> "
                      << hp.out_channels << "\n"
                      << "levels          : " << hp.model_channels.size() << "\n"
                      << "model_channels  :";
            for (int channels : hp.model_channels) std::cout << " " << channels;
            std::cout << "\n"
                      << "num_blocks      :";
            for (int blocks : hp.num_blocks) std::cout << " " << blocks;
            std::cout << "\n"
                      << "pred_subdiv     : " << (hp.pred_subdiv ? "true" : "false") << "\n"
                      << "tensors         : " << model.tensor_count() << "\n";
            return 0;
        }
        if (command == "inspect-slat-flow" && argc == 4) {
            pixal3d::SLatFlowModel model;
            std::string error;
            if (!model.load(argv[2], argv[3], false, &error)) {
                std::cerr << "error: " << error << "\n";
                return 1;
            }
            const pixal3d::SLatFlowHParams & hp = model.hparams();
            std::cout << "pixal3d " << pixal3d::version() << "\n"
                      << "kind            : slat-flow\n"
                      << "component       : " << hp.component << "\n"
                      << "model_class     : " << hp.model_class << "\n"
                      << "resolution      : " << hp.resolution << "\n"
                      << "channels        : " << hp.in_channels << " -> "
                      << hp.out_channels << "\n"
                      << "model_channels  : " << hp.model_channels << "\n"
                      << "conditioning    : " << hp.cond_channels << "\n"
                      << "blocks/heads    : " << hp.num_blocks << "/"
                      << hp.num_heads << "\n"
                      << "attention       : " << hp.image_attn_mode << "\n"
                      << "projection      : " << hp.proj_in_channels << "\n"
                      << "pe/share_mod    : " << hp.pe_mode << "/"
                      << (hp.share_mod ? "true" : "false") << "\n"
                      << "tensors         : " << model.tensor_count() << "\n";
            return 0;
        }
        if (command != "inspect-pack") {
            std::cerr << "error: unknown command: " << command << "\n\n";
        }
        print_usage(argv[0], std::cerr);
        return 2;
    }

    std::string error;
    if (!pixal3d::inspect_pack(argv[2], std::cout, &error)) {
        std::cerr << "error: " << error << "\n";
        return 1;
    }
    return 0;
}
