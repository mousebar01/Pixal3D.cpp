#include "pixal3d/inference.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main() {
    const pixal3d::Pixal3DInferenceConfig config =
        pixal3d::default_pixal3d_inference_config();
    if (config.cascade.requested_resolution != 1024 ||
        config.cascade.max_num_tokens != 49152 ||
        config.cascade.shape_normalization.mean.size() != 32 ||
        config.cascade.texture_normalization.std.size() != 32 ||
        config.cascade.shape_sampler.steps != 12 ||
        config.cascade.texture_sampler.guidance_strength != 1.0f) return 1;

    pixal3d::DualGridMeshF32 mesh;
    mesh.vertices = {1.0f, 2.0f, 3.0f,
                     4.0f, 5.0f, 6.0f,
                     7.0f, 8.0f, 9.0f};
    mesh.faces = {0, 1, 2};
    const std::vector<float> original_vertices = mesh.vertices;
    const std::vector<std::int32_t> original_faces = mesh.faces;
    const std::string path = "/tmp/pixal3d-inference-test.obj";
    std::string error;
    if (!pixal3d::write_pixal3d_obj(mesh, path, &error)) return 1;
    std::ifstream file(path);
    std::stringstream contents;
    contents << file.rdbuf();
    const std::string text = contents.str();
    if (text.find("v -1 -3 -2\n") == std::string::npos ||
        text.find("v -4 -6 -5\n") == std::string::npos ||
        text.find("v -7 -9 -8\n") == std::string::npos ||
        text.find("f 1 2 3\n") == std::string::npos ||
        mesh.vertices != original_vertices || mesh.faces != original_faces) return 1;
    std::remove(path.c_str());

    pixal3d::Pixal3DCascadeOutputF32 output;
    if (pixal3d::run_pixal3d_from_multiview_condition_stages(
            "/missing/shared.gguf", "/missing/flow.gguf", {}, config, output, &error) ||
        error.find("between one and four stages") == std::string::npos) {
        return 1;
    }
    return 0;
}
