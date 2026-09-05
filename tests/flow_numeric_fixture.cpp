#include "pixal3d/flow.h"

#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {

void emit(const char * name, const std::vector<float> & values) {
    std::cout << name << " " << values.size();
    for (float value : values) {
        std::cout << " " << value;
    }
    std::cout << "\n";
}

} // namespace

int main() {
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    const std::vector<float> timesteps = {-1.25f, 0.0f, 0.75f, 2.5f};
    std::vector<float> frequency;
    std::string error;
    if (!pixal3d::timestep_embedding(timesteps.data(), timesteps.size(), 9,
                                     10000, frequency, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    emit("timestep_frequency", frequency);

    // PyTorch [out_features, in_features] layout.  The values are deliberately
    // non-symmetric so a transposition or row/column stride mistake is visible.
    const float linear0_weight[] = {
        0.10f, -0.20f, 0.30f, -0.40f, 0.50f, -0.60f, 0.70f, -0.80f,
        -0.15f, 0.25f, -0.35f, 0.45f, -0.55f, 0.65f, -0.75f, 0.85f,
        0.05f, 0.15f, -0.25f, -0.35f, 0.45f, 0.55f, -0.65f, -0.75f,
        -0.08f, 0.18f, 0.28f, -0.38f, -0.48f, 0.58f, 0.68f, -0.78f,
        0.12f, 0.22f, 0.32f, 0.42f, -0.52f, -0.62f, -0.72f, 0.82f,
        -0.11f, -0.21f, 0.31f, 0.41f, 0.51f, -0.61f, -0.71f, -0.81f,
    };
    const float linear0_bias[] = {0.03f, -0.04f, 0.05f, -0.06f, 0.07f, -0.08f};
    const float linear2_weight[] = {
        0.21f, -0.31f, 0.41f, -0.51f, 0.61f, -0.71f,
        -0.17f, 0.27f, -0.37f, 0.47f, -0.57f, 0.67f,
        0.13f, 0.23f, -0.33f, -0.43f, 0.53f, 0.63f,
        -0.19f, -0.29f, 0.39f, 0.49f, -0.59f, -0.69f,
        0.16f, 0.26f, 0.36f, -0.46f, -0.56f, 0.66f,
        -0.14f, 0.24f, 0.34f, 0.44f, 0.54f, -0.64f,
    };
    const float linear2_bias[] = {-0.02f, 0.04f, -0.06f, 0.08f, -0.10f, 0.12f};
    std::vector<float> mlp;
    if (!pixal3d::timestep_embed_mlp(
            timesteps.data(), timesteps.size(), 6, 8, linear0_weight,
            linear0_bias, linear2_weight, linear2_bias, mlp, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    emit("timestep_mlp", mlp);

    const std::vector<float> rows = {
        0.15f, -0.20f, 0.35f, 0.40f, -0.55f,
        0.65f, 0.70f, -0.85f, 0.90f, 1.05f,
        -1.15f, 1.20f, 1.35f, -1.40f, 1.55f,
        1.65f, -1.70f, 1.85f, 1.90f, -2.05f,
        2.15f, 2.20f, -2.35f, 2.40f, 2.55f,
        -2.65f, 2.70f, 2.85f, -2.90f, 3.05f,
    };
    const float gamma[] = {0.90f, 1.05f, -0.80f, 1.15f, 0.70f};
    const float beta[] = {-0.10f, 0.20f, 0.30f, -0.40f, 0.50f};
    std::vector<float> norm;
    if (!pixal3d::layer_norm_rows(rows.data(), 2, 3, 5, 1e-6f, nullptr,
                                  nullptr, norm, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    emit("layer_norm", norm);
    std::vector<float> affine;
    if (!pixal3d::layer_norm_rows(rows.data(), 2, 3, 5, 1e-6f, gamma, beta,
                                  affine, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    emit("layer_norm_affine", affine);

    const float shift_scale[] = {
        0.01f, -0.02f, 0.03f, -0.04f, 0.05f, 0.10f, -0.20f, 0.30f, -0.40f, 0.50f,
        -0.06f, 0.07f, -0.08f, 0.09f, -0.10f, -0.15f, 0.25f, -0.35f, 0.45f, -0.55f,
    };
    std::vector<float> modulated;
    if (!pixal3d::adaln_modulate_rows(norm.data(), 2, 3, 5, shift_scale,
                                      modulated, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    emit("adaln_modulated", modulated);
    return 0;
}
