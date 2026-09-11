#include "pixal3d/slat_decoder.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr int kLatentChannels = 3;
constexpr int kModelChannels = 4;
constexpr int kOutputChannels = 2;
constexpr int kMlpHidden = 16;

void require(bool condition, const std::string & message) {
    if (!condition) {
        std::cerr << "SLat GPU-first fixture: " << message << "\n";
        std::exit(EXIT_FAILURE);
    }
}

std::vector<float> values_for(const std::string & name, std::size_t count) {
    std::vector<float> values(count, 0.0f);
    std::size_t seed = 0;
    for (std::size_t index = 0; index < name.size(); ++index) {
        seed += (index + 1) * static_cast<unsigned char>(name[index]);
    }
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = (static_cast<float>((index + seed) % 19) - 9.0f) * 0.011f;
    }
    return values;
}

void add_tensor(ggml_context * context,
                gguf_context * writer,
                std::map<std::string, std::vector<float>> & storage,
                const std::string & name,
                int n_dims,
                const int64_t * dimensions,
                const std::vector<float> & values) {
    ggml_tensor * tensor = ggml_new_tensor(context, GGML_TYPE_F32, n_dims, dimensions);
    require(tensor != nullptr, "failed to allocate " + name);
    ggml_set_name(tensor, name.c_str());
    require(ggml_nelements(tensor) == static_cast<int64_t>(values.size()),
            "element count mismatch for " + name);
    std::memcpy(tensor->data, values.data(), values.size() * sizeof(float));
    storage.emplace(name, values);
    gguf_add_tensor(writer, tensor);
    gguf_set_tensor_data(writer, name.c_str(), storage.at(name).data());
}

void add_1d(ggml_context * context,
            gguf_context * writer,
            std::map<std::string, std::vector<float>> & storage,
            const std::string & name,
            int64_t dim,
            std::vector<float> values) {
    add_tensor(context, writer, storage, name, 1, &dim, values);
}

void add_2d(ggml_context * context,
            gguf_context * writer,
            std::map<std::string, std::vector<float>> & storage,
            const std::string & name,
            int64_t dim0,
            int64_t dim1,
            std::vector<float> values) {
    const int64_t dimensions[] = {dim0, dim1};
    add_tensor(context, writer, storage, name, 2, dimensions, values);
}

void add_3d(ggml_context * context,
            gguf_context * writer,
            std::map<std::string, std::vector<float>> & storage,
            const std::string & name,
            int64_t dim0,
            int64_t dim1,
            int64_t dim2,
            std::vector<float> values) {
    const int64_t dimensions[] = {dim0, dim1, dim2};
    add_tensor(context, writer, storage, name, 3, dimensions, values);
}

std::vector<float> make_conv_weight(const std::string & name) {
    std::vector<float> values(static_cast<std::size_t>(27) * kModelChannels *
                              kModelChannels, 0.0f);
    const std::vector<float> perturbation = values_for(name, values.size());
    for (int kernel = 0; kernel < 27; ++kernel) {
        for (int output = 0; output < kModelChannels; ++output) {
            for (int input = 0; input < kModelChannels; ++input) {
                const std::size_t index =
                    (static_cast<std::size_t>(kernel) * kModelChannels + output) *
                    kModelChannels + input;
                if (kernel == 13 && output == input) {
                    values[index] = 0.25f;
                } else if (kernel != 13) {
                    values[index] = perturbation[index] * 0.015f;
                }
            }
        }
    }
    return values;
}

std::string make_fixture_pack() {
    const std::string path = "/tmp/pixal3d-slat-decoder-gpu-first-fixture.gguf";
    ggml_init_params params{};
    params.mem_size = 1u << 20;
    params.no_alloc = false;
    ggml_context * context = ggml_init(params);
    require(context != nullptr, "failed to create ggml context");
    gguf_context * writer = gguf_init_empty();
    require(writer != nullptr, "failed to create GGUF writer");
    std::map<std::string, std::vector<float>> storage;

    gguf_set_val_str(writer, "general.architecture", "pixal3d");
    gguf_set_val_str(writer, "general.name", "slat-decoder-gpu-first-fixture");
    gguf_set_val_u32(writer, "general.file_type", 0);
    gguf_set_val_u32(writer, "general.alignment", 32);
    gguf_set_val_u32(writer, "pixal3d.bundle_format", 1);
    gguf_set_val_str(writer, "pixal3d.tensor_name_scheme", "compact-v1");
    gguf_set_val_str(writer, "pixal3d.bundle_kind", "component");
    gguf_set_val_u32(writer, "pixal3d.component_count", 1);
    gguf_set_val_str(writer, "pixal3d.component.0", "shape_decoder");
    gguf_set_val_str(writer, "pixal3d.shape_decoder.model_class",
                     "FlexiDualGridVaeDecoder");
    gguf_set_val_str(writer, "pixal3d.shape_decoder.checkpoint", "gpu-first-fixture");
    gguf_set_val_str(writer, "pixal3d.shape_decoder.variant", "fixture");
    gguf_set_val_u32(writer, "pixal3d.shape_decoder.resolution", 2);
    gguf_set_val_u32(writer, "pixal3d.shape_decoder.out_channels", kOutputChannels);
    gguf_set_val_u32(writer, "pixal3d.shape_decoder.latent_channels", kLatentChannels);
    gguf_set_val_f32(writer, "pixal3d.shape_decoder.norm_eps", 1e-6f);
    gguf_set_val_u32(writer, "pixal3d.shape_decoder.n_levels", 1);
    gguf_set_val_bool(writer, "pixal3d.shape_decoder.pred_subdiv", false);
    gguf_set_val_u32(writer, "pixal3d.shape_decoder.model_channels.0", kModelChannels);
    gguf_set_val_u32(writer, "pixal3d.shape_decoder.num_blocks.0", 2);
    gguf_set_val_str(writer, "pixal3d.shape_decoder.conv_weight_source_layout",
                     "out,kd,kh,kw,in");
    gguf_set_val_str(writer, "pixal3d.shape_decoder.conv_weight_gguf_layout",
                     "kernel_volume,out,in");
    gguf_set_val_u32(writer, "pixal3d.shape_decoder.conv_kernel_size", 3);
    gguf_set_val_u32(writer, "pixal3d.shape_decoder.conv_kernel_volume", 27);

    const std::string prefix = "shape_decoder.";
    add_2d(context, writer, storage, prefix + "from_latent.weight",
           kLatentChannels, kModelChannels,
           values_for("from_latent.weight", kLatentChannels * kModelChannels));
    add_1d(context, writer, storage, prefix + "from_latent.bias", kModelChannels,
           values_for("from_latent.bias", kModelChannels));

    for (int block = 0; block < 2; ++block) {
        const std::string block_prefix = prefix + "blocks.0." + std::to_string(block) + ".";
        add_1d(context, writer, storage, block_prefix + "norm.weight", kModelChannels,
               {1.0f, 0.97f, 1.03f, 0.99f});
        add_1d(context, writer, storage, block_prefix + "norm.bias", kModelChannels,
               values_for(block_prefix + "norm.bias", kModelChannels));
        add_3d(context, writer, storage, block_prefix + "conv.weight",
               kModelChannels, kModelChannels, 27, make_conv_weight(block_prefix + "conv.weight"));
        add_1d(context, writer, storage, block_prefix + "conv.bias", kModelChannels,
               values_for(block_prefix + "conv.bias", kModelChannels));
        add_2d(context, writer, storage, block_prefix + "mlp.0.weight",
               kModelChannels, kMlpHidden,
               values_for(block_prefix + "mlp.0.weight", kModelChannels * kMlpHidden));
        add_1d(context, writer, storage, block_prefix + "mlp.0.bias", kMlpHidden,
               values_for(block_prefix + "mlp.0.bias", kMlpHidden));
        add_2d(context, writer, storage, block_prefix + "mlp.2.weight",
               kMlpHidden, kModelChannels,
               values_for(block_prefix + "mlp.2.weight", kMlpHidden * kModelChannels));
        add_1d(context, writer, storage, block_prefix + "mlp.2.bias", kModelChannels,
               values_for(block_prefix + "mlp.2.bias", kModelChannels));
    }

    add_2d(context, writer, storage, prefix + "output_layer.weight",
           kModelChannels, kOutputChannels,
           values_for("output_layer.weight", kModelChannels * kOutputChannels));
    add_1d(context, writer, storage, prefix + "output_layer.bias", kOutputChannels,
           values_for("output_layer.bias", kOutputChannels));

    require(gguf_write_to_file(writer, path.c_str(), false), "failed to write fixture pack");
    gguf_free(writer);
    ggml_free(context);
    return path;
}

struct ErrorStats {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double mean_abs = 0.0;
};

ErrorStats compare(const pixal3d::SparseTensorF32 & cpu,
                   const pixal3d::SparseTensorF32 & gpu) {
    require(cpu.coords == gpu.coords, "GPU-first changed coordinates");
    require(cpu.channels == gpu.channels && cpu.batch_size == gpu.batch_size &&
                cpu.spatial_x == gpu.spatial_x && cpu.spatial_y == gpu.spatial_y &&
                cpu.spatial_z == gpu.spatial_z,
            "GPU-first changed output metadata");
    require(cpu.feats.size() == gpu.feats.size(), "GPU-first changed feature size");
    ErrorStats result;
    for (std::size_t index = 0; index < cpu.feats.size(); ++index) {
        require(std::isfinite(cpu.feats[index]) && std::isfinite(gpu.feats[index]),
                "GPU-first produced a non-finite feature");
        const double absolute = std::fabs(static_cast<double>(cpu.feats[index]) -
                                          static_cast<double>(gpu.feats[index]));
        const double relative = absolute / std::max(
            std::fabs(static_cast<double>(cpu.feats[index])), 1e-12);
        result.max_abs = std::max(result.max_abs, absolute);
        result.max_rel = std::max(result.max_rel, relative);
        result.mean_abs += absolute;
    }
    if (!cpu.feats.empty()) result.mean_abs /= cpu.feats.size();
    return result;
}

} // namespace

int main() {
    const std::string pack = make_fixture_pack();
    pixal3d::SparseTensorF32 input;
    input.batch_size = 1;
    input.channels = kLatentChannels;
    input.spatial_x = input.spatial_y = input.spatial_z = 2;
    input.coords = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0};
    input.feats = {-0.21f, 0.04f, 0.17f,
                  0.08f, -0.12f, 0.23f,
                  -0.05f, 0.19f, -0.16f,
                  0.14f, 0.07f, -0.09f};
    std::vector<pixal3d::SparseTensorF32> no_guides;

    setenv("PIXAL3D_SLAT_DECODER_BACKEND", "gpu:0", 1);
    setenv("PIXAL3D_SLAT_DECODER_GPU_FIRST", "1", 1);
    pixal3d::SLatDecoderModel gpu_model;
    std::string error;
    require(gpu_model.load(pack, "shape_decoder", true, &error), error);
    pixal3d::SparseTensorF32 gpu_output;
    require(gpu_model.decode(input, &no_guides, gpu_output, nullptr, &error), error);

    setenv("PIXAL3D_SLAT_DECODER_BACKEND", "cpu", 1);
    pixal3d::SLatDecoderModel cpu_model;
    require(cpu_model.load(pack, "shape_decoder", true, &error), error);
    pixal3d::SparseTensorF32 cpu_output;
    require(cpu_model.decode(input, &no_guides, cpu_output, nullptr, &error), error);

    const ErrorStats stats = compare(cpu_output, gpu_output);
    // The fixture records the raw metrics rather than hiding reduction-order
    // differences.  The absolute gate is the primary correctness criterion;
    // the relative gate is only a guard for non-zero values near this fixture
    // scale.
    require(stats.max_abs < 1e-4 && stats.max_rel < 5e-3,
            "GPU-first parity error exceeds fixture tolerance");
    std::cout << "SLat GPU-first fixture: PASS"
              << " max_abs=" << stats.max_abs
              << " max_rel=" << stats.max_rel
              << " mean_abs=" << stats.mean_abs << "\n";

    std::remove(pack.c_str());
    return 0;
}
