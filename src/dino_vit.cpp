#include "pixal3d/dino_vit.h"

#include "pixal3d/backend.h"
#include "pixal3d/pack.h"

#include "ggml-alloc.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pixal3d {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

bool prefix_name(const std::string & name, const char * prefix) {
    const std::size_t length = std::strlen(prefix);
    return name.size() >= length && name.compare(0, length, prefix) == 0;
}

bool checked_product(std::size_t a, std::size_t b, std::size_t & out) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) return false;
    out = a * b;
    return true;
}

bool read_dino_hparams(const Pixal3DPackReader & reader,
                       DinoV3HParams & hp,
                       std::string * error) {
    const std::string p = "pixal3d.dino.";
    std::uint32_t u = 0;
    if (!reader.metadata_u32(p + "hidden_size", u, true, error)) return false;
    hp.hidden_size = static_cast<int>(u);
    if (!reader.metadata_u32(p + "intermediate_size", u, true, error)) return false;
    hp.intermediate_size = static_cast<int>(u);
    if (!reader.metadata_u32(p + "num_hidden_layers", u, true, error)) return false;
    hp.num_hidden_layers = static_cast<int>(u);
    if (!reader.metadata_u32(p + "num_attention_heads", u, true, error)) return false;
    hp.num_attention_heads = static_cast<int>(u);
    if (!reader.metadata_u32(p + "patch_size", u, true, error)) return false;
    hp.patch_size = static_cast<int>(u);
    if (!reader.metadata_u32(p + "num_channels", u, true, error)) return false;
    hp.num_channels = static_cast<int>(u);
    if (!reader.metadata_u32(p + "num_register_tokens", u, true, error)) return false;
    hp.num_register_tokens = static_cast<int>(u);
    if (!reader.metadata_f32(p + "layer_norm_eps", hp.layer_norm_eps, true, error) ||
        !reader.metadata_f32(p + "rope_theta", hp.rope_theta, true, error)) return false;
    if (!reader.metadata_bool(p + "use_gated_mlp", hp.use_gated_mlp, true, error) ||
        !reader.metadata_bool(p + "query_bias", hp.query_bias, false, error) ||
        !reader.metadata_bool(p + "key_bias", hp.key_bias, false, error) ||
        !reader.metadata_bool(p + "value_bias", hp.value_bias, false, error) ||
        !reader.metadata_bool(p + "proj_bias", hp.proj_bias, false, error) ||
        !reader.metadata_bool(p + "mlp_bias", hp.mlp_bias, false, error)) return false;
    if (hp.hidden_size <= 0 || hp.intermediate_size <= 0 || hp.num_hidden_layers <= 0 ||
        hp.num_attention_heads <= 0 || hp.hidden_size % hp.num_attention_heads != 0 ||
        hp.patch_size <= 0 || hp.num_channels <= 0 || hp.num_register_tokens < 0 ||
        !(hp.layer_norm_eps > 0.0f) || !std::isfinite(hp.layer_norm_eps) ||
        !(hp.rope_theta > 0.0f) || !std::isfinite(hp.rope_theta) || hp.use_gated_mlp) {
        set_error(error, "invalid DINOv3 hyperparameters");
        return false;
    }
    return true;
}

} // namespace

bool DinoV3FeaturesF32::valid(std::string * error) const {
    if (image_height <= 0 || image_width <= 0 || patch_height <= 0 || patch_width <= 0 ||
        channels <= 0 || num_register_tokens < 0) {
        if (error) *error = "invalid DINOv3 feature dimensions";
        return false;
    }
    if (!global.valid(error) || !patch_map.valid(error)) return false;
    const std::size_t prefix = static_cast<std::size_t>(1 + num_register_tokens);
    if (global.batch_size != 1 || global.channels != channels || global.tokens() != prefix ||
        global.offsets.size() != 2 || global.offsets[0] != 0 || global.offsets[1] != prefix ||
        patch_map.height != patch_height || patch_map.width != patch_width ||
        patch_map.channels != channels ||
        patch_map.features.size() != static_cast<std::size_t>(patch_height) *
                                      static_cast<std::size_t>(patch_width) *
                                      static_cast<std::size_t>(channels)) {
        if (error) *error = "DINOv3 feature tensor shape mismatch";
        return false;
    }
    return true;
}

struct DinoV3Model::Impl {
    Pixal3DPackReader reader;
    DinoV3HParams hp;
    ggml_context * weights_ctx = nullptr;
    BackendManager backend_manager;
    ggml_backend_buffer_t weights_buffer = nullptr;
    bool has_data = false;
    std::unordered_map<std::string, ggml_tensor *> tensors;

    ~Impl() { close(); }

    void close() noexcept {
        if (weights_buffer) {
            ggml_backend_buffer_free(weights_buffer);
            weights_buffer = nullptr;
        }
        backend_manager.close();
        if (weights_ctx) {
            ggml_free(weights_ctx);
            weights_ctx = nullptr;
        }
        tensors.clear();
        has_data = false;
        hp = DinoV3HParams{};
        reader.close();
    }

    ggml_tensor * weight(const std::string & local_name) const {
        const auto it = tensors.find("dino." + local_name);
        return it == tensors.end() ? nullptr : it->second;
    }
};

DinoV3Model::~DinoV3Model() {
    close();
}

void DinoV3Model::close() noexcept {
    delete impl_;
    impl_ = nullptr;
}

bool DinoV3Model::load(const std::string & path,
                       bool load_tensors,
                       std::string * error) {
    close();
    std::unique_ptr<Impl> impl(new Impl());
    if (!impl->reader.open(path, error)) return false;
    if (!impl->reader.info().has_tensor("dino.embeddings.patch_embeddings.weight")) {
        set_error(error, "GGUF pack does not contain DINOv3 tensors");
        return false;
    }
    if (!read_dino_hparams(impl->reader, impl->hp, error)) return false;

    std::size_t tensor_count = 0;
    for (const Pixal3DTensorInfo & info : impl->reader.info().tensors) {
        if (prefix_name(info.name, "dino.")) ++tensor_count;
    }
    if (tensor_count == 0 || tensor_count >
            (std::numeric_limits<std::size_t>::max() / ggml_tensor_overhead()) - 1) {
        set_error(error, "invalid DINOv3 tensor count");
        return false;
    }
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * (tensor_count + 1) + 4096;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    impl->weights_ctx = ggml_init(params);
    if (!impl->weights_ctx) {
        set_error(error, "failed to allocate DINOv3 tensor context");
        return false;
    }
    for (const Pixal3DTensorInfo & info : impl->reader.info().tensors) {
        if (!prefix_name(info.name, "dino.")) continue;
        ggml_tensor * tensor = ggml_new_tensor(impl->weights_ctx,
                                                static_cast<ggml_type>(info.ggml_type),
                                                info.n_dims, info.ne);
        if (!tensor) {
            set_error(error, "failed to create DINOv3 tensor descriptor: " + info.name);
            return false;
        }
        ggml_set_name(tensor, info.name.c_str());
        if (ggml_nbytes(tensor) != info.n_bytes) {
            set_error(error, "DINOv3 GGUF tensor byte size mismatch: " + info.name);
            return false;
        }
        impl->tensors.emplace(info.name, tensor);
    }
    if (load_tensors) {
        if (!impl->backend_manager.initialize_from_environment(
                "PIXAL3D_DINO_BACKEND", error)) {
            if (error && error->empty()) {
                set_error(error, "failed to initialize ggml backend for DINOv3");
            }
            return false;
        }
        impl->weights_buffer = ggml_backend_alloc_ctx_tensors(impl->weights_ctx,
                                                               impl->backend_manager.primary());
        if (!impl->weights_buffer) {
            set_error(error, "failed to allocate DINOv3 weights on backend " +
                             impl->backend_manager.primary_name());
            return false;
        }
        ggml_backend_buffer_set_usage(impl->weights_buffer,
                                      GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        impl->backend_manager.log_buffer(
            "weights_allocated", impl->backend_manager.primary(),
            ggml_backend_buffer_get_size(impl->weights_buffer));
        std::vector<std::uint8_t> payload;
        for (const Pixal3DTensorInfo & info : impl->reader.info().tensors) {
            if (!prefix_name(info.name, "dino.")) continue;
            if (!impl->reader.read_tensor(info.name, payload, error)) return false;
            ggml_backend_tensor_set(impl->tensors.at(info.name), payload.data(),
                                     0, payload.size());
        }
        impl->has_data = true;
    }
    impl_ = impl.release();
    return true;
}

bool DinoV3Model::is_loaded() const noexcept { return impl_ != nullptr; }
bool DinoV3Model::has_data() const noexcept { return impl_ && impl_->has_data; }
const DinoV3HParams & DinoV3Model::hparams() const noexcept {
    static const DinoV3HParams empty;
    return impl_ ? impl_->hp : empty;
}
const std::string & DinoV3Model::backend_name() const noexcept {
    static const std::string none = "none";
    return impl_ && impl_->backend_manager.initialized()
        ? impl_->backend_manager.primary_name() : none;
}
int DinoV3Model::tensor_count() const noexcept {
    return impl_ ? static_cast<int>(impl_->tensors.size()) : 0;
}
bool DinoV3Model::has_tensor(const std::string & name) const noexcept {
    return impl_ && impl_->tensors.find("dino." + name) != impl_->tensors.end();
}

bool DinoV3Model::encode(const float * pixels,
                         int height,
                         int width,
                         DinoV3FeaturesF32 & output,
                         std::string * error) const {
    output = DinoV3FeaturesF32{};
    if (!impl_) {
        set_error(error, "DINOv3 model is not loaded");
        return false;
    }
    if (!impl_->has_data) {
        set_error(error, "DINOv3 model was loaded metadata-only");
        return false;
    }
    const DinoV3HParams & hp = impl_->hp;
    if (!pixels || height <= 0 || width <= 0 || height % hp.patch_size != 0 ||
        width % hp.patch_size != 0 || height != width) {
        set_error(error, "DINOv3 input must be a non-null square CHW image divisible by patch_size");
        return false;
    }
    const std::size_t hw = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    std::size_t pixel_count = 0;
    if (!checked_product(hw, static_cast<std::size_t>(hp.num_channels), pixel_count)) {
        set_error(error, "DINOv3 input size overflows size_t");
        return false;
    }
    for (std::size_t i = 0; i < pixel_count; ++i) {
        if (!std::isfinite(pixels[i])) {
            set_error(error, "DINOv3 input contains a non-finite value");
            return false;
        }
    }
    const int patch_h = height / hp.patch_size;
    const int patch_w = width / hp.patch_size;
    const int patch_count = patch_h * patch_w;
    const int patch_dim = hp.num_channels * hp.patch_size * hp.patch_size;
    const int prefix_count = 1 + hp.num_register_tokens;
    const int token_count = prefix_count + patch_count;
    std::cerr << "pixal3d: DINO encode " << height << "x" << width
              << " (patches=" << patch_count << ")" << std::endl;
    const int C = hp.hidden_size;
    const int H = hp.num_attention_heads;
    const int hd = hp.head_dim();
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(hd));

    std::string missing;
    auto W = [&](const std::string & local_name) -> ggml_tensor * {
        ggml_tensor * tensor = impl_->weight(local_name);
        if (!tensor && missing.empty()) missing = "dino." + local_name;
        return tensor;
    };
    auto require = [&](const std::string & local_name) -> bool {
        return W(local_name) != nullptr;
    };
    if (!require("embeddings.cls_token") || !require("embeddings.patch_embeddings.weight") ||
        !require("embeddings.patch_embeddings.bias") ||
        (hp.num_register_tokens > 0 && !require("embeddings.register_tokens"))) {
        set_error(error, "missing DINOv3 tensor: " + missing);
        return false;
    }
    for (int block = 0; block < hp.num_hidden_layers; ++block) {
        const std::string p = "layer." + std::to_string(block);
        const bool common =
            require(p + ".attention.k_proj.weight") &&
            require(p + ".attention.o_proj.weight") &&
            require(p + ".attention.q_proj.weight") &&
            require(p + ".attention.v_proj.weight") &&
            require(p + ".layer_scale1.lambda1") && require(p + ".layer_scale2.lambda1") &&
            require(p + ".mlp.down_proj.weight") && require(p + ".mlp.up_proj.weight") &&
            require(p + ".norm1.weight") && require(p + ".norm1.bias") &&
            require(p + ".norm2.weight") && require(p + ".norm2.bias");
        const bool biases =
            (!hp.query_bias || require(p + ".attention.q_proj.bias")) &&
            (!hp.key_bias || require(p + ".attention.k_proj.bias")) &&
            (!hp.value_bias || require(p + ".attention.v_proj.bias")) &&
            (!hp.proj_bias || require(p + ".attention.o_proj.bias")) &&
            (!hp.mlp_bias || (require(p + ".mlp.down_proj.bias") &&
                              require(p + ".mlp.up_proj.bias")));
        if (!common || !biases) {
            set_error(error, "missing DINOv3 tensor: " + missing);
            return false;
        }
    }

    // A single graph keeps patch projection and all 24 pre-norm blocks on the
    // selected backend.  Patches are materialized as [patch_dim, P], exactly
    // the flattened Conv2d receptive fields in PyTorch's CHW order.
    std::vector<float> patch_values(static_cast<std::size_t>(patch_dim) *
                                    static_cast<std::size_t>(patch_count));
    for (int py = 0; py < patch_h; ++py) {
        for (int px = 0; px < patch_w; ++px) {
            const int patch = py * patch_w + px;
            int dst = 0;
            for (int channel = 0; channel < hp.num_channels; ++channel) {
                for (int ky = 0; ky < hp.patch_size; ++ky) {
                    for (int kx = 0; kx < hp.patch_size; ++kx) {
                        const std::size_t src = static_cast<std::size_t>(channel) * hw +
                            static_cast<std::size_t>(py * hp.patch_size + ky) *
                            static_cast<std::size_t>(width) +
                            static_cast<std::size_t>(px * hp.patch_size + kx);
                        patch_values[static_cast<std::size_t>(patch) * patch_dim + dst++] =
                            pixels[src];
                    }
                }
            }
        }
    }
    std::vector<float> cos_values(static_cast<std::size_t>(hd) * token_count, 1.0f);
    std::vector<float> sin_values(static_cast<std::size_t>(hd) * token_count, 0.0f);
    const int rope_freqs = hd / 4;
    for (int patch = 0; patch < patch_count; ++patch) {
        const int py = patch / patch_w;
        const int px = patch % patch_w;
        const float y = 2.0f * ((static_cast<float>(py) + 0.5f) / patch_h) - 1.0f;
        const float x = 2.0f * ((static_cast<float>(px) + 0.5f) / patch_w) - 1.0f;
        const int token = prefix_count + patch;
        for (int i = 0; i < rope_freqs; ++i) {
            const float inv = 1.0f / std::pow(hp.rope_theta,
                                                static_cast<float>(4 * i) / hd);
            static constexpr float pi = 3.14159265358979323846f;
            const float angles[2] = {2.0f * pi * y * inv,
                                     2.0f * pi * x * inv};
            for (int axis = 0; axis < 2; ++axis) {
                const int d = axis * rope_freqs + i;
                cos_values[static_cast<std::size_t>(token) * hd + d] = std::cos(angles[axis]);
                sin_values[static_cast<std::size_t>(token) * hd + d] = std::sin(angles[axis]);
                cos_values[static_cast<std::size_t>(token) * hd + d + 2 * rope_freqs] =
                    std::cos(angles[axis]);
                sin_values[static_cast<std::size_t>(token) * hd + d + 2 * rope_freqs] =
                    std::sin(angles[axis]);
            }
        }
    }

    const std::size_t graph_memory = ggml_tensor_overhead() * 65536 +
                                     ggml_graph_overhead_custom(65536, false);
    ggml_init_params graph_params{};
    graph_params.mem_size = graph_memory;
    graph_params.mem_buffer = nullptr;
    graph_params.no_alloc = true;
    ggml_context * ctx = ggml_init(graph_params);
    if (!ctx) {
        set_error(error, "failed to allocate DINOv3 graph context");
        return false;
    }
    ggml_tensor * patches = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, patch_dim, patch_count);
    ggml_tensor * cos_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, 1, token_count);
    ggml_tensor * sin_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, 1, token_count);
    if (!patches || !cos_t || !sin_t) {
        ggml_free(ctx);
        set_error(error, "failed to allocate DINOv3 graph inputs");
        return false;
    }
    ggml_set_input(patches);
    ggml_set_input(cos_t);
    ggml_set_input(sin_t);

    auto lin = [&](ggml_tensor * input, const std::string & name) -> ggml_tensor * {
        ggml_tensor * result = ggml_mul_mat(ctx, W(name + ".weight"), input);
        ggml_tensor * bias = W(name + ".bias");
        bool use_bias = bias != nullptr;
        if (name.find("q_proj") != std::string::npos) use_bias = hp.query_bias;
        if (name.find("k_proj") != std::string::npos) use_bias = hp.key_bias;
        if (name.find("v_proj") != std::string::npos) use_bias = hp.value_bias;
        if (name.find("o_proj") != std::string::npos) use_bias = hp.proj_bias;
        if (name.find("mlp.") != std::string::npos) use_bias = hp.mlp_bias;
        return use_bias ? ggml_add(ctx, result, bias) : result;
    };
    auto affine_norm = [&](ggml_tensor * input, const std::string & name) {
        return ggml_add(ctx,
                        ggml_mul(ctx, ggml_norm(ctx, input, hp.layer_norm_eps),
                                 W(name + ".weight")),
                        W(name + ".bias"));
    };
    auto rope = [&](ggml_tensor * input) {
        // DINO's rotate_half splits the final head dimension in two.  This
        // differs from the adjacent-pair convention used by the 3D flow.
        ggml_tensor * reshaped = ggml_reshape_4d(ctx, input, hd / 2, 2, H, token_count);
        ggml_tensor * first = ggml_cont(ctx, ggml_view_4d(
            ctx, reshaped, hd / 2, 1, H, token_count,
            reshaped->nb[1], reshaped->nb[2], reshaped->nb[3], 0));
        ggml_tensor * second = ggml_cont(ctx, ggml_view_4d(
            ctx, reshaped, hd / 2, 1, H, token_count,
            reshaped->nb[1], reshaped->nb[2], reshaped->nb[3], reshaped->nb[1]));
        ggml_tensor * rotated = ggml_concat(ctx, ggml_neg(ctx, second), first, 0);
        rotated = ggml_reshape_3d(ctx, ggml_cont(ctx, rotated), hd, H, token_count);
        return ggml_add(ctx, ggml_mul(ctx, input, cos_t),
                        ggml_mul(ctx, rotated, sin_t));
    };
    auto sdpa = [&](ggml_tensor * q, ggml_tensor * k, ggml_tensor * v) {
        ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        ggml_tensor * scores = ggml_mul_mat(ctx, kp, qp);
        scores = ggml_soft_max_ext(ctx, scores, nullptr, attention_scale, 0.0f);
        ggml_tensor * values = ggml_cont(ctx, ggml_permute(ctx, vp, 1, 0, 2, 3));
        ggml_tensor * result = ggml_mul_mat(ctx, values, scores);
        result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3));
        return ggml_reshape_2d(ctx, result, C, token_count);
    };

    ggml_tensor * patch_weight = ggml_reshape_2d(
        ctx, W("embeddings.patch_embeddings.weight"), patch_dim, C);
    ggml_tensor * h = ggml_add(ctx,
        ggml_mul_mat(ctx, patch_weight, patches),
        W("embeddings.patch_embeddings.bias"));
    ggml_tensor * cls = ggml_reshape_2d(ctx, W("embeddings.cls_token"), C, 1);
    ggml_tensor * prefix = cls;
    if (hp.num_register_tokens > 0) {
        ggml_tensor * registers = ggml_reshape_2d(
            ctx, W("embeddings.register_tokens"), C, hp.num_register_tokens);
        prefix = ggml_concat(ctx, cls, registers, 1);
    }
    h = ggml_concat(ctx, prefix, h, 1);

    for (int block = 0; block < hp.num_hidden_layers; ++block) {
        const std::string p = "layer." + std::to_string(block);
        ggml_tensor * residual = h;
        ggml_tensor * n1 = affine_norm(h, p + ".norm1");
        ggml_tensor * q = ggml_reshape_3d(ctx, lin(n1, p + ".attention.q_proj"), hd, H, token_count);
        ggml_tensor * k = ggml_reshape_3d(ctx, lin(n1, p + ".attention.k_proj"), hd, H, token_count);
        ggml_tensor * v = ggml_reshape_3d(ctx, lin(n1, p + ".attention.v_proj"), hd, H, token_count);
        q = rope(q);
        k = rope(k);
        ggml_tensor * attn = lin(sdpa(q, k, v), p + ".attention.o_proj");
        attn = ggml_mul(ctx, attn, W(p + ".layer_scale1.lambda1"));
        h = ggml_add(ctx, residual, attn);

        residual = h;
        ggml_tensor * n2 = affine_norm(h, p + ".norm2");
        ggml_tensor * mlp = lin(n2, p + ".mlp.up_proj");
        mlp = ggml_gelu_erf(ctx, mlp);
        mlp = lin(mlp, p + ".mlp.down_proj");
        mlp = ggml_mul(ctx, mlp, W(p + ".layer_scale2.lambda1"));
        h = ggml_add(ctx, residual, mlp);
    }
    ggml_tensor * result = ggml_cont(ctx, ggml_norm(ctx, h, hp.layer_norm_eps));
    ggml_set_output(result);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 65536, false);
    if (!graph) {
        ggml_free(ctx);
        set_error(error, "failed to allocate DINOv3 graph");
        return false;
    }
    ggml_build_forward_expand(graph, result);
    std::cerr << "pixal3d: DINO graph built (nodes=" << ggml_graph_n_nodes(graph)
              << ")" << std::endl;
    std::string scheduler_error;
    BackendScheduler scheduler(impl_->backend_manager, 65536, false, true,
                                &scheduler_error, "DINOv3");
    if (!scheduler.valid() ||
        (impl_->backend_manager.requires_primary_backend() &&
         !scheduler.require_primary_graph(graph, &scheduler_error)) ||
        !scheduler.allocate_graph(graph, &scheduler_error)) {
        ggml_free(ctx);
        set_error(error, "failed to allocate DINOv3 compute graph: " + scheduler_error);
        return false;
    }
    std::cerr << "pixal3d: DINO graph allocated" << std::endl;
    impl_->backend_manager.set_n_threads(4);
    ggml_backend_tensor_set(patches, patch_values.data(), 0,
                            patch_values.size() * sizeof(float));
    ggml_backend_tensor_set(cos_t, cos_values.data(), 0,
                            cos_values.size() * sizeof(float));
    ggml_backend_tensor_set(sin_t, sin_values.data(), 0,
                            sin_values.size() * sizeof(float));
    std::cerr << "pixal3d: DINO graph compute start" << std::endl;
    const ggml_status status = scheduler.compute(graph, &scheduler_error);
    std::cerr << "pixal3d: DINO graph compute complete" << std::endl;
    std::vector<float> raw(static_cast<std::size_t>(C) * token_count);
    bool ok = status == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_tensor_get(result, raw.data(), 0, raw.size() * sizeof(float));
    } else {
        set_error(error, "DINOv3 graph compute failed: " + scheduler_error);
    }
    scheduler.synchronize();
    ggml_free(ctx);
    if (!ok) return false;

    output.image_height = height;
    output.image_width = width;
    output.patch_height = patch_h;
    output.patch_width = patch_w;
    output.channels = C;
    output.num_register_tokens = hp.num_register_tokens;
    output.global.batch_size = 1;
    output.global.channels = C;
    output.global.offsets = {0, static_cast<std::size_t>(prefix_count)};
    output.global.feats.resize(static_cast<std::size_t>(prefix_count) * C);
    for (int token = 0; token < prefix_count; ++token) {
        for (int channel = 0; channel < C; ++channel) {
            output.global.feats[static_cast<std::size_t>(token) * C + channel] =
                raw[static_cast<std::size_t>(token) * C + channel];
        }
    }
    output.patch_map.image_resolution = height;
    output.patch_map.height = patch_h;
    output.patch_map.width = patch_w;
    output.patch_map.channels = C;
    output.patch_map.features.resize(static_cast<std::size_t>(patch_count) * C);
    for (int patch = 0; patch < patch_count; ++patch) {
        const int token = prefix_count + patch;
        for (int channel = 0; channel < C; ++channel) {
            output.patch_map.features[static_cast<std::size_t>(patch) * C + channel] =
                raw[static_cast<std::size_t>(token) * C + channel];
        }
    }
    if (!output.valid(error)) {
        output = DinoV3FeaturesF32{};
        return false;
    }
    return true;
}

} // namespace pixal3d
