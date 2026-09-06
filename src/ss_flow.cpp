#include "pixal3d/ss_flow.h"

#include "pixal3d/backend.h"
#include "pixal3d/flow.h"
#include "pixal3d/pack.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pixal3d {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
}

bool checked_cube(int resolution, std::size_t & points, std::string * error) {
    if (resolution <= 0) {
        set_error(error, "resolution must be positive");
        return false;
    }
    const std::size_t r = static_cast<std::size_t>(resolution);
    if (r > std::numeric_limits<std::size_t>::max() / r ||
        r * r > std::numeric_limits<std::size_t>::max() / r) {
        set_error(error, "resolution^3 overflows size_t");
        return false;
    }
    points = r * r * r;
    return true;
}

bool validate_full_grid(const SparseTensorF32 & tensor,
                        int resolution,
                        int channels,
                        std::string * error) {
    if (!tensor.valid(error)) return false;
    std::size_t points = 0;
    if (!checked_cube(resolution, points, error)) return false;
    if (tensor.batch_size != 1 || tensor.channels != channels ||
        tensor.spatial_x != resolution || tensor.spatial_y != resolution ||
        tensor.spatial_z != resolution || tensor.points() != points) {
        set_error(error, "SS-flow sampler requires one complete resolution grid");
        return false;
    }
    for (std::size_t index = 0; index < points; ++index) {
        const std::int32_t x = static_cast<std::int32_t>(
            index / static_cast<std::size_t>(resolution * resolution));
        const std::int32_t y = static_cast<std::int32_t>(
            (index / static_cast<std::size_t>(resolution)) %
            static_cast<std::size_t>(resolution));
        const std::int32_t z = static_cast<std::int32_t>(
            index % static_cast<std::size_t>(resolution));
        const std::size_t base = index * 4;
        if (tensor.coords[base + 0] != 0 || tensor.coords[base + 1] != x ||
            tensor.coords[base + 2] != y || tensor.coords[base + 3] != z) {
            set_error(error, "SS-flow sampler grid coordinates are not lexicographic");
            return false;
        }
    }
    return true;
}

void sparse_to_channel_major(const SparseTensorF32 & input,
                             std::vector<float> & output) {
    const std::size_t points = input.points();
    output.resize(points * static_cast<std::size_t>(input.channels));
    for (std::size_t point = 0; point < points; ++point) {
        for (int channel = 0; channel < input.channels; ++channel) {
            output[static_cast<std::size_t>(channel) * points + point] =
                input.feats[point * static_cast<std::size_t>(input.channels) +
                            static_cast<std::size_t>(channel)];
        }
    }
}

void channel_major_to_sparse(const SparseTensorF32 & state,
                             const std::vector<float> & channel_major,
                             int channels,
                             SparseTensorF32 & output) {
    output = state;
    output.channels = channels;
    const std::size_t points = state.points();
    output.feats.resize(points * static_cast<std::size_t>(channels));
    for (std::size_t point = 0; point < points; ++point) {
        for (int channel = 0; channel < channels; ++channel) {
            output.feats[point * static_cast<std::size_t>(channels) +
                         static_cast<std::size_t>(channel)] =
                channel_major[static_cast<std::size_t>(channel) * points + point];
        }
    }
}

VarLenTensorF32 zero_varlen_like(const VarLenTensorF32 & input) {
    VarLenTensorF32 output = input;
    std::fill(output.feats.begin(), output.feats.end(), 0.0f);
    return output;
}

SparseTensorF32 zero_sparse_like(const SparseTensorF32 & input) {
    SparseTensorF32 output = input;
    std::fill(output.feats.begin(), output.feats.end(), 0.0f);
    return output;
}

bool prefix_name(const std::string & name, const char * prefix) {
    const std::size_t length = std::strlen(prefix);
    return name.size() >= length && name.compare(0, length, prefix) == 0;
}

bool read_flow_hparams(const Pixal3DPackReader & reader,
                       SSFlowHParams & hp,
                       std::string * error) {
    const std::string prefix = "pixal3d.ss_flow.";
    std::uint32_t value = 0;
    if (!reader.metadata_u32(prefix + "resolution", value, true, error)) return false;
    hp.resolution = static_cast<int>(value);
    if (!reader.metadata_u32(prefix + "in_channels", value, true, error)) return false;
    hp.in_channels = static_cast<int>(value);
    if (!reader.metadata_u32(prefix + "out_channels", value, true, error)) return false;
    hp.out_channels = static_cast<int>(value);
    if (!reader.metadata_u32(prefix + "model_channels", value, true, error)) return false;
    hp.model_channels = static_cast<int>(value);
    if (!reader.metadata_u32(prefix + "cond_channels", value, true, error)) return false;
    hp.cond_channels = static_cast<int>(value);
    if (!reader.metadata_u32(prefix + "num_blocks", value, true, error)) return false;
    hp.num_blocks = static_cast<int>(value);
    if (!reader.metadata_u32(prefix + "num_heads", value, true, error)) return false;
    hp.num_heads = static_cast<int>(value);
    if (!reader.metadata_f32(prefix + "mlp_ratio", hp.mlp_ratio, true, error)) return false;
    if (!reader.metadata_string(prefix + "pe_mode", hp.pe_mode, false, error)) return false;
    if (hp.pe_mode.empty()) hp.pe_mode = "rope";
    if (!reader.metadata_bool(prefix + "share_mod", hp.share_mod, false, error)) return false;
    if (!reader.metadata_bool(prefix + "qk_rms_norm", hp.qk_rms_norm, false, error)) return false;
    if (!reader.metadata_bool(prefix + "qk_rms_norm_cross", hp.qk_rms_norm_cross, false, error)) return false;
    if (!reader.metadata_f32(prefix + "rope_freq_min", hp.rope_freq_min, false, error)) return false;
    if (!reader.metadata_f32(prefix + "rope_freq_base", hp.rope_freq_base, false, error)) return false;
    if (!reader.metadata_string(prefix + "image_attn_mode", hp.image_attn_mode, false, error)) return false;
    if (hp.image_attn_mode.empty()) hp.image_attn_mode = "cross";
    if (!reader.metadata_u32(prefix + "proj_in_channels", value, false, error)) return false;
    hp.proj_in_channels = value == 0 ? hp.cond_channels : static_cast<int>(value);

    std::size_t points = 0;
    if (!checked_cube(hp.resolution, points, error) || hp.in_channels <= 0 ||
        hp.out_channels <= 0 || hp.model_channels <= 0 || hp.cond_channels <= 0 ||
        hp.num_blocks <= 0 || hp.num_heads <= 0 || hp.model_channels % hp.num_heads != 0 ||
        hp.head_dim() % 2 != 0 || hp.mlp_ratio <= 0.0f ||
        !std::isfinite(hp.mlp_ratio) || hp.rope_freq_min <= 0.0f ||
        hp.rope_freq_base <= 0.0f) {
        set_error(error, "invalid SS-flow hyperparameters");
        return false;
    }
    if (hp.pe_mode != "rope") {
        set_error(error, "only pe_mode=rope is currently supported");
        return false;
    }
    if (!hp.share_mod) {
        set_error(error, "only share_mod=true is currently supported");
        return false;
    }
    if (hp.image_attn_mode != "cross" && hp.image_attn_mode != "proj") {
        set_error(error, "SS-flow supports image_attn_mode cross or proj");
        return false;
    }
    return true;
}

} // namespace

struct SSFlowModel::Impl {
    Pixal3DPackReader reader;
    SSFlowHParams hp;
    ggml_context * weights_ctx = nullptr;
    BackendManager backend_manager;
    ggml_backend_buffer_t weights_buffer = nullptr;
    std::string backend_name;
    bool has_data = false;
    std::unordered_map<std::string, ggml_tensor *> tensors;

    ~Impl() {
        close();
    }

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
        backend_name.clear();
        has_data = false;
        hp = SSFlowHParams{};
        reader.close();
    }

    ggml_tensor * weight(const std::string & local_name) const {
        const auto it = tensors.find("ss." + local_name);
        return it == tensors.end() ? nullptr : it->second;
    }
};

SSFlowModel::~SSFlowModel() {
    delete impl_;
    impl_ = nullptr;
}

void SSFlowModel::close() noexcept {
    if (impl_) {
        delete impl_;
        impl_ = nullptr;
    }
}

bool SSFlowModel::load(const std::string & path,
                       bool load_tensors,
                       std::string * error) {
    close();
    std::unique_ptr<Impl> impl(new Impl());
    if (!impl->reader.open(path, error)) {
        return false;
    }
    if (!impl->reader.info().has_tensor("ss.input_layer.weight")) {
        set_error(error, "GGUF pack does not contain ss-flow tensors");
        return false;
    }
    if (!read_flow_hparams(impl->reader, impl->hp, error)) {
        return false;
    }

    std::size_t tensor_count = 0;
    for (const Pixal3DTensorInfo & info : impl->reader.info().tensors) {
        if (prefix_name(info.name, "ss.")) {
            ++tensor_count;
        }
    }
    if (tensor_count == 0 || tensor_count > std::numeric_limits<std::size_t>::max() /
                                         ggml_tensor_overhead()) {
        set_error(error, "invalid ss-flow tensor count");
        return false;
    }
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * (tensor_count + 1) + 4096;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    impl->weights_ctx = ggml_init(params);
    if (!impl->weights_ctx) {
        set_error(error, "failed to allocate SS-flow tensor context");
        return false;
    }
    for (const Pixal3DTensorInfo & info : impl->reader.info().tensors) {
        if (!prefix_name(info.name, "ss.")) {
            continue;
        }
        const ggml_type type = static_cast<ggml_type>(info.ggml_type);
        ggml_tensor * tensor = ggml_new_tensor(impl->weights_ctx, type,
                                                info.n_dims, info.ne);
        if (!tensor) {
            set_error(error, "failed to create tensor descriptor: " + info.name);
            return false;
        }
        ggml_set_name(tensor, info.name.c_str());
        if (ggml_nbytes(tensor) != info.n_bytes) {
            set_error(error, "GGUF tensor byte size mismatch: " + info.name);
            return false;
        }
        impl->tensors.emplace(info.name, tensor);
    }

    if (load_tensors) {
        if (!impl->backend_manager.initialize_from_environment(
                "PIXAL3D_SS_FLOW_BACKEND", error)) {
            return false;
        }
        impl->backend_name = impl->backend_manager.primary_name();
        impl->weights_buffer = ggml_backend_alloc_ctx_tensors(
            impl->weights_ctx, impl->backend_manager.primary());
        if (!impl->weights_buffer) {
            set_error(error, "failed to allocate SS-flow weights on backend " +
                             impl->backend_name);
            return false;
        }
        ggml_backend_buffer_set_usage(impl->weights_buffer,
                                      GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        impl->backend_manager.log_buffer(
            "weights_allocated", impl->backend_manager.primary(),
            ggml_backend_buffer_get_size(impl->weights_buffer));
        std::vector<std::uint8_t> payload;
        for (const Pixal3DTensorInfo & info : impl->reader.info().tensors) {
            if (!prefix_name(info.name, "ss.")) {
                continue;
            }
            if (!impl->reader.read_tensor(info.name, payload, error)) {
                return false;
            }
            ggml_backend_tensor_set(impl->tensors.at(info.name), payload.data(),
                                     0, payload.size());
        }
        impl->has_data = true;
    }
    impl_ = impl.release();
    return true;
}

bool SSFlowModel::is_loaded() const noexcept {
    return impl_ != nullptr;
}

bool SSFlowModel::has_data() const noexcept {
    return impl_ && impl_->has_data;
}

const SSFlowHParams & SSFlowModel::hparams() const noexcept {
    static const SSFlowHParams empty;
    return impl_ ? impl_->hp : empty;
}

const std::string & SSFlowModel::backend_name() const noexcept {
    static const std::string none = "none";
    return impl_ && !impl_->backend_name.empty() ? impl_->backend_name : none;
}

int SSFlowModel::tensor_count() const noexcept {
    return impl_ ? static_cast<int>(impl_->tensors.size()) : 0;
}

bool SSFlowModel::has_tensor(const std::string & name) const noexcept {
    return impl_ && impl_->tensors.find(name) != impl_->tensors.end();
}

bool SSFlowModel::forward(const float * x,
                          float timestep,
                          const float * cond,
                          int cond_tokens,
                          int cond_channels,
                          const float * projected,
                          int projected_points,
                          int projected_channels,
                          float * out,
                          std::string * error) {
    if (!impl_) {
        set_error(error, "SS-flow model is not loaded");
        return false;
    }
    if (!impl_->has_data) {
        set_error(error, "SS-flow model was loaded metadata-only");
        return false;
    }
    const SSFlowHParams & hp = impl_->hp;
    std::size_t points = 0;
    if (!checked_cube(hp.resolution, points, error)) return false;
    if (!x || !out || !cond || cond_tokens <= 0 || cond_channels != hp.cond_channels) {
        set_error(error, "invalid SS-flow input or conditioning buffer");
        return false;
    }
    if (hp.image_attn_mode == "proj") {
        if (!projected || projected_points != static_cast<int>(points) ||
            projected_channels != hp.proj_in_channels) {
            set_error(error, "projected conditioning shape mismatch");
            return false;
        }
    }
    if (!std::isfinite(timestep)) {
        set_error(error, "timestep must be finite");
        return false;
    }
    const int C = hp.model_channels;
    const int H = hp.num_heads;
    const int hd = hp.head_dim();
    const int N = static_cast<int>(points);
    const int Lkv = cond_tokens;
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(hd));

    std::string missing;
    auto W = [&](const std::string & local_name) -> ggml_tensor * {
        ggml_tensor * tensor = impl_->weight(local_name);
        if (!tensor && missing.empty()) {
            missing = "ss." + local_name;
        }
        return tensor;
    };
    auto require_weight = [&](const std::string & local_name) -> bool {
        return W(local_name) != nullptr;
    };
    if (!require_weight("input_layer.weight") || !require_weight("input_layer.bias") ||
        !require_weight("out_layer.weight") || !require_weight("out_layer.bias") ||
        !require_weight("t_embedder.mlp.0.weight") ||
        !require_weight("t_embedder.mlp.0.bias") ||
        !require_weight("t_embedder.mlp.2.weight") ||
        !require_weight("t_embedder.mlp.2.bias") ||
        !require_weight("adaLN_modulation.1.weight") ||
        !require_weight("adaLN_modulation.1.bias")) {
        set_error(error, "missing SS-flow tensor: " + missing);
        return false;
    }
    for (int block = 0; block < hp.num_blocks; ++block) {
        const std::string p = "blocks." + std::to_string(block);
        const bool common =
            require_weight(p + ".modulation") &&
            require_weight(p + ".norm2.weight") && require_weight(p + ".norm2.bias") &&
            require_weight(p + ".self_attn.to_qkv.weight") &&
            require_weight(p + ".self_attn.to_qkv.bias") &&
            require_weight(p + ".self_attn.to_out.weight") &&
            require_weight(p + ".self_attn.to_out.bias") &&
            require_weight(p + ".mlp.mlp.0.weight") &&
            require_weight(p + ".mlp.mlp.0.bias") &&
            require_weight(p + ".mlp.mlp.2.weight") &&
            require_weight(p + ".mlp.mlp.2.bias");
        const bool self_norm = !hp.qk_rms_norm ||
            (require_weight(p + ".self_attn.q_rms_norm.gamma") &&
             require_weight(p + ".self_attn.k_rms_norm.gamma"));
        const std::string cross = p + ".cross_attn.cross_attn_block";
        const bool cross_norm = !hp.qk_rms_norm_cross ||
            (require_weight(cross + ".q_rms_norm.gamma") &&
             require_weight(cross + ".k_rms_norm.gamma"));
        const bool cross_weights =
            require_weight(cross + ".to_q.weight") && require_weight(cross + ".to_q.bias") &&
            require_weight(cross + ".to_kv.weight") && require_weight(cross + ".to_kv.bias") &&
            require_weight(cross + ".to_out.weight") && require_weight(cross + ".to_out.bias");
        const bool proj_weights = hp.image_attn_mode != "proj" ||
            (require_weight(p + ".cross_attn.proj_linear.weight") &&
             require_weight(p + ".cross_attn.proj_linear.bias"));
        if (!common || !self_norm || !cross_norm || !cross_weights || !proj_weights) {
            set_error(error, "missing SS-flow tensor: " + missing);
            return false;
        }
    }

    const std::size_t graph_memory = ggml_tensor_overhead() * 32768 +
                                     ggml_graph_overhead_custom(32768, false);
    ggml_init_params graph_params{};
    graph_params.mem_size = graph_memory;
    graph_params.mem_buffer = nullptr;
    graph_params.no_alloc = true;
    ggml_context * ctx = ggml_init(graph_params);
    if (!ctx) {
        set_error(error, "failed to allocate SS-flow graph context");
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32768, false);
    if (!graph) {
        ggml_free(ctx);
        set_error(error, "failed to allocate SS-flow graph");
        return false;
    }

    ggml_tensor * x_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, hp.in_channels);
    ggml_tensor * temb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    ggml_tensor * cos_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, 1, N);
    ggml_tensor * sin_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, 1, N);
    ggml_tensor * cnd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.cond_channels, Lkv);
    ggml_tensor * proj = hp.image_attn_mode == "proj"
        ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.proj_in_channels, N) : nullptr;
    if (!x_t || !temb || !cos_t || !sin_t || !cnd ||
        (hp.image_attn_mode == "proj" && !proj)) {
        ggml_free(ctx);
        set_error(error, "failed to allocate SS-flow graph inputs");
        return false;
    }
    ggml_set_input(x_t);
    ggml_set_input(temb);
    ggml_set_input(cos_t);
    ggml_set_input(sin_t);
    ggml_set_input(cnd);
    if (proj) ggml_set_input(proj);

    auto lin = [&](ggml_tensor * input, const std::string & local_name) {
        ggml_tensor * result = ggml_mul_mat(ctx, W(local_name + ".weight"), input);
        ggml_tensor * bias = W(local_name + ".bias");
        return bias ? ggml_add(ctx, result, bias) : result;
    };
    auto modulate = [&](ggml_tensor * input, ggml_tensor * scale,
                        ggml_tensor * shift) {
        return ggml_add(ctx, ggml_add(ctx, ggml_mul(ctx, input, scale), input), shift);
    };
    auto qk_norm = [&](ggml_tensor * input, const std::string & gamma_name) {
        return ggml_mul(ctx, ggml_rms_norm(ctx, input, 1e-12f), W(gamma_name));
    };
    auto rope = [&](ggml_tensor * input) {
        ggml_tensor * reshaped = ggml_reshape_4d(ctx, input, 2, hd / 2, H, N);
        ggml_tensor * even = ggml_cont(ctx, ggml_view_4d(
            ctx, reshaped, 1, hd / 2, H, N, reshaped->nb[1], reshaped->nb[2],
            reshaped->nb[3], 0));
        ggml_tensor * odd = ggml_cont(ctx, ggml_view_4d(
            ctx, reshaped, 1, hd / 2, H, N, reshaped->nb[1], reshaped->nb[2],
            reshaped->nb[3], reshaped->nb[0]));
        ggml_tensor * swapped = ggml_concat(ctx, ggml_neg(ctx, odd), even, 0);
        swapped = ggml_reshape_3d(ctx, swapped, hd, H, N);
        return ggml_add(ctx, ggml_mul(ctx, input, cos_t),
                        ggml_mul(ctx, swapped, sin_t));
    };
    auto sdpa = [&](ggml_tensor * q, ggml_tensor * k, ggml_tensor * v) {
        ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        // The SS grid has 4096 queries.  Materializing the F32 score matrix
        // here costs about 0.8 GiB for 12 heads and keeps this development
        // path numerically identical to the released Python/trellis2cpp
        // reference.  The larger SLat grids use the streaming path in
        // slat_flow.cpp because their dense score matrix is not viable.
        if (qp->ne[1] <= 4096 && kp->ne[1] <= 4096) {
            ggml_tensor * scores = ggml_mul_mat(ctx, kp, qp);
            scores = ggml_soft_max_ext(ctx, scores, nullptr,
                                       attention_scale, 0.0f);
            ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, vp, 1, 0, 2, 3));
            ggml_tensor * result = ggml_mul_mat(ctx, vt, scores);
            result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3));
            return ggml_reshape_2d(ctx, result, C, result->ne[2]);
        }
        ggml_tensor * result = ggml_flash_attn_ext(
            ctx, qp, kp, vp, nullptr, attention_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(result, GGML_PREC_F32);
        return ggml_reshape_2d(ctx, ggml_cont(ctx, result), C, result->ne[2]);
    };
    auto split_heads = [&](ggml_tensor * value, int offset, int sequence) {
        const std::size_t element_size = ggml_element_size(value);
        ggml_tensor * view = ggml_view_2d(ctx, value, C, sequence, value->nb[1],
                                           static_cast<std::size_t>(offset) * C * element_size);
        return ggml_reshape_3d(ctx, ggml_cont(ctx, view), hd, H, sequence);
    };

    ggml_tensor * h = ggml_cont(ctx, ggml_transpose(ctx, x_t));
    h = lin(h, "input_layer");
    std::vector<float> embedding;
    if (!timestep_embedding(&timestep, 1, 256, 10000, embedding, error)) {
        ggml_free(ctx);
        return false;
    }
    std::vector<float> cos_values(static_cast<std::size_t>(hd) * points, 1.0f);
    std::vector<float> sin_values(static_cast<std::size_t>(hd) * points, 0.0f);
    const int rope_freq_dim = hd / 2 / 3;
    std::vector<float> frequencies(static_cast<std::size_t>(std::max(rope_freq_dim, 0)));
    for (int index = 0; index < rope_freq_dim; ++index) {
        frequencies[static_cast<std::size_t>(index)] = hp.rope_freq_min /
            std::pow(hp.rope_freq_base, static_cast<float>(index) /
                     static_cast<float>(rope_freq_dim));
    }
    const int rope_pairs = hd / 2;
    for (std::size_t token = 0; token < points; ++token) {
        const int coordinate[3] = {
            static_cast<int>(token / static_cast<std::size_t>(hp.resolution * hp.resolution)),
            static_cast<int>((token / static_cast<std::size_t>(hp.resolution)) %
                             static_cast<std::size_t>(hp.resolution)),
            static_cast<int>(token % static_cast<std::size_t>(hp.resolution)),
        };
        for (int pair = 0; pair < rope_pairs; ++pair) {
            float phase = 0.0f;
            if (pair < 3 * rope_freq_dim) {
                phase = static_cast<float>(coordinate[pair / rope_freq_dim]) *
                        frequencies[static_cast<std::size_t>(pair % rope_freq_dim)];
            }
            const std::size_t base = token * static_cast<std::size_t>(hd) +
                                     static_cast<std::size_t>(2 * pair);
            cos_values[base] = cos_values[base + 1] = std::cos(phase);
            sin_values[base] = sin_values[base + 1] = std::sin(phase);
        }
    }

    ggml_tensor * te = lin(temb, "t_embedder.mlp.0");
    te = ggml_silu(ctx, te);
    te = lin(te, "t_embedder.mlp.2");
    ggml_tensor * tmod = lin(ggml_silu(ctx, te), "adaLN_modulation.1");
    ggml_tensor * global_cond = cnd;
    ggml_tensor * projected_cond = proj;

    for (int block = 0; block < hp.num_blocks; ++block) {
        const std::string p = "blocks." + std::to_string(block);
        ggml_tensor * mods = ggml_add(ctx, W(p + ".modulation"), tmod);
        auto chunk = [&](int index) {
            return ggml_view_1d(ctx, mods, C,
                                static_cast<std::size_t>(index) * C *
                                ggml_element_size(mods));
        };
        ggml_tensor * shift_msa = chunk(0);
        ggml_tensor * scale_msa = chunk(1);
        ggml_tensor * gate_msa = chunk(2);
        ggml_tensor * shift_mlp = chunk(3);
        ggml_tensor * scale_mlp = chunk(4);
        ggml_tensor * gate_mlp = chunk(5);

        ggml_tensor * normalized = modulate(ggml_norm(ctx, h, 1e-6f),
                                            scale_msa, shift_msa);
        ggml_tensor * qkv = lin(normalized, p + ".self_attn.to_qkv");
        ggml_tensor * q = split_heads(qkv, 0, N);
        ggml_tensor * k = split_heads(qkv, 1, N);
        ggml_tensor * v = split_heads(qkv, 2, N);
        if (hp.qk_rms_norm) {
            q = qk_norm(q, p + ".self_attn.q_rms_norm.gamma");
            k = qk_norm(k, p + ".self_attn.k_rms_norm.gamma");
        }
        q = rope(q);
        k = rope(k);
        ggml_tensor * self_out = lin(sdpa(q, k, v), p + ".self_attn.to_out");
        h = ggml_add(ctx, h, ggml_mul(ctx, self_out, gate_msa));

        ggml_tensor * cross_input = ggml_norm(ctx, h, 1e-6f);
        cross_input = ggml_add(ctx,
                               ggml_mul(ctx, cross_input, W(p + ".norm2.weight")),
                               W(p + ".norm2.bias"));
        const std::string cross = p + ".cross_attn.cross_attn_block";
        ggml_tensor * cross_q = split_heads(lin(cross_input, cross + ".to_q"), 0, N);
        ggml_tensor * cross_kv = lin(global_cond, cross + ".to_kv");
        ggml_tensor * cross_k = split_heads(cross_kv, 0, Lkv);
        ggml_tensor * cross_v = split_heads(cross_kv, 1, Lkv);
        if (hp.qk_rms_norm_cross) {
            cross_q = qk_norm(cross_q, cross + ".q_rms_norm.gamma");
            cross_k = qk_norm(cross_k, cross + ".k_rms_norm.gamma");
        }
        ggml_tensor * cross_out = lin(sdpa(cross_q, cross_k, cross_v),
                                      cross + ".to_out");
        if (hp.image_attn_mode == "proj") {
            ggml_tensor * projected_out = lin(projected_cond,
                                              p + ".cross_attn.proj_linear");
            cross_out = ggml_add(ctx, cross_out, projected_out);
        }
        h = ggml_add(ctx, h, cross_out);

        ggml_tensor * mlp_input = modulate(ggml_norm(ctx, h, 1e-6f),
                                           scale_mlp, shift_mlp);
        mlp_input = lin(mlp_input, p + ".mlp.mlp.0");
        mlp_input = ggml_gelu(ctx, mlp_input);
        mlp_input = lin(mlp_input, p + ".mlp.mlp.2");
        h = ggml_add(ctx, h, ggml_mul(ctx, mlp_input, gate_mlp));
    }

    h = ggml_norm(ctx, h, 1e-5f);
    h = lin(h, "out_layer");
    ggml_tensor * result = ggml_cont(ctx, ggml_transpose(ctx, h));
    ggml_set_output(result);
    if (!missing.empty()) {
        set_error(error, "missing SS-flow tensor: " + missing);
        ggml_free(ctx);
        return false;
    }
    ggml_build_forward_expand(graph, result);
    std::string scheduler_error;
    BackendScheduler scheduler(impl_->backend_manager, 32768, false, true,
                                &scheduler_error, "SS-flow");
    if (!scheduler.valid() ||
        (impl_->backend_manager.requires_primary_backend() &&
         !scheduler.require_primary_graph(graph, &scheduler_error)) ||
        !scheduler.allocate_graph(graph, &scheduler_error)) {
        ggml_free(ctx);
        set_error(error, "failed to allocate SS-flow compute graph: " + scheduler_error);
        return false;
    }
    impl_->backend_manager.set_n_threads(4);
    ggml_backend_tensor_set(x_t, x, 0,
                            static_cast<std::size_t>(hp.in_channels) * points * sizeof(float));
    ggml_backend_tensor_set(temb, embedding.data(), 0, embedding.size() * sizeof(float));
    ggml_backend_tensor_set(cos_t, cos_values.data(), 0,
                            cos_values.size() * sizeof(float));
    ggml_backend_tensor_set(sin_t, sin_values.data(), 0,
                            sin_values.size() * sizeof(float));
    ggml_backend_tensor_set(cnd, cond, 0,
                            static_cast<std::size_t>(cond_channels) *
                            static_cast<std::size_t>(cond_tokens) * sizeof(float));
    if (proj) {
        ggml_backend_tensor_set(proj, projected, 0,
                                static_cast<std::size_t>(projected_channels) * points *
                                sizeof(float));
    }
    const ggml_status status = scheduler.compute(graph, &scheduler_error);
    bool ok = status == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_tensor_get(result, out, 0,
                                static_cast<std::size_t>(hp.out_channels) * points *
                                sizeof(float));
    } else {
        set_error(error, "SS-flow graph compute failed: " + scheduler_error);
    }
    ggml_free(ctx);
    return ok;
}

bool SSFlowModel::sample(const SparseTensorF32 & noise,
                         const FlowEulerSamplerConfig & sampler_config,
                         const VarLenTensorF32 & global_context,
                         const SparseTensorF32 * projection_context,
                         FlowEulerSampleF32 & output,
                         std::string * error) {
    output = FlowEulerSampleF32{};
    if (!impl_) {
        set_error(error, "SS-flow model is not loaded");
        return false;
    }
    if (!impl_->has_data) {
        set_error(error, "SS-flow model was loaded metadata-only");
        return false;
    }
    const SSFlowHParams & hp = impl_->hp;
    if (hp.in_channels != hp.out_channels) {
        set_error(error, "SS-flow sampler requires equal input and output channels");
        return false;
    }
    if (!validate_full_grid(noise, hp.resolution, hp.in_channels, error)) {
        return false;
    }
    if (!global_context.valid(error) || global_context.batch_size != 1 ||
        global_context.channels != hp.cond_channels || global_context.tokens() == 0) {
        set_error(error, "SS-flow global conditioning shape mismatch");
        return false;
    }

    const bool use_projection = hp.image_attn_mode == "proj";
    if (use_projection) {
        if (!projection_context ||
            !validate_full_grid(*projection_context, hp.resolution,
                                hp.proj_in_channels, error) ||
            projection_context->coords != noise.coords) {
            set_error(error, "SS-flow projection conditioning shape mismatch");
            return false;
        }
    }

    const VarLenTensorF32 negative_global = zero_varlen_like(global_context);
    const SparseTensorF32 negative_projection = use_projection
        ? zero_sparse_like(*projection_context) : SparseTensorF32{};
    const std::size_t points = noise.points();
    const int cond_tokens = static_cast<int>(global_context.tokens());
    std::vector<float> channel_major_input;
    std::vector<float> channel_major_output;
    FlowVelocityFn callback = [&](const SparseTensorF32 & state,
                                  float model_timestep,
                                  bool conditional,
                                  SparseTensorF32 & velocity,
                                  std::string * callback_error) {
        if (!validate_full_grid(state, hp.resolution, hp.in_channels,
                                callback_error)) {
            return false;
        }
        sparse_to_channel_major(state, channel_major_input);
        channel_major_output.assign(
            points * static_cast<std::size_t>(hp.out_channels), 0.0f);
        const VarLenTensorF32 & context = conditional ? global_context : negative_global;
        const SparseTensorF32 * projected = nullptr;
        if (use_projection) {
            projected = conditional ? projection_context : &negative_projection;
        }
        const float * projected_data = projected ? projected->feats.data() : nullptr;
        const int projected_points = projected ? static_cast<int>(points) : 0;
        const int projected_channels = projected ? hp.proj_in_channels : 0;
        if (!forward(channel_major_input.data(), model_timestep,
                     context.feats.data(), cond_tokens, context.channels,
                     projected_data, projected_points, projected_channels,
                     channel_major_output.data(), callback_error)) {
            return false;
        }
        channel_major_to_sparse(state, channel_major_output, hp.out_channels, velocity);
        return true;
    };

    return flow_euler_sample_f32(noise, sampler_config, callback, output, error);
}

} // namespace pixal3d
