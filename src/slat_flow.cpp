#include "pixal3d/slat_flow.h"

#include "pixal3d/flow.h"
#include "pixal3d/pack.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pixal3d {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

bool require_pointer(const float * pointer, const char * name, std::string * error) {
    if (pointer) return true;
    set_error(error, std::string("missing SLat flow tensor: ") + name);
    return false;
}

bool checked_count(std::size_t rows, int channels, std::size_t & count,
                   std::string * error) {
    if (channels <= 0 || (rows != 0 && rows > std::numeric_limits<std::size_t>::max() /
                                      static_cast<std::size_t>(channels))) {
        set_error(error, "SLat flow row count overflows size_t");
        return false;
    }
    count = rows * static_cast<std::size_t>(channels);
    return true;
}

bool concat_sparse_features(const SparseTensorF32 & left,
                            const SparseTensorF32 & right,
                            SparseTensorF32 & output,
                            std::string * error) {
    if (!left.valid(error) || !right.valid(error) ||
        left.batch_size != right.batch_size ||
        left.spatial_x != right.spatial_x || left.spatial_y != right.spatial_y ||
        left.spatial_z != right.spatial_z || left.coords != right.coords ||
        right.channels <= 0 || left.channels >
            std::numeric_limits<int>::max() - right.channels) {
        set_error(error, "SLat flow concat condition shape or coordinates mismatch");
        return false;
    }
    output = left;
    output.channels = left.channels + right.channels;
    output.feats.resize(left.points() * static_cast<std::size_t>(output.channels));
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t point = 0; point < left.points(); ++point) {
        const float * lhs = left.feats.data() + point * static_cast<std::size_t>(left.channels);
        const float * rhs = right.feats.data() + point * static_cast<std::size_t>(right.channels);
        float * dst = output.feats.data() + point * static_cast<std::size_t>(output.channels);
        std::copy(lhs, lhs + left.channels, dst);
        std::copy(rhs, rhs + right.channels, dst + left.channels);
    }
    return true;
}

float silu(float value) {
    if (value >= 0.0f) return value / (1.0f + std::exp(-value));
    const float e = std::exp(value);
    return value * e / (1.0f + e);
}

bool row_linear(const std::vector<float> & input, std::size_t rows, int input_channels,
                const float * weight, const float * bias, int output_channels,
                std::vector<float> & output, std::string * error) {
    std::size_t expected = 0;
    if (!checked_count(rows, input_channels, expected, error) || input.size() != expected ||
        output_channels <= 0 || !weight || !bias) {
        set_error(error, "invalid SLat flow row linear input or weights");
        return false;
    }
    if (!checked_count(rows, output_channels, expected, error)) return false;
    output.assign(expected, 0.0f);
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t row = 0; row < rows; ++row) {
        const float * source = input.data() + row * static_cast<std::size_t>(input_channels);
        float * destination = output.data() + row * static_cast<std::size_t>(output_channels);
        for (int out = 0; out < output_channels; ++out) {
            float value = bias[out];
            for (int in = 0; in < input_channels; ++in) {
                value += weight[static_cast<std::size_t>(out) * input_channels + in] * source[in];
            }
            destination[out] = value;
        }
    }
    return true;
}

bool valid_hparams(const SLatFlowHParams & hp, std::string * error) {
    if (hp.resolution <= 0 || hp.in_channels <= 0 || hp.out_channels <= 0 ||
        hp.model_channels <= 0 || hp.cond_channels <= 0 || hp.num_blocks <= 0 ||
        hp.num_heads <= 0 || hp.model_channels % hp.num_heads != 0 ||
        hp.head_dim() <= 0 || hp.head_dim() % 2 != 0 || hp.mlp_ratio <= 0.0f ||
        !std::isfinite(hp.mlp_ratio) || hp.rope_freq_min <= 0.0f ||
        hp.rope_freq_base <= 0.0f || !std::isfinite(hp.rope_freq_min) ||
        !std::isfinite(hp.rope_freq_base) || hp.norm_eps <= 0.0f ||
        !std::isfinite(hp.norm_eps)) {
        set_error(error, "invalid SLat flow hyperparameters");
        return false;
    }
    if (hp.pe_mode != "rope" || !hp.share_mod) {
        set_error(error, "SLat flow currently requires pe_mode=rope and share_mod=true");
        return false;
    }
    if (hp.image_attn_mode != "cross" && hp.image_attn_mode != "proj") {
        set_error(error, "SLat flow supports image_attn_mode cross or proj");
        return false;
    }
    if (hp.image_attn_mode == "proj" && hp.proj_in_channels <= 0) {
        set_error(error, "SLat flow projection channels must be positive");
        return false;
    }
    return true;
}

bool check_model_weights(const SLatFlowHParams & hp, const SLatFlowWeightsF32 & weights,
                         std::string * error) {
    if (!require_pointer(weights.input_weight, "input_layer.weight", error) ||
        !require_pointer(weights.input_bias, "input_layer.bias", error) ||
        !require_pointer(weights.timestep_weight0, "t_embedder.mlp.0.weight", error) ||
        !require_pointer(weights.timestep_bias0, "t_embedder.mlp.0.bias", error) ||
        !require_pointer(weights.timestep_weight2, "t_embedder.mlp.2.weight", error) ||
        !require_pointer(weights.timestep_bias2, "t_embedder.mlp.2.bias", error) ||
        !require_pointer(weights.modulation_weight, "adaLN_modulation.1.weight", error) ||
        !require_pointer(weights.modulation_bias, "adaLN_modulation.1.bias", error) ||
        !require_pointer(weights.output_weight, "out_layer.weight", error) ||
        !require_pointer(weights.output_bias, "out_layer.bias", error)) return false;
    if (weights.blocks.size() != static_cast<std::size_t>(hp.num_blocks)) {
        set_error(error, "SLat flow block count does not match metadata");
        return false;
    }
    return true;
}

} // namespace

bool slat_flow_forward_f32(
    const SparseTensorF32 & input,
    const float * timesteps,
    std::size_t timestep_count,
    const VarLenTensorF32 & global_context,
    const SparseTensorF32 * projection_context,
    const SLatFlowHParams & hparams,
    const SLatFlowWeightsF32 & weights,
    SparseTensorF32 & output,
    std::string * error,
    const SparseTensorF32 * concat_condition) {
    if (!input.valid(error) || !global_context.valid(error) || !valid_hparams(hparams, error) ||
        !check_model_weights(hparams, weights, error)) return false;
    SparseTensorF32 model_input;
    const SparseTensorF32 * input_for_model = &input;
    if (concat_condition) {
        if (!concat_sparse_features(input, *concat_condition, model_input, error)) {
            return false;
        }
        input_for_model = &model_input;
    }
    if (input_for_model->channels != hparams.in_channels ||
        input_for_model->batch_size != global_context.batch_size ||
        timestep_count != static_cast<std::size_t>(input.batch_size) || !timesteps) {
        set_error(error, "SLat flow input, timestep, or context shape mismatch");
        return false;
    }
    if (hparams.image_attn_mode == "proj") {
        if (!projection_context || !projection_context->valid(error) ||
            projection_context->batch_size != input_for_model->batch_size ||
            projection_context->channels != hparams.proj_in_channels ||
            projection_context->coords != input_for_model->coords) {
            set_error(error, "SLat flow projection context shape mismatch");
            return false;
        }
    } else if (projection_context) {
        if (!projection_context->valid(error) ||
            projection_context->coords != input_for_model->coords) {
            set_error(error, "SLat flow optional projection context coordinates mismatch");
            return false;
        }
    }

    SparseTensorF32 hidden;
    if (!sparse_linear(*input_for_model, weights.input_weight, weights.input_bias,
                       hparams.model_channels, hidden, error)) return false;

    std::vector<float> timestep_embedding_values;
    if (!timestep_embed_mlp(timesteps, timestep_count, hparams.model_channels, 256,
                            weights.timestep_weight0, weights.timestep_bias0,
                            weights.timestep_weight2, weights.timestep_bias2,
                            timestep_embedding_values, error)) return false;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t index = 0; index < timestep_embedding_values.size(); ++index) {
        timestep_embedding_values[index] = silu(timestep_embedding_values[index]);
    }
    std::vector<float> timestep_modulation;
    if (!row_linear(timestep_embedding_values, timestep_count, hparams.model_channels,
                    weights.modulation_weight, weights.modulation_bias,
                    6 * hparams.model_channels, timestep_modulation, error)) return false;

    SparseTransformerBlockConfig block_config;
    block_config.channels = hparams.model_channels;
    block_config.num_heads = hparams.num_heads;
    block_config.mlp_hidden = static_cast<int>(hparams.model_channels * hparams.mlp_ratio);
    block_config.context_channels = hparams.cond_channels;
    block_config.proj_in_channels = hparams.proj_in_channels;
    block_config.norm_eps = hparams.norm_eps;
    block_config.rope_freq_min = hparams.rope_freq_min;
    block_config.rope_freq_base = hparams.rope_freq_base;
    block_config.use_rope = true;
    block_config.qk_rms_norm = hparams.qk_rms_norm;
    block_config.qk_rms_norm_cross = hparams.qk_rms_norm_cross;
    block_config.use_projection = hparams.image_attn_mode == "proj";
    for (int block = 0; block < hparams.num_blocks; ++block) {
        if (!sparse_transformer_cross_block_f32(
                hidden, timestep_modulation.data(), global_context,
                projection_context, block_config, weights.blocks[static_cast<std::size_t>(block)],
                hidden, error)) return false;
    }
    SparseTensorF32 normalized;
    // SLatFlowModel's final F.layer_norm keeps PyTorch's default eps=1e-5;
    // block norms intentionally use 1e-6.
    return sparse_layer_norm(hidden, 1e-5f, nullptr, nullptr, normalized, error) &&
           sparse_linear(normalized, weights.output_weight, weights.output_bias,
                         hparams.out_channels, output, error);
}

namespace {

bool has_prefix(const std::string & value, const std::string & prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

std::string compact_prefix(const std::string & component) {
    if (component == "shape_flow_512") return "sh512";
    if (component == "shape_flow_1024") return "sh1024";
    if (component == "texture_flow_1024") return "tx1024";
    return {};
}

bool read_tensor_f32(const Pixal3DPackReader & reader,
                     const Pixal3DTensorInfo & info,
                     std::vector<float> & output,
                     std::string * error) {
    std::vector<std::uint8_t> payload;
    if (!reader.read_tensor(info.name, payload, error)) return false;
    const ggml_type type = static_cast<ggml_type>(info.ggml_type);
    const std::size_t element_size = type == GGML_TYPE_F32 ? sizeof(float) :
        (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 ? sizeof(std::uint16_t) : 0);
    if (element_size == 0 || payload.size() % element_size != 0) {
        set_error(error, "unsupported or misaligned SLat flow tensor: " + info.name);
        return false;
    }
    const std::size_t count = payload.size() / element_size;
    if (count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        set_error(error, "SLat flow tensor is too large: " + info.name);
        return false;
    }
    output.resize(count);
    if (type == GGML_TYPE_F32) {
        std::memcpy(output.data(), payload.data(), payload.size());
    } else if (type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(payload.data()),
                              output.data(), static_cast<std::int64_t>(count));
    } else {
        ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(payload.data()),
                              output.data(), static_cast<std::int64_t>(count));
    }
    for (float value : output) {
        if (!std::isfinite(value)) {
            set_error(error, "SLat flow tensor contains a non-finite value: " + info.name);
            return false;
        }
    }
    return true;
}

bool read_hparams(const Pixal3DPackReader & reader, const std::string & component,
                  SLatFlowHParams & hp, std::string * error) {
    const std::string prefix = "pixal3d." + component + ".";
    std::uint32_t value = 0;
    if (!reader.metadata_string(prefix + "model_class", hp.model_class, true, error) ||
        !reader.metadata_u32(prefix + "resolution", value, true, error)) return false;
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
    if (!reader.metadata_f32(prefix + "mlp_ratio", hp.mlp_ratio, true, error) ||
        !reader.metadata_string(prefix + "pe_mode", hp.pe_mode, false, error) ||
        !reader.metadata_bool(prefix + "share_mod", hp.share_mod, false, error) ||
        !reader.metadata_bool(prefix + "qk_rms_norm", hp.qk_rms_norm, false, error) ||
        !reader.metadata_bool(prefix + "qk_rms_norm_cross", hp.qk_rms_norm_cross, false, error) ||
        !reader.metadata_f32(prefix + "rope_freq_min", hp.rope_freq_min, false, error) ||
        !reader.metadata_f32(prefix + "rope_freq_base", hp.rope_freq_base, false, error) ||
        !reader.metadata_string(prefix + "image_attn_mode", hp.image_attn_mode, false, error) ||
        !reader.metadata_u32(prefix + "proj_in_channels", value, false, error)) return false;
    if (hp.pe_mode.empty()) hp.pe_mode = "rope";
    if (hp.image_attn_mode.empty()) hp.image_attn_mode = "cross";
    hp.proj_in_channels = value == 0 ? hp.cond_channels : static_cast<int>(value);
    return valid_hparams(hp, error);
}

} // namespace

struct SLatFlowModel::Impl {
    Pixal3DPackReader reader;
    SLatFlowHParams hp;
    std::string alias;
    std::unordered_map<std::string, std::vector<float>> tensors;
    ggml_context * weights_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    std::string backend_name;
    std::unordered_map<std::string, ggml_tensor *> backend_tensors;
    int tensor_count = 0;
    bool has_data = false;

    ~Impl() { close_gpu(); }

    void close_gpu() noexcept {
        backend_tensors.clear();
        if (weights_buffer) {
            ggml_backend_buffer_free(weights_buffer);
            weights_buffer = nullptr;
        }
        if (backend) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
        if (weights_ctx) {
            ggml_free(weights_ctx);
            weights_ctx = nullptr;
        }
        backend_name.clear();
    }

    bool init_gpu(std::string * error);
    bool forward_gpu(const SparseTensorF32 & input,
                     const float * timesteps,
                     std::size_t timestep_count,
                     const VarLenTensorF32 & global_context,
                     const SparseTensorF32 * projection_context,
                     SparseTensorF32 & output,
                     std::string * error,
                     const SparseTensorF32 * concat_condition);
    ggml_tensor * backend_weight(const std::string & local_name) const {
        const auto found = backend_tensors.find(alias + "." + local_name);
        return found == backend_tensors.end() ? nullptr : found->second;
    }

    const float * tensor(const std::string & local_name, std::string * error) const {
        const std::string full_name = alias + "." + local_name;
        const auto found = tensors.find(full_name);
        if (found != tensors.end()) return found->second.data();
        set_error(error, "missing SLat flow tensor: " + full_name);
        return nullptr;
    }
};

bool SLatFlowModel::Impl::init_gpu(std::string * error) {
    if (backend && weights_buffer) return true;
    close_gpu();

    ggml_backend_t selected = nullptr;
    std::string selected_name;
    for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        selected = ggml_backend_dev_init(device, nullptr);
        if (!selected) continue;
        const char * description = ggml_backend_dev_description(device);
        selected_name = description ? description : ggml_backend_dev_name(device);
        break;
    }
    if (!selected) {
        set_error(error, "no ggml GPU backend is available for SLat flow");
        return false;
    }

    std::size_t count = 0;
    for (const Pixal3DTensorInfo & info : reader.info().tensors) {
        if (has_prefix(info.name, alias + ".")) ++count;
    }
    if (count == 0 || count > std::numeric_limits<std::size_t>::max() /
                                      ggml_tensor_overhead()) {
        ggml_backend_free(selected);
        set_error(error, "invalid SLat flow tensor count for GPU weights");
        return false;
    }

    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * (count + 1) + 4096;
    params.no_alloc = true;
    weights_ctx = ggml_init(params);
    if (!weights_ctx) {
        ggml_backend_free(selected);
        set_error(error, "failed to allocate SLat flow GPU tensor context");
        return false;
    }
    for (const Pixal3DTensorInfo & info : reader.info().tensors) {
        if (!has_prefix(info.name, alias + ".")) continue;
        const auto found = tensors.find(info.name);
        if (found == tensors.end()) {
            close_gpu();
            ggml_backend_free(selected);
            set_error(error, "missing host SLat flow tensor for GPU upload: " + info.name);
            return false;
        }
        ggml_tensor * tensor = ggml_new_tensor(weights_ctx, GGML_TYPE_F32,
                                               info.n_dims, info.ne);
        if (!tensor || ggml_nelements(tensor) !=
                            static_cast<int64_t>(found->second.size()) ||
            ggml_nbytes(tensor) != found->second.size() * sizeof(float)) {
            close_gpu();
            ggml_backend_free(selected);
            set_error(error, "SLat flow GPU tensor shape mismatch: " + info.name);
            return false;
        }
        ggml_set_name(tensor, info.name.c_str());
        backend_tensors.emplace(info.name, tensor);
    }

    backend = selected;
    backend_name = std::move(selected_name);
    weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, backend);
    if (!weights_buffer) {
        close_gpu();
        set_error(error, "failed to allocate SLat flow weights on GPU backend");
        return false;
    }
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const auto & entry : tensors) {
        const auto found = backend_tensors.find(entry.first);
        if (found == backend_tensors.end()) continue;
        ggml_backend_tensor_set(found->second, entry.second.data(), 0,
                                entry.second.size() * sizeof(float));
    }
    std::cerr << "pixal3d: " << hp.component << " SLat CUDA weights ready ("
              << backend_name << ")" << std::endl;
    return true;
}

bool SLatFlowModel::Impl::forward_gpu(
    const SparseTensorF32 & input,
    const float * timesteps,
    std::size_t timestep_count,
    const VarLenTensorF32 & global_context,
    const SparseTensorF32 * projection_context,
    SparseTensorF32 & output,
    std::string * error,
    const SparseTensorF32 * concat_condition) {
    if (!backend || !weights_buffer) {
        set_error(error, "SLat flow GPU weights are not initialized");
        return false;
    }
    if (!input.valid(error) || !global_context.valid(error) ||
        !valid_hparams(hp, error) || input.batch_size != 1 ||
        global_context.batch_size != 1 || timestep_count != 1 || !timesteps) {
        set_error(error, "SLat flow GPU path currently requires one batch");
        return false;
    }

    SparseTensorF32 model_input;
    const SparseTensorF32 * input_for_model = &input;
    if (concat_condition) {
        if (!concat_sparse_features(input, *concat_condition, model_input, error)) {
            return false;
        }
        input_for_model = &model_input;
    }
    if (input_for_model->channels != hp.in_channels ||
        input_for_model->batch_size != 1 ||
        global_context.channels != hp.cond_channels ||
        global_context.tokens() == 0) {
        set_error(error, "SLat flow GPU input or context shape mismatch");
        return false;
    }
    if (hp.image_attn_mode == "proj") {
        if (!projection_context || !projection_context->valid(error) ||
            projection_context->batch_size != 1 ||
            projection_context->channels != hp.proj_in_channels ||
            projection_context->coords != input_for_model->coords) {
            set_error(error, "SLat flow GPU projection context shape mismatch");
            return false;
        }
    } else if (projection_context &&
               (!projection_context->valid(error) ||
                projection_context->coords != input_for_model->coords)) {
        set_error(error, "SLat flow GPU optional projection coordinates mismatch");
        return false;
    }
    if (!std::isfinite(timesteps[0])) {
        set_error(error, "SLat flow timestep must be finite");
        return false;
    }

    const std::size_t points = input_for_model->points();
    const std::size_t context_tokens = global_context.tokens();
    if (points == 0 || points > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        context_tokens > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        set_error(error, "SLat flow GPU sequence is too large");
        return false;
    }
    const int C = hp.model_channels;
    const int H = hp.num_heads;
    const int hd = hp.head_dim();
    const int N = static_cast<int>(points);
    const int Lkv = static_cast<int>(context_tokens);
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(hd));

    auto W = [&](const std::string & local_name) -> ggml_tensor * {
        return backend_weight(local_name);
    };
    auto require_weight = [&](const std::string & local_name) -> bool {
        if (W(local_name)) return true;
        if (error) *error = "missing SLat flow GPU tensor: " + alias + "." + local_name;
        return false;
    };
    if (!require_weight("input_layer.weight") || !require_weight("input_layer.bias") ||
        !require_weight("out_layer.weight") || !require_weight("out_layer.bias") ||
        !require_weight("t_embedder.mlp.0.weight") ||
        !require_weight("t_embedder.mlp.0.bias") ||
        !require_weight("t_embedder.mlp.2.weight") ||
        !require_weight("t_embedder.mlp.2.bias") ||
        !require_weight("adaLN_modulation.1.weight") ||
        !require_weight("adaLN_modulation.1.bias")) return false;
    for (int block = 0; block < hp.num_blocks; ++block) {
        const std::string p = "blocks." + std::to_string(block);
        const std::string cross = p + ".cross_attn.cross_attn_block";
        const std::string mlp = p + ".mlp.mlp";
        const bool common =
            require_weight(p + ".modulation") &&
            require_weight(p + ".norm2.weight") && require_weight(p + ".norm2.bias") &&
            require_weight(p + ".self_attn.to_qkv.weight") &&
            require_weight(p + ".self_attn.to_qkv.bias") &&
            require_weight(p + ".self_attn.to_out.weight") &&
            require_weight(p + ".self_attn.to_out.bias") &&
            require_weight(mlp + ".0.weight") && require_weight(mlp + ".0.bias") &&
            require_weight(mlp + ".2.weight") && require_weight(mlp + ".2.bias") &&
            require_weight(cross + ".to_q.weight") && require_weight(cross + ".to_q.bias") &&
            require_weight(cross + ".to_kv.weight") && require_weight(cross + ".to_kv.bias") &&
            require_weight(cross + ".to_out.weight") && require_weight(cross + ".to_out.bias");
        const bool self_norm = !hp.qk_rms_norm ||
            (require_weight(p + ".self_attn.q_rms_norm.gamma") &&
             require_weight(p + ".self_attn.k_rms_norm.gamma"));
        const bool cross_norm = !hp.qk_rms_norm_cross ||
            (require_weight(cross + ".q_rms_norm.gamma") &&
             require_weight(cross + ".k_rms_norm.gamma"));
        const bool projection = hp.image_attn_mode != "proj" ||
            (require_weight(p + ".cross_attn.proj_linear.weight") &&
             require_weight(p + ".cross_attn.proj_linear.bias"));
        if (!common || !self_norm || !cross_norm || !projection) return false;
    }

    const std::size_t graph_memory = ggml_tensor_overhead() * 32768 +
                                     ggml_graph_overhead_custom(32768, false);
    ggml_init_params graph_params{};
    graph_params.mem_size = graph_memory;
    graph_params.no_alloc = true;
    ggml_context * ctx = ggml_init(graph_params);
    if (!ctx) {
        set_error(error, "failed to allocate SLat flow GPU graph context");
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32768, false);
    if (!graph) {
        ggml_free(ctx);
        set_error(error, "failed to allocate SLat flow GPU graph");
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
        set_error(error, "failed to allocate SLat flow GPU graph inputs");
        return false;
    }
    ggml_set_input(x_t);
    ggml_set_input(temb);
    ggml_set_input(cos_t);
    ggml_set_input(sin_t);
    ggml_set_input(cnd);
    if (proj) ggml_set_input(proj);

    auto lin = [&](ggml_tensor * input_tensor, const std::string & local_name) {
        ggml_tensor * result = ggml_mul_mat(ctx, W(local_name + ".weight"), input_tensor);
        ggml_tensor * bias = W(local_name + ".bias");
        return bias ? ggml_add(ctx, result, bias) : result;
    };
    auto modulate = [&](ggml_tensor * input_tensor, ggml_tensor * scale,
                        ggml_tensor * shift) {
        return ggml_add(ctx, ggml_add(ctx, ggml_mul(ctx, input_tensor, scale), input_tensor), shift);
    };
    auto qk_norm = [&](ggml_tensor * input_tensor, const std::string & gamma_name) {
        return ggml_mul(ctx, ggml_rms_norm(ctx, input_tensor, 1e-12f), W(gamma_name));
    };
    auto rope = [&](ggml_tensor * input_tensor) {
        ggml_tensor * reshaped = ggml_reshape_4d(ctx, input_tensor, 2, hd / 2, H, N);
        ggml_tensor * even = ggml_cont(ctx, ggml_view_4d(
            ctx, reshaped, 1, hd / 2, H, N, reshaped->nb[1], reshaped->nb[2],
            reshaped->nb[3], 0));
        ggml_tensor * odd = ggml_cont(ctx, ggml_view_4d(
            ctx, reshaped, 1, hd / 2, H, N, reshaped->nb[1], reshaped->nb[2],
            reshaped->nb[3], reshaped->nb[0]));
        ggml_tensor * swapped = ggml_concat(ctx, ggml_neg(ctx, odd), even, 0);
        swapped = ggml_reshape_3d(ctx, swapped, hd, H, N);
        return ggml_add(ctx, ggml_mul(ctx, input_tensor, cos_t),
                        ggml_mul(ctx, swapped, sin_t));
    };
    auto sdpa = [&](ggml_tensor * q, ggml_tensor * k, ggml_tensor * v) {
        // ggml_flash_attn_ext consumes [head_dim, sequence, heads, batch].
        // q/k/v are produced as [head_dim, heads, sequence].
        ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

        // Keep short sequences on the generic path (as in trellis2cpp).  This
        // is also important for compact/reference packs whose head dimension
        // is not one of the production flash-kernel specializations.
        // qp/kp are [head_dim, sequence, heads, batch].
        if (qp->ne[1] <= 4096 && kp->ne[1] <= 4096) {
            ggml_tensor * scores = ggml_mul_mat(ctx, kp, qp);
            scores = ggml_soft_max_ext(ctx, scores, nullptr,
                                       attention_scale, 0.0f);
            ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, vp, 1, 0, 2, 3));
            ggml_tensor * result = ggml_mul_mat(ctx, vt, scores);
            result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3));
            return ggml_reshape_2d(ctx, result, C, result->ne[2]);
        }

        // Development/reference packs are F32.  Keep the attention inputs in
        // F32 here so the CUDA graph is numerically comparable with the CPU
        // oracle.  The CUDA backend's F32 flash path can still use a fast
        // F16 K/V kernel for long sequences; the build therefore disables
        // TF32 globally and the long-sequence path is audited separately.
        ggml_tensor * result = ggml_flash_attn_ext(
            ctx, qp, kp, vp, nullptr, attention_scale, 0.0f, 0.0f);
        // Keep attention accumulation in F32.  The flow weights and all graph
        // inputs are F32; the CUDA backend's default fast-F16 attention can
        // otherwise introduce avoidable drift that is amplified by the 30
        // denoiser blocks and the Euler sampler.
        ggml_flash_attn_ext_set_prec(result, GGML_PREC_F32);
        result = ggml_cont(ctx, result);
        return ggml_reshape_2d(ctx, result, C, result->ne[2]);
    };
    auto split_heads = [&](ggml_tensor * value, int offset, int sequence) {
        const std::size_t element_size = ggml_element_size(value);
        ggml_tensor * view = ggml_view_2d(
            ctx, value, C, sequence, value->nb[1],
            static_cast<std::size_t>(offset) * C * element_size);
        return ggml_reshape_3d(ctx, ggml_cont(ctx, view), hd, H, sequence);
    };

    std::vector<float> channel_major_input(points * static_cast<std::size_t>(hp.in_channels));
    for (std::size_t point = 0; point < points; ++point) {
        for (int channel = 0; channel < hp.in_channels; ++channel) {
            channel_major_input[static_cast<std::size_t>(channel) * points + point] =
                input_for_model->feats[point * static_cast<std::size_t>(hp.in_channels) +
                                        static_cast<std::size_t>(channel)];
        }
    }
    std::vector<float> embedding;
    if (!timestep_embedding(timesteps, 1, 256, 10000, embedding, error)) {
        ggml_free(ctx);
        return false;
    }
    std::vector<float> cos_values(points * static_cast<std::size_t>(hd), 1.0f);
    std::vector<float> sin_values(points * static_cast<std::size_t>(hd), 0.0f);
    const int rope_freq_dim = hd / 2 / 3;
    const int rope_pairs = hd / 2;
    std::vector<float> frequencies(static_cast<std::size_t>(std::max(rope_freq_dim, 0)));
    for (int index = 0; index < rope_freq_dim; ++index) {
        frequencies[static_cast<std::size_t>(index)] = hp.rope_freq_min /
            std::pow(hp.rope_freq_base, static_cast<float>(index) /
                     static_cast<float>(rope_freq_dim));
    }
    for (std::size_t point = 0; point < points; ++point) {
        const int coordinate[3] = {
            input_for_model->coords[point * 4 + 1],
            input_for_model->coords[point * 4 + 2],
            input_for_model->coords[point * 4 + 3]};
        for (int pair = 0; pair < rope_pairs; ++pair) {
            float phase = 0.0f;
            if (rope_freq_dim > 0 && pair < 3 * rope_freq_dim) {
                phase = static_cast<float>(coordinate[pair / rope_freq_dim]) *
                        frequencies[static_cast<std::size_t>(pair % rope_freq_dim)];
            }
            const std::size_t base = point * static_cast<std::size_t>(hd) +
                                     static_cast<std::size_t>(pair * 2);
            cos_values[base] = cos_values[base + 1] = std::cos(phase);
            sin_values[base] = sin_values[base + 1] = std::sin(phase);
        }
    }

    ggml_tensor * h = ggml_cont(ctx, ggml_transpose(ctx, x_t));
    h = lin(h, "input_layer");
    ggml_tensor * te = lin(temb, "t_embedder.mlp.0");
    te = ggml_silu(ctx, te);
    te = lin(te, "t_embedder.mlp.2");
    // The CPU path applies the second SiLU after t_embedder.mlp.2 and before
    // adaLN_modulation.1; keep that activation in the ggml graph as well.
    ggml_tensor * tmod = lin(ggml_silu(ctx, te), "adaLN_modulation.1");

    for (int block = 0; block < hp.num_blocks; ++block) {
        const std::string p = "blocks." + std::to_string(block);
        const std::string cross = p + ".cross_attn.cross_attn_block";
        ggml_tensor * mods = ggml_add(ctx, W(p + ".modulation"), tmod);
        auto chunk = [&](int index) {
            return ggml_view_1d(ctx, mods, C,
                                static_cast<std::size_t>(index) * C * ggml_element_size(mods));
        };
        ggml_tensor * shift_msa = chunk(0);
        ggml_tensor * scale_msa = chunk(1);
        ggml_tensor * gate_msa = chunk(2);
        ggml_tensor * shift_mlp = chunk(3);
        ggml_tensor * scale_mlp = chunk(4);
        ggml_tensor * gate_mlp = chunk(5);

        ggml_tensor * normalized = modulate(ggml_norm(ctx, h, hp.norm_eps),
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

        ggml_tensor * cross_input = ggml_norm(ctx, h, hp.norm_eps);
        cross_input = ggml_add(ctx, ggml_mul(ctx, cross_input, W(p + ".norm2.weight")),
                               W(p + ".norm2.bias"));
        ggml_tensor * cross_q = split_heads(lin(cross_input, cross + ".to_q"), 0, N);
        ggml_tensor * cross_kv = lin(cnd, cross + ".to_kv");
        ggml_tensor * cross_k = split_heads(cross_kv, 0, Lkv);
        ggml_tensor * cross_v = split_heads(cross_kv, 1, Lkv);
        if (hp.qk_rms_norm_cross) {
            cross_q = qk_norm(cross_q, cross + ".q_rms_norm.gamma");
            cross_k = qk_norm(cross_k, cross + ".k_rms_norm.gamma");
        }
        ggml_tensor * cross_out = lin(sdpa(cross_q, cross_k, cross_v),
                                      cross + ".to_out");
        if (hp.image_attn_mode == "proj") {
            cross_out = ggml_add(ctx, cross_out,
                                 lin(proj, p + ".cross_attn.proj_linear"));
        }
        h = ggml_add(ctx, h, cross_out);

        ggml_tensor * mlp_input = modulate(ggml_norm(ctx, h, hp.norm_eps),
                                           scale_mlp, shift_mlp);
        mlp_input = lin(mlp_input, p + ".mlp.mlp.0");
        mlp_input = ggml_gelu(ctx, mlp_input);
        mlp_input = lin(mlp_input, p + ".mlp.mlp.2");
        h = ggml_add(ctx, h, ggml_mul(ctx, mlp_input, gate_mlp));
    }
    h = ggml_norm(ctx, h, 1e-5f);
    h = lin(h, "out_layer");
    ggml_tensor * result = ggml_cont(ctx, h);
    ggml_set_output(result);
    ggml_build_forward_expand(graph, result);
    if (std::getenv("PIXAL3D_SLAT_TRACE")) {
        const int graph_nodes = ggml_graph_n_nodes(graph);
        std::cerr << "pixal3d: SLat GPU graph built nodes=" << graph_nodes
                  << " points=" << points << " channels=" << hp.out_channels
                  << std::endl;
        int unsupported = 0;
        for (int index = 0; index < graph_nodes; ++index) {
            ggml_tensor * node = ggml_graph_node(graph, index);
            if (!ggml_backend_supports_op(backend, node)) {
                ++unsupported;
                std::cerr << "pixal3d: SLat GPU unsupported node " << index
                          << " op=" << ggml_op_name(node->op)
                          << " name=" << node->name << std::endl;
            }
        }
        std::cerr << "pixal3d: SLat GPU unsupported count=" << unsupported
                  << std::endl;
    }
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (std::getenv("PIXAL3D_SLAT_TRACE")) {
        std::cerr << "pixal3d: SLat GPU graph allocation start" << std::endl;
    }
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        if (allocator) ggml_gallocr_free(allocator);
        ggml_free(ctx);
        set_error(error, "failed to allocate SLat flow GPU compute graph");
        return false;
    }
    if (std::getenv("PIXAL3D_SLAT_TRACE")) {
        std::cerr << "pixal3d: SLat GPU graph allocation complete" << std::endl;
    }

    ggml_backend_tensor_set(x_t, channel_major_input.data(), 0,
                            channel_major_input.size() * sizeof(float));
    ggml_backend_tensor_set(temb, embedding.data(), 0, embedding.size() * sizeof(float));
    ggml_backend_tensor_set(cos_t, cos_values.data(), 0, cos_values.size() * sizeof(float));
    ggml_backend_tensor_set(sin_t, sin_values.data(), 0, sin_values.size() * sizeof(float));
    // The public condition buffers are row-major [token, channel] / [point,
    // channel].  With ggml tensors shaped [channel, token/point], that byte
    // order is exactly the required contiguous layout (ne[0] is the channel
    // dimension), so upload the buffers directly as in the reference graph.
    ggml_backend_tensor_set(cnd, global_context.feats.data(), 0,
                            global_context.feats.size() * sizeof(float));
    if (proj) {
        ggml_backend_tensor_set(proj, projection_context->feats.data(), 0,
                                projection_context->feats.size() * sizeof(float));
    }
    if (std::getenv("PIXAL3D_SLAT_TRACE")) {
        std::cerr << "pixal3d: SLat GPU graph compute start" << std::endl;
    }
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (std::getenv("PIXAL3D_SLAT_TRACE")) {
        std::cerr << "pixal3d: SLat GPU graph compute complete status="
                  << static_cast<int>(status) << std::endl;
    }
    std::vector<float> backend_output(points * static_cast<std::size_t>(hp.out_channels));
    bool ok = status == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_tensor_get(result, backend_output.data(), 0,
                                backend_output.size() * sizeof(float));
        output = *input_for_model;
        output.channels = hp.out_channels;
        // The final graph tensor is already returned in the public
        // [point, channel] byte order.  Do not transpose it a second time:
        // doing so scrambles channels across neighboring sparse points.
        output.feats = std::move(backend_output);
        std::string finite_error;
        if (!output.valid(&finite_error)) {
            set_error(error, finite_error.empty()
                ? "SLat flow GPU output contains a non-finite value" : finite_error);
            ok = false;
        }
    } else {
        set_error(error, "SLat flow GPU graph compute failed");
    }
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return ok;
}

SLatFlowModel::~SLatFlowModel() {
    close();
}

void SLatFlowModel::close() noexcept {
    delete impl_;
    impl_ = nullptr;
}

bool SLatFlowModel::load(const std::string & path, const std::string & component,
                         bool load_tensors, std::string * error) {
    close();
    const std::string alias = compact_prefix(component);
    if (alias.empty()) {
        set_error(error, "unsupported SLat flow component: " + component);
        return false;
    }
    std::unique_ptr<Impl> impl(new Impl());
    if (!impl->reader.open(path, error)) return false;
    impl->hp.component = component;
    impl->alias = alias;
    const std::string prefix = alias + ".";
    if (!impl->reader.info().has_tensor(prefix + "input_layer.weight")) {
        set_error(error, "GGUF pack does not contain SLat flow component: " + component);
        return false;
    }
    if (!read_hparams(impl->reader, component, impl->hp, error)) return false;
    for (const Pixal3DTensorInfo & info : impl->reader.info().tensors) {
        if (has_prefix(info.name, prefix)) ++impl->tensor_count;
    }
    if (impl->tensor_count <= 0) {
        set_error(error, "SLat flow component has no tensors: " + component);
        return false;
    }
    if (load_tensors) {
        for (const Pixal3DTensorInfo & info : impl->reader.info().tensors) {
            if (!has_prefix(info.name, prefix)) continue;
            std::vector<float> values;
            if (!read_tensor_f32(impl->reader, info, values, error)) return false;
            impl->tensors.emplace(info.name, std::move(values));
        }
        impl->has_data = true;
    }
    impl_ = impl.release();
    return true;
}

bool SLatFlowModel::is_loaded() const noexcept { return impl_ != nullptr; }

bool SLatFlowModel::has_data() const noexcept { return impl_ && impl_->has_data; }

const SLatFlowHParams & SLatFlowModel::hparams() const noexcept {
    static const SLatFlowHParams empty;
    return impl_ ? impl_->hp : empty;
}

int SLatFlowModel::tensor_count() const noexcept { return impl_ ? impl_->tensor_count : 0; }

bool SLatFlowModel::has_tensor(const std::string & name) const noexcept {
    if (!impl_) return false;
    const std::string full_name = has_prefix(name, impl_->alias + ".")
        ? name : impl_->alias + "." + name;
    return impl_->reader.info().has_tensor(full_name);
}

bool SLatFlowModel::forward(const SparseTensorF32 & input,
                            const float * timesteps,
                            std::size_t timestep_count,
                            const VarLenTensorF32 & global_context,
                            const SparseTensorF32 * projection_context,
                            SparseTensorF32 & output,
                            std::string * error,
                            const SparseTensorF32 * concat_condition) {
    if (!impl_) {
        set_error(error, "SLat flow model is not loaded");
        return false;
    }
    if (!impl_->has_data) {
        set_error(error, "SLat flow model was loaded metadata-only");
        return false;
    }
    // Production cascade stages are batch-1.  Keep the established CPU/F32
    // implementation for batched fixtures and as a safe fallback when a GPU
    // backend cannot allocate or execute the sparse graph.
    const char * backend_mode = std::getenv("PIXAL3D_SLAT_BACKEND");
    const bool force_cpu = backend_mode && std::string(backend_mode) == "cpu";
    if (!force_cpu && input.batch_size == 1 && global_context.batch_size == 1) {
        std::string gpu_error;
        if (impl_->init_gpu(&gpu_error) &&
            impl_->forward_gpu(input, timesteps, timestep_count, global_context,
                               projection_context, output, &gpu_error,
                               concat_condition)) {
            return true;
        }
        if (std::getenv("PIXAL3D_SLAT_VERBOSE")) {
            std::cerr << "pixal3d: SLat GPU forward unavailable; using CPU fallback"
                      << (gpu_error.empty() ? "" : ": " + gpu_error) << std::endl;
        }
        impl_->close_gpu();
    }
    SLatFlowWeightsF32 weights;
    weights.input_weight = impl_->tensor("input_layer.weight", error);
    weights.input_bias = impl_->tensor("input_layer.bias", error);
    weights.timestep_weight0 = impl_->tensor("t_embedder.mlp.0.weight", error);
    weights.timestep_bias0 = impl_->tensor("t_embedder.mlp.0.bias", error);
    weights.timestep_weight2 = impl_->tensor("t_embedder.mlp.2.weight", error);
    weights.timestep_bias2 = impl_->tensor("t_embedder.mlp.2.bias", error);
    weights.modulation_weight = impl_->tensor("adaLN_modulation.1.weight", error);
    weights.modulation_bias = impl_->tensor("adaLN_modulation.1.bias", error);
    weights.output_weight = impl_->tensor("out_layer.weight", error);
    weights.output_bias = impl_->tensor("out_layer.bias", error);
    if (!weights.input_weight || !weights.input_bias || !weights.timestep_weight0 ||
        !weights.timestep_bias0 || !weights.timestep_weight2 || !weights.timestep_bias2 ||
        !weights.modulation_weight || !weights.modulation_bias || !weights.output_weight ||
        !weights.output_bias) return false;
    weights.blocks.reserve(static_cast<std::size_t>(impl_->hp.num_blocks));
    for (int block = 0; block < impl_->hp.num_blocks; ++block) {
        const std::string prefix = "blocks." + std::to_string(block);
        SparseTransformerBlockWeightsF32 current;
        current.modulation = impl_->tensor(prefix + ".modulation", error);
        current.self_to_qkv_weight = impl_->tensor(prefix + ".self_attn.to_qkv.weight", error);
        current.self_to_qkv_bias = impl_->tensor(prefix + ".self_attn.to_qkv.bias", error);
        current.self_q_gamma = impl_->tensor(prefix + ".self_attn.q_rms_norm.gamma", error);
        current.self_k_gamma = impl_->tensor(prefix + ".self_attn.k_rms_norm.gamma", error);
        current.self_to_out_weight = impl_->tensor(prefix + ".self_attn.to_out.weight", error);
        current.self_to_out_bias = impl_->tensor(prefix + ".self_attn.to_out.bias", error);
        current.norm2_weight = impl_->tensor(prefix + ".norm2.weight", error);
        current.norm2_bias = impl_->tensor(prefix + ".norm2.bias", error);
        const std::string cross = prefix + ".cross_attn.cross_attn_block";
        current.cross_to_q_weight = impl_->tensor(cross + ".to_q.weight", error);
        current.cross_to_q_bias = impl_->tensor(cross + ".to_q.bias", error);
        current.cross_to_kv_weight = impl_->tensor(cross + ".to_kv.weight", error);
        current.cross_to_kv_bias = impl_->tensor(cross + ".to_kv.bias", error);
        current.cross_q_gamma = impl_->tensor(cross + ".q_rms_norm.gamma", error);
        current.cross_k_gamma = impl_->tensor(cross + ".k_rms_norm.gamma", error);
        current.cross_to_out_weight = impl_->tensor(cross + ".to_out.weight", error);
        current.cross_to_out_bias = impl_->tensor(cross + ".to_out.bias", error);
        if (impl_->hp.image_attn_mode == "proj") {
            current.proj_weight = impl_->tensor(prefix + ".cross_attn.proj_linear.weight", error);
            current.proj_bias = impl_->tensor(prefix + ".cross_attn.proj_linear.bias", error);
        }
        const std::string mlp = prefix + ".mlp.mlp";
        current.mlp0_weight = impl_->tensor(mlp + ".0.weight", error);
        current.mlp0_bias = impl_->tensor(mlp + ".0.bias", error);
        current.mlp2_weight = impl_->tensor(mlp + ".2.weight", error);
        current.mlp2_bias = impl_->tensor(mlp + ".2.bias", error);
        weights.blocks.push_back(current);
    }
    const bool ok = slat_flow_forward_f32(input, timesteps, timestep_count, global_context,
                                          projection_context, impl_->hp, weights, output, error,
                                          concat_condition);
    return ok;
}

bool SLatFlowModel::sample(const SparseTensorF32 & noise,
                           const FlowEulerSamplerConfig & sampler_config,
                           const VarLenTensorF32 & global_context,
                           const SparseTensorF32 * projection_context,
                           FlowEulerSampleF32 & output,
                           std::string * error,
                           const SparseTensorF32 * concat_condition) {
    if (!impl_) {
        set_error(error, "SLat flow model is not loaded");
        return false;
    }
    if (!impl_->has_data) {
        set_error(error, "SLat flow model was loaded metadata-only");
        return false;
    }
    if (!noise.valid(error) || !global_context.valid(error) ||
        (projection_context && !projection_context->valid(error)) ||
        (concat_condition && !concat_condition->valid(error))) return false;
    VarLenTensorF32 negative_global = global_context;
    std::fill(negative_global.feats.begin(), negative_global.feats.end(), 0.0f);
    SparseTensorF32 negative_projection;
    const SparseTensorF32 * negative_projection_ptr = nullptr;
    if (projection_context) {
        negative_projection = *projection_context;
        std::fill(negative_projection.feats.begin(), negative_projection.feats.end(), 0.0f);
        negative_projection_ptr = &negative_projection;
    }
    FlowVelocityFn callback = [this, &global_context, &negative_global,
                               projection_context, negative_projection_ptr,
                               concat_condition]
                              (const SparseTensorF32 & state, float timestep,
                               bool conditional, SparseTensorF32 & velocity,
                               std::string * callback_error) {
        const VarLenTensorF32 & context = conditional ? global_context : negative_global;
        const SparseTensorF32 * projected = conditional
            ? projection_context : negative_projection_ptr;
        std::vector<float> timesteps(static_cast<std::size_t>(state.batch_size), timestep);
        return forward(state, timesteps.data(), timesteps.size(), context, projected,
                       velocity, callback_error, concat_condition);
    };
    const bool ok = flow_euler_sample_f32(noise, sampler_config, callback, output, error);
    // A stage owns its GPU weights only while its sampler is active.  This
    // keeps the three SLat components from simultaneously occupying VRAM.
    impl_->close_gpu();
    return ok;
}

} // namespace pixal3d
