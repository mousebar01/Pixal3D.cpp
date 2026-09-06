#include "pixal3d/backend.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace {

bool check(bool condition, const char * message) {
    if (condition) return true;
    std::fprintf(stderr, "CUDA backend matmul test: %s\n", message);
    return false;
}

} // namespace

int main() {
    pixal3d::BackendManager manager;
    std::string error;
    if (!check(manager.initialize(
                   {pixal3d::BackendPolicyKind::gpu, 0}, &error),
               error.empty() ? "failed to initialize forced GPU backend"
                             : error.c_str())) {
        return 1;
    }
    if (!check(!manager.backends().empty() && manager.primary() != nullptr,
               "forced GPU manager has no primary backend")) {
        manager.close();
        return 1;
    }
    if (!check(manager.devices().front().type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                   manager.devices().front().type == GGML_BACKEND_DEVICE_TYPE_IGPU,
               "primary backend is not a GPU device")) {
        manager.close();
        return 1;
    }

    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 8 +
                      ggml_graph_overhead_custom(8, false);
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    if (!check(context != nullptr, "failed to create ggml context")) {
        manager.close();
        return 1;
    }

    ggml_tensor * weight = ggml_new_tensor_2d(context, GGML_TYPE_F16, 3, 2);
    ggml_tensor * input = ggml_new_tensor_2d(context, GGML_TYPE_F32, 3, 1);
    ggml_tensor * result = weight && input ? ggml_mul_mat(context, weight, input) : nullptr;
    if (!check(weight != nullptr && input != nullptr && result != nullptr,
               "failed to create F16/F32 matmul")) {
        ggml_free(context);
        manager.close();
        return 1;
    }
    ggml_set_name(weight, "cuda_f16_weight");
    ggml_set_name(input, "cuda_f32_input");
    ggml_set_name(result, "cuda_f32_result");
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    ggml_set_input(weight);
    ggml_set_input(input);
    ggml_set_output(result);

    ggml_cgraph * graph = ggml_new_graph_custom(context, 8, false);
    if (!check(graph != nullptr, "failed to create matmul graph")) {
        ggml_free(context);
        manager.close();
        return 1;
    }
    ggml_build_forward_expand(graph, result);

    bool ok = false;
    {
        pixal3d::BackendScheduler scheduler(manager, 8, false, true,
                                            &error, "CUDA F16 matmul");
        if (!check(scheduler.valid(), error.empty() ? "scheduler construction failed"
                                                    : error.c_str())) {
            ggml_free(context);
            manager.close();
            return 1;
        }
        error.clear();
        if (!check(scheduler.require_primary(result, &error),
                   error.empty() ? "failed to require matmul on GPU"
                                 : error.c_str()) ||
            !check(scheduler.allocate_graph(graph, &error),
                   error.empty() ? "failed to allocate matmul graph"
                                 : error.c_str())) {
            ggml_free(context);
            manager.close();
            return 1;
        }
        if (!check(scheduler.tensor_backend(result) == manager.primary(),
                   "matmul result was not placed on the forced GPU backend")) {
            ggml_free(context);
            manager.close();
            return 1;
        }
        const ggml_fp16_t weight_values[] = {
            ggml_fp32_to_fp16(1.0f), ggml_fp32_to_fp16(2.0f), ggml_fp32_to_fp16(3.0f),
            ggml_fp32_to_fp16(4.0f), ggml_fp32_to_fp16(5.0f), ggml_fp32_to_fp16(6.0f),
        };
        const float input_values[] = {1.0f, -1.0f, 0.5f};
        ggml_backend_tensor_set(weight, weight_values, 0, sizeof(weight_values));
        ggml_backend_tensor_set(input, input_values, 0, sizeof(input_values));
        error.clear();
        const ggml_status status = scheduler.compute(graph, &error);
        if (!check(status == GGML_STATUS_SUCCESS,
                   error.empty() ? "F16 weight/F32 activation compute failed"
                                 : error.c_str())) {
            ggml_free(context);
            manager.close();
            return 1;
        }
        float output[2] = {};
        ggml_backend_tensor_get(result, output, 0, sizeof(output));
        ok = std::isfinite(output[0]) && std::isfinite(output[1]) &&
             std::fabs(output[0] - 0.5f) < 1e-3f &&
             std::fabs(output[1] - 2.0f) < 1e-3f &&
             scheduler.buffer_size(manager.primary()) > 0;
        if (!ok) {
            std::fprintf(stderr, "CUDA backend matmul test: output [%g, %g]\n",
                         output[0], output[1]);
        }
    }

    ggml_free(context);
    manager.close();
    return ok && !manager.initialized() ? 0 : 1;
}
