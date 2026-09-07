#include "pixal3d/backend.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

bool check(bool condition, const char * message) {
    if (condition) return true;
    std::fprintf(stderr, "backend manager test: %s\n", message);
    return false;
}

} // namespace

void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

int main() {
    pixal3d::BackendPolicy policy;
    std::string error;

    if (!check(pixal3d::BackendPolicy::parse("auto", policy, &error),
               "failed to parse auto policy")) return 1;
    if (!check(policy.kind == pixal3d::BackendPolicyKind::auto_select &&
                   policy.name() == "auto",
               "auto policy has unexpected values")) return 1;

    if (!check(pixal3d::BackendPolicy::parse("CPU", policy, &error),
               "failed to parse CPU policy")) return 1;
    if (!check(policy.kind == pixal3d::BackendPolicyKind::cpu &&
                   policy.name() == "cpu",
               "CPU policy has unexpected values")) return 1;

    if (!check(pixal3d::BackendPolicy::parse("gpu:3", policy, &error),
               "failed to parse GPU policy")) return 1;
    if (!check(policy.kind == pixal3d::BackendPolicyKind::gpu &&
                   policy.gpu_index == 3 && policy.name() == "gpu:3",
               "GPU policy has unexpected values")) return 1;

    if (!check(!pixal3d::BackendPolicy::parse("gpu:-1", policy, &error) &&
                   !error.empty(),
               "negative GPU policy was accepted")) return 1;
    error.clear();
    if (!check(!pixal3d::BackendPolicy::parse("metal", policy, &error) &&
                   !error.empty(),
               "unknown backend policy was accepted")) return 1;

    set_env("PIXAL3D_BACKEND", "cpu");
    set_env("PIXAL3D_TEST_STAGE_BACKEND", "gpu:99");
    std::string source;
    error.clear();
    const pixal3d::BackendPolicy stage_policy =
        pixal3d::BackendPolicy::from_environment("PIXAL3D_TEST_STAGE_BACKEND",
                                                  &source, &error);
    if (!check(error.empty() && source == "PIXAL3D_TEST_STAGE_BACKEND" &&
                   stage_policy.kind == pixal3d::BackendPolicyKind::gpu &&
                   stage_policy.gpu_index == 99,
               "stage backend policy did not override global policy")) return 1;
    set_env("PIXAL3D_TEST_STAGE_BACKEND", nullptr);
    source.clear();
    error.clear();
    const pixal3d::BackendPolicy global_policy =
        pixal3d::BackendPolicy::from_environment("PIXAL3D_TEST_STAGE_BACKEND",
                                                  &source, &error);
    if (!check(error.empty() && source == "PIXAL3D_BACKEND" &&
                   global_policy.kind == pixal3d::BackendPolicyKind::cpu,
               "global backend policy was not used after stage override removal")) return 1;
    set_env("PIXAL3D_BACKEND", nullptr);
    set_env("PIXAL3D_TEST_STAGE_BACKEND", "");
    source.clear();
    error.clear();
    const pixal3d::BackendPolicy default_policy =
        pixal3d::BackendPolicy::from_environment("PIXAL3D_TEST_STAGE_BACKEND",
                                                  &source, &error);
    if (!check(error.empty() && source == "default" &&
                   default_policy.kind == pixal3d::BackendPolicyKind::auto_select,
               "empty environment did not select auto policy")) return 1;
    set_env("PIXAL3D_TEST_STAGE_BACKEND", nullptr);

    int gpu_count = 0;
    for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        const enum ggml_backend_dev_type type =
            ggml_backend_dev_type(ggml_backend_dev_get(index));
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU ||
            type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            ++gpu_count;
        }
    }
    pixal3d::BackendManager unavailable_gpu;
    error.clear();
    if (!check(!unavailable_gpu.initialize(
                   {pixal3d::BackendPolicyKind::gpu, gpu_count}, &error) &&
                   !error.empty(),
               "forced unavailable GPU policy unexpectedly succeeded")) return 1;

    pixal3d::BackendManager manager;
    error.clear();
    if (!check(manager.initialize({pixal3d::BackendPolicyKind::cpu, 0}, &error),
               error.empty() ? "CPU backend manager initialization failed"
                             : error.c_str())) return 1;
    if (!check(manager.initialized() && manager.backends().size() == 1 &&
                   manager.devices().size() == 1 && manager.primary() != nullptr,
               "CPU backend manager has unexpected state")) return 1;
    if (!check(manager.devices().front().type == GGML_BACKEND_DEVICE_TYPE_CPU,
               "forced CPU manager did not select CPU")) return 1;
    if (!check(!manager.primary_name().empty(),
               "CPU backend manager has no primary name")) return 1;
    if (!check(manager.cpu_threadpool_n_threads() == GGML_DEFAULT_N_THREADS,
               "CPU backend manager did not configure the default threadpool")) return 1;
    manager.set_n_threads(2);
    if (!check(manager.cpu_threadpool_n_threads() == 2,
               "CPU backend manager did not resize the threadpool")) return 1;
    manager.set_n_threads(GGML_DEFAULT_N_THREADS);
    if (!check(manager.cpu_threadpool_n_threads() == GGML_DEFAULT_N_THREADS,
               "CPU backend manager did not restore the default threadpool")) return 1;

    pixal3d::BackendManager auto_manager;
    error.clear();
    if (!check(auto_manager.initialize(
                   {pixal3d::BackendPolicyKind::auto_select, 0}, &error),
               error.empty() ? "auto backend manager initialization failed"
                             : error.c_str())) return 1;
    if (!check(auto_manager.initialized() &&
                   auto_manager.backends().size() == auto_manager.devices().size() &&
                   !auto_manager.backends().empty(),
               "auto backend manager has inconsistent device state")) return 1;
    if (!check(auto_manager.devices().back().type == GGML_BACKEND_DEVICE_TYPE_CPU,
               "auto backend manager did not keep CPU as the final fallback")) return 1;
    if (!check(!auto_manager.primary_name().empty(),
               "auto backend manager has no primary name")) return 1;
    auto_manager.close();

    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 8 +
                      ggml_graph_overhead_custom(8, false);
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    if (!check(context != nullptr, "ggml_init failed")) {
        manager.close();
        return 1;
    }

    ggml_tensor * left = ggml_new_tensor_1d(context, GGML_TYPE_F32, 2);
    ggml_tensor * right = ggml_new_tensor_1d(context, GGML_TYPE_F32, 2);
    ggml_tensor * result = ggml_add(context, left, right);
    if (!check(left != nullptr && right != nullptr && result != nullptr,
               "failed to create add graph tensors")) {
        ggml_free(context);
        manager.close();
        return 1;
    }
    ggml_set_input(left);
    ggml_set_input(right);
    ggml_set_output(result);

    ggml_cgraph * graph = ggml_new_graph_custom(context, 8, false);
    if (!check(graph != nullptr, "failed to create graph")) {
        ggml_free(context);
        manager.close();
        return 1;
    }
    ggml_build_forward_expand(graph, result);

    bool graph_ok = false;
    {
        std::string scheduler_error;
        pixal3d::BackendScheduler scheduler(manager, 8, false, true,
                                            &scheduler_error);
        if (!scheduler.valid()) {
            std::fprintf(stderr, "backend manager test: scheduler construction failed: %s\n",
                         scheduler_error.c_str());
        } else if (!scheduler.require_primary(result, &scheduler_error)) {
            std::fprintf(stderr, "backend manager test: required-node registration failed: %s\n",
                         scheduler_error.c_str());
        } else if (!scheduler.allocate_graph(graph, &scheduler_error)) {
            std::fprintf(stderr, "backend manager test: graph allocation failed: %s\n",
                         scheduler_error.c_str());
        } else {
            const float lhs[] = {1.25f, -2.0f};
            const float rhs[] = {2.75f, 0.5f};
            ggml_backend_tensor_set(left, lhs, 0, sizeof(lhs));
            ggml_backend_tensor_set(right, rhs, 0, sizeof(rhs));

            error.clear();
            if (!manager.supports_op(result, &error)) {
                std::fprintf(stderr, "backend manager test: add probe failed: %s\n",
                             error.c_str());
            } else if (scheduler.compute(graph, &scheduler_error) !=
                       GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "backend manager test: scheduler compute failed: %s\n",
                             scheduler_error.c_str());
            } else {
                float values[2] = {};
                ggml_backend_tensor_get(result, values, 0, sizeof(values));
                graph_ok = std::isfinite(values[0]) && std::isfinite(values[1]) &&
                           std::fabs(values[0] - 4.0f) < 1e-6f &&
                           std::fabs(values[1] + 1.5f) < 1e-6f;
                if (!graph_ok) {
                    std::fprintf(stderr,
                                 "backend manager test: unexpected add output [%g, %g]\n",
                                 values[0], values[1]);
                }
            }
        }
    }

    ggml_free(context);
    manager.close();
    if (!check(graph_ok && !manager.initialized(),
               "scheduler graph or manager close check failed")) return 1;
    return 0;
}
