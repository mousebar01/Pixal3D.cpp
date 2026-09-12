#include "pixal3d/profile.h"
#include "pixal3d/backend.h"
#include "pixal3d/mesh_postprocess.h"
#include "../src/sparse_profile.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {

void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

class Environment {
public:
    Environment() {
        for (const char * name : {"PIXAL3D_PROFILE", "PIXAL3D_BACKEND_TRACE",
                                 "PIXAL3D_SLAT_DECODER_PROFILE", "PIXAL3D_SLAT_DECODER_TRACE",
                                 "PIXAL3D_VALIDATE_SPARSE_MAP_CACHE"}) {
            const char * value = std::getenv(name);
            entries_.push_back({name, value != nullptr, value ? value : ""});
            set_env(name, nullptr);
        }
    }
    ~Environment() {
        for (const auto & entry : entries_) {
            set_env(entry.name, entry.present ? entry.value.c_str() : nullptr);
        }
    }
private:
    struct Entry { const char * name; bool present; std::string value; };
    std::vector<Entry> entries_;
};

// Capture both C fprintf and C++ cerr, without assuming a POSIX-only test host.
class Capture {
public:
    Capture() {
        std::cerr.flush();
        std::fflush(stderr);
        file_ = std::tmpfile();
        check(file_ != nullptr, "tmpfile failed");
#if defined(_WIN32)
        saved_ = _dup(_fileno(stderr));
        check(saved_ >= 0 && _dup2(_fileno(file_), _fileno(stderr)) == 0, "redirect failed");
#else
        saved_ = dup(fileno(stderr));
        check(saved_ >= 0 && dup2(fileno(file_), fileno(stderr)) >= 0, "redirect failed");
#endif
    }
    ~Capture() {
        restore();
        std::fclose(file_);
    }
    std::string finish() {
        restore();
        std::rewind(file_);
        std::string result;
        char buffer[4096];
        std::size_t count;
        while ((count = std::fread(buffer, 1, sizeof(buffer), file_)) != 0) {
            result.append(buffer, count);
        }
        return result;
    }
private:
    void restore() {
        if (saved_ < 0) return;
        std::cerr.flush();
        std::fflush(stderr);
#if defined(_WIN32)
        _dup2(saved_, _fileno(stderr));
        _close(saved_);
#else
        dup2(saved_, fileno(stderr));
        close(saved_);
#endif
        saved_ = -1;
    }
    FILE * file_ = nullptr;
    int saved_ = -1;
};

int clock_calls = 0;
double fake_clock() noexcept { return static_cast<double>(++clock_calls); }

void test_configuration() {
    using pixal3d::ProfileMode;
    using pixal3d::profile_mode_from_environment;
    Environment environment;
    check(profile_mode_from_environment() == ProfileMode::off, "default is not off");
    set_env("PIXAL3D_BACKEND_TRACE", "1");
    check(profile_mode_from_environment(nullptr, "PIXAL3D_BACKEND_TRACE") == ProfileMode::trace,
          "legacy backend trace lost");
    set_env("PIXAL3D_PROFILE", "SuMmArY");
    check(profile_mode_from_environment() == ProfileMode::summary,
          "profile mode is not case-insensitive");
    set_env("PIXAL3D_PROFILE", nullptr);
    for (const char * disabled : {"0", "off", "false", ""}) {
        set_env("PIXAL3D_BACKEND_TRACE", disabled);
        check(profile_mode_from_environment(nullptr, "PIXAL3D_BACKEND_TRACE") == ProfileMode::off,
              "disabled legacy flag enabled profiling");
    }
    set_env("PIXAL3D_SLAT_DECODER_PROFILE", "1");
    check(profile_mode_from_environment("PIXAL3D_SLAT_DECODER_PROFILE") == ProfileMode::trace,
          "legacy detailed profile lost");
    set_env("PIXAL3D_PROFILE", "summary");
    check(profile_mode_from_environment("PIXAL3D_SLAT_DECODER_PROFILE") == ProfileMode::summary,
          "global summary did not override legacy detail");
    set_env("PIXAL3D_PROFILE", "off");
    check(profile_mode_from_environment("PIXAL3D_SLAT_DECODER_PROFILE") == ProfileMode::off,
          "global off did not override legacy detail");
    set_env("PIXAL3D_PROFILE", "trace");
    check(profile_mode_from_environment() == ProfileMode::trace, "global trace rejected");
    set_env("PIXAL3D_PROFILE", "typo");
    {
        Capture capture;
        check(profile_mode_from_environment() == ProfileMode::off, "invalid mode not off");
        check(capture.finish().find("invalid PIXAL3D_PROFILE") != std::string::npos,
              "invalid mode was silent");
    }
    set_env("PIXAL3D_PROFILE", nullptr);
    set_env("PIXAL3D_SLAT_DECODER_PROFILE", nullptr);
    set_env("PIXAL3D_VALIDATE_SPARSE_MAP_CACHE", "1");
    check(profile_mode_from_environment("PIXAL3D_SLAT_DECODER_PROFILE",
                                       "PIXAL3D_SLAT_DECODER_TRACE") == ProfileMode::off,
          "validation unexpectedly enabled profiling");
}

void test_timers() {
    pixal3d::ProfileTimer disabled(false, fake_clock);
    for (int index = 0; index < 1000; ++index) {
        check(disabled.elapsed_ms() == 0.0, "inactive timer has elapsed time");
    }
    check(clock_calls == 0, "inactive timer read the clock");
    pixal3d::ProfileTimer enabled(true, fake_clock);
    check(clock_calls == 1 && enabled.elapsed_ms() == 1.0 && clock_calls == 2,
          "active timer did not use clock");

    // Inactive sparse scopes must not record or allocate per-op records.
    pixal3d::detail::SparseProfileStats stats;
    {
        pixal3d::detail::SparseProfileScope active(&stats);
        {
            pixal3d::detail::SparseProfileScope inactive(nullptr);
            auto timer = pixal3d::detail::make_sparse_profile_timer(
                &pixal3d::detail::SparseProfileStats::valid_calls,
                &pixal3d::detail::SparseProfileStats::valid_ms);
        }
        check(stats.valid_calls == 0 && stats.validation_records.empty(),
              "inactive sparse scope recorded work");
        auto timer = pixal3d::detail::make_sparse_profile_timer(
            &pixal3d::detail::SparseProfileStats::valid_calls,
            &pixal3d::detail::SparseProfileStats::valid_ms);
        auto moved = std::move(timer);
        moved.stop();
        moved.stop();
    }
    check(stats.valid_calls == 1 && pixal3d::detail::sparse_profile_current() == nullptr,
          "sparse timer ownership/scope restoration failed");
}

void test_scope(pixal3d::ProfileMode mode) {
    Capture capture;
    {
        pixal3d::ProfileScope scope(mode, "fixture");
        scope.phase("work");
    }
    const std::string text = capture.finish();
    check((text.find("wall_ms=") != std::string::npos) == (mode != pixal3d::ProfileMode::off),
          "scope summary mode mismatch");
    check((text.find("phase=work") != std::string::npos) == (mode == pixal3d::ProfileMode::trace),
          "scope detail mode mismatch");
    if (mode == pixal3d::ProfileMode::off) check(text.empty(), "inactive scope logged");
}

void test_graph(const char * setting, pixal3d::ProfileMode expected) {
    Environment environment;
    set_env("PIXAL3D_PROFILE", setting);
    Capture capture;
    pixal3d::BackendManager manager;
    std::string error;
    check(manager.initialize({pixal3d::BackendPolicyKind::cpu, 0}, &error), error.c_str());
    check(manager.profile_mode() == expected, "manager mode mismatch");
    // The manager snapshots config; later environment changes do not alter it.
    set_env("PIXAL3D_PROFILE", "off");
    check(manager.profile_mode() == expected, "manager did not snapshot mode");
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false);
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    check(context != nullptr, "ggml init failed");
    {
        ggml_tensor * left = ggml_new_tensor_1d(context, GGML_TYPE_F32, 2);
        ggml_tensor * right = ggml_new_tensor_1d(context, GGML_TYPE_F32, 2);
        ggml_tensor * result = ggml_add(context, left, right);
        ggml_set_name(result, "profile_fixture.add");
        ggml_set_input(left);
        ggml_set_input(right);
        ggml_set_output(result);
        ggml_cgraph * graph = ggml_new_graph_custom(context, 8, false);
        ggml_build_forward_expand(graph, result);
        pixal3d::BackendScheduler scheduler(manager, 8, false, true, &error, "profile-fixture");
        check(scheduler.valid() && scheduler.require_primary(result, &error) &&
                  scheduler.allocate_graph(graph, &error), error.c_str());
        for (int iteration = 0; iteration < 2; ++iteration) {
            const float lhs[] = {1.25f, -2.0f}, rhs[] = {2.75f, 0.5f};
            ggml_backend_tensor_set(left, lhs, 0, sizeof(lhs));
            ggml_backend_tensor_set(right, rhs, 0, sizeof(rhs));
            check(scheduler.compute(graph, &error) == GGML_STATUS_SUCCESS, error.c_str());
            check(scheduler.tensor_backend(result) == manager.primary(), "placement changed");
            float values[2] = {};
            ggml_backend_tensor_get(result, values, 0, sizeof(values));
            check(values[0] == 4.0f && values[1] == -1.5f, "profiling changed F32 output");
        }
    }
    ggml_free(context);
    manager.close();
    manager.close(); // Summary must be flushed only once.
    const std::string text = capture.finish();
    check(text.find("policy=cpu") != std::string::npos, "profiling suppressed required diagnostics");
    const std::string summary = "profile stage=backend";
    const auto first = text.find(summary);
    check((first != std::string::npos) == (expected != pixal3d::ProfileMode::off),
          "backend summary mode mismatch");
    if (first != std::string::npos) {
        check(text.find(summary, first + 1) == std::string::npos, "duplicate summary");
        check(text.find("allocations=1 computes=2") != std::string::npos,
              "scheduler timing was not aggregated");
    }
    check((text.find("scheduler node=") != std::string::npos) ==
              (expected == pixal3d::ProfileMode::trace), "node trace mode mismatch");
    check((text.find("backend timing") != std::string::npos) ==
              (expected == pixal3d::ProfileMode::trace), "timing trace mode mismatch");
}

void test_mesh() {
    Environment environment;
    const std::vector<float> vertices = {0, 0, 0, 1, 0, 0, 0, 1, 0};
    const std::vector<std::int32_t> faces = {0, 1, 2};
    const std::vector<float> pbr = {1, 0, 0, 0, 0.5f, 1,
                                  1, 0, 0, 0, 0.5f, 1,
                                  1, 0, 0, 0, 0.5f, 1};
    pixal3d::BakedMesh golden;
    for (const char * setting : {"off", "summary", "trace"}) {
        set_env("PIXAL3D_PROFILE", setting);
        Capture capture;
        auto mesh = pixal3d::uv_bake(vertices, 3, faces, 1, pbr, 32, nullptr);
        const auto text = capture.finish();
        check(mesh.ok(), "tiny UV bake failed");
        const bool off = std::string(setting) == "off";
        const bool trace = std::string(setting) == "trace";
        check((text.find("profile stage=uv_bake wall_ms=") != std::string::npos) == !off,
              "UV summary mode mismatch");
        check((text.find("phase=xatlas_pack_charts") != std::string::npos) == trace,
              "UV detail mode mismatch");
        if (off) {
            check(text.find("uv_bake phase=") == std::string::npos, "old unconditional timer survived");
            golden = std::move(mesh);
        } else {
            check(mesh.T == golden.T && mesh.verts == golden.verts && mesh.faces == golden.faces &&
                      mesh.uv == golden.uv && mesh.base == golden.base && mesh.mr == golden.mr,
                  "profiling changed UV or texture output");
        }
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--mesh") {
            test_mesh();
        } else {
            check(argc == 1, "unexpected arguments");
            test_configuration();
            test_timers();
            for (auto mode : {pixal3d::ProfileMode::off, pixal3d::ProfileMode::summary,
                              pixal3d::ProfileMode::trace}) test_scope(mode);
            test_graph(nullptr, pixal3d::ProfileMode::off);
            test_graph("off", pixal3d::ProfileMode::off);
            test_graph("summary", pixal3d::ProfileMode::summary);
            test_graph("trace", pixal3d::ProfileMode::trace);
        }
        std::puts("profile tests passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "profile test: %s\n", error.what());
        return 1;
    }
}
