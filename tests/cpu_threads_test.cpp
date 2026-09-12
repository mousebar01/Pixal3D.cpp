#include "pixal3d/cpu_threads.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void set_env(const char * name, const char * value) {
    if (setenv(name, value, 1) != 0) throw std::runtime_error("setenv failed");
}

void unset_env(const char * name) {
    unsetenv(name);
}

void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        pixal3d::clear_cpu_thread_override();
        set_env("PIXAL3D_CPU_THREADS", "7");
        set_env("OMP_NUM_THREADS", "3,2");
        auto policy = pixal3d::cpu_thread_policy_from_environment();
        check(policy.threads == 7 && policy.source == "PIXAL3D_CPU_THREADS",
              "PIXAL3D_CPU_THREADS did not take precedence");

        set_env("PIXAL3D_CPU_THREADS", "invalid");
        policy = pixal3d::cpu_thread_policy_from_environment();
        check(policy.threads == 3 && policy.source == "OMP_NUM_THREADS",
              "OMP_NUM_THREADS fallback did not parse its first value");
        check(!policy.warning.empty(), "invalid CPU thread configuration was not recorded");

        int threads = 0;
        std::string error;
        check(pixal3d::parse_cpu_thread_count(" 12 ", threads, &error) && threads == 12,
              "positive CLI thread count did not parse");
        check(!pixal3d::parse_cpu_thread_count("0", threads, &error) && !error.empty(),
              "zero CLI thread count was accepted");
        check(!pixal3d::parse_cpu_thread_count("3x", threads, &error) && !error.empty(),
              "invalid CLI thread suffix was accepted");

        check(pixal3d::set_cpu_thread_override(9, &error), error.c_str());
        policy = pixal3d::cpu_thread_policy_from_environment();
        check(policy.threads == 9 && policy.source == "override",
              "explicit CPU thread override did not take precedence");
        check(pixal3d::cpu_thread_count() == 9, "cpu_thread_count ignored override");
        check(pixal3d::cpu_should_parallelize(9 * 256),
              "large operation was classified as serial");
        check(!pixal3d::cpu_should_parallelize(9 * 256 - 1),
              "small operation was classified as parallel");

        pixal3d::clear_cpu_thread_override();
        unset_env("PIXAL3D_CPU_THREADS");
        unset_env("OMP_NUM_THREADS");
        check(pixal3d::cpu_thread_count() == pixal3d::default_cpu_thread_count(),
              "default CPU thread count mismatch");
        std::cout << "cpu thread tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "cpu thread test: " << error.what() << '\n';
        return 1;
    }
}
