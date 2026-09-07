#include "pixal3d/slat_flow.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

std::vector<std::int32_t> generated_coords(std::size_t count, int resolution) {
    std::vector<std::int32_t> result;
    result.reserve(count * 4);
    const std::size_t side = static_cast<std::size_t>(resolution);
    const std::size_t plane = side * side;
    for (std::size_t index = 0; index < count; ++index) {
        const int z = static_cast<int>((index / plane) % side);
        const std::size_t rem = index % plane;
        const int y = static_cast<int>(rem / side);
        const int x = static_cast<int>(rem % side);
        result.insert(result.end(), {0, x, y, z});
    }
    return result;
}

struct Fingerprint {
    std::uint64_t hash = 1469598103934665603ull;
    double sum = 0.0;
    double squared_sum = 0.0;
    bool finite = true;
};

Fingerprint fingerprint(const std::vector<float> & values) {
    Fingerprint result;
    for (float value : values) {
        result.finite = result.finite && std::isfinite(value);
        result.sum += static_cast<double>(value);
        result.squared_sum += static_cast<double>(value) * static_cast<double>(value);
        std::uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "unexpected float size");
        std::memcpy(&bits, &value, sizeof(bits));
        for (int byte = 0; byte < 4; ++byte) {
            result.hash ^= static_cast<std::uint8_t>(bits >> (byte * 8));
            result.hash *= 1099511628211ull;
        }
    }
    return result;
}

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void print_stats(const char * name, std::vector<double> values) {
    if (values.empty()) return;
    std::sort(values.begin(), values.end());
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    const auto percentile = [&values](double fraction) {
        const std::size_t index = static_cast<std::size_t>(
            fraction * static_cast<double>(values.size() - 1));
        return values[index];
    };
    std::cout << name
              << " count=" << values.size()
              << " min_ms=" << values.front()
              << " mean_ms=" << (sum / static_cast<double>(values.size()))
              << " median_ms=" << percentile(0.50)
              << " p95_ms=" << percentile(0.95)
              << " max_ms=" << values.back() << "\n";
}

bool parse_size(const char * text, std::size_t & value) {
    if (!text || !*text) return false;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0) return false;
    value = static_cast<std::size_t>(parsed);
    return static_cast<unsigned long long>(value) == parsed;
}

std::vector<std::int32_t> unused_coordinate(const pixal3d::SparseTensorF32 & input) {
    for (int z = 0; z < input.spatial_z; ++z) {
        for (int y = 0; y < input.spatial_y; ++y) {
            for (int x = 0; x < input.spatial_x; ++x) {
                bool used = false;
                for (std::size_t point = 0; point < input.points(); ++point) {
                    const std::size_t base = point * 4;
                    if (input.coords[base + 0] == 0 && input.coords[base + 1] == x &&
                        input.coords[base + 2] == y && input.coords[base + 3] == z) {
                        used = true;
                        break;
                    }
                }
                if (!used) return {0, x, y, z};
            }
        }
    }
    return {};
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 7) {
        std::cerr << "usage: pixal3d_slat_flow_cuda_benchmark <pack> "
                     "[component] [points] [warmup] [iterations] [--coords-regression]\n";
        return 2;
    }

    const bool coordinate_regression = argc == 7 &&
        std::string(argv[6]) == "--coords-regression";
    if (argc == 7 && !coordinate_regression) {
        std::cerr << "unknown benchmark option: " << argv[6] << "\n";
        return 2;
    }
    const std::string component = argc >= 3 ? argv[2] : "shape_flow_512";
    std::size_t points = 512;
    std::size_t warmup = 2;
    std::size_t iterations = 5;
    if (argc >= 4 && !parse_size(argv[3], points)) {
        std::cerr << "invalid point count: " << argv[3] << "\n";
        return 2;
    }
    if (argc >= 5 && !parse_size(argv[4], warmup)) {
        std::cerr << "invalid warmup count: " << argv[4] << "\n";
        return 2;
    }
    if (argc >= 6 && !parse_size(argv[5], iterations)) {
        std::cerr << "invalid iteration count: " << argv[5] << "\n";
        return 2;
    }

    const char * backend_policy = std::getenv("PIXAL3D_SLAT_BACKEND");
    if (!backend_policy || !*backend_policy) {
        if (setenv("PIXAL3D_SLAT_BACKEND", "gpu:0", 0) != 0) {
            std::cerr << "failed to set default PIXAL3D_SLAT_BACKEND=gpu:0\n";
            return 1;
        }
        backend_policy = "gpu:0";
    }

    pixal3d::SLatFlowModel model;
    std::string error;
    const auto load_begin = Clock::now();
    if (!model.load(argv[1], component, true, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    const double load_ms = elapsed_ms(load_begin, Clock::now());
    const pixal3d::SLatFlowHParams & hp = model.hparams();
    if (hp.resolution <= 0 || hp.in_channels <= 0 || hp.cond_channels <= 0 ||
        hp.proj_in_channels < 0) {
        std::cerr << "invalid SLat flow hyperparameters\n";
        return 1;
    }

    pixal3d::SparseTensorF32 input;
    input.batch_size = 1;
    input.channels = hp.in_channels;
    input.spatial_x = input.spatial_y = input.spatial_z = hp.resolution;
    input.coords = generated_coords(points, hp.resolution);
    input.feats.resize(points * static_cast<std::size_t>(input.channels));
    for (std::size_t index = 0; index < input.feats.size(); ++index) {
        input.feats[index] = -0.27f + 0.013f * static_cast<float>(index % 257);
    }

    pixal3d::SparseTensorF32 projection;
    projection.batch_size = 1;
    projection.channels = hp.proj_in_channels;
    projection.spatial_x = projection.spatial_y = projection.spatial_z = hp.resolution;
    projection.coords = input.coords;
    projection.feats.resize(points * static_cast<std::size_t>(projection.channels));
    for (std::size_t index = 0; index < projection.feats.size(); ++index) {
        projection.feats[index] = 0.19f - 0.017f * static_cast<float>(index % 257);
    }

    pixal3d::VarLenTensorF32 global;
    global.batch_size = 1;
    global.channels = hp.cond_channels;
    global.offsets = {0, 5};
    global.feats.resize(5 * static_cast<std::size_t>(global.channels));
    for (std::size_t index = 0; index < global.feats.size(); ++index) {
        global.feats[index] = -0.11f + 0.021f * static_cast<float>(index % 257);
    }
    const float timestep = 0.17f;
    const pixal3d::SparseTensorF32 * projection_ptr =
        hp.image_attn_mode == "proj" ? &projection : nullptr;

    pixal3d::SparseTensorF32 output;
    const auto cold_begin = Clock::now();
    if (!model.forward(input, &timestep, 1, global, projection_ptr, output, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    const double cold_ms = elapsed_ms(cold_begin, Clock::now());
    const Fingerprint cold_fingerprint = fingerprint(output.feats);
    if (!cold_fingerprint.finite) {
        std::cerr << "cold output contains non-finite values\n";
        return 1;
    }

    if (coordinate_regression) {
        const auto original_coords = input.coords;
        const auto replacement = unused_coordinate(input);
        if (replacement.empty()) {
            std::cerr << "coordinate regression requires an unused coordinate\n";
            return 1;
        }
        const std::size_t point = points / 2;
        const auto run_regression_forward = [&](const char * label) {
            if (!model.forward(input, &timestep, 1, global, projection_ptr,
                               output, &error)) {
                std::cerr << "coordinate regression " << label << " failed: "
                          << error << "\n";
                return Fingerprint{};
            }
            const Fingerprint result = fingerprint(output.feats);
            if (!result.finite) {
                std::cerr << "coordinate regression " << label
                          << " output contains non-finite values\n";
            }
            std::cout << "coordinate_regression_step=" << label
                      << " hash=0x" << std::hex << result.hash << std::dec
                      << " finite=" << (result.finite ? 1 : 0) << "\n";
            return result;
        };

        const Fingerprint repeated = run_regression_forward("A_repeat");
        input.coords[point * 4 + 0] = replacement[0];
        input.coords[point * 4 + 1] = replacement[1];
        input.coords[point * 4 + 2] = replacement[2];
        input.coords[point * 4 + 3] = replacement[3];
        projection.coords = input.coords;
        const Fingerprint changed = run_regression_forward("B_changed_coord");
        input.coords = original_coords;
        projection.coords = original_coords;
        const Fingerprint restored = run_regression_forward("A_restored");
        const bool repeated_ok = repeated.finite && repeated.hash == cold_fingerprint.hash;
        const bool restored_ok = restored.finite && restored.hash == cold_fingerprint.hash;
        std::cout << "coordinate_regression= "
                  << (repeated_ok && restored_ok ? "PASS" : "FAIL")
                  << " repeated_matches=" << (repeated_ok ? 1 : 0)
                  << " changed_differs="
                  << (changed.finite && changed.hash != cold_fingerprint.hash ? 1 : 0)
                  << " restored_matches=" << (restored_ok ? 1 : 0) << "\n";
        if (!repeated_ok || !restored_ok) return 1;
    }

    std::vector<double> warm_times;
    warm_times.reserve(iterations);
    Fingerprint final_fingerprint;
    for (std::size_t index = 0; index < warmup + iterations; ++index) {
        const auto begin = Clock::now();
        if (!model.forward(input, &timestep, 1, global, projection_ptr, output, &error)) {
            std::cerr << error << "\n";
            return 1;
        }
        const double elapsed = elapsed_ms(begin, Clock::now());
        final_fingerprint = fingerprint(output.feats);
        if (!final_fingerprint.finite) {
            std::cerr << "warm output contains non-finite values\n";
            return 1;
        }
        if (index >= warmup) warm_times.push_back(elapsed);
    }

    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::cout << "benchmark=slat_flow_cuda"
              << " component=" << component
              << " backend_policy=" << backend_policy
              << " points=" << points
              << " warmup=" << warmup
              << " iterations=" << iterations << "\n";
    std::cout << "load_ms=" << load_ms << " cold_forward_ms=" << cold_ms << "\n";
    std::cout << "cold_hash=0x" << std::hex << cold_fingerprint.hash << std::dec
              << " cold_sum=" << cold_fingerprint.sum
              << " cold_l2_sq=" << cold_fingerprint.squared_sum << "\n";
    std::cout << "warm_hash=0x" << std::hex << final_fingerprint.hash << std::dec
              << " warm_sum=" << final_fingerprint.sum
              << " warm_l2_sq=" << final_fingerprint.squared_sum
              << " stable=" << (cold_fingerprint.hash == final_fingerprint.hash ? 1 : 0)
              << "\n";
    print_stats("warm_forward", std::move(warm_times));
    return cold_fingerprint.hash == final_fingerprint.hash ? 0 : 1;
}
