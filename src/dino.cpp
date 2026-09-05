#include "pixal3d/dino.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

namespace pixal3d {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

bool read_u32(const std::uint8_t * & cursor, const std::uint8_t * end,
              std::uint32_t & value) {
    if (static_cast<std::size_t>(end - cursor) < 4) return false;
    value = static_cast<std::uint32_t>(cursor[0]) |
            (static_cast<std::uint32_t>(cursor[1]) << 8) |
            (static_cast<std::uint32_t>(cursor[2]) << 16) |
            (static_cast<std::uint32_t>(cursor[3]) << 24);
    cursor += 4;
    return true;
}

bool checked_product(std::size_t left, std::size_t right, std::size_t & output) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        return false;
    }
    output = left * right;
    return true;
}

} // namespace

bool load_dino_condition(const std::string & path,
                         DinoConditionF32 & output,
                         std::string * error) {
    output = DinoConditionF32{};
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        set_error(error, "cannot open DINO condition file: " + path);
        return false;
    }
    const std::vector<std::uint8_t> bytes(
        (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (bytes.size() < 8 + 3 * sizeof(std::uint32_t)) {
        set_error(error, "DINO condition file is too small for its header");
        return false;
    }
    static constexpr char magic[] = {'D', 'I', 'N', 'O', 'C', 'O', 'N', 'D'};
    if (std::memcmp(bytes.data(), magic, sizeof(magic)) != 0) {
        set_error(error, "bad DINO condition magic (expected DINOCOND)");
        return false;
    }
    const std::uint8_t * cursor = bytes.data() + sizeof(magic);
    const std::uint8_t * end = bytes.data() + bytes.size();
    std::uint32_t version = 0;
    std::uint32_t dtype = 0;
    std::uint32_t ndim = 0;
    if (!read_u32(cursor, end, version) || !read_u32(cursor, end, dtype) ||
        !read_u32(cursor, end, ndim)) {
        set_error(error, "truncated DINO condition header");
        return false;
    }
    if (dtype != 0) {
        set_error(error, "unsupported DINO condition dtype (only F32 is supported)");
        return false;
    }
    if (ndim != 2 && ndim != 3) {
        set_error(error, "DINO condition shape must have two or three dimensions");
        return false;
    }
    std::vector<std::int64_t> shape(ndim);
    std::size_t elements = 1;
    for (std::uint32_t index = 0; index < ndim; ++index) {
        std::uint32_t dimension = 0;
        if (!read_u32(cursor, end, dimension) || dimension == 0 ||
            !checked_product(elements, static_cast<std::size_t>(dimension), elements)) {
            set_error(error, "invalid or overflowing DINO condition shape");
            return false;
        }
        shape[index] = static_cast<std::int64_t>(dimension);
    }
    const std::size_t payload_bytes = end - cursor;
    std::size_t wanted_bytes = 0;
    if (!checked_product(elements, sizeof(float), wanted_bytes) ||
        payload_bytes < wanted_bytes) {
        set_error(error, "truncated DINO condition payload");
        return false;
    }
    const std::size_t batch = ndim == 3 ? static_cast<std::size_t>(shape[0]) : 1;
    const std::size_t tokens = ndim == 3 ? static_cast<std::size_t>(shape[1]) :
                                           static_cast<std::size_t>(shape[0]);
    const std::size_t channels = ndim == 3 ? static_cast<std::size_t>(shape[2]) :
                                             static_cast<std::size_t>(shape[1]);
    if (batch != 1 || channels > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        tokens > std::numeric_limits<std::size_t>::max() / channels) {
        set_error(error, "DINO condition must contain one batch and valid token channels");
        return false;
    }

    output.format_version = version;
    output.shape = std::move(shape);
    output.global.batch_size = 1;
    output.global.channels = static_cast<int>(channels);
    output.global.offsets = {0, tokens};
    output.global.feats.resize(elements);
    std::memcpy(output.global.feats.data(), cursor, wanted_bytes);
    for (float value : output.global.feats) {
        if (!std::isfinite(value)) {
            output = DinoConditionF32{};
            set_error(error, "DINO condition payload contains a non-finite value");
            return false;
        }
    }
    return true;
}

DinoFingerprintF32 fingerprint_dino_condition(const DinoConditionF32 & condition) {
    DinoFingerprintF32 fingerprint;
    fingerprint.count = condition.global.feats.size();
    if (fingerprint.count == 0) return fingerprint;
    fingerprint.minimum = std::numeric_limits<float>::infinity();
    fingerprint.maximum = -std::numeric_limits<float>::infinity();
    for (float value : condition.global.feats) {
        fingerprint.minimum = std::min(fingerprint.minimum, value);
        fingerprint.maximum = std::max(fingerprint.maximum, value);
        fingerprint.sum += static_cast<double>(value);
        fingerprint.l2 += static_cast<double>(value) * static_cast<double>(value);
    }
    fingerprint.mean = fingerprint.sum / static_cast<double>(fingerprint.count);
    fingerprint.l2 = std::sqrt(fingerprint.l2);
    return fingerprint;
}

} // namespace pixal3d
