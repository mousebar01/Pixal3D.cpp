#include "pixal3d/dino.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

void put_u32(std::ofstream & file, std::uint32_t value) {
    const char bytes[4] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
        static_cast<char>((value >> 16) & 0xffu),
        static_cast<char>((value >> 24) & 0xffu),
    };
    file.write(bytes, sizeof(bytes));
}

bool write_condition(const std::string & path,
                     const std::vector<std::uint32_t> & shape,
                     const std::vector<float> & values,
                     std::uint32_t dtype = 0) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) return false;
    file.write("DINOCOND", 8);
    put_u32(file, 7);
    put_u32(file, dtype);
    put_u32(file, static_cast<std::uint32_t>(shape.size()));
    for (std::uint32_t dimension : shape) put_u32(file, dimension);
    file.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
    return static_cast<bool>(file);
}

bool close(float left, float right) {
    return std::fabs(left - right) < 1e-6f;
}

} // namespace

int main() {
    const std::string path = "/tmp/pixal3d-dino-condition-test.dinodata";
    const std::vector<float> values = {-1.0f, 0.5f, 2.0f, 4.0f, -3.0f, 1.5f};
    if (!write_condition(path, {1, 2, 3}, values)) return 1;
    pixal3d::DinoConditionF32 condition;
    std::string error;
    if (!pixal3d::load_dino_condition(path, condition, &error)) return 1;
    if (condition.format_version != 7 || condition.shape !=
        std::vector<std::int64_t>({1, 2, 3}) || condition.global.batch_size != 1 ||
        condition.global.channels != 3 || condition.global.tokens() != 2 ||
        condition.global.feats != values) return 1;
    const auto fp = pixal3d::fingerprint_dino_condition(condition);
    if (!close(fp.minimum, -3.0f) || !close(fp.maximum, 4.0f) ||
        !close(static_cast<float>(fp.mean), 4.0f / 6.0f) ||
        !close(static_cast<float>(fp.sum), 4.0f) || fp.count != values.size()) return 1;

    if (!write_condition(path, {2, 3}, values)) return 1;
    if (!pixal3d::load_dino_condition(path, condition, &error) ||
        condition.shape != std::vector<std::int64_t>({2, 3}) ||
        condition.global.tokens() != 2 || condition.global.channels != 3) return 1;

    if (!write_condition(path, {1, 2, 3}, values, 1) ||
        pixal3d::load_dino_condition(path, condition, &error)) return 1;
    std::remove(path.c_str());
    return 0;
}
