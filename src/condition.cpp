#include "pixal3d/condition.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>

namespace pixal3d {
namespace {

constexpr char kMagic[8] = {'P', '3', 'D', 'C', 'O', 'N', 'D', '\0'};
constexpr char kMultiViewMagic[8] = {'P', '3', 'D', 'M', 'V', 'C', 'O', 'N'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kDinoMap = 0;
constexpr std::uint32_t kNafMap = 1;

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

bool read_exact(std::ifstream & file, void * destination, std::size_t bytes) {
    if (bytes == 0) return true;
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        return false;
    }
    file.read(static_cast<char *>(destination), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(file);
}

bool read_u32(std::ifstream & file, std::uint32_t & value) {
    std::uint8_t bytes[4]{};
    if (!read_exact(file, bytes, sizeof(bytes))) return false;
    value = static_cast<std::uint32_t>(bytes[0]) |
            (static_cast<std::uint32_t>(bytes[1]) << 8) |
            (static_cast<std::uint32_t>(bytes[2]) << 16) |
            (static_cast<std::uint32_t>(bytes[3]) << 24);
    return true;
}

bool read_u64(std::ifstream & file, std::uint64_t & value) {
    std::uint8_t bytes[8]{};
    if (!read_exact(file, bytes, sizeof(bytes))) return false;
    value = 0;
    for (int index = 0; index < 8; ++index) {
        value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8);
    }
    return true;
}

bool read_f32(std::ifstream & file, float & value) {
    std::uint32_t bits = 0;
    if (!read_u32(file, bits)) return false;
    std::memcpy(&value, &bits, sizeof(value));
    return true;
}

bool checked_product(std::size_t left, std::size_t right, std::size_t & output) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        return false;
    }
    output = left * right;
    return true;
}

bool payload_available(std::ifstream & file, std::size_t bytes) {
    const std::streampos current = file.tellg();
    if (current < 0) return false;
    file.seekg(0, std::ios::end);
    const std::streampos end = file.tellg();
    file.seekg(current, std::ios::beg);
    if (end < current || end < 0) return false;
    const std::uintmax_t remaining = static_cast<std::uintmax_t>(end - current);
    return bytes <= remaining;
}

bool read_payload(std::ifstream & file, std::size_t elements,
                  std::vector<float> & output, std::string * error,
                  const char * what) {
    std::size_t bytes = 0;
    if (!checked_product(elements, sizeof(float), bytes)) {
        set_error(error, std::string("condition ") + what + " payload size overflows");
        return false;
    }
    if (!payload_available(file, bytes)) {
        set_error(error, std::string("truncated condition ") + what + " payload");
        return false;
    }
    output.resize(elements);
    if (!read_exact(file, output.data(), bytes)) {
        output.clear();
        set_error(error, std::string("truncated condition ") + what + " payload");
        return false;
    }
    for (float value : output) {
        if (!std::isfinite(value)) {
            output.clear();
            set_error(error, std::string("condition ") + what + " payload is non-finite");
            return false;
        }
    }
    return true;
}

bool read_map(std::ifstream & file, Pixal3DFeatureMapF32 & map,
              std::string * error, const char * role) {
    std::uint32_t image_resolution = 0;
    std::uint32_t height = 0;
    std::uint32_t width = 0;
    std::uint32_t channels = 0;
    if (!read_u32(file, image_resolution) || !read_u32(file, height) ||
        !read_u32(file, width) || !read_u32(file, channels)) {
        set_error(error, std::string("truncated ") + role + " feature-map header");
        return false;
    }
    if (image_resolution == 0 || height == 0 || width == 0 || channels == 0 ||
        image_resolution > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        height > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        width > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        channels > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        set_error(error, std::string("invalid ") + role + " feature-map dimensions");
        return false;
    }
    std::size_t elements = 0;
    if (!checked_product(static_cast<std::size_t>(height),
                         static_cast<std::size_t>(width), elements) ||
        !checked_product(elements, static_cast<std::size_t>(channels), elements)) {
        set_error(error, std::string(role) + " feature-map dimensions overflow");
        return false;
    }
    map.image_resolution = static_cast<int>(image_resolution);
    map.height = static_cast<int>(height);
    map.width = static_cast<int>(width);
    map.channels = static_cast<int>(channels);
    return read_payload(file, elements, map.features, error, role);
}

bool read_camera(std::ifstream & file, ProjectionCamera & camera,
                 std::string * error, const std::string & stage_name) {
    std::uint32_t has_transform = 0;
    if (!read_f32(file, camera.camera_angle_x) ||
        !read_f32(file, camera.distance) ||
        !read_f32(file, camera.mesh_scale) ||
        !read_u32(file, has_transform) || has_transform > 1) {
        set_error(error, "truncated or invalid multi-view camera: " + stage_name);
        return false;
    }
    camera.has_transform = has_transform != 0;
    if (!camera.has_transform) {
        set_error(error, "multi-view camera must contain an absolute c2w transform: " +
                           stage_name);
        return false;
    }
    for (float & value : camera.transform_matrix) {
        if (!read_f32(file, value)) {
            set_error(error, "truncated multi-view camera transform: " + stage_name);
            return false;
        }
    }
    return true;
}

bool at_end(std::ifstream & file) {
    char extra = 0;
    file.read(&extra, 1);
    return file.eof();
}

bool write_u32(std::ofstream & file, std::uint32_t value) {
    std::uint8_t bytes[4] = {
        static_cast<std::uint8_t>(value & 0xffu),
        static_cast<std::uint8_t>((value >> 8) & 0xffu),
        static_cast<std::uint8_t>((value >> 16) & 0xffu),
        static_cast<std::uint8_t>((value >> 24) & 0xffu),
    };
    file.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
    return static_cast<bool>(file);
}

bool write_f32(std::ofstream & file, float value) {
    if (!std::isfinite(value)) return false;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return write_u32(file, bits);
}

bool write_payload(std::ofstream & file, const std::vector<float> & values) {
    if (values.size() > static_cast<std::size_t>(
            std::numeric_limits<std::streamsize>::max()) / sizeof(float)) {
        return false;
    }
    if (!values.empty()) {
        file.write(reinterpret_cast<const char *>(values.data()),
                   static_cast<std::streamsize>(values.size() * sizeof(float)));
    }
    return static_cast<bool>(file);
}

} // namespace

bool Pixal3DFeatureMapF32::valid(std::string * error) const {
    if (image_resolution <= 0 || height <= 0 || width <= 0 || channels <= 0) {
        set_error(error, "feature map dimensions must be positive");
        return false;
    }
    std::size_t elements = 0;
    if (!checked_product(static_cast<std::size_t>(height),
                         static_cast<std::size_t>(width), elements) ||
        !checked_product(elements, static_cast<std::size_t>(channels), elements) ||
        features.size() != elements) {
        set_error(error, "feature map payload size does not match dimensions");
        return false;
    }
    for (float value : features) {
        if (!std::isfinite(value)) {
            set_error(error, "feature map contains a non-finite value");
            return false;
        }
    }
    return true;
}

bool Pixal3DConditionStageF32::valid(std::string * error) const {
    if (name.empty()) {
        set_error(error, "condition stage name is empty");
        return false;
    }
    if (!global.valid(error) || global.batch_size != 1 || global.tokens() == 0) {
        if (!error || error->empty()) set_error(error, "condition global tensor is invalid");
        return false;
    }
    if (!dino_map.valid(error)) return false;
    if (!naf_map.empty() && !naf_map.valid(error)) return false;
    if (!naf_map.empty() && naf_map.image_resolution <= 0) {
        set_error(error, "NAF feature map image resolution is invalid");
        return false;
    }
    return true;
}

bool Pixal3DConditionViewF32::valid(std::string * error) const {
    if (!(camera.camera_angle_x > 0.0f && camera.camera_angle_x < 3.14159265358979323846f) ||
        !std::isfinite(camera.camera_angle_x) ||
        !(camera.distance > 0.0f) || !std::isfinite(camera.distance) ||
        !(camera.mesh_scale > 0.0f) || !std::isfinite(camera.mesh_scale)) {
        set_error(error, "multi-view camera parameters are invalid");
        return false;
    }
    if (!camera.has_transform) {
        set_error(error, "multi-view camera must provide an absolute c2w transform");
        return false;
    }
    for (float value : camera.transform_matrix) {
        if (!std::isfinite(value)) {
            set_error(error, "multi-view camera transform contains a non-finite value");
            return false;
        }
    }
    if (!global.valid(error) || global.batch_size != 1 || global.tokens() == 0) {
        if (!error || error->empty()) set_error(error, "multi-view global tensor is invalid");
        return false;
    }
    if (!dino_map.valid(error)) return false;
    if (!naf_map.empty() && !naf_map.valid(error)) return false;
    return true;
}

bool Pixal3DConditionStageMVF32::valid(std::string * error) const {
    if (name.empty()) {
        set_error(error, "multi-view condition stage name is empty");
        return false;
    }
    if (views.empty() || views.size() > 64) {
        set_error(error, "multi-view condition stage must contain between one and 64 views");
        return false;
    }
    std::size_t global_elements = 0;
    int global_channels = 0;
    std::size_t global_tokens = 0;
    int dino_channels = 0;
    int naf_channels = 0;
    bool have_naf = false;
    float mesh_scale = 0.0f;
    for (std::size_t index = 0; index < views.size(); ++index) {
        const Pixal3DConditionViewF32 & view = views[index];
        if (!view.valid(error)) return false;
        if (index == 0) {
            global_channels = view.global.channels;
            global_tokens = view.global.tokens();
            dino_channels = view.dino_map.channels;
            have_naf = view.has_naf();
            naf_channels = have_naf ? view.naf_map.channels : 0;
            mesh_scale = view.camera.mesh_scale;
        } else if (view.global.channels != global_channels ||
                   view.global.tokens() != global_tokens ||
                   view.dino_map.channels != dino_channels ||
                   view.has_naf() != have_naf ||
                   (have_naf && view.naf_map.channels != naf_channels) ||
                   std::fabs(view.camera.mesh_scale - mesh_scale) > 1e-6f) {
            set_error(error, "multi-view condition feature shapes or mesh scale do not match");
            return false;
        }
    }
    if (!checked_product(global_tokens, static_cast<std::size_t>(global_channels), global_elements)) {
        set_error(error, "multi-view global tensor size overflows");
        return false;
    }
    return true;
}

const Pixal3DConditionStageF32 *
Pixal3DConditionBundleF32::find(const std::string & name) const noexcept {
    for (const Pixal3DConditionStageF32 & stage : stages) {
        if (stage.name == name) return &stage;
    }
    return nullptr;
}

Pixal3DConditionStageF32 *
Pixal3DConditionBundleF32::find(const std::string & name) noexcept {
    for (Pixal3DConditionStageF32 & stage : stages) {
        if (stage.name == name) return &stage;
    }
    return nullptr;
}

const Pixal3DConditionStageMVF32 *
Pixal3DMultiViewConditionBundleF32::find(const std::string & name) const noexcept {
    for (const Pixal3DConditionStageMVF32 & stage : stages) {
        if (stage.name == name) return &stage;
    }
    return nullptr;
}

Pixal3DConditionStageMVF32 *
Pixal3DMultiViewConditionBundleF32::find(const std::string & name) noexcept {
    for (Pixal3DConditionStageMVF32 & stage : stages) {
        if (stage.name == name) return &stage;
    }
    return nullptr;
}

bool Pixal3DMultiViewConditionBundleF32::valid(std::string * error) const {
    if (format_version != kVersion) {
        set_error(error, "unsupported P3DMVCON version");
        return false;
    }
    if (stages.empty() || stages.size() > 4) {
        set_error(error, "P3DMVCON must contain between one and four stages");
        return false;
    }
    std::unordered_set<std::string> names;
    std::size_t expected_views = 0;
    for (const Pixal3DConditionStageMVF32 & stage : stages) {
        if (!names.insert(stage.name).second || !stage.valid(error)) {
            if (!error || error->empty()) {
                set_error(error, "invalid or duplicate P3DMVCON stage");
            }
            return false;
        }
        if (expected_views == 0) expected_views = stage.views.size();
        if (stage.views.size() != expected_views) {
            set_error(error, "all P3DMVCON stages must contain the same number of views");
            return false;
        }
    }
    return true;
}

bool Pixal3DConditionBundleF32::valid(std::string * error) const {
    if (format_version != kVersion) {
        set_error(error, "unsupported P3DCOND version");
        return false;
    }
    if (stages.empty() || stages.size() > 4) {
        set_error(error, "P3DCOND must contain between one and four stages");
        return false;
    }
    std::unordered_set<std::string> names;
    for (const Pixal3DConditionStageF32 & stage : stages) {
        if (!names.insert(stage.name).second || !stage.valid(error)) {
            if (!error || error->empty()) set_error(error, "invalid or duplicate P3DCOND stage");
            return false;
        }
    }
    return true;
}

bool load_pixal3d_condition_bundle(const std::string & path,
                                   Pixal3DConditionBundleF32 & output,
                                   std::string * error) {
    output = Pixal3DConditionBundleF32{};
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        set_error(error, "cannot open P3DCOND file: " + path);
        return false;
    }
    char magic[sizeof(kMagic)]{};
    if (!read_exact(file, magic, sizeof(magic)) ||
        std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        set_error(error, "bad P3DCOND magic");
        return false;
    }
    std::uint32_t version = 0;
    std::uint32_t stage_count = 0;
    if (!read_u32(file, version) || !read_u32(file, stage_count)) {
        set_error(error, "truncated P3DCOND header");
        return false;
    }
    if (version != kVersion || stage_count == 0 || stage_count > 4) {
        set_error(error, "unsupported P3DCOND version or stage count");
        return false;
    }
    output.format_version = version;
    output.stages.reserve(stage_count);
    for (std::uint32_t stage_index = 0; stage_index < stage_count; ++stage_index) {
        std::uint32_t name_length = 0;
        if (!read_u32(file, name_length) || name_length == 0 || name_length > 128) {
            set_error(error, "invalid P3DCOND stage name length");
            return false;
        }
        Pixal3DConditionStageF32 stage;
        stage.name.resize(name_length);
        if (!read_exact(file, stage.name.data(), name_length)) {
            set_error(error, "truncated P3DCOND stage name");
            return false;
        }

        std::uint32_t tokens = 0;
        std::uint32_t channels = 0;
        std::uint32_t map_count = 0;
        if (!read_u32(file, tokens) || !read_u32(file, channels) ||
            !read_u32(file, map_count) || tokens == 0 || channels == 0 ||
            channels > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
            map_count == 0 || map_count > 2) {
            set_error(error, "invalid P3DCOND stage header: " + stage.name);
            return false;
        }
        std::size_t global_elements = 0;
        if (!checked_product(static_cast<std::size_t>(tokens),
                             static_cast<std::size_t>(channels), global_elements)) {
            set_error(error, "P3DCOND global tensor size overflows: " + stage.name);
            return false;
        }
        stage.global.batch_size = 1;
        stage.global.channels = static_cast<int>(channels);
        stage.global.offsets = {0, static_cast<std::size_t>(tokens)};
        if (!read_payload(file, global_elements, stage.global.feats, error,
                          (stage.name + " global").c_str())) return false;

        bool have_dino = false;
        bool have_naf = false;
        for (std::uint32_t map_index = 0; map_index < map_count; ++map_index) {
            std::uint32_t role = 0;
            if (!read_u32(file, role) || (role != kDinoMap && role != kNafMap)) {
                set_error(error, "invalid P3DCOND feature-map role: " + stage.name);
                return false;
            }
            if (role == kDinoMap) {
                if (have_dino || !read_map(file, stage.dino_map, error, "DINO")) return false;
                have_dino = true;
            } else {
                if (have_naf || !read_map(file, stage.naf_map, error, "NAF")) return false;
                have_naf = true;
            }
        }
        if (!have_dino || !stage.valid(error)) {
            if (!error || error->empty()) set_error(error, "P3DCOND stage lacks a valid DINO map");
            return false;
        }
        output.stages.push_back(std::move(stage));
    }
    if (!at_end(file)) {
        set_error(error, "P3DCOND contains trailing bytes");
        output = Pixal3DConditionBundleF32{};
        return false;
    }
    if (!output.valid(error)) {
        output = Pixal3DConditionBundleF32{};
        return false;
    }
    return true;
}

bool save_pixal3d_condition_bundle(const std::string & path,
                                   const Pixal3DConditionBundleF32 & bundle,
                                   std::string * error) {
    if (path.empty() || !bundle.valid(error)) return false;
    if (bundle.stages.size() > std::numeric_limits<std::uint32_t>::max()) {
        set_error(error, "too many P3DCOND stages");
        return false;
    }
    const std::string temporary = path + ".part";
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    if (!file) {
        set_error(error, "cannot create temporary P3DCOND file: " + temporary);
        return false;
    }
    auto fail = [&](const std::string & message) {
        file.close();
        std::remove(temporary.c_str());
        set_error(error, message);
        return false;
    };
    file.write(kMagic, sizeof(kMagic));
    if (!file || !write_u32(file, kVersion) ||
        !write_u32(file, static_cast<std::uint32_t>(bundle.stages.size()))) {
        return fail("failed writing P3DCOND header");
    }
    for (const Pixal3DConditionStageF32 & stage : bundle.stages) {
        if (stage.name.size() > 128 || stage.global.tokens() >
                std::numeric_limits<std::uint32_t>::max() ||
            stage.global.channels > std::numeric_limits<std::uint32_t>::max()) {
            return fail("P3DCOND stage metadata exceeds uint32 range: " + stage.name);
        }
        const std::uint32_t name_length = static_cast<std::uint32_t>(stage.name.size());
        if (!write_u32(file, name_length)) return fail("failed writing P3DCOND stage name length");
        file.write(stage.name.data(), static_cast<std::streamsize>(stage.name.size()));
        if (!file || !write_u32(file, static_cast<std::uint32_t>(stage.global.tokens())) ||
            !write_u32(file, static_cast<std::uint32_t>(stage.global.channels)) ||
            !write_u32(file, stage.has_naf() ? 2u : 1u) ||
            !write_payload(file, stage.global.feats)) {
            return fail("failed writing P3DCOND stage: " + stage.name);
        }
        const auto write_map = [&](std::uint32_t role,
                                   const Pixal3DFeatureMapF32 & map) -> bool {
            return write_u32(file, role) &&
                   write_u32(file, static_cast<std::uint32_t>(map.image_resolution)) &&
                   write_u32(file, static_cast<std::uint32_t>(map.height)) &&
                   write_u32(file, static_cast<std::uint32_t>(map.width)) &&
                   write_u32(file, static_cast<std::uint32_t>(map.channels)) &&
                   write_payload(file, map.features);
        };
        if (!write_map(kDinoMap, stage.dino_map) ||
            (stage.has_naf() && !write_map(kNafMap, stage.naf_map))) {
            return fail("failed writing P3DCOND feature maps: " + stage.name);
        }
    }
    file.flush();
    if (!file) return fail("failed flushing temporary P3DCOND file");
    file.close();
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        std::remove(temporary.c_str());
        set_error(error, "cannot atomically replace P3DCOND file: " + path);
        return false;
    }
    return true;
}

bool load_pixal3d_multiview_condition_bundle(
    const std::string & path,
    Pixal3DMultiViewConditionBundleF32 & output,
    std::string * error) {
    output = Pixal3DMultiViewConditionBundleF32{};
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        set_error(error, "cannot open P3DMVCON file: " + path);
        return false;
    }
    char magic[sizeof(kMultiViewMagic)]{};
    if (!read_exact(file, magic, sizeof(magic)) ||
        std::memcmp(magic, kMultiViewMagic, sizeof(kMultiViewMagic)) != 0) {
        set_error(error, "bad P3DMVCON magic");
        return false;
    }
    std::uint32_t version = 0;
    std::uint32_t stage_count = 0;
    if (!read_u32(file, version) || !read_u32(file, stage_count) ||
        version != kVersion || stage_count == 0 || stage_count > 4) {
        set_error(error, "unsupported P3DMVCON version or stage count");
        return false;
    }
    output.format_version = version;
    output.stages.reserve(stage_count);
    for (std::uint32_t stage_index = 0; stage_index < stage_count; ++stage_index) {
        std::uint32_t name_length = 0;
        std::uint32_t view_count = 0;
        if (!read_u32(file, name_length) || name_length == 0 || name_length > 128) {
            set_error(error, "invalid P3DMVCON stage header");
            return false;
        }
        Pixal3DConditionStageMVF32 stage;
        stage.name.resize(name_length);
        if (!read_exact(file, stage.name.data(), name_length)) {
            set_error(error, "truncated P3DMVCON stage name");
            return false;
        }
        if (!read_u32(file, view_count) || view_count == 0 || view_count > 64) {
            set_error(error, "invalid P3DMVCON view count: " + stage.name);
            return false;
        }
        stage.views.reserve(view_count);
        for (std::uint32_t view_index = 0; view_index < view_count; ++view_index) {
            Pixal3DConditionViewF32 view;
            if (!read_camera(file, view.camera, error, stage.name)) return false;

            std::uint32_t tokens = 0;
            std::uint32_t channels = 0;
            std::uint32_t map_count = 0;
            if (!read_u32(file, tokens) || !read_u32(file, channels) ||
                !read_u32(file, map_count) || tokens == 0 || channels == 0 ||
                channels > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
                map_count == 0 || map_count > 2) {
                set_error(error, "invalid P3DMVCON view header: " + stage.name);
                return false;
            }
            std::size_t global_elements = 0;
            if (!checked_product(static_cast<std::size_t>(tokens),
                                 static_cast<std::size_t>(channels), global_elements)) {
                set_error(error, "P3DMVCON global tensor size overflows: " + stage.name);
                return false;
            }
            view.global.batch_size = 1;
            view.global.channels = static_cast<int>(channels);
            view.global.offsets = {0, static_cast<std::size_t>(tokens)};
            const std::string global_label = stage.name + " view global";
            if (!read_payload(file, global_elements, view.global.feats, error,
                              global_label.c_str())) return false;

            bool have_dino = false;
            bool have_naf = false;
            for (std::uint32_t map_index = 0; map_index < map_count; ++map_index) {
                std::uint32_t role = 0;
                if (!read_u32(file, role) || (role != kDinoMap && role != kNafMap)) {
                    set_error(error, "invalid P3DMVCON feature-map role: " + stage.name);
                    return false;
                }
                if (role == kDinoMap) {
                    if (have_dino || !read_map(file, view.dino_map, error, "DINO")) {
                        return false;
                    }
                    have_dino = true;
                } else {
                    if (have_naf || !read_map(file, view.naf_map, error, "NAF")) {
                        return false;
                    }
                    have_naf = true;
                }
            }
            if (!have_dino || !view.valid(error)) {
                if (!error || error->empty()) {
                    set_error(error, "P3DMVCON view lacks a valid DINO map: " + stage.name);
                }
                return false;
            }
            stage.views.push_back(std::move(view));
        }
        if (!stage.valid(error)) return false;
        output.stages.push_back(std::move(stage));
    }
    if (!at_end(file)) {
        set_error(error, "P3DMVCON contains trailing bytes");
        output = Pixal3DMultiViewConditionBundleF32{};
        return false;
    }
    if (!output.valid(error)) {
        output = Pixal3DMultiViewConditionBundleF32{};
        return false;
    }
    return true;
}

bool save_pixal3d_multiview_condition_bundle(
    const std::string & path,
    const Pixal3DMultiViewConditionBundleF32 & bundle,
    std::string * error) {
    if (path.empty() || !bundle.valid(error)) return false;
    if (bundle.stages.size() > std::numeric_limits<std::uint32_t>::max()) {
        set_error(error, "too many P3DMVCON stages");
        return false;
    }
    const std::string temporary = path + ".part";
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    if (!file) {
        set_error(error, "cannot create temporary P3DMVCON file: " + temporary);
        return false;
    }
    auto fail = [&](const std::string & message) {
        file.close();
        std::remove(temporary.c_str());
        set_error(error, message);
        return false;
    };
    file.write(kMultiViewMagic, sizeof(kMultiViewMagic));
    if (!file || !write_u32(file, kVersion) ||
        !write_u32(file, static_cast<std::uint32_t>(bundle.stages.size()))) {
        return fail("failed writing P3DMVCON header");
    }
    for (const Pixal3DConditionStageMVF32 & stage : bundle.stages) {
        if (stage.name.size() > 128 || stage.views.size() > 64) {
            return fail("P3DMVCON stage metadata exceeds limits: " + stage.name);
        }
        const std::uint32_t name_length = static_cast<std::uint32_t>(stage.name.size());
        if (!write_u32(file, name_length)) {
            return fail("failed writing P3DMVCON stage name length");
        }
        file.write(stage.name.data(), static_cast<std::streamsize>(stage.name.size()));
        if (!file || !write_u32(file, static_cast<std::uint32_t>(stage.views.size()))) {
            return fail("failed writing P3DMVCON stage header: " + stage.name);
        }
        for (const Pixal3DConditionViewF32 & view : stage.views) {
            if (!write_f32(file, view.camera.camera_angle_x) ||
                !write_f32(file, view.camera.distance) ||
                !write_f32(file, view.camera.mesh_scale) ||
                !write_u32(file, view.camera.has_transform ? 1u : 0u)) {
                return fail("failed writing P3DMVCON camera header: " + stage.name);
            }
            for (float value : view.camera.transform_matrix) {
                if (!write_f32(file, value)) {
                    return fail("failed writing P3DMVCON camera transform: " + stage.name);
                }
            }
            if (view.global.tokens() > std::numeric_limits<std::uint32_t>::max() ||
                view.global.channels > std::numeric_limits<std::uint32_t>::max()) {
                return fail("P3DMVCON global metadata exceeds uint32 range: " + stage.name);
            }
            if (!write_u32(file, static_cast<std::uint32_t>(view.global.tokens())) ||
                !write_u32(file, static_cast<std::uint32_t>(view.global.channels)) ||
                !write_u32(file, view.has_naf() ? 2u : 1u) ||
                !write_payload(file, view.global.feats)) {
                return fail("failed writing P3DMVCON global tensor: " + stage.name);
            }
            const auto write_map = [&](std::uint32_t role,
                                       const Pixal3DFeatureMapF32 & map) -> bool {
                return write_u32(file, role) &&
                       write_u32(file, static_cast<std::uint32_t>(map.image_resolution)) &&
                       write_u32(file, static_cast<std::uint32_t>(map.height)) &&
                       write_u32(file, static_cast<std::uint32_t>(map.width)) &&
                       write_u32(file, static_cast<std::uint32_t>(map.channels)) &&
                       write_payload(file, map.features);
            };
            if (!write_map(kDinoMap, view.dino_map) ||
                (view.has_naf() && !write_map(kNafMap, view.naf_map))) {
                return fail("failed writing P3DMVCON feature maps: " + stage.name);
            }
        }
    }
    file.flush();
    if (!file) return fail("failed flushing temporary P3DMVCON file");
    file.close();
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        std::remove(temporary.c_str());
        set_error(error, "cannot atomically replace P3DMVCON file: " + path);
        return false;
    }
    return true;
}

bool project_condition_stage_f32(
    const Pixal3DConditionStageF32 & stage,
    const ProjectionCamera & camera,
    int grid_resolution,
    const std::vector<std::int32_t> & coords,
    VarLenTensorF32 & global,
    SparseTensorF32 & projection,
    std::string * error) {
    global = VarLenTensorF32{};
    projection = SparseTensorF32{};
    if (!stage.valid(error) || grid_resolution <= 0) {
        if (!error || error->empty()) set_error(error, "invalid condition stage or grid resolution");
        return false;
    }
    std::vector<float> dino_features;
    std::vector<float> naf_features;
    std::vector<std::uint8_t> dino_valid;
    std::vector<std::uint8_t> naf_valid;
    ProjectionGridOptions dino_options;
    dino_options.grid_resolution = grid_resolution;
    dino_options.image_resolution = stage.dino_map.image_resolution;
    if (!project_grid_features_at_coords(
            stage.dino_map.features.data(), stage.dino_map.height, stage.dino_map.width,
            stage.dino_map.channels, dino_options, camera, coords,
            dino_features, dino_valid, error)) return false;
    if (!stage.naf_map.empty()) {
        ProjectionGridOptions naf_options;
        naf_options.grid_resolution = grid_resolution;
        naf_options.image_resolution = stage.naf_map.image_resolution;
        if (!project_grid_features_at_coords(
                stage.naf_map.features.data(), stage.naf_map.height, stage.naf_map.width,
                stage.naf_map.channels, naf_options, camera, coords,
                naf_features, naf_valid, error)) return false;
    }
    const std::size_t points = coords.size() / 4;
    if (dino_features.size() != points * static_cast<std::size_t>(stage.dino_map.channels) ||
        (!stage.naf_map.empty() &&
         naf_features.size() != points * static_cast<std::size_t>(stage.naf_map.channels))) {
        set_error(error, "condition projection gather returned an unexpected size");
        return false;
    }
    const int output_channels = stage.dino_map.channels +
                                (stage.naf_map.empty() ? 0 : stage.naf_map.channels);
    if (output_channels <= 0 || output_channels > std::numeric_limits<int>::max()) {
        set_error(error, "condition projection channel count overflows");
        return false;
    }
    projection.batch_size = 1;
    projection.channels = output_channels;
    projection.spatial_x = projection.spatial_y = projection.spatial_z = grid_resolution;
    projection.coords = coords;
    projection.feats.resize(points * static_cast<std::size_t>(output_channels));
    for (std::size_t point = 0; point < points; ++point) {
        float * destination = projection.feats.data() +
                              point * static_cast<std::size_t>(output_channels);
        const float * dino = dino_features.data() +
                             point * static_cast<std::size_t>(stage.dino_map.channels);
        std::copy(dino, dino + stage.dino_map.channels, destination);
        if (!stage.naf_map.empty()) {
            const float * naf = naf_features.data() +
                                point * static_cast<std::size_t>(stage.naf_map.channels);
            std::copy(naf, naf + stage.naf_map.channels,
                      destination + stage.dino_map.channels);
        }
    }
    global = stage.global;
    return projection.valid(error);
}

bool project_condition_stage_multiview_average_f32(
    const Pixal3DConditionStageMVF32 & stage,
    int grid_resolution,
    const std::vector<std::int32_t> & coords,
    VarLenTensorF32 & global,
    SparseTensorF32 & projection,
    std::string * error) {
    global = VarLenTensorF32{};
    projection = SparseTensorF32{};
    if (!stage.valid(error) || grid_resolution <= 0 || coords.size() % 4 != 0) {
        if (!error || error->empty()) {
            set_error(error, "invalid multi-view condition stage or projection request");
        }
        return false;
    }

    std::vector<std::array<float, 16>> camera_to_world;
    std::vector<float> distances;
    camera_to_world.reserve(stage.views.size());
    distances.reserve(stage.views.size());
    for (const Pixal3DConditionViewF32 & view : stage.views) {
        camera_to_world.push_back(view.camera.transform_matrix);
        distances.push_back(view.camera.distance);
    }
    std::vector<std::array<float, 16>> relative;
    if (!compute_relative_calc_matrices(camera_to_world, distances, relative, error)) {
        return false;
    }

    const std::size_t points = coords.size() / 4;
    const std::size_t view_count = stage.views.size();
    const Pixal3DConditionViewF32 & first = stage.views.front();
    const int dino_channels = first.dino_map.channels;
    const int naf_channels = first.has_naf() ? first.naf_map.channels : 0;
    const int output_channels = dino_channels + naf_channels;
    if (output_channels <= 0 ||
        points > std::numeric_limits<std::size_t>::max() /
                      static_cast<std::size_t>(output_channels)) {
        set_error(error, "multi-view projection output size overflows size_t");
        return false;
    }
    global = first.global;
    for (float & value : global.feats) value = 0.0f;
    projection.batch_size = 1;
    projection.channels = output_channels;
    projection.spatial_x = projection.spatial_y = projection.spatial_z = grid_resolution;
    projection.coords = coords;
    projection.feats.assign(points * static_cast<std::size_t>(output_channels), 0.0f);

    for (std::size_t view_index = 0; view_index < view_count; ++view_index) {
        const Pixal3DConditionViewF32 & view = stage.views[view_index];
        ProjectionCamera relative_camera = view.camera;
        relative_camera.has_transform = true;
        relative_camera.transform_matrix = relative[view_index];

        ProjectionGridOptions dino_options;
        dino_options.grid_resolution = grid_resolution;
        dino_options.image_resolution = view.dino_map.image_resolution;
        std::vector<float> dino_features;
        std::vector<std::uint8_t> dino_valid;
        if (!project_grid_features_at_coords(
                view.dino_map.features.data(), view.dino_map.height, view.dino_map.width,
                view.dino_map.channels, dino_options, relative_camera, coords,
                dino_features, dino_valid, error)) return false;
        if (dino_features.size() != points * static_cast<std::size_t>(dino_channels)) {
            set_error(error, "multi-view DINO projection returned an unexpected size");
            return false;
        }

        std::vector<float> naf_features;
        std::vector<std::uint8_t> naf_valid;
        if (view.has_naf()) {
            ProjectionGridOptions naf_options;
            naf_options.grid_resolution = grid_resolution;
            naf_options.image_resolution = view.naf_map.image_resolution;
            if (!project_grid_features_at_coords(
                    view.naf_map.features.data(), view.naf_map.height, view.naf_map.width,
                    view.naf_map.channels, naf_options, relative_camera, coords,
                    naf_features, naf_valid, error)) return false;
            if (naf_features.size() != points * static_cast<std::size_t>(naf_channels)) {
                set_error(error, "multi-view NAF projection returned an unexpected size");
                return false;
            }
        }

        for (std::size_t index = 0; index < global.feats.size(); ++index) {
            global.feats[index] += view.global.feats[index];
            if (!std::isfinite(global.feats[index])) {
                set_error(error, "multi-view global feature average is non-finite");
                return false;
            }
        }
        for (std::size_t point = 0; point < points; ++point) {
            float * destination = projection.feats.data() +
                                  point * static_cast<std::size_t>(output_channels);
            const float * dino = dino_features.data() +
                                 point * static_cast<std::size_t>(dino_channels);
            for (int channel = 0; channel < dino_channels; ++channel) {
                destination[channel] += dino[channel];
            }
            if (view.has_naf()) {
                const float * naf = naf_features.data() +
                                   point * static_cast<std::size_t>(naf_channels);
                for (int channel = 0; channel < naf_channels; ++channel) {
                    destination[dino_channels + channel] += naf[channel];
                }
            }
        }
    }

    const float inverse_views = 1.0f / static_cast<float>(view_count);
    for (float & value : global.feats) value *= inverse_views;
    for (float & value : projection.feats) {
        value *= inverse_views;
        if (!std::isfinite(value)) {
            set_error(error, "multi-view projection average is non-finite");
            return false;
        }
    }
    return global.valid(error) && projection.valid(error);
}

} // namespace pixal3d
