#include "pixal3d/pack.h"

#include "gguf.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

namespace pixal3d {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
}

bool read_u32(const gguf_context * context,
              const char * key,
              std::uint32_t & value,
              std::string * error,
              bool required) {
    const int64_t id = gguf_find_key(context, key);
    if (id < 0) {
        if (required) {
            set_error(error, std::string("missing GGUF metadata: ") + key);
            return false;
        }
        value = 0;
        return true;
    }
    switch (gguf_get_kv_type(context, id)) {
        case GGUF_TYPE_UINT8:  value = gguf_get_val_u8(context, id);  return true;
        case GGUF_TYPE_UINT16: value = gguf_get_val_u16(context, id); return true;
        case GGUF_TYPE_UINT32: value = gguf_get_val_u32(context, id); return true;
        case GGUF_TYPE_INT8: {
            const int value_i = gguf_get_val_i8(context, id);
            if (value_i < 0) break;
            value = static_cast<std::uint32_t>(value_i);
            return true;
        }
        case GGUF_TYPE_INT16: {
            const int value_i = gguf_get_val_i16(context, id);
            if (value_i < 0) break;
            value = static_cast<std::uint32_t>(value_i);
            return true;
        }
        case GGUF_TYPE_INT32: {
            const std::int32_t value_i = gguf_get_val_i32(context, id);
            if (value_i < 0) break;
            value = static_cast<std::uint32_t>(value_i);
            return true;
        }
        default:
            break;
    }
    set_error(error, std::string("GGUF metadata is not a non-negative integer: ") + key);
    return false;
}

bool read_string(const gguf_context * context,
                 const char * key,
                 std::string & value,
                 std::string * error,
                 bool required) {
    const int64_t id = gguf_find_key(context, key);
    if (id < 0) {
        if (required) {
            set_error(error, std::string("missing GGUF metadata: ") + key);
            return false;
        }
        value.clear();
        return true;
    }
    if (gguf_get_kv_type(context, id) != GGUF_TYPE_STRING) {
        set_error(error, std::string("GGUF metadata is not a string: ") + key);
        return false;
    }
    const char * raw = gguf_get_val_str(context, id);
    value = raw ? raw : "";
    return true;
}

bool file_size(const std::string & path, std::uint64_t & size, std::string * error) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        set_error(error, "cannot open GGUF file: " + path);
        return false;
    }
    const std::streamoff end = file.tellg();
    if (end < 0) {
        set_error(error, "cannot determine GGUF file size: " + path);
        return false;
    }
    size = static_cast<std::uint64_t>(end);
    return true;
}

} // namespace

std::size_t Pixal3DPackInfo::tensor_bytes() const {
    std::size_t total = 0;
    for (const Pixal3DTensorInfo & tensor : tensors) {
        if (tensor.n_bytes > std::numeric_limits<std::size_t>::max() - total) {
            return std::numeric_limits<std::size_t>::max();
        }
        total += tensor.n_bytes;
    }
    return total;
}

bool Pixal3DPackInfo::has_tensor(const std::string & tensor_name) const {
    for (const Pixal3DTensorInfo & tensor : tensors) {
        if (tensor.name == tensor_name) {
            return true;
        }
    }
    return false;
}

bool load_pack_info(const std::string & path,
                    Pixal3DPackInfo & output,
                    std::string * error) {
    output = Pixal3DPackInfo{};

    ggml_context * context = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &context;
    gguf_context * gguf = gguf_init_from_file(path.c_str(), params);
    if (!gguf) {
        set_error(error, "gguf_init_from_file failed: " + path);
        return false;
    }

    bool ok = true;
    output.gguf_version = gguf_get_version(gguf);
    output.alignment = gguf_get_alignment(gguf);
    output.data_offset = gguf_get_data_offset(gguf);
    if (output.gguf_version != 3) {
        set_error(error, "unsupported GGUF version: " +
                           std::to_string(output.gguf_version));
        ok = false;
    }
    if (ok && !read_string(gguf, "general.architecture", output.architecture,
                           error, true)) {
        ok = false;
    }
    if (ok && output.architecture != "pixal3d") {
        set_error(error, "unexpected GGUF architecture: " + output.architecture);
        ok = false;
    }
    if (ok && !read_string(gguf, "general.name", output.name, error, true)) {
        ok = false;
    }
    if (ok && !read_string(gguf, "pixal3d.tensor_name_scheme",
                           output.tensor_name_scheme, error, true)) {
        ok = false;
    }
    if (ok && output.tensor_name_scheme != "compact-v1") {
        set_error(error, "unsupported Pixal3D tensor name scheme: " +
                           output.tensor_name_scheme);
        ok = false;
    }
    if (ok && !read_u32(gguf, "general.file_type", output.file_type, error, true)) {
        ok = false;
    }
    if (ok && !read_u32(gguf, "pixal3d.bundle_format", output.bundle_format,
                        error, true)) {
        ok = false;
    }

    if (ok) {
        for (std::size_t i = 0;; ++i) {
            const std::string prefix = "pixal3d.component." + std::to_string(i);
            const int64_t id = gguf_find_key(gguf, prefix.c_str());
            if (id < 0) {
                break;
            }
            if (gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) {
                set_error(error, "component metadata is not a string: " + prefix);
                ok = false;
                break;
            }
            const char * raw = gguf_get_val_str(gguf, id);
            const std::string value = raw ? raw : "";
            Pixal3DComponentInfo component;
            component.key = value;
            const std::size_t first = value.find('|');
            const std::size_t second = first == std::string::npos
                                           ? std::string::npos
                                           : value.find('|', first + 1);
            if (first == std::string::npos || second == std::string::npos) {
                // Current converter format stores the component identifier in
                // this KV and keeps model_class/variant in namespaced KVs.
                // Accept the compact form as the canonical representation.
                component.stage = value;
                const std::string prefix = "pixal3d." + value + ".";
                if (!read_string(gguf, (prefix + "model_class").c_str(),
                                 component.model_class, error, false) ||
                    !read_string(gguf, (prefix + "variant").c_str(),
                                 component.variant, error, false)) {
                    ok = false;
                    break;
                }
            } else {
                // Keep accepting the self-describing form used by early
                // development packs: stage|model_class|variant.
                component.stage = value.substr(0, first);
                component.model_class = value.substr(first + 1, second - first - 1);
                component.variant = value.substr(second + 1);
            }
            output.components.push_back(std::move(component));
        }
    }

    std::uint64_t size = 0;
    if (ok && !file_size(path, size, error)) {
        ok = false;
    }
    if (ok && output.data_offset > size) {
        set_error(error, "GGUF tensor data offset is beyond end of file");
        ok = false;
    }

    std::unordered_set<std::string> names;
    if (ok) {
        const int64_t count = gguf_get_n_tensors(gguf);
        output.tensors.reserve(static_cast<std::size_t>(count));
        for (int64_t i = 0; i < count; ++i) {
            const char * raw_name = gguf_get_tensor_name(gguf, i);
            const std::string tensor_name = raw_name ? raw_name : "";
            if (tensor_name.empty() || tensor_name.size() >= 64 ||
                !names.insert(tensor_name).second) {
                set_error(error, "GGUF tensor names must be non-empty, unique, and <64 bytes");
                ok = false;
                break;
            }

            const std::size_t offset = gguf_get_tensor_offset(gguf, i);
            const std::size_t bytes = gguf_get_tensor_size(gguf, i);
            if (offset > size || bytes > size - offset ||
                output.data_offset > size - offset - bytes) {
                set_error(error, "GGUF tensor payload is outside the file: " + tensor_name);
                ok = false;
                break;
            }
            if (output.alignment == 0 || offset % output.alignment != 0) {
                set_error(error, "GGUF tensor offset is not aligned: " + tensor_name);
                ok = false;
                break;
            }

            ggml_tensor * tensor = context ? ggml_get_tensor(context, tensor_name.c_str()) : nullptr;
            if (!tensor) {
                set_error(error, "GGUF tensor descriptor is missing: " + tensor_name);
                ok = false;
                break;
            }
            Pixal3DTensorInfo info;
            info.name = tensor_name;
            info.n_dims = ggml_n_dims(tensor);
            for (int dim = 0; dim < 4; ++dim) {
                info.ne[dim] = dim < info.n_dims ? tensor->ne[dim] : 1;
            }
            info.ggml_type = static_cast<int>(gguf_get_tensor_type(gguf, i));
            info.type_name = ggml_type_name(tensor->type);
            info.n_bytes = bytes;
            info.offset = offset;
            output.tensors.push_back(std::move(info));
        }
    }

    if (context) {
        ggml_free(context);
    }
    gguf_free(gguf);
    return ok;
}

Pixal3DPackReader::~Pixal3DPackReader() {
    close();
}

bool Pixal3DPackReader::open(const std::string & path, std::string * error) {
    close();
    Pixal3DPackInfo parsed;
    if (!load_pack_info(path, parsed, error)) {
        return false;
    }
    path_ = path;
    info_ = std::move(parsed);
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;
    metadata_ = gguf_init_from_file(path.c_str(), params);
    if (!metadata_) {
        close();
        set_error(error, "cannot open GGUF metadata: " + path);
        return false;
    }
    open_ = true;
    return true;
}

void Pixal3DPackReader::close() noexcept {
    if (metadata_) {
        gguf_free(metadata_);
        metadata_ = nullptr;
    }
    path_.clear();
    info_ = Pixal3DPackInfo{};
    open_ = false;
}

bool Pixal3DPackReader::is_open() const noexcept {
    return open_;
}

const Pixal3DPackInfo & Pixal3DPackReader::info() const noexcept {
    return info_;
}

bool Pixal3DPackReader::read_tensor(const std::string & name,
                                    std::vector<std::uint8_t> & output,
                                    std::string * error) const {
    output.clear();
    if (!open_) {
        set_error(error, "pack reader is not open");
        return false;
    }
    const Pixal3DTensorInfo * wanted = nullptr;
    for (const Pixal3DTensorInfo & tensor : info_.tensors) {
        if (tensor.name == name) {
            wanted = &tensor;
            break;
        }
    }
    if (!wanted) {
        set_error(error, "tensor not found: " + name);
        return false;
    }
    if (wanted->offset > std::numeric_limits<std::size_t>::max() - info_.data_offset ||
        wanted->n_bytes > std::numeric_limits<std::size_t>::max() -
                               (info_.data_offset + wanted->offset)) {
        set_error(error, "tensor offset overflows size_t: " + name);
        return false;
    }
    const std::size_t absolute_offset = info_.data_offset + wanted->offset;
    output.resize(wanted->n_bytes);
    std::ifstream file(path_, std::ios::binary);
    if (!file) {
        output.clear();
        set_error(error, "cannot open GGUF file: " + path_);
        return false;
    }
    if (absolute_offset > static_cast<std::size_t>(std::numeric_limits<std::streamoff>::max())) {
        output.clear();
        set_error(error, "tensor offset is not representable by stream: " + name);
        return false;
    }
    file.seekg(static_cast<std::streamoff>(absolute_offset), std::ios::beg);
    if (!file) {
        output.clear();
        set_error(error, "cannot seek to tensor: " + name);
        return false;
    }
    if (wanted->n_bytes > 0) {
        if (wanted->n_bytes >
            static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
            output.clear();
            set_error(error, "tensor is too large for one stream read: " + name);
            return false;
        }
        file.read(reinterpret_cast<char *>(output.data()),
                  static_cast<std::streamsize>(wanted->n_bytes));
        if (!file || static_cast<std::size_t>(file.gcount()) != wanted->n_bytes) {
            output.clear();
            set_error(error, "short read for tensor: " + name);
            return false;
        }
    }
    return true;
}

bool Pixal3DPackReader::metadata_u32(const std::string & key,
                                     std::uint32_t & value,
                                     bool required,
                                     std::string * error) const {
    if (!open_ || !metadata_) {
        set_error(error, "pack reader is not open");
        return false;
    }
    return read_u32(metadata_, key.c_str(), value, error, required);
}

bool Pixal3DPackReader::metadata_f32(const std::string & key,
                                     float & value,
                                     bool required,
                                     std::string * error) const {
    if (!open_ || !metadata_) {
        set_error(error, "pack reader is not open");
        return false;
    }
    const int64_t id = gguf_find_key(metadata_, key.c_str());
    if (id < 0) {
        if (required) {
            set_error(error, "missing GGUF metadata: " + key);
            return false;
        }
        value = 0.0f;
        return true;
    }
    if (gguf_get_kv_type(metadata_, id) == GGUF_TYPE_FLOAT32) {
        value = gguf_get_val_f32(metadata_, id);
        return true;
    }
    set_error(error, "GGUF metadata is not float32: " + key);
    return false;
}

bool Pixal3DPackReader::metadata_bool(const std::string & key,
                                      bool & value,
                                      bool required,
                                      std::string * error) const {
    if (!open_ || !metadata_) {
        set_error(error, "pack reader is not open");
        return false;
    }
    const int64_t id = gguf_find_key(metadata_, key.c_str());
    if (id < 0) {
        if (required) {
            set_error(error, "missing GGUF metadata: " + key);
            return false;
        }
        value = false;
        return true;
    }
    if (gguf_get_kv_type(metadata_, id) == GGUF_TYPE_BOOL) {
        value = gguf_get_val_bool(metadata_, id);
        return true;
    }
    set_error(error, "GGUF metadata is not bool: " + key);
    return false;
}

bool Pixal3DPackReader::metadata_string(const std::string & key,
                                        std::string & value,
                                        bool required,
                                        std::string * error) const {
    if (!open_ || !metadata_) {
        set_error(error, "pack reader is not open");
        return false;
    }
    return read_string(metadata_, key.c_str(), value, error, required);
}

} // namespace pixal3d
