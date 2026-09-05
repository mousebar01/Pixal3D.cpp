#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;

namespace pixal3d {

// Stable Pixal3D-owned tensor descriptor.  This deliberately does not expose
// the reference trellis2 ABI so the production library can be built without
// ref/trellis2cpp.
struct Pixal3DTensorInfo {
    std::string name;
    int n_dims = 0;
    std::int64_t ne[4] = {1, 1, 1, 1};
    int ggml_type = 0;
    std::string type_name;
    std::size_t n_bytes = 0;
    // Byte offset relative to Pixal3DPackInfo::data_offset.
    std::size_t offset = 0;
};

struct Pixal3DComponentInfo {
    std::string key;
    std::string stage;
    std::string model_class;
    std::string variant;
};

struct Pixal3DPackInfo {
    std::uint32_t gguf_version = 0;
    std::size_t alignment = 0;
    std::size_t data_offset = 0;
    std::string architecture;
    std::string name;
    std::string tensor_name_scheme;
    std::uint32_t file_type = 0;
    std::uint32_t bundle_format = 0;
    std::vector<Pixal3DComponentInfo> components;
    std::vector<Pixal3DTensorInfo> tensors;

    std::size_t tensor_bytes() const;
    bool has_tensor(const std::string & name) const;
};

// Parse a Pixal3D GGUF pack without allocating tensor payloads.  This is the
// common boundary for shared/base-flow/mv-flow files and component packs.
// The parser also checks the tensor table, alignment, duplicate names, and
// that every tensor payload lies inside the file.
bool load_pack_info(const std::string & path,
                    Pixal3DPackInfo & output,
                    std::string * error = nullptr);

// Read-only, on-demand access to tensors in a validated GGUF pack.  The
// reader intentionally does not load the complete (multi-gigabyte) bundle:
// each call reads exactly one tensor payload into the supplied vector.
class Pixal3DPackReader {
public:
    Pixal3DPackReader() = default;
    ~Pixal3DPackReader();

    Pixal3DPackReader(const Pixal3DPackReader &) = delete;
    Pixal3DPackReader & operator=(const Pixal3DPackReader &) = delete;

    bool open(const std::string & path, std::string * error = nullptr);
    void close() noexcept;
    bool is_open() const noexcept;
    const Pixal3DPackInfo & info() const noexcept;

    bool read_tensor(const std::string & name,
                     std::vector<std::uint8_t> & output,
                     std::string * error = nullptr) const;

    // Read scalar/string metadata without exposing gguf_context in the public
    // API.  These accessors are used by model loaders after open() has already
    // validated the Pixal3D pack boundary.
    bool metadata_u32(const std::string & key,
                      std::uint32_t & value,
                      bool required = true,
                      std::string * error = nullptr) const;
    bool metadata_f32(const std::string & key,
                      float & value,
                      bool required = true,
                      std::string * error = nullptr) const;
    bool metadata_bool(const std::string & key,
                       bool & value,
                       bool required = true,
                       std::string * error = nullptr) const;
    bool metadata_string(const std::string & key,
                         std::string & value,
                         bool required = true,
                         std::string * error = nullptr) const;

private:
    std::string path_;
    Pixal3DPackInfo info_;
    ::gguf_context * metadata_ = nullptr;
    bool open_ = false;
};

} // namespace pixal3d
