#include "pixal3d/scaffold.h"

#include "pixal3d/pack.h"

#include <ostream>
#include <string>

namespace pixal3d {

const char * version() noexcept {
    return "0.3.0";
}

bool inspect_pack(const std::string & path,
                  std::ostream & out,
                  std::string * error) {
    Pixal3DPackInfo info;
    if (!load_pack_info(path, info, error)) {
        return false;
    }

    out << "pixal3d " << version() << "\n"
        << "kind            : gguf-pack\n"
        << "file            : " << path << "\n"
        << "name            : " << info.name << "\n"
        << "architecture    : " << info.architecture << "\n"
        << "name_scheme     : " << info.tensor_name_scheme << "\n"
        << "bundle_format   : " << info.bundle_format << "\n"
        << "file_type       : " << info.file_type << "\n"
        << "components      : " << info.components.size() << "\n"
        << "tensors         : " << info.tensors.size() << "\n"
        << "tensor_bytes    : " << info.tensor_bytes() << "\n";
    for (const Pixal3DComponentInfo & component : info.components) {
        out << "  component      : " << component.stage
            << " | " << component.model_class
            << " | " << component.variant << "\n";
    }
    return true;
}

} // namespace pixal3d
