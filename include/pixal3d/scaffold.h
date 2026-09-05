#pragma once

#include <iosfwd>
#include <string>

namespace pixal3d {

// This is the production library version. Reference implementations under
// ref/ are intentionally not part of the public ABI.
const char * version() noexcept;

// Read and print a Pixal3D GGUF bundle without allocating tensor payloads.
bool inspect_pack(const std::string & path,
                  std::ostream & out,
                  std::string * error = nullptr);

} // namespace pixal3d
