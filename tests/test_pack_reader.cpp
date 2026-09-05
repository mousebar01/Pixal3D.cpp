#include "pixal3d/pack.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main() {
    const char * path = "/tmp/pixal3d-pack-reader-test.gguf";
    std::remove(path);

    ggml_init_params params{};
    params.mem_size = 1u << 20;
    params.no_alloc = false;
    ggml_context * context = ggml_init(params);
    require(context != nullptr, "create ggml context");
    ggml_tensor * tensor = ggml_new_tensor_2d(context, GGML_TYPE_F32, 3, 2);
    require(tensor != nullptr, "create test tensor");
    ggml_set_name(tensor, "ss.input_layer.bias");
    const float values[] = {1.25f, -2.5f, 3.75f, 4.5f, -5.25f, 6.0f};
    std::memcpy(tensor->data, values, sizeof(values));

    gguf_context * writer = gguf_init_empty();
    require(writer != nullptr, "create GGUF writer");
    gguf_set_val_str(writer, "general.architecture", "pixal3d");
    gguf_set_val_str(writer, "general.name", "pack-reader-test");
    gguf_set_val_u32(writer, "general.file_type", 0);
    gguf_set_val_u32(writer, "general.alignment", 32);
    gguf_set_val_u32(writer, "pixal3d.bundle_format", 1);
    gguf_set_val_str(writer, "pixal3d.tensor_name_scheme", "compact-v1");
    gguf_add_tensor(writer, tensor);
    gguf_set_tensor_data(writer, "ss.input_layer.bias", values);
    require(gguf_write_to_file(writer, path, false), "write test GGUF");
    gguf_free(writer);
    ggml_free(context);

    pixal3d::Pixal3DPackReader reader;
    std::string error;
    require(reader.open(path, &error), error);
    require(reader.is_open(), "reader open state");
    require(reader.info().tensors.size() == 1, "reader tensor count");
    require(reader.info().tensors[0].offset == 0, "first tensor offset");
    std::vector<std::uint8_t> raw;
    require(reader.read_tensor("ss.input_layer.bias", raw, &error), error);
    require(raw.size() == sizeof(values), "tensor byte count");
    require(std::memcmp(raw.data(), values, sizeof(values)) == 0,
            "tensor bytes round trip");
    require(!reader.read_tensor("missing", raw, &error),
            "missing tensor is rejected");
    reader.close();
    require(!reader.is_open(), "reader close state");
    require(!reader.read_tensor("ss.input_layer.bias", raw, &error),
            "closed reader is rejected");
    std::remove(path);
    std::cout << "pack reader tests: PASS\n";
    return EXIT_SUCCESS;
}
