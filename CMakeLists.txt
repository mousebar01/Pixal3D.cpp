cmake_minimum_required(VERSION 3.16)

project(Pixal3D
    VERSION 0.3.0
    DESCRIPTION "Pixal3D C++/ggml migration scaffold"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include(GNUInstallDirs)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(PIXAL3D_USE_BUNDLED_GGML
       "Use the top-level ggml submodule" ON)
option(PIXAL3D_ENABLE_CUDA
       "Enable the ggml CUDA backend (requires a CUDA toolchain)" OFF)
option(PIXAL3D_CUDA_TF32
       "Allow CUDA TF32 tensor operations (faster, but not F32-exact)" OFF)
option(PIXAL3D_GGML_NATIVE
       "Tune ggml for the build host (reduces binary portability)" OFF)
option(PIXAL3D_BUILD_CLI "Build the pixal3d diagnostic CLI" ON)

# Image decoding is optional at configure time.  PNM remains available without
# third-party libraries; PNG/JPEG are enabled when the host provides them.
find_package(PNG QUIET)
find_package(JPEG QUIET)
find_package(OpenMP QUIET)

set(PIXAL3D_GGML_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml"
    CACHE PATH "Path to the project ggml submodule")

if(PIXAL3D_USE_BUNDLED_GGML)
    if(NOT EXISTS "${PIXAL3D_GGML_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "ggml was not found at ${PIXAL3D_GGML_SOURCE_DIR}. "
            "Initialize the project submodule with: "
            "git submodule update --init --recursive")
    endif()

    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_NATIVE ${PIXAL3D_GGML_NATIVE} CACHE BOOL "" FORCE)
    set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA ${PIXAL3D_ENABLE_CUDA} CACHE BOOL "" FORCE)
    if(PIXAL3D_ENABLE_CUDA AND NOT PIXAL3D_CUDA_TF32)
        # This definition must be visible while the ggml subdirectory is
        # configured: its CUDA handle is created in the bundled backend.
        add_compile_definitions(GGML_CUDA_DISABLE_TF32=1)
    endif()
    add_subdirectory("${PIXAL3D_GGML_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/ggml")
    set(PIXAL3D_GGML_TARGET ggml)
else()
    find_package(ggml CONFIG REQUIRED)
    set(PIXAL3D_GGML_TARGET ggml::ggml)
endif()

add_library(pixal3d_core
    src/backend.cpp
    src/condition.cpp
    src/dino.cpp
    src/dino_vit.cpp
    src/naf.cpp
    src/flow.cpp
    src/flow_sampler.cpp
    src/inference.cpp
    src/dual_grid.cpp
    src/mesh_topology.cpp
    src/pack.cpp
    src/pipeline.cpp
    src/projection.cpp
    src/scaffold.cpp
    src/sparse.cpp
    src/sparse_transformer.cpp
    src/slat_decoder.cpp
    src/slat_flow.cpp
    src/moge_camera.cpp
    src/ss_decoder.cpp
    src/ss_flow.cpp
    src/texture_export.cpp
    src/vision_condition.cpp
    src/image.cpp
)

target_include_directories(pixal3d_core
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_compile_features(pixal3d_core PUBLIC cxx_std_17)
target_link_libraries(pixal3d_core PUBLIC ${PIXAL3D_GGML_TARGET})

# Vendored chart parameterization backend for the GLB texture bake
# (third_party/xatlas, MIT; jpcy/xatlas master f700c7790aaa030e794b52ba7791a05c085faf0c).
add_library(pixal3d_xatlas STATIC third_party/xatlas/xatlas.cpp)
target_compile_features(pixal3d_xatlas PUBLIC cxx_std_17)
target_include_directories(pixal3d_xatlas PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/xatlas>")
target_link_libraries(pixal3d_core PUBLIC pixal3d_xatlas)

# Vendored mesh simplifier for the GLB export decimation pass
# (third_party/meshoptimizer, MIT; zeux/meshoptimizer v1.2).
add_library(pixal3d_meshopt STATIC third_party/meshoptimizer/simplifier.cpp)
target_compile_features(pixal3d_meshopt PUBLIC cxx_std_17)
target_include_directories(pixal3d_meshopt PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/meshoptimizer>")
target_link_libraries(pixal3d_core PUBLIC pixal3d_meshopt)

# Vendored mesh postprocess chain for the GLB export path
# (third_party/trellis-postprocess, MIT; pwilkin/trellis.cpp 2516c48).
# weld/clean/orient/decimate/holes + xatlas unwrap with trilinear voxel-PBR
# bake; also uses the vendored xatlas and meshoptimizer targets above.
add_library(pixal3d_trellis_post STATIC
    third_party/trellis-postprocess/uv_bake.cpp
    third_party/trellis-postprocess/decimate_qem.cpp
    third_party/trellis-postprocess/tri_bvh.cpp
)
target_compile_features(pixal3d_trellis_post PUBLIC cxx_std_17)
target_include_directories(pixal3d_trellis_post PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/trellis-postprocess>")
target_link_libraries(pixal3d_trellis_post PUBLIC pixal3d_xatlas pixal3d_meshopt)
target_link_libraries(pixal3d_core PUBLIC pixal3d_trellis_post)

if(OpenMP_CXX_FOUND)
    target_link_libraries(pixal3d_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(PNG_FOUND)
    target_compile_definitions(pixal3d_core PUBLIC PIXAL3D_HAVE_PNG=1)
    target_include_directories(pixal3d_core PRIVATE ${PNG_INCLUDE_DIRS})
    target_link_libraries(pixal3d_core PUBLIC ${PNG_LIBRARIES})
endif()

if(JPEG_FOUND)
    target_compile_definitions(pixal3d_core PUBLIC PIXAL3D_HAVE_JPEG=1)
    target_include_directories(pixal3d_core PRIVATE ${JPEG_INCLUDE_DIR})
    target_link_libraries(pixal3d_core PUBLIC ${JPEG_LIBRARIES})
endif()

# Optional MoGe-2 ONNX camera estimation (pixal3d/moge_camera.h).  Point
# PIXAL3D_ONNXRUNTIME_ROOT at a prebuilt onnxruntime tree that contains
# include/onnxruntime_cxx_api.h and lib/libonnxruntime.so.
set(PIXAL3D_ONNXRUNTIME_ROOT "" CACHE PATH
    "Prebuilt onnxruntime root directory (optional)")
find_path(ONNXRUNTIME_INCLUDE_DIR
    onnxruntime_cxx_api.h
    HINTS "${PIXAL3D_ONNXRUNTIME_ROOT}/include")
find_library(ONNXRUNTIME_LIBRARY
    onnxruntime
    HINTS "${PIXAL3D_ONNXRUNTIME_ROOT}/lib")
if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    set(PIXAL3D_HAVE_ONNXRUNTIME TRUE)
    target_compile_definitions(pixal3d_core PUBLIC PIXAL3D_HAVE_ONNXRUNTIME=1)
    target_include_directories(pixal3d_core PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(pixal3d_core PUBLIC ${ONNXRUNTIME_LIBRARY})
    message(STATUS "MoGe camera estimation enabled (onnxruntime: ${ONNXRUNTIME_LIBRARY})")
else()
    message(STATUS "MoGe camera estimation disabled (set PIXAL3D_ONNXRUNTIME_ROOT to enable)")
endif()

if(PIXAL3D_ENABLE_CUDA)
    target_compile_definitions(pixal3d_core PUBLIC PIXAL3D_ENABLE_CUDA=1)
endif()

if(PIXAL3D_BUILD_CLI)
    add_executable(pixal3d src/main.cpp)
    target_compile_features(pixal3d PRIVATE cxx_std_17)
    target_link_libraries(pixal3d PRIVATE pixal3d_core)
endif()

include(CTest)
if(BUILD_TESTING AND PIXAL3D_BUILD_CLI)
    add_test(NAME pixal3d_help COMMAND pixal3d --help)
    add_test(NAME pixal3d_version COMMAND pixal3d --version)
    add_executable(pixal3d_projection_test tests/test_projection.cpp)
    target_compile_features(pixal3d_projection_test PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_projection_test PRIVATE pixal3d_core)
    add_test(NAME pixal3d_projection COMMAND pixal3d_projection_test)
    add_executable(pixal3d_projection_fixture tests/projection_numeric_fixture.cpp)
    target_compile_features(pixal3d_projection_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_projection_fixture PRIVATE pixal3d_core)
    add_executable(pixal3d_flow_test tests/test_flow.cpp)
    target_compile_features(pixal3d_flow_test PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_flow_test PRIVATE pixal3d_core)
    add_test(NAME pixal3d_flow COMMAND pixal3d_flow_test)
    add_executable(pixal3d_flow_fixture tests/flow_numeric_fixture.cpp)
    target_compile_features(pixal3d_flow_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_flow_fixture PRIVATE pixal3d_core)
    add_executable(pixal3d_flow_sampler_fixture tests/flow_sampler_numeric_fixture.cpp)
    target_compile_features(pixal3d_flow_sampler_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_flow_sampler_fixture PRIVATE pixal3d_core)
    add_test(NAME pixal3d_flow_sampler_fixture COMMAND pixal3d_flow_sampler_fixture)
    add_executable(pixal3d_pack_reader_test tests/test_pack_reader.cpp)
    target_compile_features(pixal3d_pack_reader_test PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_pack_reader_test PRIVATE pixal3d_core)
    add_test(NAME pixal3d_pack_reader COMMAND pixal3d_pack_reader_test)
    add_executable(pixal3d_dino_test tests/test_dino.cpp)
    target_compile_features(pixal3d_dino_test PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_dino_test PRIVATE pixal3d_core)
    add_test(NAME pixal3d_dino COMMAND pixal3d_dino_test)
    set_tests_properties(pixal3d_dino PROPERTIES LABELS "unit;cpu")
    add_executable(pixal3d_backend_manager_test tests/backend_manager_test.cpp)
    target_compile_features(pixal3d_backend_manager_test PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_backend_manager_test PRIVATE pixal3d_core)
    add_test(NAME pixal3d_backend_manager COMMAND pixal3d_backend_manager_test)
    set_tests_properties(pixal3d_backend_manager PROPERTIES LABELS "unit;backend;cpu")
    if(PIXAL3D_ENABLE_CUDA)
        add_executable(pixal3d_backend_cuda_matmul_test
            tests/backend_cuda_matmul_test.cpp)
        target_compile_features(pixal3d_backend_cuda_matmul_test PRIVATE cxx_std_17)
        target_link_libraries(pixal3d_backend_cuda_matmul_test PRIVATE pixal3d_core)
        add_test(NAME pixal3d_backend_cuda_matmul
                 COMMAND pixal3d_backend_cuda_matmul_test)
        set_tests_properties(pixal3d_backend_cuda_matmul
            PROPERTIES LABELS "backend;cuda;parity;f16")
    endif()
    add_executable(pixal3d_dino_vit_numeric_fixture
        tests/dino_vit_numeric_fixture.cpp)
    target_compile_features(pixal3d_dino_vit_numeric_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_dino_vit_numeric_fixture PRIVATE pixal3d_core)
    add_executable(pixal3d_vision_encoder_fixture
        tests/vision_encoder_numeric_fixture.cpp)
    target_compile_features(pixal3d_vision_encoder_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_vision_encoder_fixture PRIVATE pixal3d_core)
    add_executable(pixal3d_vision_condition_test tests/vision_condition_test.cpp)
    target_compile_features(pixal3d_vision_condition_test PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_vision_condition_test PRIVATE pixal3d_core)
    add_test(NAME pixal3d_vision_condition COMMAND pixal3d_vision_condition_test)
    add_executable(pixal3d_image_test tests/test_image.cpp)
    target_compile_features(pixal3d_image_test PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_image_test PRIVATE pixal3d_core)
    add_test(NAME pixal3d_image COMMAND pixal3d_image_test)
    add_executable(pixal3d_image_resize_numeric_fixture
        tests/image_resize_numeric_fixture.cpp)
    target_compile_features(pixal3d_image_resize_numeric_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_image_resize_numeric_fixture PRIVATE pixal3d_core)
    add_executable(pixal3d_naf_numeric_fixture tests/naf_numeric_fixture.cpp)
    target_compile_features(pixal3d_naf_numeric_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_naf_numeric_fixture PRIVATE pixal3d_core)
    add_executable(pixal3d_condition_test tests/test_condition.cpp)
    target_compile_features(pixal3d_condition_test PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_condition_test PRIVATE pixal3d_core)
    add_test(NAME pixal3d_condition COMMAND pixal3d_condition_test)
    add_executable(pixal3d_condition_multiview_test
        tests/test_condition_multiview.cpp)
    target_compile_features(pixal3d_condition_multiview_test PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_condition_multiview_test PRIVATE pixal3d_core)
    add_test(NAME pixal3d_condition_multiview
             COMMAND pixal3d_condition_multiview_test)
    add_executable(pixal3d_multiview_condition_fixture
        tests/multiview_condition_numeric_fixture.cpp)
    target_compile_features(pixal3d_multiview_condition_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_multiview_condition_fixture PRIVATE pixal3d_core)
    add_executable(pixal3d_inference_test tests/test_inference.cpp)
    target_compile_features(pixal3d_inference_test PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_inference_test PRIVATE pixal3d_core)
    add_test(NAME pixal3d_inference COMMAND pixal3d_inference_test)
    add_executable(pixal3d_texture_export_test tests/texture_export_test.cpp)
    target_compile_features(pixal3d_texture_export_test PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_texture_export_test PRIVATE pixal3d_core)
    add_test(NAME pixal3d_texture_export COMMAND pixal3d_texture_export_test)
    set_tests_properties(pixal3d_texture_export PROPERTIES LABELS "unit;cpu")
    add_executable(pixal3d_moge_camera_test tests/moge_camera_test.cpp)
    target_compile_features(pixal3d_moge_camera_test PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_moge_camera_test PRIVATE pixal3d_core)
    add_test(NAME pixal3d_moge_camera COMMAND pixal3d_moge_camera_test)
    set_tests_properties(pixal3d_moge_camera PROPERTIES LABELS "unit;cpu")
    add_executable(pixal3d_ss_flow_fixture tests/ss_flow_numeric_fixture.cpp)
    target_compile_features(pixal3d_ss_flow_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_ss_flow_fixture PRIVATE pixal3d_core)
    add_executable(pixal3d_ss_stage1_fixture tests/ss_stage1_fixture.cpp)
    target_compile_features(pixal3d_ss_stage1_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_ss_stage1_fixture PRIVATE pixal3d_core)
    add_executable(pixal3d_ss_flow_compare tests/ss_flow_forward_compare.cpp)
    target_compile_features(pixal3d_ss_flow_compare PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_ss_flow_compare PRIVATE pixal3d_core)
    add_executable(pixal3d_ss_decoder_fixture tests/ss_decoder_numeric_fixture.cpp)
    target_compile_features(pixal3d_ss_decoder_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_ss_decoder_fixture PRIVATE pixal3d_core)
    add_executable(pixal3d_ss_occupancy_coords_test
        tests/ss_occupancy_coords_test.cpp)
    target_compile_features(pixal3d_ss_occupancy_coords_test PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_ss_occupancy_coords_test PRIVATE pixal3d_core)
    add_test(NAME pixal3d_ss_occupancy_coords
             COMMAND pixal3d_ss_occupancy_coords_test)
    add_executable(pixal3d_sparse_fixture tests/sparse_numeric_fixture.cpp)
    target_compile_features(pixal3d_sparse_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_sparse_fixture PRIVATE pixal3d_core)
    add_executable(pixal3d_dual_grid_fixture tests/dual_grid_numeric_fixture.cpp)
    target_compile_features(pixal3d_dual_grid_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_dual_grid_fixture PRIVATE pixal3d_core)
    add_test(NAME pixal3d_dual_grid_fixture COMMAND pixal3d_dual_grid_fixture)
    add_executable(pixal3d_mesh_topology_test tests/mesh_topology_test.cpp)
    target_compile_features(pixal3d_mesh_topology_test PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_mesh_topology_test PRIVATE pixal3d_core)
    add_test(NAME pixal3d_mesh_topology COMMAND pixal3d_mesh_topology_test)
    set_tests_properties(pixal3d_mesh_topology PROPERTIES LABELS "unit;cpu")
    add_executable(pixal3d_sparse_transformer_fixture
        tests/sparse_transformer_numeric_fixture.cpp)
    target_compile_features(pixal3d_sparse_transformer_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_sparse_transformer_fixture PRIVATE pixal3d_core)
    add_executable(pixal3d_slat_decoder_fixture
        tests/slat_decoder_numeric_fixture.cpp)
    target_compile_features(pixal3d_slat_decoder_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_slat_decoder_fixture PRIVATE pixal3d_core)
    add_test(NAME pixal3d_slat_decoder_fixture COMMAND pixal3d_slat_decoder_fixture --upsample)
    set_tests_properties(pixal3d_slat_decoder_fixture PROPERTIES LABELS "unit;parity;cpu")
    add_executable(pixal3d_slat_decoder_loader_fixture
        tests/slat_decoder_loader_fixture.cpp)
    target_compile_features(pixal3d_slat_decoder_loader_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_slat_decoder_loader_fixture PRIVATE pixal3d_core)
    add_executable(pixal3d_slat_decoder_dump_probe
        tests/slat_decoder_dump_probe.cpp)
    target_compile_features(pixal3d_slat_decoder_dump_probe PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_slat_decoder_dump_probe PRIVATE pixal3d_core)
    add_executable(pixal3d_slat_flow_fixture tests/slat_flow_numeric_fixture.cpp)
    target_compile_features(pixal3d_slat_flow_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_slat_flow_fixture PRIVATE pixal3d_core)
    add_test(NAME pixal3d_slat_flow_fixture COMMAND pixal3d_slat_flow_fixture)
    set_tests_properties(pixal3d_slat_flow_fixture PROPERTIES LABELS "unit;parity;cpu")
    add_executable(pixal3d_slat_flow_loader_fixture tests/slat_flow_loader_fixture.cpp)
    target_compile_features(pixal3d_slat_flow_loader_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_slat_flow_loader_fixture PRIVATE pixal3d_core)
    add_executable(pixal3d_slat_flow_cuda_benchmark
        tests/slat_flow_cuda_benchmark.cpp)
    target_compile_features(pixal3d_slat_flow_cuda_benchmark PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_slat_flow_cuda_benchmark PRIVATE pixal3d_core)
    add_executable(pixal3d_slat_flow_concat_fixture
        tests/slat_flow_concat_fixture.cpp)
    target_compile_features(pixal3d_slat_flow_concat_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_slat_flow_concat_fixture PRIVATE pixal3d_core)
    add_executable(pixal3d_pipeline_utils_test tests/pipeline_utils_test.cpp)
    target_compile_features(pixal3d_pipeline_utils_test PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_pipeline_utils_test PRIVATE pixal3d_core)
    add_test(NAME pixal3d_pipeline_utils COMMAND pixal3d_pipeline_utils_test)
    add_executable(pixal3d_cascade_fixture tests/pixal3d_cascade_fixture.cpp)
    target_compile_features(pixal3d_cascade_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_cascade_fixture PRIVATE pixal3d_core)
endif()

install(TARGETS pixal3d_core
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

if(PIXAL3D_BUILD_CLI)
    install(TARGETS pixal3d RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()

install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES
    LICENSE
    NOTICE
    DESTINATION "${CMAKE_INSTALL_DATADIR}/pixal3d"
)
