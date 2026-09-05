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
    src/condition.cpp
    src/dino.cpp
    src/dino_vit.cpp
    src/naf.cpp
    src/flow.cpp
    src/flow_sampler.cpp
    src/inference.cpp
    src/dual_grid.cpp
    src/pack.cpp
    src/pipeline.cpp
    src/projection.cpp
    src/scaffold.cpp
    src/sparse.cpp
    src/sparse_transformer.cpp
    src/slat_decoder.cpp
    src/slat_flow.cpp
    src/ss_decoder.cpp
    src/ss_flow.cpp
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
    add_executable(pixal3d_sparse_transformer_fixture
        tests/sparse_transformer_numeric_fixture.cpp)
    target_compile_features(pixal3d_sparse_transformer_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_sparse_transformer_fixture PRIVATE pixal3d_core)
    add_executable(pixal3d_slat_decoder_fixture
        tests/slat_decoder_numeric_fixture.cpp)
    target_compile_features(pixal3d_slat_decoder_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_slat_decoder_fixture PRIVATE pixal3d_core)
    add_test(NAME pixal3d_slat_decoder_fixture COMMAND pixal3d_slat_decoder_fixture)
    add_executable(pixal3d_slat_decoder_loader_fixture
        tests/slat_decoder_loader_fixture.cpp)
    target_compile_features(pixal3d_slat_decoder_loader_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_slat_decoder_loader_fixture PRIVATE pixal3d_core)
    add_executable(pixal3d_slat_flow_fixture tests/slat_flow_numeric_fixture.cpp)
    target_compile_features(pixal3d_slat_flow_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_slat_flow_fixture PRIVATE pixal3d_core)
    add_test(NAME pixal3d_slat_flow_fixture COMMAND pixal3d_slat_flow_fixture)
    add_executable(pixal3d_slat_flow_loader_fixture tests/slat_flow_loader_fixture.cpp)
    target_compile_features(pixal3d_slat_flow_loader_fixture PRIVATE cxx_std_17)
    target_link_libraries(pixal3d_slat_flow_loader_fixture PRIVATE pixal3d_core)
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
