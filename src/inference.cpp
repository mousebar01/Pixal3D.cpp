#include "pixal3d/inference.h"
#include "pixal3d/pack.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <cstdlib>
#include <utility>

namespace pixal3d {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

bool checked_add(std::size_t left, std::size_t right, std::size_t & output) {
    if (right > std::numeric_limits<std::size_t>::max() - left) return false;
    output = left + right;
    return true;
}

bool checked_mul(std::size_t left, std::size_t right, std::size_t & output) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        return false;
    }
    output = left * right;
    return true;
}

bool pack_host_f32_bytes(const Pixal3DPackInfo & info, std::size_t & output,
                         std::string * error) {
    output = 0;
    for (const Pixal3DTensorInfo & tensor : info.tensors) {
        std::size_t elements = 1;
        for (int dim = 0; dim < tensor.n_dims; ++dim) {
            if (tensor.ne[dim] <= 0 ||
                !checked_mul(elements, static_cast<std::size_t>(tensor.ne[dim]), elements)) {
                set_error(error, "invalid tensor dimensions while estimating model memory: " +
                                   tensor.name);
                return false;
            }
        }
        std::size_t tensor_bytes = 0;
        if (!checked_mul(elements, sizeof(float), tensor_bytes) ||
            !checked_add(output, tensor_bytes, output)) {
            set_error(error, "model memory estimate overflows size_t: " + tensor.name);
            return false;
        }
    }
    return true;
}

const char * condition_stage_name(Pixal3DCascadeStage stage) {
    switch (stage) {
    case Pixal3DCascadeStage::sparse_structure: return "ss";
    case Pixal3DCascadeStage::shape_slat_low: return "shape_512";
    case Pixal3DCascadeStage::shape_slat_high: return "shape_1024";
    case Pixal3DCascadeStage::texture_slat: return "tex_1024";
    }
    return nullptr;
}

std::vector<float> make_shape_mean() {
    return {0.781296f, 0.018091f, -0.495192f, -0.558457f, 1.06053f,
            0.093252f, 1.518149f, -0.933218f, -0.732996f, 2.604095f,
            -0.118341f, -2.143904f, 0.495076f, -2.179512f, -2.130751f,
            -0.996944f, 0.261421f, -2.217463f, 1.260067f, -0.150213f,
            3.790713f, 1.481266f, -1.046058f, -1.523667f, -0.059621f,
            2.22078f, 1.621212f, 0.87723f, 0.567247f, -3.175944f,
            -3.186688f, 1.578665f};
}

std::vector<float> make_shape_std() {
    return {5.972266f, 4.706852f, 5.44501f, 5.209927f, 5.32022f,
            4.547237f, 5.020802f, 5.444004f, 5.226681f, 5.683095f,
            4.831436f, 5.286469f, 5.652043f, 5.367606f, 5.525084f,
            4.730578f, 4.805265f, 5.124013f, 5.530808f, 5.619001f,
            5.10393f, 5.41767f, 5.269677f, 5.547194f, 5.634698f,
            5.235274f, 6.110351f, 5.511298f, 6.237273f, 4.879207f,
            5.347008f, 5.405691f};
}

std::vector<float> make_texture_mean() {
    return {3.501659f, 2.212398f, 2.226094f, 0.251093f, -0.026248f,
            -0.687364f, 0.439898f, -0.928075f, 0.029398f, -0.339596f,
            -0.869527f, 1.038479f, -0.972385f, 0.126042f, -1.129303f,
            0.455149f, -1.209521f, 2.069067f, 0.544735f, 2.569128f,
            -0.323407f, 2.293f, -1.925608f, -1.217717f, 1.213905f,
            0.971588f, -0.023631f, 0.10675f, 2.021786f, 0.250524f,
            -0.662387f, -0.768862f};
}

std::vector<float> make_texture_std() {
    return {2.665652f, 2.743913f, 2.765121f, 2.595319f, 3.037293f,
            2.291316f, 2.144656f, 2.911822f, 2.969419f, 2.501689f,
            2.154811f, 3.163343f, 2.621215f, 2.381943f, 3.186697f,
            3.021588f, 2.295916f, 3.234985f, 3.233086f, 2.26014f,
            2.874801f, 2.810596f, 3.29272f, 2.674999f, 2.680878f,
            2.372054f, 2.451546f, 2.353556f, 2.995195f, 2.379849f,
            2.786195f, 2.77519f};
}

bool run_pixal3d_with_condition_builder(
    const std::string & shared_pack,
    const std::string & flow_pack,
    const Pixal3DInferenceConfig & config,
    const Pixal3DConditionBuilderF32 & condition_builder,
    Pixal3DCascadeOutputF32 & output,
    std::string * error) {
    output = Pixal3DCascadeOutputF32{};
    std::size_t estimated_bytes = 0;
    if (!estimate_pixal3d_model_bytes(shared_pack, flow_pack, estimated_bytes, error)) {
        return false;
    }
    if (config.max_model_bytes > 0 && estimated_bytes > config.max_model_bytes) {
        set_error(error, "Pixal3D model F32 memory estimate exceeds configured limit: " +
                           std::to_string(estimated_bytes) + " > " +
                           std::to_string(config.max_model_bytes) + " bytes");
        return false;
    }

    SSFlowModel ss_flow;
    SSDecoderModel ss_decoder;
    SLatFlowModel shape_flow_low;
    SLatFlowModel shape_flow_high;
    SLatFlowModel texture_flow;
    SLatDecoderModel shape_decoder;
    SLatDecoderModel texture_decoder;
    std::cerr << "pixal3d: loading ss-flow tensors" << std::endl;
    if (!ss_flow.load(flow_pack, true, error)) return false;
    std::cerr << "pixal3d: ss-flow backend " << ss_flow.backend_name() << std::endl;
    std::cerr << "pixal3d: loading ss decoder tensors" << std::endl;
    if (!ss_decoder.load(shared_pack, true, error)) return false;
    std::cerr << "pixal3d: ss decoder backend " << ss_decoder.backend_name() << std::endl;
    std::cerr << "pixal3d: loading shape 512 flow tensors" << std::endl;
    if (!shape_flow_low.load(flow_pack, "shape_flow_512", true, error)) return false;
    std::cerr << "pixal3d: loading shape 1024 flow tensors" << std::endl;
    if (!shape_flow_high.load(flow_pack, "shape_flow_1024", true, error)) return false;
    std::cerr << "pixal3d: loading texture 1024 flow tensors" << std::endl;
    if (!texture_flow.load(flow_pack, "texture_flow_1024", true, error)) return false;
    std::cerr << "pixal3d: loading shape decoder tensors" << std::endl;
    if (!shape_decoder.load(shared_pack, "shape_decoder", true, error)) return false;
    std::cerr << "pixal3d: loading texture decoder tensors" << std::endl;
    if (!texture_decoder.load(shared_pack, "texture_decoder", true, error)) return false;
    std::cerr << "pixal3d: cascade tensors loaded" << std::endl;

    std::mt19937_64 generator(config.seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> ss_noise_override;
    const char * ss_noise_path = std::getenv("PIXAL3D_SS_NOISE_F32");
    if (ss_noise_path && ss_noise_path[0] != '\0') {
        std::ifstream stream(ss_noise_path, std::ios::binary);
        if (!stream) {
            set_error(error, "failed to open PIXAL3D_SS_NOISE_F32: " +
                               std::string(ss_noise_path));
            return false;
        }
        stream.seekg(0, std::ios::end);
        const std::streamoff bytes = stream.tellg();
        stream.seekg(0, std::ios::beg);
        if (bytes < 0 || static_cast<std::size_t>(bytes) % sizeof(float) != 0) {
            set_error(error, "PIXAL3D_SS_NOISE_F32 is not a raw F32 file");
            return false;
        }
        ss_noise_override.resize(static_cast<std::size_t>(bytes) / sizeof(float));
        stream.read(reinterpret_cast<char *>(ss_noise_override.data()), bytes);
        if (!stream) {
            set_error(error, "failed to read PIXAL3D_SS_NOISE_F32: " +
                               std::string(ss_noise_path));
            return false;
        }
        std::cerr << "pixal3d: loaded SS noise override values="
                  << ss_noise_override.size() << std::endl;
    }
    const Pixal3DNoiseBuilderF32 noise_builder =
        [&generator, &normal, &ss_noise_override](Pixal3DCascadeStage stage,
                              const std::vector<std::int32_t> & coords,
                              int channels,
                              int grid_resolution,
                              SparseTensorF32 & result,
                              std::string * error) {
            if (channels <= 0 || grid_resolution <= 0 || coords.empty() ||
                coords.size() % 4 != 0) {
                set_error(error, "invalid random-noise request");
                return false;
            }
            result = SparseTensorF32{};
            result.batch_size = 1;
            result.channels = channels;
            result.spatial_x = result.spatial_y = result.spatial_z = grid_resolution;
            result.coords = coords;
            result.feats.resize(result.points() * static_cast<std::size_t>(channels));
            if (stage == Pixal3DCascadeStage::sparse_structure &&
                !ss_noise_override.empty()) {
                const std::size_t expected = result.points() *
                                             static_cast<std::size_t>(channels);
                if (ss_noise_override.size() != expected) {
                    set_error(error, "PIXAL3D_SS_NOISE_F32 element count does not match "
                                      "the sparse-structure noise request");
                    return false;
                }
                for (std::size_t point = 0; point < result.points(); ++point) {
                    for (int channel = 0; channel < channels; ++channel) {
                        result.feats[point * static_cast<std::size_t>(channels) +
                                     static_cast<std::size_t>(channel)] =
                            ss_noise_override[static_cast<std::size_t>(channel) *
                                               result.points() + point];
                    }
                }
            } else {
                for (float & value : result.feats) value = normal(generator);
            }
            return result.valid(error);
        };

    return run_pixal3d_cascade_f32(
        ss_flow, ss_decoder, shape_flow_low, shape_flow_high, texture_flow,
        shape_decoder, texture_decoder, config.cascade, noise_builder,
        condition_builder, output, error);
}

} // namespace

Pixal3DInferenceConfig default_pixal3d_inference_config() {
    Pixal3DInferenceConfig config;
    config.camera = ProjectionCamera::front(0.8575560450553894f, 2.0f, 1.0f);
    config.cascade.sparse_structure.sampler.steps = 12;
    config.cascade.sparse_structure.sampler.sigma_min = 1e-5f;
    config.cascade.sparse_structure.sampler.rescale_t = 5.0f;
    config.cascade.sparse_structure.sampler.guidance_strength = 7.5f;
    config.cascade.sparse_structure.sampler.guidance_rescale = 0.7f;
    config.cascade.sparse_structure.sampler.guidance_interval_min = 0.6f;
    config.cascade.sparse_structure.sampler.guidance_interval_max = 1.0f;
    config.cascade.sparse_structure.occupancy_threshold = 0.0f;
    config.cascade.sparse_structure.target_resolution = 32;

    config.cascade.shape_sampler.steps = 12;
    config.cascade.shape_sampler.sigma_min = 1e-5f;
    config.cascade.shape_sampler.rescale_t = 3.0f;
    config.cascade.shape_sampler.guidance_strength = 7.5f;
    config.cascade.shape_sampler.guidance_rescale = 0.5f;
    config.cascade.shape_sampler.guidance_interval_min = 0.6f;
    config.cascade.shape_sampler.guidance_interval_max = 1.0f;
    config.cascade.texture_sampler.steps = 12;
    config.cascade.texture_sampler.sigma_min = 1e-5f;
    config.cascade.texture_sampler.rescale_t = 3.0f;
    config.cascade.texture_sampler.guidance_strength = 1.0f;
    config.cascade.texture_sampler.guidance_rescale = 0.0f;
    config.cascade.texture_sampler.guidance_interval_min = 0.6f;
    config.cascade.texture_sampler.guidance_interval_max = 0.9f;

    config.cascade.shape_normalization.mean = make_shape_mean();
    config.cascade.shape_normalization.std = make_shape_std();
    config.cascade.texture_normalization.mean = make_texture_mean();
    config.cascade.texture_normalization.std = make_texture_std();
    config.cascade.requested_resolution = 1024;
    config.cascade.max_num_tokens = 49152;
    config.cascade.decoder_upsample_times = 4;
    config.cascade.voxel_margin = 0.5f;
    config.seed = 42;
    return config;
}

bool estimate_pixal3d_model_bytes(const std::string & shared_pack,
                                  const std::string & flow_pack,
                                  std::size_t & bytes,
                                  std::string * error) {
    bytes = 0;
    Pixal3DPackInfo shared_info;
    Pixal3DPackInfo flow_info;
    if (!load_pack_info(shared_pack, shared_info, error) ||
        !load_pack_info(flow_pack, flow_info, error)) return false;
    std::size_t shared_bytes = 0;
    std::size_t flow_bytes = 0;
    if (!pack_host_f32_bytes(shared_info, shared_bytes, error) ||
        !pack_host_f32_bytes(flow_info, flow_bytes, error) ||
        !checked_add(shared_bytes, flow_bytes, bytes)) {
        set_error(error, "Pixal3D model memory estimate overflows size_t");
        return false;
    }
    return true;
}

bool run_pixal3d_from_condition_bundle(
    const std::string & shared_pack,
    const std::string & flow_pack,
    const std::string & condition_path,
    const Pixal3DInferenceConfig & config,
    Pixal3DCascadeOutputF32 & output,
    std::string * error) {
    Pixal3DConditionBundleF32 bundle;
    if (!load_pixal3d_condition_bundle(condition_path, bundle, error)) return false;
    for (const char * name : {"ss", "shape_512", "shape_1024", "tex_1024"}) {
        if (!bundle.find(name)) {
            set_error(error, std::string("condition bundle is missing stage: ") + name);
            return false;
        }
    }
    const Pixal3DConditionBuilderF32 condition_builder =
        [&bundle, &config](Pixal3DCascadeStage stage,
                           const std::vector<std::int32_t> & coords,
                           int grid_resolution,
                           Pixal3DImageConditionF32 & result,
                           std::string * error) {
            const char * name = condition_stage_name(stage);
            if (!name) {
                set_error(error, "unknown Pixal3D condition stage");
                return false;
            }
            const Pixal3DConditionStageF32 * source = bundle.find(name);
            if (!source) {
                set_error(error, std::string("missing condition stage: ") + name);
                return false;
            }
            result = Pixal3DImageConditionF32{};
            return project_condition_stage_f32(
                *source, config.camera, grid_resolution, coords,
                result.global, result.projection, error);
        };
    return run_pixal3d_with_condition_builder(
        shared_pack, flow_pack, config, condition_builder, output, error);
}

bool run_pixal3d_from_condition_stages(
    const std::string & shared_pack,
    const std::string & flow_pack,
    const std::vector<Pixal3DConditionStageF32> & stages,
    const Pixal3DInferenceConfig & config,
    Pixal3DCascadeOutputF32 & output,
    std::string * error) {
    Pixal3DConditionBundleF32 bundle;
    bundle.format_version = 1;
    bundle.stages = stages;
    if (!bundle.valid(error)) return false;
    for (const char * name : {"ss", "shape_512", "shape_1024", "tex_1024"}) {
        if (!bundle.find(name)) {
            set_error(error, std::string("native condition stages are missing: ") + name);
            return false;
        }
    }
    const Pixal3DConditionBuilderF32 condition_builder =
        [&bundle, &config](Pixal3DCascadeStage stage,
                           const std::vector<std::int32_t> & coords,
                           int grid_resolution,
                           Pixal3DImageConditionF32 & result,
                           std::string * error) {
            const char * name = condition_stage_name(stage);
            if (!name) {
                set_error(error, "unknown Pixal3D condition stage");
                return false;
            }
            const Pixal3DConditionStageF32 * source = bundle.find(name);
            if (!source) {
                set_error(error, std::string("missing native condition stage: ") + name);
                return false;
            }
            result = Pixal3DImageConditionF32{};
            return project_condition_stage_f32(
                *source, config.camera, grid_resolution, coords,
                result.global, result.projection, error);
        };
    return run_pixal3d_with_condition_builder(
        shared_pack, flow_pack, config, condition_builder, output, error);
}

bool run_pixal3d_from_multiview_condition_bundle(
    const std::string & shared_pack,
    const std::string & flow_pack,
    const std::string & condition_path,
    const Pixal3DInferenceConfig & config,
    Pixal3DCascadeOutputF32 & output,
    std::string * error) {
    Pixal3DMultiViewConditionBundleF32 bundle;
    if (!load_pixal3d_multiview_condition_bundle(condition_path, bundle, error)) {
        return false;
    }
    for (const char * name : {"ss", "shape_512", "shape_1024", "tex_1024"}) {
        if (!bundle.find(name)) {
            set_error(error, std::string("multi-view condition bundle is missing stage: ") +
                               name);
            return false;
        }
    }
    const Pixal3DConditionBuilderF32 condition_builder =
        [&bundle](Pixal3DCascadeStage stage,
                  const std::vector<std::int32_t> & coords,
                  int grid_resolution,
                  Pixal3DImageConditionF32 & result,
                  std::string * error) {
            const char * name = condition_stage_name(stage);
            if (!name) {
                set_error(error, "unknown Pixal3D condition stage");
                return false;
            }
            const Pixal3DConditionStageMVF32 * source = bundle.find(name);
            if (!source) {
                set_error(error, std::string("missing multi-view condition stage: ") + name);
                return false;
            }
            result = Pixal3DImageConditionF32{};
            return project_condition_stage_multiview_average_f32(
                *source, grid_resolution, coords,
                result.global, result.projection, error);
        };
    return run_pixal3d_with_condition_builder(
        shared_pack, flow_pack, config, condition_builder, output, error);
}

bool run_pixal3d_from_multiview_condition_stages(
    const std::string & shared_pack,
    const std::string & flow_pack,
    const std::vector<Pixal3DConditionStageMVF32> & stages,
    const Pixal3DInferenceConfig & config,
    Pixal3DCascadeOutputF32 & output,
    std::string * error) {
    Pixal3DMultiViewConditionBundleF32 bundle;
    bundle.format_version = 1;
    bundle.stages = stages;
    if (!bundle.valid(error)) return false;
    for (const char * name : {"ss", "shape_512", "shape_1024", "tex_1024"}) {
        if (!bundle.find(name)) {
            set_error(error, std::string("native multi-view condition stages are missing: ") +
                               name);
            return false;
        }
    }
    const Pixal3DConditionBuilderF32 condition_builder =
        [&bundle](Pixal3DCascadeStage stage,
                  const std::vector<std::int32_t> & coords,
                  int grid_resolution,
                  Pixal3DImageConditionF32 & result,
                  std::string * error) {
            const char * name = condition_stage_name(stage);
            if (!name) {
                set_error(error, "unknown Pixal3D condition stage");
                return false;
            }
            const Pixal3DConditionStageMVF32 * source = bundle.find(name);
            if (!source) {
                set_error(error, std::string("missing native multi-view condition stage: ") +
                                   name);
                return false;
            }
            result = Pixal3DImageConditionF32{};
            return project_condition_stage_multiview_average_f32(
                *source, grid_resolution, coords,
                result.global, result.projection, error);
        };
    return run_pixal3d_with_condition_builder(
        shared_pack, flow_pack, config, condition_builder, output, error);
}

bool write_pixal3d_obj(const DualGridMeshF32 & mesh,
                       const std::string & path,
                       std::string * error) {
    if (mesh.vertices.size() % 3 != 0 || mesh.faces.size() % 3 != 0) {
        set_error(error, "mesh vertices/faces are not packed xyz/index triples");
        return false;
    }
    const std::size_t vertex_count = mesh.vertices.size() / 3;
    for (std::int32_t index : mesh.faces) {
        if (index < 0 || static_cast<std::size_t>(index) >= vertex_count) {
            set_error(error, "mesh face index is outside the vertex buffer");
            return false;
        }
    }
    std::ofstream file(path, std::ios::trunc);
    if (!file) {
        set_error(error, "cannot open OBJ output: " + path);
        return false;
    }
    file.setf(std::ios::fmtflags(0), std::ios::floatfield);
    file.precision(std::numeric_limits<float>::max_digits10);
    file << "# Pixal3D.cpp Flexible Dual Grid mesh\n";
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        const float x = mesh.vertices[vertex * 3 + 0];
        const float y = mesh.vertices[vertex * 3 + 1];
        const float z = mesh.vertices[vertex * 3 + 2];
        // Preserve the legacy shape-only OBJ compatibility frame.  The native
        // textured GLB writer uses the Python main-path frame separately.
        file << "v " << -x << " " << -z << " " << -y << "\n";
    }
    for (std::size_t face = 0; face < mesh.faces.size(); face += 3) {
        file << "f " << (mesh.faces[face + 0] + 1) << " "
             << (mesh.faces[face + 1] + 1) << " "
             << (mesh.faces[face + 2] + 1) << "\n";
    }
    if (!file) {
        set_error(error, "failed while writing OBJ output: " + path);
        return false;
    }
    return true;
}

} // namespace pixal3d
