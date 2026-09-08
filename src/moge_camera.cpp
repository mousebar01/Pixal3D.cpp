#include "pixal3d/moge_camera.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#ifdef PIXAL3D_HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace pixal3d {
namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

// normalized_view_plane_uv() (moge/utils/geometry_torch.py): uv spans
// [-span, span] with span = aspect / sqrt(1 + aspect^2) horizontally and
// 1 / sqrt(1 + aspect^2) vertically; v grows toward the bottom row.
double uv_span_x(double aspect_ratio) {
    return aspect_ratio / std::sqrt(1.0 + aspect_ratio * aspect_ratio);
}

double uv_span_y(double aspect_ratio) {
    return 1.0 / std::sqrt(1.0 + aspect_ratio * aspect_ratio);
}

} // namespace

bool moge_camera_supported() {
#ifdef PIXAL3D_HAVE_ONNXRUNTIME
    return true;
#else
    return false;
#endif
}

float moge_distance_from_fov_f32(
    float camera_angle_x,
    float mesh_scale,
    int image_resolution) {
    // inference.py compute_f_pixels(): focal_length = 16 / tan(fov / 2),
    // f_pixels = focal_length * resolution / 32 (sensor width 32 mm).
    const double focal_length = 16.0 / std::tan(static_cast<double>(camera_angle_x) * 0.5);
    const double f_pixels = focal_length * static_cast<double>(image_resolution) / 32.0;
    // distance_from_fov(): the grid point (-1, 0, 0) rotates to itself and
    // halves to (-0.5 / mesh_scale, 0, 0); the target is the left image
    // border (x_ndc = -resolution / 2) and the world y component is 0.
    const double grid_x = -0.5 / static_cast<double>(mesh_scale);
    const double x_ndc = -static_cast<double>(image_resolution) * 0.5;
    return static_cast<float>(f_pixels * grid_x / x_ndc);
}

bool moge_recover_focal_shift_f32(
    const float * points,
    const float * mask,
    int width,
    int height,
    const MoGeCameraOptions & options,
    float & focal,
    float & shift,
    std::string * error) {
    focal = 1.0f;
    shift = 0.0f;
    if (!points || !mask || width <= 0 || height <= 0) {
        set_error(error, "invalid point map for MoGe focal recovery");
        return false;
    }
    const int recovery = options.recovery_resolution > 0 ? options.recovery_resolution : 64;
    const std::size_t plane = static_cast<std::size_t>(width) * height;
    for (std::size_t index = 0; index < plane * 3; ++index) {
        if (!std::isfinite(points[index])) {
            set_error(error, "point map contains a non-finite value");
            return false;
        }
    }

    // Nearest downsample: torch F.interpolate(mode='nearest') maps output
    // row d to source row floor(d * in / out).
    std::vector<int> rows(recovery);
    std::vector<int> cols(recovery);
    for (int d = 0; d < recovery; ++d) {
        rows[d] = static_cast<int>(static_cast<std::int64_t>(d) * height / recovery);
        cols[d] = static_cast<int>(static_cast<std::int64_t>(d) * width / recovery);
    }
    const double aspect = static_cast<double>(width) / height;
    const double span_x = uv_span_x(aspect);
    const double span_y = uv_span_y(aspect);
    std::vector<double> uv_x;
    std::vector<double> uv_y;
    std::vector<double> xy_x;
    std::vector<double> xy_y;
    std::vector<double> xy_z;
    uv_x.reserve(static_cast<std::size_t>(recovery) * recovery);
    for (int d = 0; d < recovery; ++d) {
        for (int e = 0; e < recovery; ++e) {
            const std::size_t source =
                static_cast<std::size_t>(rows[d]) * width + cols[e];
            if (mask[source] <= 0.5f) continue;
            uv_x.push_back(-span_x * (width - 1) / width +
                           2.0 * span_x * (width - 1) / width * cols[e] / (width - 1));
            uv_y.push_back(-span_y * (height - 1) / height +
                           2.0 * span_y * (height - 1) / height * rows[d] / (height - 1));
            xy_x.push_back(points[source * 3 + 0]);
            xy_y.push_back(points[source * 3 + 1]);
            xy_z.push_back(points[source * 3 + 2]);
        }
    }
    const std::size_t count = uv_x.size();
    if (count < 2) {
        // The reference returns focal=1, shift=0 for degenerate masks.
        return true;
    }

    // One-parameter Levenberg-Marquardt on
    //   r_i(s) = f(s) * xy_i / (z_i + s) - uv_i,
    //   f(s) = sum(p * uv) / sum(p * p),
    // mirroring scipy.optimize.least_squares(method='lm', x0=0, ftol=1e-3)
    // with its default two-point finite-difference Jacobian.
    std::vector<double> projection_x(count);
    std::vector<double> projection_y(count);
    std::vector<double> residual(count * 2);
    std::vector<double> residual_plus(count * 2);
    std::vector<double> residual_minus(count * 2);
    // r_i(s) with f(s) folded in, exactly as scipy sees the residual vector.
    auto residuals_at = [&](double s) {
        double numerator = 0.0;
        double denominator = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            const double inverse_z = 1.0 / (xy_z[i] + s);
            projection_x[i] = xy_x[i] * inverse_z;
            projection_y[i] = xy_y[i] * inverse_z;
            numerator += projection_x[i] * uv_x[i] + projection_y[i] * uv_y[i];
            denominator += projection_x[i] * projection_x[i] +
                           projection_y[i] * projection_y[i];
        }
        const double focal_value = numerator / denominator;
        double cost = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            residual[i * 2 + 0] = focal_value * projection_x[i] - uv_x[i];
            residual[i * 2 + 1] = focal_value * projection_y[i] - uv_y[i];
            cost += residual[i * 2 + 0] * residual[i * 2 + 0] +
                    residual[i * 2 + 1] * residual[i * 2 + 1];
        }
        return focal_value;
    };
    double s = 0.0;
    double focal_value = residuals_at(s);
    double cost = 0.0;
    for (std::size_t i = 0; i < count * 2; ++i) {
        cost += residual[i] * residual[i];
    }
    if (!std::isfinite(cost)) {
        focal = static_cast<float>(focal_value);
        shift = static_cast<float>(s);
        return true;
    }
    const double epsilon = 1.0e-6;
    double damping = 1.0e-3;
    for (int iteration = 0; iteration < 100; ++iteration) {
        const double cost_before = cost;
        bool improved = false;
        for (int damping_try = 0; damping_try < 60; ++damping_try) {
            // Two-sided finite-difference Jacobian of the full residual
            // vector (f(s) included), then the damped 1x1 normal equation.
            double jacobian_square = 0.0;
            double jacobian_residual = 0.0;
            const double focal_plus = residuals_at(s + epsilon);
            std::copy(residual.begin(), residual.end(), residual_plus.begin());
            const double focal_minus = residuals_at(s - epsilon);
            std::copy(residual.begin(), residual.end(), residual_minus.begin());
            residuals_at(s);
            for (std::size_t i = 0; i < count * 2; ++i) {
                const double jacobian =
                    (residual_plus[i] - residual_minus[i]) / (2.0 * epsilon);
                jacobian_square += jacobian * jacobian;
                jacobian_residual += jacobian * residual[i];
            }
            (void)focal_plus;
            (void)focal_minus;
            const double denominator = (1.0 + damping) * jacobian_square;
            if (!(denominator > 0.0) || !std::isfinite(denominator)) break;
            const double step = -jacobian_residual / denominator;
            const double candidate_focal = residuals_at(s + step);
            double candidate_cost = 0.0;
            for (std::size_t i = 0; i < count * 2; ++i) {
                candidate_cost += residual[i] * residual[i];
            }
            if (std::isfinite(candidate_cost) && candidate_cost < cost_before) {
                s += step;
                focal_value = candidate_focal;
                cost = candidate_cost;
                improved = true;
                damping = std::max(damping * 0.5, 1.0e-12);
                break;
            }
            damping *= 4.0;
        }
        if (!improved) break;
        // scipy ftol=1e-3: terminate when the relative cost decrease stalls.
        if (cost_before - cost <= 1.0e-3 * cost_before) break;
    }
    focal = static_cast<float>(focal_value);
    shift = static_cast<float>(s);
    return true;
}

bool estimate_pixal3d_camera_with_moge_f32(
    const Pixal3DImageF32 & image,
    const std::string & onnx_path,
    const MoGeCameraOptions & options,
    ProjectionCamera & camera,
    std::string * error) {
#ifdef PIXAL3D_HAVE_ONNXRUNTIME
    camera = ProjectionCamera{};
    if (!image.valid(error) || image.channels != 3 || onnx_path.empty()) {
        set_error(error, "invalid image or MoGe ONNX path");
        return false;
    }
    try {
        static Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "pixal3d-moge");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        Ort::Session session(environment, onnx_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        auto image_name = session.GetInputNameAllocated(0, allocator);
        auto tokens_name = session.GetInputNameAllocated(1, allocator);

        // Pixal3DImageF32 is channel-major float RGB in [0, 1]; the ONNX
        // graph consumes NCHW directly.
        const std::int64_t height = image.height;
        const std::int64_t width = image.width;
        const std::array<std::int64_t, 4> image_shape = {1, 3, height, width};
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<float> tensor(image.pixels.begin(), image.pixels.end());
        Ort::Value image_tensor = Ort::Value::CreateTensor<float>(
            memory, tensor.data(), tensor.size(), image_shape.data(), 4);
        const std::int64_t num_tokens = options.num_tokens;
        Ort::Value tokens_tensor = Ort::Value::CreateTensor<std::int64_t>(
            memory, const_cast<std::int64_t *>(&num_tokens), 1, nullptr, 0);

        const char * input_names[] = {image_name.get(), tokens_name.get()};
        const char * output_names[] = {"points", "normal", "mask", "metric_scale"};
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(image_tensor));
        input_tensors.push_back(std::move(tokens_tensor));
        auto outputs = session.Run(Ort::RunOptions{}, input_names,
                                   input_tensors.data(), input_tensors.size(),
                                   output_names, 4);
        if (outputs.size() != 4 || !outputs[0].IsTensor() || !outputs[2].IsTensor()) {
            set_error(error, "MoGe ONNX graph outputs do not match the expected "
                             "points/normal/mask/metric_scale contract");
            return false;
        }
        auto points_info = outputs[0].GetTensorTypeAndShapeInfo();
        auto mask_info = outputs[2].GetTensorTypeAndShapeInfo();
        if (points_info.GetElementCount() !=
                static_cast<std::size_t>(height) * width * 3 ||
            mask_info.GetElementCount() !=
                static_cast<std::size_t>(height) * width) {
            set_error(error, "MoGe ONNX output shape does not match the input image");
            return false;
        }
        const float * points = outputs[0].GetTensorData<float>();
        const float * mask = outputs[2].GetTensorData<float>();

        float focal = 0.0f;
        float shift = 0.0f;
        if (!moge_recover_focal_shift_f32(points, mask, image.width, image.height,
                                          options, focal, shift, error)) {
            return false;
        }
        // v2.py: fx = focal / 2 * sqrt(1 + a^2) / a with a = width / height;
        // the focal is expressed relative to half the image diagonal, and the
        // reference wild path normalizes fx by the image width.
        const double aspect = static_cast<double>(width) / height;
        const double fx_normalized = focal * 0.5 * std::sqrt(1.0 + aspect * aspect) / aspect;
        const double camera_angle_x = 2.0 * std::atan(1.0 / (2.0 * fx_normalized));
        const double distance = moge_distance_from_fov_f32(
            static_cast<float>(camera_angle_x), options.mesh_scale,
            options.estimation_resolution);
        camera = ProjectionCamera::front(
            static_cast<float>(camera_angle_x),
            static_cast<float>(distance),
            options.mesh_scale);
        return true;
    } catch (const Ort::Exception & exception) {
        set_error(error, std::string("MoGe ONNX inference failed: ") + exception.what());
        return false;
    }
#else
    (void)image;
    (void)onnx_path;
    (void)options;
    (void)camera;
    set_error(error, "MoGe camera estimation requires a build with onnxruntime "
                     "(PIXAL3D_HAVE_ONNXRUNTIME)");
    return false;
#endif
}

} // namespace pixal3d
