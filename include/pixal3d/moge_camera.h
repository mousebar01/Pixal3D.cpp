#ifndef PIXAL3D_MOGE_CAMERA_H
#define PIXAL3D_MOGE_CAMERA_H

#include "pixal3d/image.h"
#include "pixal3d/projection.h"

#include <string>

namespace pixal3d {

// MoGe-2 ONNX camera estimation.
//
// The official MoGe-2 ONNX export (https://huggingface.co/Ruicheng/moge-2-vitl-normal-onnx,
// see microsoft/MoGe docs/onnx.md) exposes the raw forward pass only:
// image [B,3,H,W] + num_tokens -> affine point map [B,H,W,3], normal,
// mask [B,H,W], metric scale.  Camera intrinsics are not an output head; the
// reference pipeline recovers them from the point map by fitting an unknown z
// shift and focal length on a 64x64 nearest downsample of the masked map
// (moge/utils/geometry_numpy.py solve_optimal_focal_shift), then converts the
// focal to a horizontal FOV.  The wild-image path pairs the FOV with a
// distance derived so that the unit-conditioning-volume edge projects onto
// the image border (inference.py distance_from_fov).
//
// This module implements the same chain in C++:
//   ONNX forward -> masked 64x64 recovery -> 1-D Levenberg-Marquardt shift
//   fit + closed-form focal -> horizontal FOV -> reference distance rule.
// The recovery and distance helpers are pure math and always compiled; the
// ONNX forward requires an onnxruntime build (PIXAL3D_HAVE_ONNXRUNTIME).
struct MoGeCameraOptions {
    // Number of base ViT tokens; the reference wild path uses infer() with
    // resolution_level 9, which resolves to the upper end of the suggested
    // num_tokens_range [1200, 3600].
    int num_tokens = 3600;
    // Recovery grid; recover_focal_shift downsamples to this size.
    int recovery_resolution = 64;
    // image_resolution argument of the reference distance_from_fov call.
    int estimation_resolution = 512;
    float mesh_scale = 1.0f;
};

// True when the build contains the optional onnxruntime backend.
bool moge_camera_supported();

// Recover the focal length (relative to half the image diagonal) and z shift
// from an affine point map exactly like the reference implementation.  The
// point map is row-major HxWx3; mask is row-major HxW (values > 0.5 count as
// foreground).  Both must contain finite values.
bool moge_recover_focal_shift_f32(
    const float * points,
    const float * mask,
    int width,
    int height,
    const MoGeCameraOptions & options,
    float & focal,
    float & shift,
    std::string * error = nullptr);

// Reference inference.py distance_from_fov(): distance placing the
// unit-conditioning-volume edge point (-0.5 / mesh_scale on the camera axis)
// at the left image border for the given horizontal FOV.
float moge_distance_from_fov_f32(
    float camera_angle_x,
    float mesh_scale,
    int image_resolution);

// Estimate a front ProjectionCamera from an already-preprocessed condition
// image using the MoGe-2 ONNX model at onnx_path.  Requires a build with
// onnxruntime; otherwise reports an error.
bool estimate_pixal3d_camera_with_moge_f32(
    const Pixal3DImageF32 & image,
    const std::string & onnx_path,
    const MoGeCameraOptions & options,
    ProjectionCamera & camera,
    std::string * error = nullptr);

} // namespace pixal3d

#endif
