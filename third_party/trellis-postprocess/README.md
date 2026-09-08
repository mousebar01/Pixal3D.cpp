# trellis-postprocess

Vendored mesh postprocess modules from pwilkin/trellis.cpp at commit
2516c48b677050c570f47eba2e68dc8a5bc918b0 (MIT). Files:

- `decimate_qem.cpp`: CuMesh QEM edge-collapse decimator (CPU path of the
  reference simplify.cu).
- `uv_bake.{h,cpp}`: weld/clean/orient/decimate/simplify/holes + xatlas UV
  unwrap with trilinear voxel-PBR bake and seam dilation.
- `remesh_dc.{h,cpp}`: narrow-band dual-contouring remesh (needs a volume
  field; used by the pipeline remesh path).
- `tri_bvh.{h,cpp}`: triangle BVH for surface snapping.
- `Simplify.h`: Fast-Quadric-Mesh-Simplify (MIT, Spacerat) used by the
  CuMesh-port QEM decimator.

The modules are used by the GLB export remesh and UV paths. They are modified
from upstream only where noted in the integration change summary.
