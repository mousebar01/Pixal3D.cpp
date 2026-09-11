# Texture Output Appears Gray: Diagnosis Record

- **Recorded:** 2026-09-11
- **Scope:** the end-to-end texture output for the fixed `armor_knight.png`
  image
- **Status:** diagnosed; no model, decoder, UV-bake, or GLB-export change is
  justified by this incident

This record is a guardrail for future debugging. The result below applies to
this fixture and must not be generalized to a high-saturation color test.

## Fixed regression fixture

Use this image for repeatable end-to-end runs:

```text
weights/Pixal3D/test_inputs/armor_knight.png
```

SHA-256:

```text
d4e48ed407101749dc7bfe0bcafcdadd27278fbe563038a9c49ad9e5ff58bc19
```

The image is dominated by a silver-gray metal knight, dark-gray chain mail,
small brown leather areas, and a black background. It is therefore useful for
pipeline stability and output-structure checks, but it is **not** a suitable
standalone fixture for deciding whether red, green, and blue channels are
being preserved.

## Observed symptom

The rendered textured result looked gray. The first assumption was that the
C++ path had lost color in one of these places:

- decoder output or channel ordering;
- the `value * 0.5f + 0.5f` output conversion;
- UV projection or baking;
- GLB base-color texture export.

The intermediate and reference comparisons below rule out that diagnosis for
this input.

## Evidence

### Color is already low-chroma before UV baking

The C++ `texture_decoded` tensor was low-chroma before UV baking. Per-channel
means and mean absolute channel differences were:

| measurement | value |
| --- | ---: |
| C++ mean R | 0.28026 |
| C++ mean G | 0.25714 |
| C++ mean B | 0.24046 |
| C++ mean `|R-G|` | 0.02305 |
| C++ mean `|G-B|` | 0.01681 |
| C++ mean `|R-B|` | 0.03975 |

Consequently, the GLB writer and UV bake cannot be the point at which this
fixture first became gray: the decoded texture was already close to neutral
before either stage ran.

### The Python reference shows the same behavior

The same image was run through the Python reference with the same seed and
camera parameters:

```text
seed       = 42
fov        = 0.558434
distance   = 1.743943
mesh_scale = 1.0
```

The complete 1024 cascade also produced low-chroma texture values:

| measurement | value |
| --- | ---: |
| Python mean R | 0.26124 |
| Python mean G | 0.24864 |
| Python mean B | 0.24482 |
| Python mean `|R-G|` | 0.01325 |
| Python mean `|G-B|` | 0.00634 |
| Python mean `|R-B|` | 0.01693 |

Both implementations therefore agree on the important qualitative result:
this silver-gray input produces a low-saturation texture. The evidence does
not support changing the C++ decoder or exporter merely to make the preview
more colorful.

### Projection and preprocessing were not the source

For the same input, the C++ Texture projection condition compared with the
Python reference as follows:

```text
correlation  ~= 0.9999886
relative L2  ~= 0.00481
```

Swapping coordinate axes increased the relative error to approximately
`0.35`–`0.46`, so the observed result is not explained by an XYZ coordinate
order mistake.

The preprocessed images also differed only slightly:

```text
maximum pixel difference = 3/255
mean pixel difference    ~= 0.149/255
```

That preprocessing difference is far too small to explain an entire texture
becoming gray.

## Conclusion and code-change guardrail

For `armor_knight.png`, the low saturation is a behavior shared by the Python
reference and C++ implementation and is consistent with the mostly gray input
material. Do **not** treat this observation alone as evidence of a bug in:

- `slat_decoder.cpp` or decoder channel layout;
- the `value * 0.5f + 0.5f` conversion;
- UV baking or GLB base-color export;
- projection coordinate order.

In particular, do not relax numerical checks or make an output look more
colorful by applying an unvalidated channel swap, contrast/saturation boost,
or decoder rewrite. Any such change would make the C++ path diverge from the
reference without fixing the reported case.

## Short triage order for a future gray-output report

1. **Verify the fixture.** Check the input path and SHA-256. Record whether the
   image itself contains saturated red, green, or blue regions.
2. **Inspect the first colored intermediate.** Measure per-channel means and
   `mean(abs(R-G))`, `mean(abs(G-B))`, and `mean(abs(R-B))` on
   `texture_decoded`, before UV baking.
3. **Compare the Python oracle.** Use the same image, seed, camera parameters,
   resolution, and precision policy. If both paths are low-chroma, stop treating
   this as a C++ gray-conversion bug.
4. **Check projection parity.** Compare correlation and relative L2; test axis
   permutations only as a diagnostic. Do not change axis order without a
   reference mismatch that identifies it as the cause.
5. **Only if the C++ intermediate is gray while the reference is not**, inspect
   decoder tensor layout, channel ordering, normalization, dtype conversion,
   and the output/export path in that order.
6. **Use a separate saturated-color fixture** (for example, a synthetic image
   with distinct red, green, and blue patches) before making any color-channel
   change.

The `armor_knight.png` fixture should remain a regression image for deterministic
pipeline behavior, not the sole color-fidelity test.

## Separate issue: texture voxel-count mismatch

The investigation also observed a discrete output-size difference:

```text
Python final texture voxel count = 4,644,079
C++ final texture voxel count    = 4,127,527
```

This may involve BF16/F32 handling or the sparse decoder path, but there is no
evidence that it caused the gray appearance. Track it as a separate numerical
parity task; do not use the gray preview as proof of its cause.

## Validation context and local evidence

The comparison was performed in the Docker development environment. The CUDA
build used CUDA `12.6.3`, and the recorded CTest run passed all 22 tests. The
following files were diagnostic artifacts only and are not project sources:

```text
/tmp/pixal3d-gray-debug-0910b/cascade/texture_decoded.bin
/tmp/pixal3d-condition-compare/cpp_tex_projection.bin
/tmp/pixal3d-condition-compare/reference_texture_attrs.npy
/tmp/python_full_20260911_retry.log
```

Do not add these artifacts, model weights, generated meshes, or build
directories to version control.
