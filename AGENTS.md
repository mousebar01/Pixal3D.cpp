# Pixal3D.cpp Agent Rules

## Project scope

Pixal3D.cpp is a C++17/ggml migration of the Pixal3D image-to-3D pipeline.
The outer project is the source of truth for new C++ code. The original
Python implementation and the `ref/trellis2cpp` port are reference or bundled
dependencies and must not be changed incidentally.

Keep these boundaries explicit:

- `src/` and `include/` contain the outer Pixal3D API and CLI.
- `scripts/` contains download, conversion, and validation helpers.
- `ref/Pixal3D/` is the original Python reference implementation.
- `ref/trellis2cpp/` is the current ggml stage-1 backend.
- `weights/`, `build/`, downloaded models, and generated artifacts are local
  development data and must not be added to version control.
- Docker images are development conveniences; do not treat a local image or
  container state as project source.

The NAF image upsampler is a CPU-resident F32 path. OpenMP may parallelize
independent output work, but per-pixel softmax workspaces must remain
thread-local and the reference accumulation order must be preserved.

Do not claim that a model or pipeline stage is supported merely because its
weights can be serialized. A stage is supported only after its GGUF metadata,
tensor layout, loader, forward pass, and reference comparison are complete.

## Before changing code

1. Read the relevant public header, implementation, tests, and README.
2. Check the matching Python reference in `ref/Pixal3D/` before changing model
   math, tensor names, shapes, normalization, attention, sampling, or decoder
   behavior.
3. Check the bundled `ref/trellis2cpp` API and ggml conventions before adding
   a backend call.
4. Keep a change focused. Separate model support, loader changes, numerical
   fixes, build changes, and documentation changes when they can be reviewed
   independently.
5. Do not overwrite user changes or regenerate large files without an explicit
   request.

If a change affects a vendored/reference directory, call out the reason and
the exact files in the change summary. Do not silently fork the reference
implementation inside the outer project.

## Build and test gates

The normal local build is:

```sh
git -C ref/trellis2cpp submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
(cd build && ctest --output-on-failure)
```

For a quick API smoke check:

```sh
./build/bin/pixal3d --help
./build/bin/pixal3d --version
```

Use `PIXAL3D_GGML_NATIVE=ON` only when the binary is intentionally tied to
the build machine's CPU family. Keep the default build portable.

For every C++ change:

- build the affected target;
- run the relevant CTest tests;
- run a CLI smoke test when the command or public API changes;
- include the exact command and result in the handoff.

For model or numerical changes, use this order:

1. Convert lossless F32 weights.
2. Run the original Python reference on a fixed input.
3. Run the C++ implementation on the same input.
4. Compare shapes, finite values, fingerprints, relative L2 error, and any
   discrete result such as occupancy sign agreement.
5. Only after F32 agrees, test F16/BF16 and then any quantized format.

Do not hide a numerical mismatch behind a relaxed threshold. Explain the
source of the mismatch and record the chosen tolerance.

### Backend policy and multi-backend execution

Use ggml's backend registry as the only source of device discovery. New model
code must not enumerate CUDA, CPU, or other backends with separate hard-coded
selection loops. The shared backend manager must:

- expose an explicit policy: `auto`, `cpu`, or `gpu:<index>`; future
  accelerators must be additive rather than changing the meaning of `cpu`;
- enumerate devices with `ggml_backend_dev_count()`, report device type and
  description, and treat GPU, IGPU, ACCEL, and CPU as distinct device classes;
- always keep a CPU backend available as the final fallback;
- make forced modes fail loudly when the requested backend is unavailable or
  does not support the graph, while `auto` may fall back only after logging the
  reason;
- return the selected backend list and policy to every pipeline stage so that
  backend choice is not hidden in a model constructor.

Production graphs must use `ggml_backend_sched` for placement, allocation, and
cross-backend copies. Model code should build the graph once, allocate or size
the scheduler buffers once, compute through the scheduler, and reset/reuse it
per invocation. Direct calls to `ggml_backend_graph_compute()` are restricted
to small isolated probes and temporary migration code; do not add new direct
backend paths. Existing direct paths are migration debt and must be listed in
the change summary until they are moved behind the shared scheduler.

Weights must be placed according to backend capability. Before assigning a
weight buffer, check the required operation with `ggml_backend_dev_supports_op`
or the equivalent ggml backend API. Unsupported operations must be reported
with the stage, operation, tensor, and backend name; never silently assume that
all GPU backends implement the same operation set.

Every backend initialization and graph execution must log the policy,
selected device/backend, weight-buffer sizes, compute-buffer sizes, and any
fallback. Backend options and compile definitions must be verified against the
pinned official ggml commit; an option whose macro or API is absent upstream
must be removed, marked unsupported, or covered by an explicit downstream
patch and test.

### GPU-first execution invariant

For every stage declared GPU-capable, use `GPU > CPU` as the placement priority
for every validated GPU-capable numerical operation. Backend selection alone is
not sufficient: a stage is GPU-first only when its eligible feature operations
and intermediate activation residency satisfy this invariant.

- GPU-capable numerical operations include, where the selected backend
  supports them, linear/matrix multiplication, convolution, normalization,
  activation, bias, residual arithmetic, and feature gather/scatter or
  reshape/permute operations. Once a GPU implementation has passed the
  required correctness checks, do not call an equivalent CPU helper in the
  GPU stage.
- Keep eligible weights and intermediate feature activations on the selected
  GPU for the duration of a GPU stage. H2D/D2H transfers are allowed at stage
  boundaries or at an explicitly documented topology transition, but must not
  be introduced implicitly inside a hot loop. Every transfer must be counted
  and profiled.
- CPU execution is allowed only for an explicit host-only allowlist, such as
  coordinate/topology construction, metadata and serialization, validation
  required by the current policy, or a mesh/export algorithm that has no
  validated GPU implementation yet. Such work must be named and logged as
  host-only; it must not make the enclosing stage appear fully GPU-resident.
- `gpu:<index>` is strict for GPU-capable stages: an unavailable required GPU
  operation or an unapproved CPU fallback is an error, not a silent downgrade.
  `auto` may choose a CPU fallback only after logging the stage, operation,
  tensor, backend, and reason. If a partial GPU plan would cause repeated
  activation transfers, prefer a complete CPU plan or an explicitly declared
  migration/probe plan rather than silently alternating between devices.
- A stage must not be described as GPU-first while it contains an untracked
  partial GPU path. Diagnostics must report GPU operations, host-only
  operations, CPU fallbacks, and H2D/D2H counts and bytes.
- Forced-GPU tests must assert placement for all eligible operations and must
  fail when an eligible operation executes on the CPU. CPU-only host work and
  explicit migration probes must be tested separately.

This is a placement and residency rule, not a requirement to immediately
rewrite every CPU-only postprocess algorithm. QEM, xatlas charting, UV baking,
and other export operations may remain CPU-only until they have validated GPU
implementations, but they must be explicitly classified as host-side work and
must not be presented as GPU-accelerated stages.

### Precision policy

Keep these two decisions separate:

- `--ftype auto` is the stable project deployment default, written exactly as
  `auto=F32 flow, F16 decoder`: all flow bundles select F32, while the shared
  decoder bundle selects the F16 policy (eligible matrix tensors are F16 and
  sensitive/scalar tensors remain F32). Do not change this default or describe
  it as an all-F32 development mode without an explicit maintainer decision.
- Numerical porting and Python-reference parity use `--ftype 0` explicitly.
  This creates all-F32 GGUF storage so tensor layouts and forward math can be
  compared without a storage conversion hiding an error. It is a validation
  choice, not a replacement for the `auto` deployment policy.

The DINOv3 and NAF component converters currently require F32. A decoder
source that is already FP16 cannot regain lost precision merely by promoting
it to F32; BF16 in a checkpoint filename also does not prove that the payload
is BF16, so inspect the actual tensor dtype before choosing a policy.

This policy is a repository invariant. If it changes, update the converter's
named constants and parser help, the README command examples, and the
converter contract test in the same change. Never describe `auto` as “F32”
without naming the component split; use `--ftype 0` whenever “all F32” is
intended. A generated filename such as `*-f16.gguf` or `*-f32.gguf` must match
the actual converter argument and the metadata inspection output.

## C++ and ggml style

- Use C++17 already selected by the top-level CMake project.
- Follow the surrounding code before introducing a new style.
- Use four spaces, braces on the same line, no trailing whitespace, and
  lowercase filenames with underscores, matching this repository.
- Prefer simple, readable code over clever templates or abstraction layers.
- Use `snake_case` for functions and variables; keep the existing `pixal3d_`
  and `trellis2_` prefixes for public symbols.
- Use sized integer types for public tensor dimensions, counts, byte offsets,
  and serialized fields where appropriate.
- Keep ownership explicit. Use RAII for files, buffers, ggml contexts, and
  model handles; release stage-specific resources when a pipeline stage ends.
- Do not introduce global mutable model state or hidden allocations in a hot
  inference loop.
- Preserve the existing error-reporting style at public boundaries. Errors
  must identify the file, tensor, stage, or backend that failed.
- Keep public headers dependency-light and document non-obvious tensor layouts
  or API invariants there.
- Maintain CPU portability. Backend-specific code must have a clear fallback
  or an explicit build-time requirement.

Do not put backend-specific `#ifdef` branches in model math when a ggml
backend API or scheduler can express the same behavior. Backend-specific code
belongs in the backend manager, build configuration, or a narrowly scoped
capability probe.

ggml conventions are part of the API, not implementation details:

- tensors are row-major;
- dimension 0 is the contiguous/column dimension, followed by rows and higher
  dimensions;
- `ggml_mul_mat(ctx, A, B)` uses ggml's transposed matrix convention, so verify
  the intended mathematical operation instead of assuming standard indexing;
- preserve the reference model's channel order, reshape order, RoPE layout,
  normalization epsilon, and dtype conversion rules.

## Model files and GGUF

Model files stay outside version control under `weights/Pixal3D/`. Never add
`.safetensors`, `.gguf`, `.dinodata`, latent dumps, rendered meshes, or partial
downloads to a source change.

The download and conversion scripts must:

- reject `.part` files as incomplete;
- validate expected size or checksum when metadata is available;
- write converted files atomically;
- report the source checkpoint, config, stage, dtype, and output path;
- avoid loading an entire multi-gigabyte checkpoint when a lazy tensor reader
  is available.

Use two distinct levels of model organization:

1. One logical GGUF per independent pipeline component. For Pixal3D this means
   sparse-structure flow/decoder, Shape SLat flow 512/1024, Shape decoder,
   Texture flow, and Texture decoder remain separate logical models.
2. Physical GGUF shards only when one logical model is too large. Shards must
   be loadable as one model and must not be confused with functional stages.

The 512 and 1024 Shape flow models are cascade stages, not duplicate files.
Keep their stage names and metadata unambiguous. Do not merge them or add a
sharding scheme until the C++ loader can discover, validate, and release the
corresponding files.

GGUF metadata and tensor names must be deterministic and namespaced. When a
new architecture is added, define its metadata contract and loader checks
before converting the full checkpoint. A converter warning about an
unsupported forward path is not equivalent to runtime support.

## Python reference and porting discipline

The Python implementation is the numerical oracle for behavior until an
equivalent C++ test replaces it. When porting a module:

- identify the exact Python class and call path;
- record input/output shapes and dtypes;
- port one operation or block at a time;
- compare a small deterministic fixture before optimizing;
- preserve sampler schedules, guidance intervals, random seeds, and threshold
  semantics;
- add a regression fixture for every bug found at the language boundary.

Projection attention, sparse tensors, 3D convolutions, pixel shuffle, and
RoPE are high-risk areas. Do not substitute a superficially similar ggml
operation without a reference comparison.

## Test taxonomy and CI

CTest is the authoritative list of automated tests. Every fixture executable
that is intended to be a regression test must have a matching `add_test()`;
building a `*_fixture` target without registering it does not count as test
coverage. Assign stable labels to each test, at minimum:

- `unit`: no model files, network, GPU, or nondeterministic input;
- `loader`: GGUF metadata, tensor names, layout, dtype, and truncated-file
  rejection;
- `parity`: a fixed fixture compared with the Python oracle or a golden F32
  result;
- `cpu` and `cuda`: tests that explicitly force the corresponding backend;
- `integration`: real model or image smoke tests, allowed to be skipped when
  declared assets are absent.

Keep tests in progressively more expensive tiers:

1. cheap unit and shape/finite checks run on every build;
2. loader and component numerical fixtures run on CPU with deterministic
   inputs and recorded fingerprints;
3. CPU/CUDA (and later other backend) parity uses the same graph, input, seed,
   and explicitly documented tolerances;
4. real-weight/image smoke tests check finite values and mesh structural gates;
5. quality and performance benchmarks remain separate from the default CTest
   pass and record backend, device, timings, memory, and output checksums.

Forced CPU and forced GPU tests must be separate from `auto` tests. `auto`
must never hide a failed GPU path by making the only test pass on CPU. Tests
that need downloaded models must declare their asset requirement and be
filterable by label, following the `ctest -L <label>` pattern used by
whisper.cpp.

CI must keep a portable CPU job as the required baseline, then add backend
jobs for CUDA and other enabled backends when the runner provides them. At
minimum, exercise one Debug and one Release build, a sanitizer job, and the
configured backend's CTest labels. Docker images may provide reproducible
backend build environments, but an image build is not a numerical or runtime
parity test.

## Docker and local development

`docker/Dockerfile` is the small C++ build/runtime image. The optional
`docker/Dockerfile.python` is for the original Python reference environment
and may pull very large dependencies. Keep downloaded weights mounted or in
the ignored `weights/` directory; do not bake them into images.

The Docker setup is for local development and reproducibility. Do not add
generated images, running-container state, credentials, caches, or host
mount-specific paths to source files.

## Agent conduct and handoff

Agents may inspect, edit, build, and test files needed for the requested task,
but must not commit, push, publish, or open external issues/PRs unless the
user explicitly asks for that action.

Before editing, state the intended scope when the change is more than a small
local fix.

Troubleshooting findings must be captured in `docs/` before handoff. For every
completed investigation that identifies a reproducible bug, regression, failure,
or misleading symptom, create or update a focused document with the symptom,
fixture/environment, reproduction or evidence, root cause or ruled-out causes,
and the concrete fix, workaround, or prevention rule. If the issue remains
unresolved, record that status and the next diagnostic step instead of claiming
resolution. Mention the document path in the handoff, and keep generated logs,
weights, meshes, and build artifacts outside `docs/` and version control.

After editing, report:

- files changed and why;
- tests or validation commands run;
- known limitations and unimplemented stages;
- any generated or local-only files deliberately left outside version control.

Keep changes understandable by the human maintainer. Do not add speculative
frameworks, broad refactors, or compatibility layers without a concrete use
case. If the reference behavior and the proposed C++ behavior disagree, stop
and surface the discrepancy instead of silently choosing one.

These rules are adapted from the contributor, coding, and AI-agent guidance
in [whisper.cpp's AGENTS.md](https://github.com/ggerganov/whisper.cpp/blob/master/AGENTS.md)
and [CONTRIBUTING.md](https://github.com/ggerganov/whisper.cpp/blob/master/CONTRIBUTING.md),
with Pixal3D-specific model and GGUF requirements added.
