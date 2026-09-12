# Profiler opt-in contract

- **Updated:** 2026-09-12
- **Status:** implemented in the outer project

## Runtime behavior

`PIXAL3D_PROFILE` accepts:

- `off` — default; no optional timing collection or timing output
- `summary` — stage-level summaries
- `trace` — stage and phase details

Legacy stage-specific profiling variables remain compatible. An explicit
`PIXAL3D_PROFILE` value takes precedence over legacy flags; an invalid value
warns and disables profiling.

Normal error, fallback, device-policy and required capability messages are not
profiler output and remain available when profiling is off.

## Performance guarantee

When disabled, `ProfileTimer` does not call the clock. The implementation does
not add profiler-only GPU synchronization, activation readback or tensor
inspection. Optional timing is gated at collection call sites rather than only
suppressing formatted output afterward.

Profiler output is host wall-clock information. It is not a GPU utilization
measurement and does not certify placement or parity.

## Usage

```sh
PIXAL3D_PROFILE=off     ./build/bin/pixal3d run-image ...
PIXAL3D_PROFILE=summary ./build/bin/pixal3d run-image ...
PIXAL3D_PROFILE=trace   ./build/bin/pixal3d run-image ...
```

Use `summary` to locate a stage-level long tail and `trace` only for a focused
investigation. Production timing should use `off` unless the overhead has been
explicitly accepted.

## Validation on 2026-09-12

```sh
cmake --build build-backend-test --parallel 4
(cd build-backend-test && ctest --output-on-failure)
./build-backend-test/bin/pixal3d_profile_test
./build-backend-test/bin/pixal3d_profile_test --mesh
git diff --check
```

Result: CPU build succeeded, 24/24 CTest tests passed, profile tests passed and
whitespace validation passed. This does not measure end-to-end runtime overhead
or certify GPU-specific profiling behavior.
