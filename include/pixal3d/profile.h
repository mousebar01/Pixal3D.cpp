#pragma once

namespace pixal3d {

enum class ProfileMode { off, summary, trace };

// Snapshot at a model/stage boundary, not inside an element/operation loop.
// PIXAL3D_PROFILE=off|summary|trace takes precedence over legacy flags.
// With no global setting, an enabled legacy flag selects trace (preserving
// its detailed diagnostics). Invalid global values warn and disable profiling.
ProfileMode profile_mode_from_environment(const char * legacy_profile = nullptr,
                                          const char * legacy_trace = nullptr) noexcept;

double profile_time_now_ms() noexcept;

// Disabled timers never call the clock. Clock injection permits deterministic
// tests of the inactive path without performance thresholds or global hooks.
class ProfileTimer {
public:
    using Clock = double (*)() noexcept;
    explicit ProfileTimer(bool enabled, Clock clock = profile_time_now_ms) noexcept;
    bool enabled() const noexcept { return enabled_; }
    double elapsed_ms() const noexcept;

private:
    bool enabled_;
    Clock clock_;
    double start_;
};

// Host wall-clock diagnostics only: never synchronize a GPU or read tensors.
// stage must outlive this scope. Summary prints once, on exit (including early
// returns); trace additionally permits phase records. No output is a success gate.
class ProfileScope {
public:
    ProfileScope(ProfileMode mode, const char * stage) noexcept;
    ~ProfileScope();
    ProfileMode mode() const noexcept { return mode_; }
    bool enabled() const noexcept { return mode_ != ProfileMode::off; }
    bool tracing() const noexcept { return mode_ == ProfileMode::trace; }
    ProfileScope(const ProfileScope &) = delete;
    ProfileScope & operator=(const ProfileScope &) = delete;
    void phase(const char * name) noexcept;

private:
    ProfileMode mode_;
    const char * stage_;
    ProfileTimer timer_;
    double previous_ms_ = 0.0;
};

} // namespace pixal3d
