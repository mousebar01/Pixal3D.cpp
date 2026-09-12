#include "pixal3d/profile.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace pixal3d {
namespace {

bool equals_ascii(const char * value, const char * expected) noexcept {
    if (!value || !expected) return value == expected;
    while (*value && *expected) {
        const char value_char = (*value >= 'A' && *value <= 'Z')
            ? static_cast<char>(*value - 'A' + 'a') : *value;
        if (value_char != *expected) return false;
        ++value;
        ++expected;
    }
    return *value == '\0' && *expected == '\0';
}

bool legacy_enabled(const char * name) noexcept {
    const char * value = name ? std::getenv(name) : nullptr;
    return value && *value && std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "off") != 0 && std::strcmp(value, "false") != 0;
}

} // namespace

ProfileMode profile_mode_from_environment(const char * legacy_profile,
                                          const char * legacy_trace) noexcept {
    const char * value = std::getenv("PIXAL3D_PROFILE");
    if (value && *value) {
        if (equals_ascii(value, "off")) return ProfileMode::off;
        if (equals_ascii(value, "summary")) return ProfileMode::summary;
        if (equals_ascii(value, "trace")) return ProfileMode::trace;
        std::fprintf(stderr,
                     "pixal3d: invalid PIXAL3D_PROFILE='%s'; expected off|summary|trace; profiling disabled\n",
                     value);
        return ProfileMode::off;
    }
    return legacy_enabled(legacy_profile) || legacy_enabled(legacy_trace)
        ? ProfileMode::trace : ProfileMode::off;
}

double profile_time_now_ms() noexcept {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
        clock::now().time_since_epoch()).count();
}

ProfileTimer::ProfileTimer(bool enabled, Clock clock) noexcept
    : enabled_(enabled && clock != nullptr),
      clock_(clock != nullptr ? clock : profile_time_now_ms),
      start_(enabled_ ? clock_() : 0.0) {}

double ProfileTimer::elapsed_ms() const noexcept {
    return enabled_ ? clock_() - start_ : 0.0;
}

ProfileScope::ProfileScope(ProfileMode mode, const char * stage) noexcept
    : mode_(mode), stage_(stage && *stage ? stage : "unknown"),
      timer_(mode != ProfileMode::off) {}

ProfileScope::~ProfileScope() {
    if (!timer_.enabled()) return;
    std::fprintf(stderr, "pixal3d: profile stage=%s wall_ms=%.6f\n",
                 stage_, timer_.elapsed_ms());
}

void ProfileScope::phase(const char * name) noexcept {
    if (mode_ != ProfileMode::trace) return;
    const double elapsed = timer_.elapsed_ms();
    std::fprintf(stderr,
                 "pixal3d: profile stage=%s phase=%s exclusive_ms=%.6f cumulative_ms=%.6f\n",
                 stage_, name && *name ? name : "unknown",
                 elapsed - previous_ms_, elapsed);
    previous_ms_ = elapsed;
}

} // namespace pixal3d
