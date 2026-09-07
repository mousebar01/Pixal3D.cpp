#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/package_release.sh --build-dir DIR [options]

Create a minimal Pixal3D binary archive without models or build artifacts.

Options:
  --build-dir DIR       CMake build directory containing bin/pixal3d (required)
  --binary PATH         Override the CLI binary path
  --output-dir DIR      Output directory (default: dist)
  --variant NAME        Package suffix (default: cpu)
  --backend NAME        Backend name (default: derived from variant)
  --source-commit SHA   Record an explicit source commit in metadata
  --release-tag TAG     Record the release tag (default: v<version>)
  --validation-status S Record the validation status
  --workflow-run-id ID  Record the GitHub Actions workflow run ID
  --submodule-commit SHA
                        Record the ggml submodule commit
  --build-environment S Record the build environment description
  -h, --help            Show this help
EOF
}

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=""
binary_path=""
output_dir="${root_dir}/dist"
variant="cpu"
backend="${PIXAL3D_BACKEND:-}"
source_commit="${PIXAL3D_SOURCE_COMMIT:-}"
release_tag="${PIXAL3D_RELEASE_TAG:-}"
validation_status="${PIXAL3D_VALIDATION_STATUS:-unverified}"
workflow_run_id="${PIXAL3D_WORKFLOW_RUN_ID:-}"
submodule_commit="${PIXAL3D_SUBMODULE_COMMIT:-}"
build_environment="${PIXAL3D_BUILD_ENVIRONMENT:-unknown}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            [[ $# -ge 2 ]] || { echo "error: --build-dir requires a value" >&2; exit 2; }
            build_dir=$2
            shift 2
            ;;
        --binary)
            [[ $# -ge 2 ]] || { echo "error: --binary requires a value" >&2; exit 2; }
            binary_path=$2
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || { echo "error: --output-dir requires a value" >&2; exit 2; }
            output_dir=$2
            shift 2
            ;;
        --variant)
            [[ $# -ge 2 ]] || { echo "error: --variant requires a value" >&2; exit 2; }
            variant=$2
            shift 2
            ;;
        --backend)
            [[ $# -ge 2 ]] || { echo "error: --backend requires a value" >&2; exit 2; }
            backend=$2
            shift 2
            ;;
        --source-commit)
            [[ $# -ge 2 ]] || { echo "error: --source-commit requires a value" >&2; exit 2; }
            source_commit=$2
            shift 2
            ;;
        --release-tag)
            [[ $# -ge 2 ]] || { echo "error: --release-tag requires a value" >&2; exit 2; }
            release_tag=$2
            shift 2
            ;;
        --validation-status)
            [[ $# -ge 2 ]] || { echo "error: --validation-status requires a value" >&2; exit 2; }
            validation_status=$2
            shift 2
            ;;
        --workflow-run-id)
            [[ $# -ge 2 ]] || { echo "error: --workflow-run-id requires a value" >&2; exit 2; }
            workflow_run_id=$2
            shift 2
            ;;
        --submodule-commit)
            [[ $# -ge 2 ]] || { echo "error: --submodule-commit requires a value" >&2; exit 2; }
            submodule_commit=$2
            shift 2
            ;;
        --build-environment)
            [[ $# -ge 2 ]] || { echo "error: --build-environment requires a value" >&2; exit 2; }
            build_environment=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$build_dir" ]]; then
    echo "error: --build-dir is required" >&2
    usage >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to write RELEASE-MANIFEST.json" >&2
    exit 1
fi
if [[ ! "$variant" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "error: invalid package variant: $variant" >&2
    exit 2
fi

if [[ -z "$binary_path" ]]; then
    binary_path="${build_dir%/}/bin/pixal3d"
fi

if [[ ! -f "$binary_path" ]]; then
    echo "error: Pixal3D CLI was not found at $binary_path" >&2
    exit 1
fi
if [[ ! -x "$binary_path" ]]; then
    echo "error: Pixal3D CLI is not executable: $binary_path" >&2
    exit 1
fi

version_line=$("$binary_path" --version 2>/dev/null | head -n 1 || true)
if [[ -z "$version_line" ]]; then
    echo "error: failed to run '$binary_path --version'" >&2
    exit 1
fi
version=${version_line##* }
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?$ ]]; then
    echo "error: could not parse Pixal3D version from: $version_line" >&2
    exit 1
fi

case "$(uname -m)" in
    x86_64|amd64) architecture=x86_64 ;;
    aarch64|arm64) architecture=arm64 ;;
    *) architecture=$(uname -m) ;;
esac

if [[ -z "$backend" ]]; then
    if [[ "$variant" == cuda-* ]]; then
        backend=cuda
    else
        backend=cpu
    fi
fi
release_tag=${release_tag:-v${version}}
if [[ ! "$release_tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?$ ]]; then
    echo "error: invalid release tag: $release_tag" >&2
    exit 2
fi

platform_os=linux
libc=unknown
if command -v ldd >/dev/null 2>&1; then
    ldd_version=$(ldd --version 2>&1 | head -n 1 || true)
    if [[ "$ldd_version" == *GLIBC* ]]; then
        libc=glibc
    fi
fi

if [[ -z "$source_commit" ]] && git -C "$root_dir" rev-parse --verify HEAD >/dev/null 2>&1; then
    source_commit=$(git -C "$root_dir" rev-parse HEAD)
fi
source_commit=${source_commit:-unknown}
if [[ -z "$submodule_commit" ]] && git -C "$root_dir/ggml" rev-parse --verify HEAD >/dev/null 2>&1; then
    submodule_commit=$(git -C "$root_dir/ggml" rev-parse HEAD)
fi
submodule_commit=${submodule_commit:-unknown}

output_dir=$(mkdir -p "$output_dir" && cd "$output_dir" && pwd)
package_name="pixal3d-${version}-${platform_os}-${architecture}-${variant}"
archive_name="${package_name}.tar.gz"
archive_path="${output_dir}/${archive_name}"
checksum_name="${package_name}.sha256"
checksum_path="${output_dir}/${checksum_name}"
all_checksums_path="${output_dir}/SHA256SUMS"
manifest_path="${output_dir}/RELEASE-MANIFEST.json"

staging_root=$(mktemp -d "${TMPDIR:-/tmp}/pixal3d-package.XXXXXX")
trap 'rm -rf "$staging_root"' EXIT
package_root="${staging_root}/${package_name}"
mkdir -p "$package_root/bin" "$package_root/licenses"

install -m 0755 "$binary_path" "$package_root/bin/pixal3d"
install -m 0644 "${root_dir}/README.md" "$package_root/README.md"

cat > "$package_root/RUNTIME-DEPENDENCIES.txt" <<EOF
Pixal3D ${version} (${variant}) runtime notes

This is a CLI-only archive. Model weights, GGUF files, condition files, and
generated meshes are intentionally not included. The CMake SDK library and headers
are also not included in this binary archive.

Expected runtime library families for the native Linux build:
  libc, libstdc++, libgcc_s, libgomp, libpng16, libjpeg, and zlib

The names above describe runtime requirements, not a promise of one specific
Linux distribution ABI. The CUDA compile-only variant additionally requires a
compatible NVIDIA driver, CUDA runtime, cuBLAS/cuBLASLt, and C++ runtime.
EOF

if command -v ldd >/dev/null 2>&1; then
    {
        printf '\nBuild-host detected SONAMEs (informational; paths omitted):\n'
        ldd "$binary_path" 2>/dev/null | awk '
            /=>/ { name = $1; gsub(".*/", "", name); print name; next }
            $1 ~ /\.so/ { name = $1; gsub(".*/", "", name); print name }
        ' | sort -u
    } >> "$package_root/RUNTIME-DEPENDENCIES.txt"
fi

project_license_present=false
project_license_status=missing_root_license
if [[ -f "${root_dir}/LICENSE" ]]; then
    install -m 0644 "${root_dir}/LICENSE" "$package_root/licenses/PROJECT-LICENSE"
    project_license_present=true
    project_license_status=present
else
    cat > "$package_root/licenses/PROJECT-LICENSE-STATUS.txt" <<'EOF'
The outer Pixal3D.cpp repository does not contain a root-level LICENSE file.
This archive does not assign a license to the outer project.
EOF
    echo "warning: no root LICENSE found; archive contains a license status notice" >&2
fi

notice_present=false
notice_status=missing_notice
if [[ -f "${root_dir}/NOTICE" ]]; then
    install -m 0644 "${root_dir}/NOTICE" "$package_root/licenses/NOTICE"
    notice_present=true
    notice_status=present
else
    echo "warning: no root NOTICE found; archive has no third-party notice summary" >&2
fi

if [[ -f "${root_dir}/ggml/LICENSE" ]]; then
    install -m 0644 "${root_dir}/ggml/LICENSE" "$package_root/licenses/ggml-LICENSE"
fi

cat > "$package_root/BUILD-METADATA.txt" <<EOF
project=Pixal3D
version=${version}
release_tag=${release_tag}
variant=${variant}
backend=${backend}
platform=${platform_os}-${architecture}
libc=${libc}
source_commit=${source_commit}
ggml_submodule_commit=${submodule_commit}
workflow_run_id=${workflow_run_id:-unknown}
validation_status=${validation_status}
build_environment=${build_environment}
project_license_present=${project_license_present}
notice_present=${notice_present}
EOF

rm -f "$archive_path" "$checksum_path"
(
    cd "$staging_root"
    tar --sort=name --owner=0 --group=0 --numeric-owner \
        --mtime='UTC 1970-01-01' -cf - "$package_name" | gzip -n > "$archive_path"
)

# Regenerate sidecars and the aggregate from the archives that actually exist.
# This removes stale checksum entries when an old archive was deleted.
shopt -s nullglob
for stale_checksum in "$output_dir"/pixal3d-*.sha256; do
    stale_archive="${stale_checksum%.sha256}.tar.gz"
    if [[ ! -f "$stale_archive" ]]; then
        rm -f "$stale_checksum"
    fi
done
archives=("$output_dir"/pixal3d-*.tar.gz)
if [[ "${#archives[@]}" -eq 0 ]]; then
    echo "error: no release archives found in $output_dir" >&2
    exit 1
fi
: > "$all_checksums_path"
for existing_archive in "${archives[@]}"; do
    existing_name=$(basename "$existing_archive")
    existing_checksum=$(sha256sum "$existing_archive" | awk '{print $1}')
    printf '%s  %s\n' "$existing_checksum" "$existing_name" \
        > "${existing_archive%.tar.gz}.sha256"
    printf '%s  %s\n' "$existing_checksum" "$existing_name" >> "$all_checksums_path"
done
LC_ALL=C sort -o "$all_checksums_path" "$all_checksums_path"

archive_size=$(stat -c '%s' "$archive_path")
archive_sha256=$(sha256sum "$archive_path" | awk '{print $1}')
PROJECT_LICENSE_PRESENT="$project_license_present" \
PROJECT_LICENSE_STATUS="$project_license_status" \
NOTICE_PRESENT="$notice_present" \
NOTICE_STATUS="$notice_status" \
PROJECT_VERSION="$version" \
RELEASE_TAG="$release_tag" \
SOURCE_COMMIT="$source_commit" \
GGML_SUBMODULE_COMMIT="$submodule_commit" \
WORKFLOW_RUN_ID="${workflow_run_id:-}" \
PLATFORM_OS="$platform_os" \
ARCHITECTURE="$architecture" \
LIBC="$libc" \
BACKEND="$backend" \
VARIANT="$variant" \
VALIDATION_STATUS="$validation_status" \
BUILD_ENVIRONMENT="$build_environment" \
ARCHIVE_NAME="$archive_name" \
ARCHIVE_SIZE="$archive_size" \
ARCHIVE_SHA256="$archive_sha256" \
python3 - "$manifest_path" <<'PY'
import json
import os
import sys

manifest = {
    "project": "Pixal3D",
    "version": os.environ["PROJECT_VERSION"],
    "release_tag": os.environ["RELEASE_TAG"],
    "source_commit": os.environ["SOURCE_COMMIT"],
    "ggml_submodule_commit": os.environ["GGML_SUBMODULE_COMMIT"],
    "workflow_run_id": os.environ["WORKFLOW_RUN_ID"] or None,
    "platform": {
        "os": os.environ["PLATFORM_OS"],
        "architecture": os.environ["ARCHITECTURE"],
        "libc": os.environ["LIBC"],
    },
    "backend": os.environ["BACKEND"],
    "variant": os.environ["VARIANT"],
    "validation_status": os.environ["VALIDATION_STATUS"],
    "build_environment": os.environ["BUILD_ENVIRONMENT"],
    "package_type": "cli-only",
    "models_included": False,
    "project_license": {
        "present": os.environ["PROJECT_LICENSE_PRESENT"] == "true",
        "status": os.environ["PROJECT_LICENSE_STATUS"],
    },
    "notice": {
        "present": os.environ["NOTICE_PRESENT"] == "true",
        "status": os.environ["NOTICE_STATUS"],
    },
    "artifact": {
        "name": os.environ["ARCHIVE_NAME"],
        "size_bytes": int(os.environ["ARCHIVE_SIZE"]),
        "sha256": os.environ["ARCHIVE_SHA256"],
    },
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

printf 'created %s\n' "$archive_path"
printf 'created %s\n' "$checksum_path"
printf 'updated %s\n' "$all_checksums_path"
printf 'created %s\n' "$manifest_path"
