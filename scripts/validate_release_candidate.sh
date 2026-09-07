#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/validate_release_candidate.sh --candidate-dir DIR [options]

Validate one downloaded CPU release candidate archive.

Options:
  --candidate-dir DIR   Directory containing the candidate artifact (required)
  --expected-tag TAG    Require the package version to match TAG
  --expected-commit SHA Require the package source commit to match SHA
  --require-project-license
                        Require licenses/PROJECT-LICENSE in the archive
  --require-notice      Require licenses/NOTICE in the archive
  -h, --help            Show this help
EOF
}

candidate_dir=""
expected_tag=""
expected_commit=""
require_project_license=false
require_notice=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --candidate-dir)
            [[ $# -ge 2 ]] || { echo "error: --candidate-dir requires a value" >&2; exit 2; }
            candidate_dir=$2
            shift 2
            ;;
        --expected-tag)
            [[ $# -ge 2 ]] || { echo "error: --expected-tag requires a value" >&2; exit 2; }
            expected_tag=$2
            shift 2
            ;;
        --expected-commit)
            [[ $# -ge 2 ]] || { echo "error: --expected-commit requires a value" >&2; exit 2; }
            expected_commit=$2
            shift 2
            ;;
        --require-project-license)
            require_project_license=true
            shift
            ;;
        --require-notice)
            require_notice=true
            shift
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

if [[ -z "$candidate_dir" || ! -d "$candidate_dir" ]]; then
    echo "error: candidate directory does not exist: $candidate_dir" >&2
    exit 2
fi

checksum_file="${candidate_dir%/}/SHA256SUMS"
manifest_file="${candidate_dir%/}/RELEASE-MANIFEST.json"
if [[ ! -f "$checksum_file" ]]; then
    echo "error: candidate is missing SHA256SUMS" >&2
    exit 1
fi
if [[ ! -f "$manifest_file" ]]; then
    echo "error: candidate is missing RELEASE-MANIFEST.json" >&2
    exit 1
fi
(
    cd "$candidate_dir"
    sha256sum -c SHA256SUMS
)

mapfile -t archives < <(find "$candidate_dir" -maxdepth 1 -type f -name 'pixal3d-*.tar.gz' -print | sort)
if [[ "${#archives[@]}" -ne 1 ]]; then
    echo "error: expected exactly one Pixal3D archive, found ${#archives[@]}" >&2
    exit 1
fi
archive=${archives[0]}
archive_name=$(basename "$archive")
sidecar="${archive%.tar.gz}.sha256"
if [[ ! -f "$sidecar" ]]; then
    echo "error: candidate is missing archive checksum: $(basename "$sidecar")" >&2
    exit 1
fi
(
    cd "$candidate_dir"
    sha256sum -c "$(basename "$sidecar")"
)

unpack_dir=$(mktemp -d "${TMPDIR:-/tmp}/pixal3d-release-check.XXXXXX")
trap 'rm -rf "$unpack_dir"' EXIT
tar -xzf "$archive" -C "$unpack_dir"

mapfile -t top_levels < <(find "$unpack_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n')
if [[ "${#top_levels[@]}" -ne 1 ]]; then
    echo "error: archive must contain exactly one top-level directory" >&2
    exit 1
fi
package_root="${unpack_dir}/${top_levels[0]}"
metadata="${package_root}/BUILD-METADATA.txt"
binary="${package_root}/bin/pixal3d"
if [[ ! -f "$metadata" || ! -x "$binary" ]]; then
    echo "error: archive is missing executable bin/pixal3d or BUILD-METADATA.txt" >&2
    exit 1
fi

version=$(sed -n 's/^version=//p' "$metadata")
release_tag=$(sed -n 's/^release_tag=//p' "$metadata")
variant=$(sed -n 's/^variant=//p' "$metadata")
backend=$(sed -n 's/^backend=//p' "$metadata")
source_commit=$(sed -n 's/^source_commit=//p' "$metadata")
submodule_commit=$(sed -n 's/^ggml_submodule_commit=//p' "$metadata")
validation_status=$(sed -n 's/^validation_status=//p' "$metadata")
project_license_present=$(sed -n 's/^project_license_present=//p' "$metadata")
notice_present=$(sed -n 's/^notice_present=//p' "$metadata")
if [[ -z "$version" || -z "$release_tag" || -z "$variant" || -z "$backend" ||
      -z "$source_commit" || -z "$submodule_commit" || -z "$validation_status" ||
      -z "$project_license_present" || -z "$notice_present" ]]; then
    echo "error: incomplete BUILD-METADATA.txt" >&2
    exit 1
fi
if [[ ! "$source_commit" =~ ^[0-9a-f]{40}$ || ! "$submodule_commit" =~ ^[0-9a-f]{40}$ ]]; then
    echo "error: invalid source or ggml submodule commit metadata" >&2
    exit 1
fi
if [[ "$variant" != "cpu" || "$backend" != "cpu" ]]; then
    echo "error: formal release candidates must use cpu/cpu, got ${variant}/${backend}" >&2
    exit 1
fi
if [[ -n "$expected_commit" && "$source_commit" != "$expected_commit" ]]; then
    echo "error: candidate source commit is $source_commit, expected $expected_commit" >&2
    exit 1
fi
if [[ "$release_tag" != "v${version}" ]]; then
    echo "error: metadata release tag $release_tag disagrees with version $version" >&2
    exit 1
fi
if [[ -n "$expected_tag" && "$expected_tag" != "$release_tag" ]]; then
    echo "error: candidate requires tag $release_tag, got $expected_tag" >&2
    exit 1
fi
if [[ ! "$archive_name" =~ ^pixal3d-${version}-linux-[A-Za-z0-9._-]+-cpu\.tar\.gz$ ]]; then
    echo "error: archive name does not match version/variant: $archive_name" >&2
    exit 1
fi

if [[ "$require_project_license" == true && ! -f "${package_root}/licenses/PROJECT-LICENSE" ]]; then
    echo "error: formal release requires licenses/PROJECT-LICENSE" >&2
    exit 1
fi
if [[ "$require_project_license" == true && "$project_license_present" != true ]]; then
    echo "error: formal release metadata does not confirm project license" >&2
    exit 1
fi
if [[ "$require_notice" == true && ! -f "${package_root}/licenses/NOTICE" ]]; then
    echo "error: formal release requires licenses/NOTICE" >&2
    exit 1
fi
if [[ "$require_notice" == true && "$notice_present" != true ]]; then
    echo "error: formal release metadata does not confirm NOTICE" >&2
    exit 1
fi

version_line=$("$binary" --version 2>/dev/null)
cli_version=${version_line##* }
if [[ "$cli_version" != "$version" ]]; then
    echo "error: archive metadata version $version disagrees with CLI version $cli_version" >&2
    exit 1
fi

python3 - "$manifest_file" "$archive" "$archive_name" "$version" "$release_tag" "$source_commit" "$submodule_commit" "$variant" "$backend" "$validation_status" "$project_license_present" "$notice_present" <<'PY'
import hashlib
import json
import os
import sys

(
    manifest_path,
    archive_path,
    archive_name,
    version,
    release_tag,
    source_commit,
    submodule_commit,
    variant,
    backend,
    validation_status,
    project_license_present,
    notice_present,
) = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
if manifest.get("version") != version:
    raise SystemExit("error: manifest version disagrees with package metadata")
if manifest.get("release_tag") != release_tag:
    raise SystemExit("error: manifest release tag disagrees with package metadata")
if manifest.get("source_commit") != source_commit:
    raise SystemExit("error: manifest source commit disagrees with package metadata")
if manifest.get("ggml_submodule_commit") != submodule_commit:
    raise SystemExit("error: manifest ggml commit disagrees with package metadata")
if manifest.get("variant") != variant or manifest.get("backend") != backend:
    raise SystemExit("error: manifest backend/variant disagrees with package metadata")
if manifest.get("validation_status") != validation_status:
    raise SystemExit("error: manifest validation status disagrees with package metadata")
if manifest.get("project_license", {}).get("present") != (project_license_present == "true"):
    raise SystemExit("error: manifest project license status disagrees with package metadata")
if manifest.get("notice", {}).get("present") != (notice_present == "true"):
    raise SystemExit("error: manifest NOTICE status disagrees with package metadata")
artifact = manifest.get("artifact", {})
actual_size = os.path.getsize(archive_path)
actual_sha = hashlib.sha256(open(archive_path, "rb").read()).hexdigest()
if artifact.get("name") != archive_name:
    raise SystemExit("error: manifest archive name disagrees with candidate")
if artifact.get("size_bytes") != actual_size:
    raise SystemExit("error: manifest archive size disagrees with candidate")
if artifact.get("sha256") != actual_sha:
    raise SystemExit("error: manifest archive SHA-256 disagrees with candidate")
if manifest.get("models_included") is not False:
    raise SystemExit("error: release archive must not include models")
PY

printf 'candidate archive: %s\n' "$archive_name"
printf 'version: %s\n' "$version"
printf 'tag: %s\n' "$release_tag"
printf 'backend/variant: %s/%s\n' "$backend" "$variant"
printf 'source commit: %s\n' "$source_commit"
printf 'ggml submodule: %s\n' "$submodule_commit"
printf 'validation status: %s\n' "$validation_status"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    printf 'version=%s\n' "$version" >> "$GITHUB_OUTPUT"
    printf 'tag_name=%s\n' "$release_tag" >> "$GITHUB_OUTPUT"
    printf 'source_commit=%s\n' "$source_commit" >> "$GITHUB_OUTPUT"
    printf 'submodule_commit=%s\n' "$submodule_commit" >> "$GITHUB_OUTPUT"
    printf 'archive_name=%s\n' "$archive_name" >> "$GITHUB_OUTPUT"
fi
