#!/usr/bin/env bash
# Download the pinned official valeoai/NAF release asset and verify it.

set -eu
set -o pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/weights/NAF"
TARGET="${MODEL_DIR}/naf_release.pth"
EXPECTED_SIZE=2664431
EXPECTED_SHA256="c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f"

if ! command -v gh >/dev/null 2>&1; then
    echo "error: GitHub CLI (gh) is required" >&2
    exit 2
fi
if ! command -v sha256sum >/dev/null 2>&1; then
    echo "error: sha256sum is required" >&2
    exit 2
fi

mkdir -p "${MODEL_DIR}"
verify() {
    [ -f "$1" ] || return 1
    [ "$(stat -c '%s' "$1")" -eq "${EXPECTED_SIZE}" ] || return 1
    [ "$(sha256sum "$1" | awk '{print $1}')" = "${EXPECTED_SHA256}" ]
}

if verify "${TARGET}"; then
    echo "SKIP ${TARGET} (verified)"
    exit 0
fi
if [ -f "${TARGET}" ]; then
    echo "error: existing ${TARGET} failed size/hash verification; refusing to overwrite" >&2
    exit 1
fi

TEMP_DIR="$(mktemp -d "${MODEL_DIR}/.naf-download.XXXXXX")"
cleanup() { rmdir "${TEMP_DIR}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
echo "Downloading valeoai/NAF model release to ${MODEL_DIR}..."
gh release download model --repo valeoai/NAF --pattern naf_release.pth \
    --dir "${TEMP_DIR}"
DOWNLOADED="${TEMP_DIR}/naf_release.pth"
if ! verify "${DOWNLOADED}"; then
    echo "error: downloaded NAF asset failed size/hash verification" >&2
    exit 1
fi
mv "${DOWNLOADED}" "${TARGET}"
echo "DONE ${TARGET} (${EXPECTED_SIZE} bytes, SHA-256 ${EXPECTED_SHA256})"
