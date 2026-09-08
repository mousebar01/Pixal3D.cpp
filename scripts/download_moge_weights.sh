#!/usr/bin/env bash
# Download the pinned official Ruicheng/moge-2-vitl-normal-onnx asset and
# verify it.  The model backs `pixal3d run-image --camera auto` (MoGe-2
# per-image camera estimation).

set -eu
set -o pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/weights/MoGe"
TARGET="${MODEL_DIR}/moge-2-vitl-normal.onnx"
EXPECTED_SIZE=1324265014
EXPECTED_SHA256="afbc4ccc3450298f3afb35b90f015f4c4f552dea21dc6470d5f7b78b77e2d751"
DOWNLOAD_URL="https://huggingface.co/Ruicheng/moge-2-vitl-normal-onnx/resolve/main/model.onnx"

if ! command -v curl >/dev/null 2>&1; then
    echo "error: curl is required" >&2
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
    echo "       remove the file and rerun this script" >&2
    exit 1
fi

PART="${TARGET}.part"
echo "Downloading Ruicheng/moge-2-vitl-normal-onnx to ${MODEL_DIR}..."
# Existing *.part files are resumed; the finished part is verified before the
# atomic move so an interrupted download never looks like a usable model.
curl -L --fail --continue-at - --output "${PART}" "${DOWNLOAD_URL}"
if ! verify "${PART}"; then
    echo "error: downloaded MoGe asset failed size/hash verification" >&2
    echo "       keeping ${PART} for inspection; remove it to retry from scratch" >&2
    exit 1
fi
mv "${PART}" "${TARGET}"
echo "DONE ${TARGET} (${EXPECTED_SIZE} bytes, SHA-256 ${EXPECTED_SHA256})"
