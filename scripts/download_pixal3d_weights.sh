#!/usr/bin/env bash
# Download the pinned Pixal3D Hugging Face snapshot with resumable transfers.
# Existing *.part files are resumed; completed files are size/hash checked.

set -u
set -o pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/weights/Pixal3D"
METADATA="${MODEL_DIR}/.download-metadata.json"
REVISION="b0cb2e1b794cab9aa0ac38a95d794a4d9337437f"
REPO="TencentARC/Pixal3D"
WORKERS="${PIXAL3D_DOWNLOAD_WORKERS:-8}"
CHUNK_BYTES="${PIXAL3D_DOWNLOAD_CHUNK_BYTES:-67108864}"
CURL_MAX_TIME="${PIXAL3D_CURL_MAX_TIME:-300}"

# Some workstations expose different uppercase/lowercase proxy variables.
# curl gives the lowercase names precedence, so make the selected proxy
# explicit.  PIXAL3D_PROXY can override the environment for this downloader.
if [ -n "${PIXAL3D_PROXY:-}" ]; then
    HTTPS_PROXY="${PIXAL3D_PROXY}"
    HTTP_PROXY="${PIXAL3D_PROXY}"
    ALL_PROXY="${PIXAL3D_PROXY}"
    export HTTPS_PROXY HTTP_PROXY ALL_PROXY
fi
if [ -n "${HTTPS_PROXY:-}" ]; then
    https_proxy="${HTTPS_PROXY}"
    export https_proxy
fi
if [ -n "${HTTP_PROXY:-}" ]; then
    http_proxy="${HTTP_PROXY}"
    export http_proxy
fi
if [ -n "${ALL_PROXY:-}" ]; then
    all_proxy="${ALL_PROXY}"
    export all_proxy
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "error: curl is required" >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required" >&2
    exit 2
fi
if ! command -v sha256sum >/dev/null 2>&1; then
    echo "error: sha256sum is required" >&2
    exit 2
fi
case "${WORKERS}" in
    ''|*[!0-9]*) echo "error: PIXAL3D_DOWNLOAD_WORKERS must be a positive integer" >&2; exit 2 ;;
esac
if [ "${WORKERS}" -lt 1 ]; then
    echo "error: PIXAL3D_DOWNLOAD_WORKERS must be at least 1" >&2
    exit 2
fi
case "${CHUNK_BYTES}" in
    ''|*[!0-9]*) echo "error: PIXAL3D_DOWNLOAD_CHUNK_BYTES must be a positive integer" >&2; exit 2 ;;
esac
if [ "${CHUNK_BYTES}" -lt 1 ]; then
    echo "error: PIXAL3D_DOWNLOAD_CHUNK_BYTES must be at least 1" >&2
    exit 2
fi

mkdir -p "${MODEL_DIR}"

valid_metadata() {
    python3 - "${METADATA}" <<'PY'
import json
import sys
try:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        data = json.load(handle)
    assert isinstance(data, list)
    assert any(item.get("type") == "file" for item in data)
except Exception:
    raise SystemExit(1)
PY
}

fetch_metadata() {
    if [ -s "${METADATA}" ] && valid_metadata; then
        return 0
    fi

    local tmp="${METADATA}.tmp.$$"
    local base url
    for base in \
        "https://huggingface.co" \
        "https://hf-mirror.com"; do
        url="${base}/api/models/${REPO}/tree/${REVISION}?recursive=true&expand=true"
        echo "Fetching model manifest from ${base}..."
        if curl --http1.1 -fsSL --retry 20 --retry-delay 5 \
            --retry-connrefused --connect-timeout 60 "${url}" -o "${tmp}" \
            && valid_metadata_file="${tmp}" python3 - <<'PY'
import json
import os
path = os.environ["valid_metadata_file"]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
if not isinstance(data, list) or not any(x.get("type") == "file" for x in data):
    raise SystemExit("manifest is not a file tree")
PY
        then
            mv -f "${tmp}" "${METADATA}"
            return 0
        fi
        rm -f "${tmp}"
    done

    echo "error: unable to fetch a valid model manifest" >&2
    return 1
}

fetch_metadata || exit 1

MANIFEST="$(mktemp "${TMPDIR:-/tmp}/pixal3d-manifest.XXXXXX")"
cleanup() { rm -f "${MANIFEST}"; }
trap cleanup EXIT INT TERM

# Keep the largest files first.  The worker modulo assignment gives eight
# independent curl streams while still allowing a rerun to resume every file.
python3 - "${METADATA}" "${MANIFEST}" <<'PY'
import json
import sys

metadata_path, manifest_path = sys.argv[1:]
with open(metadata_path, "r", encoding="utf-8") as handle:
    entries = json.load(handle)
files = [entry for entry in entries if entry.get("type") == "file"]
files.sort(key=lambda entry: (-int(entry.get("size", 0)), entry["path"]))
with open(manifest_path, "w", encoding="utf-8") as handle:
    for index, entry in enumerate(files):
        lfs = entry.get("lfs") or {}
        # Use a visible placeholder: Bash treats tab as IFS whitespace and
        # would otherwise collapse the empty SHA field, shifting the path.
        sha256 = lfs.get("oid") or "-"
        handle.write("{}\t{}\t{}\t{}\n".format(
            index, int(entry.get("size", 0)), sha256, entry["path"]))
PY

total_bytes="$(awk -F '\t' '{sum += $2} END {print sum + 0}' "${MANIFEST}")"
total_gib="$(awk -v n="${total_bytes}" 'BEGIN {printf "%.2f", n/1024/1024/1024}')"
echo "Pixal3D ${REVISION}: ${total_gib} GiB across $(wc -l < "${MANIFEST}") files"
echo "Destination: ${MODEL_DIR}"
echo "Workers: ${WORKERS} (set PIXAL3D_DOWNLOAD_WORKERS to change)"
echo "Range chunk size: ${CHUNK_BYTES} bytes (set PIXAL3D_DOWNLOAD_CHUNK_BYTES to change)"

download_one() {
    local expected_size="$1"
    local expected_sha="$2"
    local relpath="$3"
    local target="${MODEL_DIR}/${relpath}"
    local part="${target}.part"
    local actual_size base url start end chunk_size chunk_path chunk_ok

    if [ "${expected_sha}" = "-" ]; then
        expected_sha=""
    fi

    mkdir -p "$(dirname -- "${target}")"

    if [ -f "${target}" ]; then
        actual_size="$(stat -c '%s' "${target}")"
        if [ "${actual_size}" != "${expected_size}" ]; then
            echo "FAIL ${relpath}: completed file has ${actual_size} bytes, expected ${expected_size}" >&2
            return 1
        fi
        if [ -n "${expected_sha}" ]; then
            if [ "$(sha256sum "${target}" | awk '{print $1}')" != "${expected_sha}" ]; then
                echo "FAIL ${relpath}: SHA-256 mismatch" >&2
                return 1
            fi
        fi
        echo "SKIP ${relpath} (verified)"
        return 0
    fi

    if [ -f "${part}" ]; then
        actual_size="$(stat -c '%s' "${part}")"
        if [ "${actual_size}" -gt "${expected_size}" ]; then
            echo "FAIL ${relpath}: .part is larger than expected (${actual_size} > ${expected_size})" >&2
            return 1
        fi
        if [ "${actual_size}" -eq "${expected_size}" ] && [ -n "${expected_sha}" ]; then
            echo "RESUME ${relpath}: checking complete-size partial"
            if [ "$(sha256sum "${part}" | awk '{print $1}')" != "${expected_sha}" ]; then
                echo "  ${relpath}: complete-size partial is corrupt; restarting that file" >&2
                truncate -s 0 "${part}"
                actual_size=0
            fi
        fi
        echo "RESUME ${relpath} at ${actual_size}/${expected_size} bytes"
    else
        echo "GET ${relpath} (${expected_size} bytes)"
    fi

    start="$(stat -c '%s' "${part}" 2>/dev/null || echo 0)"
    while [ "${start}" -lt "${expected_size}" ]; do
        end=$((start + CHUNK_BYTES - 1))
        if [ "${end}" -ge "${expected_size}" ]; then
            end=$((expected_size - 1))
        fi
        chunk_size=$((end - start + 1))
        chunk_path="${part}.chunk.$$"
        chunk_ok=0

        for base in "https://huggingface.co" "https://hf-mirror.com"; do
            url="${base}/${REPO}/resolve/${REVISION}/${relpath}?download=true"
            rm -f "${chunk_path}"
            echo "  ${relpath}: range ${start}-${end} via ${base}"
            if curl --http1.1 -fL --silent --show-error \
                --retry 5 --retry-delay 3 --retry-connrefused \
                --connect-timeout 30 --max-time "${CURL_MAX_TIME}" \
                -r "${start}-${end}" -o "${chunk_path}" "${url}"; then
                actual_size="$(stat -c '%s' "${chunk_path}" 2>/dev/null || echo 0)"
                if [ "${actual_size}" -eq "${chunk_size}" ]; then
                    if cat "${chunk_path}" >> "${part}"; then
                        rm -f "${chunk_path}"
                        start=$((start + chunk_size))
                        chunk_ok=1
                        break
                    fi
                    echo "  ${relpath}: could not append range (disk/write error)" >&2
                    truncate -s "${start}" "${part}"
                fi
                echo "  ${relpath}: range returned ${actual_size}/${chunk_size} bytes" >&2
            else
                echo "  ${relpath}: range transfer failed; trying fallback" >&2
            fi
            rm -f "${chunk_path}"
        done

        if [ "${chunk_ok}" -ne 1 ]; then
            echo "FAIL ${relpath}: range ${start}-${end} failed; leaving ${part} for resume" >&2
            return 1
        fi
    done

    if [ -n "${expected_sha}" ]; then
        echo "  ${relpath}: verifying SHA-256"
        if [ "$(sha256sum "${part}" | awk '{print $1}')" != "${expected_sha}" ]; then
            echo "FAIL ${relpath}: SHA-256 mismatch; leaving .part for manual inspection" >&2
            return 1
        fi
    fi
    mv -f "${part}" "${target}"
    echo "DONE ${relpath}"
    return 0
}

worker() {
    local worker_id="$1"
    local index expected_size expected_sha relpath
    local failed=0
    while IFS=$'\t' read -r index expected_size expected_sha relpath; do
        [ "$((index % WORKERS))" -eq "${worker_id}" ] || continue
        if ! download_one "${expected_size}" "${expected_sha}" "${relpath}"; then
            failed=1
        fi
    done < "${MANIFEST}"
    return "${failed}"
}

pids=()
for ((worker_id = 0; worker_id < WORKERS; worker_id++)); do
    worker "${worker_id}" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done

if [ "${failed}" -ne 0 ]; then
    echo "Some files failed. Re-run this same command to resume the .part files." >&2
    exit 1
fi

echo "All Pixal3D files downloaded and verified under ${MODEL_DIR}."
