#!/usr/bin/env bash
# Keep retrying the resumable Pixal3D downloader until every file is complete.

set -u

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RETRY_INTERVAL="${PIXAL3D_RETRY_INTERVAL:-60}"

case "${RETRY_INTERVAL}" in
    ''|*[!0-9]*) echo "error: PIXAL3D_RETRY_INTERVAL must be an integer" >&2; exit 2 ;;
esac

while :; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] starting/resuming weight download"
    bash "${PROJECT_ROOT}/scripts/download_pixal3d_weights.sh"
    status=$?
    if [ "${status}" -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] download complete"
        exit 0
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] downloader exited with ${status}; retrying in ${RETRY_INTERVAL}s"
    sleep "${RETRY_INTERVAL}"
done
