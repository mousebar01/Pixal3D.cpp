#!/usr/bin/env python3
"""Serve the local Pixal3D workbench and bridge image jobs to the CLI.

This is intentionally a small, dependency-free development server.  It binds
only to localhost by default, accepts one image upload at a time, starts the
native CLI without a shell, and keeps generated files under the ignored
``demo-runs/`` directory.  It is not an internet-facing production service.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = ROOT / "demo"
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".ppm", ".pgm", ".pbm", ".pnm"}
BACKEND_PATTERN = re.compile(r"^(?:auto|cpu|gpu|gpu:[0-9]+)$", re.IGNORECASE)
PROFILE_VALUES = {"off", "summary", "trace"}


class DemoError(Exception):
    """An expected request/configuration error with an HTTP status."""

    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


class DemoState:
    def __init__(self, args: argparse.Namespace):
        self.repo_root = ROOT
        self.binary = resolve_path(args.pixal3d)
        self.model_paths = {
            "shared_pack": resolve_path(args.shared_pack),
            "flow_pack": resolve_path(args.flow_pack),
            "dino_pack": resolve_path(args.dino_pack),
            "naf_pack": resolve_path(args.naf_pack),
            "moge_onnx": resolve_path(args.moge_onnx),
        }
        self.host = args.host
        self.port = args.port
        self.max_upload_bytes = args.max_upload_mb * 1024 * 1024
        self.max_running_jobs = args.max_running_jobs
        self.max_jobs = args.max_jobs
        self.job_timeout = args.job_timeout
        self.default_backend = args.default_backend
        self.default_profile = args.default_profile
        self.run_root = resolve_path(args.run_dir)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.running = threading.Semaphore(self.max_running_jobs)

    def asset_status(self) -> Dict[str, Any]:
        assets = {}
        for name, path in self.model_paths.items():
            assets[name] = {"name": name, "path": str(path), "exists": path.is_file()}
        binary_exists = self.binary.is_file() and os.access(str(self.binary), os.X_OK)
        missing = [name for name, item in assets.items() if not item["exists"]]
        if not binary_exists:
            missing.insert(0, "pixal3d_binary")
        return {
            "ready": not missing,
            "binary": {"path": str(self.binary), "exists": binary_exists},
            "models": assets,
            "missing": missing,
        }

    def config_payload(self) -> Dict[str, Any]:
        status = self.asset_status()
        return {
            "ready": status["ready"],
            "binary": status["binary"],
            "models": status["models"],
            "missing": status["missing"],
            "defaults": {
                "backend": self.default_backend,
                "profile": self.default_profile,
                "seed": 42,
                "steps": 12,
                "texture_size": 512,
                "vision_resolution": 256,
            },
            "limits": {
                "max_upload_bytes": self.max_upload_bytes,
                "max_upload_mb": self.max_upload_bytes / (1024 * 1024),
                "max_running_jobs": self.max_running_jobs,
                "max_jobs": self.max_jobs,
                "job_timeout_seconds": self.job_timeout,
            },
            "server": {"host": self.host, "port": self.port},
        }

    def new_job(self, input_name: str, params: Dict[str, Any]) -> str:
        with self.lock:
            self._prune_finished_jobs_locked()
            active = len(self.jobs)
            if active >= self.max_jobs:
                raise DemoError(
                    "job queue is full; wait for an existing job to finish",
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
            job_id = uuid.uuid4().hex
            job_dir = self.run_root / job_id
            job_dir.mkdir(parents=True, exist_ok=False)
            suffix = Path(input_name).suffix.lower()
            input_path = job_dir / ("input" + suffix)
            output_path = job_dir / "output.glb"
            self.jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "stage": "Queued",
                "progress": 0,
                "logs": [],
                "input_name": input_name,
                "params": params,
                "input_path": input_path,
                "output_path": output_path,
                "created_at": time.time(),
                "started_at": None,
                "ended_at": None,
                "returncode": None,
                "error": None,
            }
            return job_id

    def _prune_finished_jobs_locked(self) -> None:
        finished = [
            job for job in self.jobs.values()
            if job["status"] in ("complete", "error")
        ]
        finished.sort(key=lambda job: job["ended_at"] or job["created_at"])
        while len(self.jobs) >= self.max_jobs and finished:
            job = finished.pop(0)
            self.jobs.pop(job["id"], None)
            shutil.rmtree(job["input_path"].parent, ignore_errors=True)

    def write_input(self, job_id: str, image_name: str, image_bytes: bytes) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise DemoError("job no longer exists", HTTPStatus.NOT_FOUND)
            path = job["input_path"]
        path.write_bytes(image_bytes)

    def start_job(self, job_id: str) -> None:
        worker = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        worker.start()

    def snapshot(self, job_id: str) -> Dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise DemoError("job not found", HTTPStatus.NOT_FOUND)
            elapsed = elapsed_seconds(job["started_at"], job["ended_at"])
            return {
                "id": job["id"],
                "status": job["status"],
                "stage": job["stage"],
                "progress": job["progress"],
                "logs": list(job["logs"]),
                "input_name": job["input_name"],
                "params": dict(job["params"]),
                "created_at": job["created_at"],
                "started_at": job["started_at"],
                "ended_at": job["ended_at"],
                "elapsed_seconds": elapsed,
                "returncode": job["returncode"],
                "error": job["error"],
                "output_url": (
                    "/api/jobs/{}/output".format(job_id)
                    if job["status"] == "complete"
                    else None
                ),
            }

    def output_path(self, job_id: str) -> Path:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise DemoError("job not found", HTTPStatus.NOT_FOUND)
            if job["status"] != "complete":
                raise DemoError("job output is not ready", HTTPStatus.CONFLICT)
            path = job["output_path"]
        if not path.is_file():
            raise DemoError("job output is missing", HTTPStatus.NOT_FOUND)
        return path

    def _run_job(self, job_id: str) -> None:
        self.running.acquire()
        try:
            try:
                self._set_job(job_id, status="running", stage="Starting native CLI", progress=2,
                              started_at=time.time())
                with self.lock:
                    job = self.jobs[job_id]
                    params = dict(job["params"])
                    input_path = job["input_path"]
                    output_path = job["output_path"]

                command = [
                    str(self.binary),
                    "run-image",
                    str(self.model_paths["shared_pack"]),
                    str(self.model_paths["flow_pack"]),
                    str(self.model_paths["dino_pack"]),
                    str(self.model_paths["naf_pack"]),
                    str(input_path),
                    str(output_path),
                    "--seed",
                    str(params["seed"]),
                    "--steps",
                    str(params["steps"]),
                    "--texture-size",
                    str(params["texture_size"]),
                    "--moge-onnx",
                    str(self.model_paths["moge_onnx"]),
                ]
                if params["vision_resolution"]:
                    command.extend(["--vision-resolution", str(params["vision_resolution"])])
                if params["max_structure_points"]:
                    command.extend(["--max-structure-points", str(params["max_structure_points"])])
                if params["occupancy_threshold"] is not None:
                    command.extend(["--occupancy-threshold", str(params["occupancy_threshold"])])
                if params["max_model_gib"] is not None:
                    command.extend(["--max-model-gib", str(params["max_model_gib"])])

                environment = os.environ.copy()
                environment["PIXAL3D_BACKEND"] = params["backend"]
                environment["PIXAL3D_PROFILE"] = params["profile"]
                self._append_log(job_id, "$ " + shell_join(command))
                self._append_log(job_id, "backend policy: " + params["backend"])
                self._set_job(job_id, stage="Loading models", progress=5)

                try:
                    process = subprocess.Popen(
                        command,
                        cwd=str(self.repo_root),
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                except OSError as exc:
                    self._fail_job(job_id, "failed to start pixal3d: {}".format(exc), -1)
                    return

                deadline = time.monotonic() + self.job_timeout if self.job_timeout > 0 else None
                assert process.stdout is not None
                while True:
                    if deadline is not None and time.monotonic() > deadline:
                        process.kill()
                        self._append_log(job_id, "job timeout reached; native process killed")
                        process.wait()
                        self._fail_job(job_id, "job exceeded the configured timeout", -9)
                        return
                    line = process.stdout.readline()
                    if line:
                        self._append_log(job_id, line.rstrip("\r\n"))
                        self._update_progress_from_line(job_id, line)
                        continue
                    if process.poll() is not None:
                        break
                    time.sleep(0.05)
                returncode = process.wait()
                if returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0:
                    self._set_job(job_id, status="complete", stage="Complete", progress=100,
                                  ended_at=time.time(), returncode=returncode)
                else:
                    message = "pixal3d exited with code {}".format(returncode)
                    if returncode == 0:
                        message = "pixal3d finished without producing a valid GLB"
                    self._fail_job(job_id, message, returncode)
            except Exception as exc:
                self._fail_job(job_id, "unexpected demo worker error: {}".format(exc), -1)
        finally:
            self.running.release()

    def _update_progress_from_line(self, job_id: str, line: str) -> None:
        lower = line.lower()
        if "loading dino" in lower or "loading naf" in lower or "loading" in lower:
            self._set_job(job_id, stage="Loading vision models", progress=8)
        elif "camera_estimated" in lower or "moge" in lower:
            self._set_job(job_id, stage="Estimating camera", progress=20)
        elif "condition" in lower or "dino graph" in lower or "naf" in lower:
            self._set_job(job_id, stage="Encoding image condition", progress=28)
        elif "sparse-structure" in lower or "ss-flow" in lower or "sampling" in lower:
            self._set_job(job_id, stage="Sampling sparse structure", progress=45)
        elif "shape" in lower or "texture" in lower or "slat" in lower:
            self._set_job(job_id, stage="Decoding 3D features", progress=62)
        elif "qem" in lower or "decimat" in lower:
            self._set_job(job_id, stage="CPU mesh decimation", progress=72)
        elif "uv bake" in lower or "xatlas" in lower or "atlas" in lower:
            self._set_job(job_id, stage="CPU UV unwrap and texture bake", progress=86)
        elif "output" in lower and ".glb" in lower:
            self._set_job(job_id, stage="Writing GLB", progress=96)

    def _append_log(self, job_id: str, line: str) -> None:
        line = line[:4096]
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            job["logs"].append(line)
            if len(job["logs"]) > 500:
                del job["logs"][: len(job["logs"]) - 500]

    def _set_job(self, job_id: str, **updates: Any) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job:
                job.update(updates)

    def _fail_job(self, job_id: str, message: str, returncode: int) -> None:
        self._append_log(job_id, "error: " + message)
        self._set_job(job_id, status="error", stage="Failed", progress=0,
                      ended_at=time.time(), returncode=returncode, error=message)


def resolve_path(value: Any) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def elapsed_seconds(started: Optional[float], ended: Optional[float]) -> Optional[float]:
    if started is None:
        return None
    return round(max(0.0, (ended if ended is not None else time.time()) - started), 3)


def shell_join(command: List[str]) -> str:
    # Diagnostic-only quoting; the actual subprocess call never uses a shell.
    import shlex

    return " ".join(shlex.quote(item) for item in command)


def parse_int_field(fields: Dict[str, Any], name: str, default: int,
                    minimum: int, maximum: int) -> int:
    raw = fields.get(name)
    if raw in (None, ""):
        return default
    try:
        value = int(str(raw), 10)
    except (TypeError, ValueError):
        raise DemoError("{} must be an integer".format(name))
    if value < minimum or value > maximum:
        raise DemoError("{} must be in the range {}..{}".format(name, minimum, maximum))
    return value


def parse_optional_float_field(fields: Dict[str, Any], name: str,
                               default: Optional[float], minimum: Optional[float] = None,
                               maximum: Optional[float] = None) -> Optional[float]:
    raw = fields.get(name)
    if raw in (None, ""):
        return default
    try:
        value = float(str(raw))
    except (TypeError, ValueError):
        raise DemoError("{} must be a number".format(name))
    if not math.isfinite(value):
        raise DemoError("{} must be finite".format(name))
    if minimum is not None and value < minimum:
        raise DemoError("{} must be >= {}".format(name, minimum))
    if maximum is not None and value > maximum:
        raise DemoError("{} must be <= {}".format(name, maximum))
    return value


def parse_job_params(fields: Dict[str, Any], state: DemoState) -> Dict[str, Any]:
    backend = str(fields.get("backend") or state.default_backend).strip().lower()
    if not BACKEND_PATTERN.match(backend):
        raise DemoError("backend must be auto, cpu, gpu, or gpu:<index>")
    profile = str(fields.get("profile") or state.default_profile).strip().lower()
    if profile not in PROFILE_VALUES:
        raise DemoError("profile must be off, summary, or trace")
    seed = parse_int_field(fields, "seed", 42, 0, (1 << 64) - 1)
    steps = parse_int_field(fields, "steps", 12, 1, 64)
    texture_size = parse_int_field(fields, "texture_size", 512, 1, 4096)
    vision_resolution = parse_int_field(fields, "vision_resolution", 256, 0, 1024)
    if vision_resolution and (vision_resolution < 16 or vision_resolution % 16 != 0):
        raise DemoError("vision_resolution must be 0 or a multiple of 16 >= 16")
    max_structure_points = parse_int_field(fields, "max_structure_points", 0, 0, 100000000)
    occupancy_threshold = parse_optional_float_field(fields, "occupancy_threshold", None)
    max_model_gib = parse_optional_float_field(fields, "max_model_gib", None, 0.1, 1024.0)
    return {
        "backend": backend,
        "profile": profile,
        "seed": seed,
        "steps": steps,
        "texture_size": texture_size,
        "vision_resolution": vision_resolution,
        "max_structure_points": max_structure_points,
        "occupancy_threshold": occupancy_threshold,
        "max_model_gib": max_model_gib,
    }


def parse_multipart(content_type: str, body: bytes) -> Dict[str, Any]:
    if not content_type.lower().startswith("multipart/form-data"):
        raise DemoError("POST /api/generate requires multipart/form-data")
    header = "Content-Type: {}\r\nMIME-Version: 1.0\r\n\r\n".format(content_type)
    message = BytesParser(policy=policy.default).parsebytes(header.encode("utf-8") + body)
    if not message.is_multipart():
        raise DemoError("invalid multipart request")
    fields: Dict[str, Any] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename is not None:
            fields[name] = {"filename": filename, "data": payload}
        else:
            fields[name] = payload.decode("utf-8", errors="strict").strip()
    return fields


def parse_device_lines(raw: str) -> List[Dict[str, Any]]:
    devices = []
    for line in raw.splitlines():
        match = re.match(
            r"device=(?P<device>\d+) type=(?P<type>\S+) name=(?P<name>\S+)"
            r"(?: policy=(?P<policy>\S+))? description=(?P<description>.*?)"
            r" memory_free_bytes=(?P<free>\d+) memory_total_bytes=(?P<total>\d+)$",
            line,
        )
        if not match:
            devices.append({"raw": line})
            continue
        item = match.groupdict()
        devices.append({
            "registry_index": int(item["device"]),
            "type": item["type"],
            "name": item["name"],
            "policy": item["policy"],
            "description": item["description"],
            "memory_free_bytes": int(item["free"]),
            "memory_total_bytes": int(item["total"]),
        })
    return devices


class DemoHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "Pixal3DDemo/0.1"

    @property
    def state(self) -> DemoState:
        return self.server.demo_state  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        try:
            if path == "/api/config":
                self.send_json(self.state.config_payload())
            elif path == "/api/devices":
                self.send_json(self.devices_payload())
            elif path.startswith("/api/jobs/") and path.endswith("/output"):
                job_id = path[len("/api/jobs/") : -len("/output")].strip("/")
                self.send_output(job_id)
            elif path.startswith("/api/jobs/"):
                job_id = path[len("/api/jobs/") :].strip("/")
                self.send_json(self.state.snapshot(job_id))
            elif path in ("/", "/index.html"):
                self.send_static("index.html")
            else:
                raise DemoError("not found", HTTPStatus.NOT_FOUND)
        except DemoError as exc:
            self.send_error_json(exc.status, str(exc))
        except Exception as exc:  # keep the local server alive for a bad request
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path != "/api/generate":
            self.send_error_json(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            fields = self.read_multipart()
            image = fields.get("image")
            if not isinstance(image, dict) or not image.get("data"):
                raise DemoError("image upload is required")
            image_name = str(image.get("filename") or "input.png")
            suffix = Path(image_name).suffix.lower()
            if suffix not in SUPPORTED_IMAGE_SUFFIXES:
                raise DemoError(
                    "unsupported image extension {}; use PNG, JPEG, or PNM".format(suffix or "(none)")
                )
            image_bytes = image["data"]
            if len(image_bytes) > self.state.max_upload_bytes:
                raise DemoError(
                    "image exceeds the {} MiB upload limit".format(
                        self.state.max_upload_bytes // (1024 * 1024)
                    )
                )
            asset_status = self.state.asset_status()
            if not asset_status["ready"]:
                raise DemoError(
                    "demo runtime is not ready; missing: " + ", ".join(asset_status["missing"]),
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
            params = parse_job_params(fields, self.state)
            job_id = self.state.new_job(image_name, params)
            self.state.write_input(job_id, image_name, image_bytes)
            self.state.start_job(job_id)
            self.send_json({"job_id": job_id, "job": self.state.snapshot(job_id)}, HTTPStatus.ACCEPTED)
        except DemoError as exc:
            self.send_error_json(exc.status, str(exc))
        except Exception as exc:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def read_multipart(self) -> Dict[str, Any]:
        length_text = self.headers.get("Content-Length")
        if not length_text:
            raise DemoError("Content-Length is required", HTTPStatus.LENGTH_REQUIRED)
        try:
            length = int(length_text, 10)
        except ValueError:
            raise DemoError("invalid Content-Length")
        max_request = self.state.max_upload_bytes + 1024 * 1024
        if length < 0 or length > max_request:
            raise DemoError(
                "request exceeds the {} MiB upload limit".format(self.state.max_upload_bytes // (1024 * 1024)),
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
        body = self.rfile.read(length)
        if len(body) != length:
            raise DemoError("incomplete upload", HTTPStatus.BAD_REQUEST)
        return parse_multipart(self.headers.get("Content-Type", ""), body)

    def devices_payload(self) -> Dict[str, Any]:
        if not self.state.binary.is_file() or not os.access(str(self.state.binary), os.X_OK):
            raise DemoError("pixal3d binary is not executable", HTTPStatus.SERVICE_UNAVAILABLE)
        try:
            result = subprocess.run(
                [str(self.state.binary), "--list-devices"],
                cwd=str(self.state.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DemoError("device query failed: {}".format(exc), HTTPStatus.BAD_GATEWAY)
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "devices": parse_device_lines(result.stdout),
            "raw": result.stdout.splitlines(),
            "stderr": result.stderr.splitlines(),
        }

    def send_output(self, job_id: str) -> None:
        path = self.state.output_path(job_id)
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "model/gltf-binary")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", 'inline; filename="pixal3d-{}.glb"'.format(job_id))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_static(self, relative_name: str) -> None:
        root = STATIC_ROOT.resolve()
        candidate = (root / relative_name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise DemoError("not found", HTTPStatus.NOT_FOUND)
        if not candidate.is_file():
            raise DemoError("not found", HTTPStatus.NOT_FOUND)
        data = candidate.read_bytes()
        content_type = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json({"error": message, "status": int(status)}, status)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("pixal3d-demo: " + (fmt % args) + "\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="bind port (default: 8765)")
    parser.add_argument("--pixal3d", type=Path, default=ROOT / "build" / "bin" / "pixal3d",
                        help="pixal3d executable")
    parser.add_argument("--shared-pack", type=Path,
                        default=ROOT / "build" / "weights" / "pixal3d-shared-f16.gguf")
    parser.add_argument("--flow-pack", type=Path,
                        default=ROOT / "build" / "weights" / "pixal3d-base-flow-f32.gguf")
    parser.add_argument("--dino-pack", type=Path,
                        default=ROOT / "build" / "weights" / "dinov3-vitl16-pretrain-lvd1689m-f32.gguf")
    parser.add_argument("--naf-pack", type=Path,
                        default=ROOT / "weights" / "NAF" / "naf_release-f32.gguf")
    parser.add_argument("--moge-onnx", type=Path,
                        default=ROOT / "weights" / "MoGe" / "moge-2-vitl-normal.onnx")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "demo-runs",
                        help="ignored directory for uploads and generated GLBs")
    parser.add_argument("--max-upload-mb", type=int, default=16,
                        help="maximum image upload size (default: 16)")
    parser.add_argument("--max-running-jobs", type=int, default=1,
                        help="maximum concurrent native jobs (default: 1)")
    parser.add_argument("--max-jobs", type=int, default=16,
                        help="maximum queued/retained jobs (default: 16)")
    parser.add_argument("--job-timeout", type=int, default=0,
                        help="kill jobs after N seconds; 0 disables the timeout")
    parser.add_argument("--default-backend", default="auto",
                        help="default backend policy: auto, cpu, gpu, or gpu:<index>")
    parser.add_argument("--default-profile", choices=sorted(PROFILE_VALUES), default="off",
                        help="default PIXAL3D_PROFILE mode")
    args = parser.parse_args(argv)
    if args.port < 0 or args.port > 65535:
        parser.error("--port must be in the range 0..65535")
    if args.max_upload_mb <= 0:
        parser.error("--max-upload-mb must be positive")
    if args.max_running_jobs <= 0 or args.max_jobs <= 0:
        parser.error("job limits must be positive")
    if args.job_timeout < 0:
        parser.error("--job-timeout must be non-negative")
    if not BACKEND_PATTERN.match(args.default_backend):
        parser.error("--default-backend must be auto, cpu, gpu, or gpu:<index>")
    args.default_backend = args.default_backend.lower()
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not STATIC_ROOT.is_dir():
        print("error: demo static directory is missing: {}".format(STATIC_ROOT), file=sys.stderr)
        return 2
    state = DemoState(args)
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    server.demo_state = state  # type: ignore[attr-defined]
    actual_host, actual_port = server.server_address[:2]
    state.port = actual_port
    print("Pixal3D demo listening on http://{}:{}/".format(actual_host, actual_port), flush=True)
    status = state.asset_status()
    if status["ready"]:
        print("pixal3d-demo: runtime assets ready", file=sys.stderr, flush=True)
    else:
        print("pixal3d-demo: runtime not ready; missing {}".format(
            ", ".join(status["missing"])), file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\npixal3d-demo: stopping", file=sys.stderr)
    finally:
        server.server_close()
        # Do not delete job outputs on shutdown: they are useful for local
        # inspection and remain under the ignored run directory.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
