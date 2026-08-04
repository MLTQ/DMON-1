"""Immutable manifests and atomic multi-worker execution for SOL2 campaigns."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = 1
JOB_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_jobs(jobs: list[dict]) -> None:
    ids = [row.get("id") for row in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("campaign job ids must be unique")
    known = set(ids)
    for row in jobs:
        job_id = row.get("id")
        if not isinstance(job_id, str) or JOB_ID.fullmatch(job_id) is None:
            raise ValueError(f"invalid campaign job id {job_id!r}")
        command = row.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(part, str) and part for part in command
        ):
            raise ValueError(f"job {job_id} needs a nonempty string command")
        expected = row.get("expected_artifact")
        if not isinstance(expected, str) or Path(expected).is_absolute():
            raise ValueError(f"job {job_id} expected artifact must be relative")
        completion_check = row.get("completion_check")
        if completion_check is not None:
            field = completion_check.get("json_field")
            if not isinstance(field, list) or not field or not all(
                isinstance(part, str) and part for part in field
            ):
                raise ValueError(f"job {job_id} completion check needs json_field")
            if "equals" not in completion_check:
                raise ValueError(f"job {job_id} completion check needs equals")
        dependencies = row.get("depends_on", [])
        if not isinstance(dependencies, list) or not set(dependencies) <= known:
            raise ValueError(f"job {job_id} has unknown dependencies")
        if job_id in dependencies:
            raise ValueError(f"job {job_id} cannot depend on itself")


def write_manifest(root: Path, jobs: list[dict], protocol: dict) -> dict:
    """Write an immutable deterministic campaign manifest."""

    _validate_jobs(jobs)
    body = {"schema": SCHEMA, "protocol": protocol, "jobs": jobs}
    payload = {**body, "manifest_hash": _canonical_hash(body)}
    path = root / "MANIFEST.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise ValueError("refusing to replace a different campaign manifest")
        return existing
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, payload)
    return payload


def load_manifest(root: Path) -> dict:
    payload = json.loads((root / "MANIFEST.json").read_text())
    body = {
        "schema": payload.get("schema"),
        "protocol": payload.get("protocol"),
        "jobs": payload.get("jobs"),
    }
    if payload.get("schema") != SCHEMA or payload.get("manifest_hash") != _canonical_hash(body):
        raise ValueError("campaign manifest schema or hash is invalid")
    _validate_jobs(payload["jobs"])
    return payload


@contextmanager
def _locked_state(root: Path, manifest: dict):
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "campaign.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state_path = root / "campaign_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get("manifest_hash") != manifest["manifest_hash"]:
                raise ValueError("campaign state belongs to a different manifest")
        else:
            state = {
                "schema": SCHEMA,
                "manifest_hash": manifest["manifest_hash"],
                "jobs": {},
            }
        yield state
        _atomic_json(state_path, state)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def claim_ready_job(
    root: Path,
    worker_id: str,
    *,
    retry_failed: bool = False,
    max_attempts: int = 2,
    reclaim_after_seconds: float | None = None,
    now: float | None = None,
) -> dict | None:
    """Atomically claim the first dependency-ready job in manifest order."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    manifest = load_manifest(root)
    current_time = time.time() if now is None else now
    with _locked_state(root, manifest) as state:
        statuses = state["jobs"]
        if reclaim_after_seconds is not None:
            for row in statuses.values():
                if (
                    row.get("status") == "running"
                    and current_time - float(row["started_at"]) >= reclaim_after_seconds
                ):
                    row.update(
                        {
                            "status": "failed",
                            "finished_at": current_time,
                            "failure": "stale_worker_reclaimed",
                        }
                    )
        for job in manifest["jobs"]:
            existing = statuses.get(job["id"], {})
            status = existing.get("status", "pending")
            if status in {"complete", "running"}:
                continue
            if status == "failed":
                if not retry_failed or int(existing.get("attempt", 0)) >= max_attempts:
                    continue
            if any(
                statuses.get(dependency, {}).get("status") != "complete"
                for dependency in job.get("depends_on", [])
            ):
                continue
            attempt = int(existing.get("attempt", 0)) + 1
            statuses[job["id"]] = {
                "status": "running",
                "worker_id": worker_id,
                "pid": os.getpid(),
                "attempt": attempt,
                "started_at": current_time,
            }
            return dict(job)
    return None


def finish_job(
    root: Path,
    job_id: str,
    worker_id: str,
    *,
    exit_code: int,
    now: float | None = None,
) -> dict:
    """Atomically finish a claimed job and verify its declared artifact."""

    manifest = load_manifest(root)
    jobs = {row["id"]: row for row in manifest["jobs"]}
    if job_id not in jobs:
        raise ValueError(f"unknown job {job_id!r}")
    current_time = time.time() if now is None else now
    with _locked_state(root, manifest) as state:
        row = state["jobs"].get(job_id)
        if row is None or row.get("status") != "running":
            raise ValueError(f"job {job_id} is not running")
        if row.get("worker_id") != worker_id:
            raise ValueError(f"job {job_id} belongs to another worker")
        job = jobs[job_id]
        artifact = root / job["expected_artifact"]
        artifact_present = artifact.is_file()
        completion_ok = artifact_present
        completion_check = job.get("completion_check")
        if completion_ok and completion_check is not None:
            try:
                value = json.loads(artifact.read_text())
                for field in completion_check["json_field"]:
                    value = value[field]
                completion_ok = value == completion_check["equals"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                completion_ok = False
        success = exit_code == 0 and completion_ok
        row.update(
            {
                "status": "complete" if success else "failed",
                "finished_at": current_time,
                "elapsed_seconds": current_time - float(row["started_at"]),
                "exit_code": exit_code,
                "artifact_present": artifact_present,
                "completion_check_passed": completion_ok,
            }
        )
        if not success:
            if exit_code:
                row["failure"] = "nonzero_exit"
            elif not artifact_present:
                row["failure"] = "expected_artifact_missing"
            else:
                row["failure"] = "completion_check_failed"
        return dict(row)


def campaign_status(root: Path) -> dict:
    manifest = load_manifest(root)
    with _locked_state(root, manifest) as state:
        rows = {}
        counts = {name: 0 for name in ("pending", "running", "complete", "failed")}
        for job in manifest["jobs"]:
            row = dict(state["jobs"].get(job["id"], {"status": "pending"}))
            rows[job["id"]] = row
            counts[row["status"]] += 1
        return {"manifest_hash": manifest["manifest_hash"], "counts": counts, "jobs": rows}


def _resolved_command(command: list[str], root: Path) -> list[str]:
    replacements = {"{python}": sys.executable, "{root}": str(root.resolve())}
    return [replacements.get(part, part.replace("{root}", str(root.resolve()))) for part in command]


def run_worker(
    root: Path,
    worker_id: str,
    *,
    cuda_visible_devices: str | None,
    retry_failed: bool,
    max_attempts: int,
    reclaim_after_seconds: float | None,
    poll_seconds: float,
) -> int:
    """Run dependency-ready jobs until the campaign reaches a terminal state."""

    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    while True:
        job = claim_ready_job(
            root,
            worker_id,
            retry_failed=retry_failed,
            max_attempts=max_attempts,
            reclaim_after_seconds=reclaim_after_seconds,
        )
        if job is not None:
            environment = os.environ.copy()
            if cuda_visible_devices is not None:
                environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            command = _resolved_command(job["command"], root)
            with (log_dir / f"{job['id']}.log").open("a") as log:
                log.write(f"\n[campaign] worker={worker_id} command={json.dumps(command)}\n")
                log.flush()
                result = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            finish_job(root, job["id"], worker_id, exit_code=result.returncode)
            continue
        status = campaign_status(root)
        if status["counts"]["running"]:
            time.sleep(poll_seconds)
            continue
        return 0 if status["counts"]["pending"] == 0 and status["counts"]["failed"] == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute an immutable SOL2 campaign")
    subparsers = parser.add_subparsers(dest="command", required=True)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("root", type=Path)
    worker = subparsers.add_parser("worker")
    worker.add_argument("root", type=Path)
    worker.add_argument("--worker-id", required=True)
    worker.add_argument("--cuda-visible-devices")
    worker.add_argument("--retry-failed", action="store_true")
    worker.add_argument("--max-attempts", type=int, default=2)
    worker.add_argument("--reclaim-after-seconds", type=float)
    worker.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(campaign_status(args.root), indent=2))
        return
    raise SystemExit(
        run_worker(
            args.root,
            args.worker_id,
            cuda_visible_devices=args.cuda_visible_devices,
            retry_failed=args.retry_failed,
            max_attempts=args.max_attempts,
            reclaim_after_seconds=args.reclaim_after_seconds,
            poll_seconds=args.poll_seconds,
        )
    )


if __name__ == "__main__":
    main()
