"""Launch the frozen classical runner with one persistent wall-clock budget.

This file only prepares process control. Do not run it before the lead writes
the authorized gate receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "exploration" / "20261022T232117Z"
CLASSICAL = RUN / "classical"
PYTHON = ROOT / "preparation" / "20261022T182444Z" / "arm3_environment" / "venv" / "Scripts" / "python.exe"
RUNNER = CLASSICAL / "runner.py"
OUTPUT_DIR = RUN / "interaction" / "launcher_runs"
STATE_PATH = RUN / "interaction" / "launcher_state.json"
WALL_SECONDS = 14_400
EXPECTED_PLAN_SHA256 = "530a7ff532fbc56b5291e9a73e4ffb0a40689a0623fced5172656f897d1d502c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_hash() -> tuple[str, dict[str, str]]:
    sources = [RUNNER, RUN / "protocol_support.py", CLASSICAL / "protocol_support.py",
               RUN / "interaction" / "interaction.py"]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required source file is missing: {missing}")
    per_file = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    combined = hashlib.sha256(json.dumps(per_file, sort_keys=True).encode("utf-8")).hexdigest()
    return combined, per_file


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-receipt", type=Path, required=True)
    parser.add_argument("--output", default=str(CLASSICAL))
    args = parser.parse_args()
    if not args.gate_receipt.is_file():
        raise FileNotFoundError(f"Lead gate receipt is missing: {args.gate_receipt}")
    if not PYTHON.is_file() or not RUNNER.is_file():
        raise FileNotFoundError("Staged Python or classical runner is missing")

    now = time.time()
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if state.get("status") == "running":
            raise RuntimeError("A prior launcher attempt is still marked running")
        if state.get("status") == "complete":
            raise RuntimeError("The classical runner already completed under this budget")
    else:
        state = {"start_unix": now, "deadline_unix": now + WALL_SECONDS,
                 "wall_budget_seconds": WALL_SECONDS, "attempt_count": 0,
                 "attempts": [], "status": "ready"}
    remaining = float(state["deadline_unix"]) - now
    if remaining <= 0:
        state["status"] = "wall_budget_exhausted"
        write_json(STATE_PATH, state)
        raise TimeoutError("The persistent 14,400-second wall budget is exhausted")

    # Read-only gate check. The runner also checks the receipt before fitting.
    gate = json.loads(args.gate_receipt.read_text(encoding="utf-8-sig"))
    plan_hash = sha256(RUN / "FROZEN_PLAN.md")
    if plan_hash != EXPECTED_PLAN_SHA256:
        raise ValueError("Frozen plan hash differs from the lead gate")
    if gate.get("execution_authorized") is not True:
        raise PermissionError("Lead gate does not authorize fitting")
    for name in ("feature_manifest_frozen", "source_review_pass", "preflight_pass"):
        if gate.get(name) is not True:
            raise ValueError(f"Lead gate check is not true: {name}")
    if int(gate.get("maximum_attempts", -1)) != 1500:
        raise ValueError("Lead gate attempt cap must be 1500")
    if int(gate.get("maximum_seconds", -1)) != WALL_SECONDS:
        raise ValueError("Lead gate wall-time cap must be 14400 seconds")
    if gate.get("expected_plan_hash") != EXPECTED_PLAN_SHA256:
        raise ValueError("Lead gate plan hash differs from the frozen plan")
    if gate.get("frozen_plan_sha256") != EXPECTED_PLAN_SHA256:
        raise ValueError("Lead gate frozen_plan_sha256 differs from the frozen plan")
    digest, file_hashes = code_hash()
    attempt = int(state["attempt_count"]) + 1
    attempt_dir = OUTPUT_DIR / f"attempt_{attempt:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    stdout_path, stderr_path = attempt_dir / "stdout.log", attempt_dir / "stderr.log"
    command = [str(PYTHON), str(RUNNER), "--output", args.output,
               "--authorize-fitting", "--gate-receipt", str(args.gate_receipt.resolve())]
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = RUN / output_path
    command[3] = str(output_path.resolve())
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = "1"

    started = time.time()
    state.update({"attempt_count": attempt, "status": "running", "last_attempt_unix": started,
                  "deadline_unix": state["deadline_unix"]})
    write_json(STATE_PATH, state)
    return_code = None
    timed_out = False
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        launch_unix = time.time()
        remaining = float(state["deadline_unix"]) - launch_unix
        if remaining <= 0:
            timed_out = True
        else:
            try:
                result = subprocess.run(command, cwd=str(ROOT), env=env, stdout=stdout_file,
                                        stderr=stderr_file, timeout=remaining, check=False)
                return_code = result.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
    ended = time.time()
    run_receipt = {
        "attempt": attempt, "start_unix": state["start_unix"],
        "deadline_unix": state["deadline_unix"], "attempt_start_unix": started,
        "subprocess_launch_unix": launch_unix,
        "attempt_end_unix": ended, "elapsed_seconds": ended - started,
        "wall_budget_seconds": WALL_SECONDS, "remaining_at_start_seconds": remaining,
        "timed_out": timed_out, "exit_code": return_code,
        "command": command, "cwd": str(ROOT), "python": str(PYTHON),
        "gate_receipt": str(args.gate_receipt.resolve()), "gate_receipt_sha256": sha256(args.gate_receipt),
        "runner_code_sha256": digest, "source_sha256": file_hashes,
        "thread_environment": {name: env[name] for name in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")},
        "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
    }
    receipt_path = attempt_dir / "process_receipt.json"
    write_json(receipt_path, run_receipt)
    state["attempts"].append(str(receipt_path.relative_to(RUN)))
    state["status"] = "complete" if return_code == 0 and not timed_out else "retryable_failure"
    state["last_result"] = {"exit_code": return_code, "timed_out": timed_out,
                             "elapsed_seconds": ended - started}
    write_json(STATE_PATH, state)
    print(json.dumps(run_receipt, indent=2))
    return 0 if return_code == 0 and not timed_out else 1


if __name__ == "__main__":
    raise SystemExit(main())
