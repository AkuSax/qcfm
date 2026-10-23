"""Read-only model readiness check. This file never fits or scores a model."""
import argparse, json
from pathlib import Path
from protocol_support import SOURCE, EXPECTED, sha256

ROOT = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authorize-fitting", action="store_true",
                    help="Record a future user's explicit fit authorization. This script still does not fit models.")
    ap.add_argument("--feature-file", type=Path, default=ROOT / "aim_features.npz")
    args = ap.parse_args()
    missing = []
    for name, digest in EXPECTED.items():
        path = SOURCE / name
        if not path.exists() or sha256(path) != digest:
            missing.append(f"frozen source file missing or hash mismatch: {name}")
    if not args.feature_file.exists():
        missing.append(f"AIM atom and scalar feature bundle missing: {args.feature_file}")
    missing.extend([
        "62 atom joins passed; 3 stereochemistry records remain held; full-cohort hydrogen/free-conformer preparation remains unrun",
        "AIM atomic export and runtime parity/invariance checks remain incomplete",
        "nested runner is implemented but has not been executed",
        "no model-execution authorization is recorded in this preparation arm",
    ])
    report = {
        "status": "BLOCKED_PREPARATION_ONLY",
        "authorization_flag_seen": bool(args.authorize_fitting),
        "authorize_flag_does_not_run_models": True,
        "fit_runner_implemented": True,
        "model_calls": 0,
        "missing_gates": missing,
        "future_fit_entrypoint": "fit_entrypoint.py; explicit --authorize-fitting and complete gate receipt required",
    }
    out = ROOT / "fit_preflight.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
