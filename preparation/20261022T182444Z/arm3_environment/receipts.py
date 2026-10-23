"""Write measured runtime and artifact receipts without changing the lock."""
import hashlib
import importlib.metadata as md
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
PY = ROOT / "venv" / "Scripts" / "python.exe"
WEIGHT = ROOT / "weights" / "aimnet2_wb97m_d3_0.pt"
EXPECTED_WEIGHT_BYTES = 8836941
EXPECTED_WEIGHT_SHA256 = "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


commit = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
tree = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD^{tree}"], text=True).strip()
freeze = subprocess.check_output([str(PY), "-m", "pip", "freeze"], text=True)
(ROOT / "observed_freeze.txt").write_text(freeze, encoding="utf-8")
lock = ROOT / "requirements.lock"
weight_present = WEIGHT.is_file()
actual_size = WEIGHT.stat().st_size if weight_present else None
actual_sha = sha256(WEIGHT) if weight_present else None
stage_path = ROOT / "weight_stage_receipt.json"
stage = json.loads(stage_path.read_text(encoding="utf-8")) if stage_path.is_file() else {}
stage_receipt_verified = bool(
    weight_present
    and stage.get("retrieved") is True
    and stage.get("url") == "https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/aimnet2_wb97m_d3_0.pt"
    and Path(stage.get("path", "")).resolve() == WEIGHT.resolve()
    and stage.get("bytes") == actual_size
    and stage.get("sha256") == actual_sha
)
import_path = ROOT / "import_receipt.json"
import_receipt = json.loads(import_path.read_text(encoding="utf-8")) if import_path.is_file() else {}

receipt = {
    "python": sys.version.split()[0],
    "packages": {
        name: md.version(name)
        for name in ("torch", "numpy", "rdkit", "aimnet", "warp-lang", "nvalchemi-toolkit-ops",
                     "requests", "click", "PyYAML", "jinja2", "h5py")
    },
    "aimnet_member": "aimnet2-wb97m-d3_0",
    "aimnet_source_commit": commit,
    "aimnet_source_tree": tree,
    "expected_final_aim_channels": 256,
    "weight": {
        "path": str(WEIGHT),
        "present": weight_present,
        "actual_bytes": actual_size,
        "expected_bytes": EXPECTED_WEIGHT_BYTES,
        "actual_sha256": actual_sha,
        "expected_sha256": EXPECTED_WEIGHT_SHA256,
        "hash_matches": actual_size == EXPECTED_WEIGHT_BYTES and actual_sha == EXPECTED_WEIGHT_SHA256,
        "stage_receipt_verified": stage_receipt_verified,
        "downloaded": stage_receipt_verified if stage_path.is_file() else None,
        "loaded_by_import_check": import_receipt.get("weights_loaded"),
        "model_initialized_by_import_check": import_receipt.get("model_initialized"),
    },
    "lock_path": str(lock),
    "lock_sha256": sha256(lock),
    "observed_freeze_path": str(ROOT / "observed_freeze.txt"),
    "observed_freeze_sha256": hashlib.sha256(freeze.encode("utf-8")).hexdigest(),
}
(ROOT / "dependency_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
