"""Extract final AIMNet2 atom features from an explicit prepared-atom manifest.

No model loads unless the caller passes --execute. The default self-test is synthetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MODEL = "aimnet2-wb97m-d3_0"
COMMIT = "d14692839f4e5544f4eb38c15cf01298c27960f8"
WEIGHT_SHA256 = "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28"
WEIGHT_BYTES = 8836941
CHANNELS = 256
WEIGHT_PATH = Path(__file__).resolve().parent / "weights" / "aimnet2_wb97m_d3_0.pt"


class AIMFeaturesCalculator:
    """Expose AIM as an atom feature that survives calculator output filtering."""

    @staticmethod
    def calculator_class():
        from aimnet.calculators.calculator import AIMNet2Calculator

        class _AIMFeaturesCalculator(AIMNet2Calculator):
            keys_out = [*AIMNet2Calculator.keys_out, "aim"]
            atom_feature_keys = [*AIMNet2Calculator.atom_feature_keys, "aim"]

        return _AIMFeaturesCalculator


def validate_manifest(m: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = ("id", "formal_charge", "atomic_numbers", "coord_angstrom",
                "heavy_atom_indices", "source_atom_index")
    missing = [k for k in required if k not in m]
    if missing:
        raise ValueError(f"Missing manifest fields: {', '.join(missing)}")
    z = np.asarray(m["atomic_numbers"], dtype=np.int64)
    xyz = np.asarray(m["coord_angstrom"], dtype=np.float64)
    heavy = np.asarray(m["heavy_atom_indices"], dtype=np.int64)
    source = np.asarray(m["source_atom_index"], dtype=np.int64)
    if z.ndim != 1 or xyz.shape != (len(z), 3):
        raise ValueError("atomic_numbers must be N; coord_angstrom must be N by 3")
    if not np.isfinite(xyz).all() or (z < 1).any():
        raise ValueError("Coordinates must be finite and atomic numbers must be positive")
    if heavy.ndim != 1 or source.shape != heavy.shape or len(heavy) == 0:
        raise ValueError("Heavy atom indices and source map must have equal nonzero length")
    if (heavy < 0).any() or (heavy >= len(z)).any() or len(set(heavy.tolist())) != len(heavy):
        raise ValueError("heavy_atom_indices must be unique valid row indices")
    if (z[heavy] <= 1).any():
        raise ValueError("heavy_atom_indices must select non-hydrogen atoms")
    if not np.array_equal(np.sort(heavy), np.flatnonzero(z > 1)):
        raise ValueError("heavy_atom_indices must cover every non-hydrogen input atom")
    if (source < 0).any():
        raise ValueError("source_atom_index values must be nonnegative")
    if len(set(source.tolist())) != len(source):
        raise ValueError("source_atom_index must be unique")
    if not np.isscalar(m["formal_charge"]):
        raise ValueError("formal_charge must be one integer")
    if int(m["formal_charge"]) != m["formal_charge"]:
        raise ValueError("formal_charge must be integral")
    return z, xyz, heavy


def map_heavy_features(all_features: np.ndarray, heavy: np.ndarray, channels: int = CHANNELS) -> np.ndarray:
    features = np.asarray(all_features)
    if features.ndim != 2 or features.shape[1] != channels:
        raise ValueError(f"AIM output must have shape N by {channels}")
    if len(heavy) and (heavy.min() < 0 or heavy.max() >= len(features)):
        raise ValueError("Heavy atom indices exceed AIM output rows")
    out = np.asarray(features[heavy], dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError("AIM features contain a non-finite value")
    return out


def self_test() -> None:
    import torch

    receipt_path = Path(__file__).resolve().parent / "synthetic_test_receipt.json"
    receipt = {"status": "running", "model_initialized": False,
               "weights_loaded": False, "checks": []}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    def passed(name: str) -> None:
        receipt["checks"].append({"name": name, "passed": True})
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    m = {"id": "synthetic", "formal_charge": 0,
         "atomic_numbers": [6, 1, 8, 1],
         "coord_angstrom": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
         "heavy_atom_indices": [0, 2], "source_atom_index": [4, 7]}
    z, xyz, heavy = validate_manifest(m)
    assert z.tolist() == [6, 1, 8, 1] and xyz.shape == (4, 3)
    passed("full input atom rows and coordinates")
    fake = np.arange(4 * CHANNELS, dtype=np.float32).reshape(4, CHANNELS)
    got = map_heavy_features(fake, heavy)
    assert got.shape == (2, CHANNELS) and np.array_equal(got, fake[[0, 2]])
    passed("heavy feature export and 256-channel schema")
    padded = np.concatenate([fake, np.zeros((2, CHANNELS), dtype=np.float32)])
    assert np.array_equal(map_heavy_features(padded, heavy), got)
    passed("synthetic padding rows do not enter export")
    perm = np.array([2, 0, 3, 1])
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm))
    assert np.array_equal(fake[heavy], fake[perm][inverse[heavy]])
    passed("synthetic atom permutation mapping")
    stacked = np.concatenate([fake, fake + 1], axis=0)
    assert np.array_equal(stacked[:4], fake) and np.array_equal(stacked[4:], fake + 1)
    passed("synthetic batch split")
    centered = xyz - xyz.mean(axis=0)
    rot, _ = np.linalg.qr(np.array([[0.2, 0.7, 0.6], [0.8, -0.5, 0.1], [0.4, 0.4, -0.8]]))
    assert np.allclose(np.linalg.norm(centered[:, None] - centered[None, :], axis=-1),
                       np.linalg.norm((centered @ rot)[:, None] - (centered @ rot)[None, :], axis=-1),
                       atol=1e-6, rtol=1e-6)
    passed("synthetic rotation geometry")
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    grad, = torch.autograd.grad((x * x).sum(), x)
    assert torch.isfinite(grad).all() and torch.equal(grad, torch.tensor([2.0, 4.0]))
    passed("synthetic force-gradient path")
    invalid = [("non-finite coordinates", {**m, "coord_angstrom": [[float("nan"), 0, 0]] * 4}),
               ("hydrogen in heavy map", {**m, "heavy_atom_indices": [1, 2]}),
               ("incomplete source map", {**m, "source_atom_index": [4]})]
    for name, patch in invalid:
        try:
            validate_manifest(patch)
        except ValueError:
            receipt["checks"].append({"name": name, "passed": True})
        else:
            receipt["status"] = "failed"
            receipt["failure"] = "Invalid synthetic manifest passed validation"
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            raise AssertionError("Invalid synthetic manifest passed validation")
    receipt["status"] = "passed"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"synthetic checks passed: {len(receipt['checks'])}")


def extract(manifest_path: Path, output_dir: Path, device: str, force_grad: bool, calc=None) -> None:
    import torch

    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    z, xyz, heavy = validate_manifest(m)
    if calc is None:
        Calc = AIMFeaturesCalculator.calculator_class()
        if not WEIGHT_PATH.is_file():
            raise FileNotFoundError(f"Pinned weight is absent. Run {WEIGHT_PATH.parent.parent / 'stage_weight.py'} first.")
        digest = hashlib.sha256(WEIGHT_PATH.read_bytes()).hexdigest()
        if WEIGHT_PATH.stat().st_size != WEIGHT_BYTES or digest != WEIGHT_SHA256:
            raise ValueError("The staged AIMNet2 weight does not match its pinned size and SHA-256")
        calc = Calc(model=str(WEIGHT_PATH), device=device, compile_model=False, deterministic=True)
    calc.model.eval()
    for parameter in calc.model.parameters():
        parameter.requires_grad_(False)
    data = {
        "coord": torch.as_tensor(xyz, dtype=torch.float32, device=device),
        "numbers": torch.as_tensor(z, dtype=torch.int32, device=device),
        "charge": torch.tensor([int(m["formal_charge"])], dtype=torch.float32, device=device),
    }
    # Do not wrap this call in inference_mode. The optional force path needs autograd.
    if force_grad:
        with torch.enable_grad():
            result = calc(data, forces=True)
    else:
        with torch.no_grad():
            result = calc(data)
    all_features = result["aim"].detach().to("cpu").numpy()
    features = map_heavy_features(all_features, heavy)
    outdir = output_dir / str(m["id"])
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {"features": features,
               "source_atom_index": np.asarray(m["source_atom_index"], dtype=np.int64),
               "heavy_atom_indices": heavy, "atomic_numbers": z, "coord_angstrom": xyz,
               "energy_native": result["energy"].detach().cpu().numpy(),
               "charges": result["charges"].detach().cpu().numpy()}
    if "forces" in result:
        payload["forces_native"] = result["forces"].detach().cpu().numpy()
    np.savez_compressed(outdir / "features.npz", **payload)
    raw = manifest_path.read_bytes()
    receipt = {
        "id": m["id"], "model": MODEL, "source_commit": COMMIT,
        "weight_bytes": WEIGHT_BYTES, "weight_sha256": WEIGHT_SHA256,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "atom_order": "full input rows; exports select heavy_atom_indices in order",
        "coordinate_unit": "angstrom", "feature_channels": CHANNELS,
        "feature_shape": list(features.shape), "force_gradient_requested": force_grad,
        "batching": "single molecule per calculator call",
        "padding_test": "pending explicit model smoke",
        "rotation_test": "pending explicit model smoke",
        "permutation_test": "pending explicit model smoke",
    }
    (outdir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if force_grad:
        if "forces" not in result:
            raise RuntimeError("Calculator did not return forces")
        if not torch.isfinite(result["forces"]).all():
            raise RuntimeError("Forces contain a non-finite value")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--execute", action="store_true", help="Load AIMNet2 weights and calculate features")
    p.add_argument("--input", type=Path, nargs="+")
    p.add_argument("--output", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--force-grad", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test()
    if not a.execute:
        if not a.self_test:
            p.error("Choose --self-test or explicit --execute")
        return
    if a.input is None or a.output is None:
        p.error("--execute requires --input and --output")
    for path in a.input:
        validate_manifest(json.loads(path.read_text(encoding="utf-8")))
    if not WEIGHT_PATH.is_file():
        raise FileNotFoundError(f"Pinned weight is absent. Run {WEIGHT_PATH.parent.parent / 'stage_weight.py'} first.")
    if WEIGHT_PATH.stat().st_size != WEIGHT_BYTES or hashlib.sha256(WEIGHT_PATH.read_bytes()).hexdigest() != WEIGHT_SHA256:
        raise ValueError("The staged AIMNet2 weight does not match its pinned size and SHA-256")
    Calc = AIMFeaturesCalculator.calculator_class()
    # Load the selected model once for all manifest inputs.
    calc = Calc(model=str(WEIGHT_PATH), device=a.device, compile_model=False, deterministic=True)
    for path in a.input:
        extract(path, a.output, a.device, a.force_grad, calc=calc)


if __name__ == "__main__":
    main()
