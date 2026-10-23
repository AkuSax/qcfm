"""Fail-fast AIMNet2 export smoke. It loads weights only with --execute."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from aimnet.calculators.calculator import AIMNet2Calculator
from extract_aim_features import (AIMFeaturesCalculator, CHANNELS, WEIGHT_BYTES,
                                  WEIGHT_PATH, WEIGHT_SHA256, validate_manifest)

ATOL = RTOL = 1e-4
MAX_EVALUATIONS = 18
DEVICE = "cpu"
COUNTS = {"api_calls": 0, "molecule_evaluations": 0}
GLOBAL_RECEIPT: Path | None = None


def verify_weight() -> None:
    if not WEIGHT_PATH.is_file():
        raise FileNotFoundError("Run stage_weight.py before the explicitly authorized smoke.")
    if WEIGHT_PATH.stat().st_size != WEIGHT_BYTES:
        raise ValueError("Staged weight size does not match the pinned receipt")
    digest = hashlib.sha256(WEIGHT_PATH.read_bytes()).hexdigest()
    if digest != WEIGHT_SHA256:
        raise ValueError("Staged weight SHA-256 does not match the pinned receipt")


def array(result, key: str) -> np.ndarray:
    value = result[key]
    if isinstance(value, (list, tuple)):
        rows = [v.detach().to("cpu").numpy() for v in value]
        return np.concatenate(rows, axis=0)
    return value.detach().to("cpu").numpy()


def run_calc(calc, data: dict, molecule_evaluations: int, forces: bool = False):
    if COUNTS["molecule_evaluations"] + molecule_evaluations > MAX_EVALUATIONS:
        raise RuntimeError("The smoke evaluation limit would be exceeded")
    COUNTS["api_calls"] += 1
    COUNTS["molecule_evaluations"] += molecule_evaluations
    if GLOBAL_RECEIPT is not None:
        receipt = json.loads(GLOBAL_RECEIPT.read_text(encoding="utf-8"))
        receipt.update(COUNTS)
        GLOBAL_RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if forces:
        with torch.enable_grad():
            return calc(data, forces=True)
    with torch.no_grad():
        return calc(data)


def call(calc, coord: np.ndarray, z: np.ndarray, charge: int,
         mol_idx: np.ndarray | None = None, forces: bool = False):
    data = {
        "coord": torch.as_tensor(coord, dtype=torch.float32, device=DEVICE),
        "numbers": torch.as_tensor(z, dtype=torch.int32, device=DEVICE),
        "charge": torch.as_tensor([charge], dtype=torch.float32, device=DEVICE),
    }
    if mol_idx is not None:
        data["mol_idx"] = torch.as_tensor(mol_idx, dtype=torch.int32, device=DEVICE)
    return run_calc(calc, data, molecule_evaluations=1, forces=forces)


def check_close(a, b, label: str, receipt: dict, atol: float = ATOL, rtol: float = RTOL) -> None:
    a = np.asarray(a)
    b = np.asarray(b)
    ok = a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False)
    receipt["checks"].append({"name": label, "passed": bool(ok),
                              "atol": atol, "rtol": rtol,
                              "max_abs_error": float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else None})
    receipt_path = Path(receipt["receipt_path"])
    if not ok:
        receipt["status"] = "failed"
        receipt["failure"] = label
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if not ok:
        raise AssertionError(f"{label} failed; see {receipt_path}")


def main() -> None:
    global DEVICE, GLOBAL_RECEIPT
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true", help="Load the verified pretrained weight")
    p.add_argument("--input", nargs="+", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = p.parse_args()
    if not a.execute:
        p.error("--execute is required. The model smoke is gated.")
    if a.input is None or len(a.input) != 3 or a.output is None:
        p.error("Provide exactly 3 --input manifests and --output")
    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable in this environment")
    DEVICE = a.device
    manifests = [json.loads(f.read_text(encoding="utf-8")) for f in a.input]
    arrays = [validate_manifest(m) for m in manifests]
    verify_weight()

    a.output.mkdir(parents=True, exist_ok=True)
    GLOBAL_RECEIPT = a.output / "smoke_global_receipt.json"
    global_receipt = {"status": "running", "model": "aimnet2-wb97m-d3_0",
                      "weight_sha256": WEIGHT_SHA256, "evaluation_limit": MAX_EVALUATIONS,
                      "api_calls": 0, "molecule_evaluations": 0,
                      "plan": ["3 single exports", "3 standard scalar checks", "3 rotations",
                               "3 permutations", "3-conformer concat batch", "3-molecule dense padded batch"],
                      "input_ids": [m["id"] for m in manifests]}
    GLOBAL_RECEIPT.write_text(json.dumps(global_receipt, indent=2) + "\n", encoding="utf-8")

    CalcWithAim = AIMFeaturesCalculator.calculator_class()
    # Build both calculators from the verified file to preserve its metadata.
    export_calc = CalcWithAim(model=str(WEIGHT_PATH), device=a.device,
                              compile_model=False, deterministic=True)
    standard_calc = AIMNet2Calculator(model=str(WEIGHT_PATH), device=a.device,
                                      compile_model=False, deterministic=True)
    export_calc.model.eval()
    standard_calc.model.eval()
    for calc in (export_calc, standard_calc):
        for parameter in calc.model.parameters():
            parameter.requires_grad_(False)
    export_meta = dict(export_calc.metadata or {})
    standard_meta = dict(standard_calc.metadata or {})
    if export_meta != standard_meta:
        raise ValueError("Standard and feature calculators have different model metadata")
    if export_meta.get("needs_dispersion") and not export_meta.get("d3_params"):
        raise ValueError("D3 metadata is missing from the selected model")
    global_receipt["model_metadata"] = {k: export_meta.get(k) for k in
                                         ("needs_coulomb", "coulomb_mode", "needs_dispersion", "d3_params")}
    global_receipt["metadata_parity"] = "passed"
    GLOBAL_RECEIPT.write_text(json.dumps(global_receipt, indent=2) + "\n", encoding="utf-8")

    # The budget is 18 molecule evaluations in 14 API calls.
    base_results = []
    for manifest, (z, xyz, heavy) in zip(manifests, arrays, strict=True):
        outdir = a.output / str(manifest["id"])
        outdir.mkdir(parents=True, exist_ok=True)
        receipt_path = outdir / "smoke_receipt.json"
        receipt = {"id": manifest["id"], "status": "running", "evaluations_used": 0,
                   "evaluation_limit": MAX_EVALUATIONS, "checks": [], "receipt_path": str(receipt_path),
                   "model": "aimnet2-wb97m-d3_0", "weight_sha256": WEIGHT_SHA256,
                   "source_commit": "d14692839f4e5544f4eb38c15cf01298c27960f8",
                   "atol": ATOL, "rtol": RTOL}
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        result = call(export_calc, xyz, z, int(manifest["formal_charge"]), forces=True)
        used = COUNTS["molecule_evaluations"]
        features = array(result, "aim")
        if features.shape != (len(z), CHANNELS) or not np.isfinite(features).all():
            raise ValueError(f"{manifest['id']}: expected finite (N,{CHANNELS}) AIM features")
        receipt["checks"].append({"name": "finite 256-channel features", "passed": True})
        if array(result, "energy").size == 0 or not np.isfinite(array(result, "energy")).all():
            raise ValueError(f"{manifest['id']}: invalid scalar energy")
        receipt["checks"].append({"name": "finite scalar energy", "passed": True})
        for key in ("charges", "forces"):
            if key not in result or not np.isfinite(array(result, key)).all():
                raise ValueError(f"{manifest['id']}: missing or non-finite {key}")
        if array(result, "forces").shape != xyz.shape:
            raise ValueError(f"{manifest['id']}: force shape does not match input coordinates")
        receipt["checks"].append({"name": "finite force gradients", "passed": True})
        receipt["evaluations_used"] = used
        receipt["feature_shape_all_atoms"] = list(features.shape)
        receipt["feature_shape_heavy_atoms"] = [len(heavy), CHANNELS]
        receipt["padding_restoration"] = {"passed": features.shape[0] == len(z), "expected_rows": len(z)}
        receipt["checks"].append({"name": "single output padding restoration",
                                  "passed": features.shape[0] == len(z)})
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(outdir / "features.npz", features=features[heavy].astype(np.float32),
                            source_atom_index=np.asarray(manifest["source_atom_index"], dtype=np.int64),
                            heavy_atom_indices=heavy, atomic_numbers=z, coord_angstrom=xyz,
                            energy_native=array(result, "energy"), charges=array(result, "charges"),
                            forces_native=array(result, "forces"))
        base_results.append((manifest, z, xyz, heavy,
                             result["aim"].detach().cpu().numpy(), array(result, "energy")))

    for manifest, z, xyz, heavy, base_aim, export_energy in base_results:
        std = call(standard_calc, xyz, z, int(manifest["formal_charge"]))
        check_close(array(std, "energy"), export_energy, f"{manifest['id']}: standard/export energy",
                    _receipt_for(a.output, manifest["id"], COUNTS["molecule_evaluations"]), atol=1e-4, rtol=0.0)

        center = xyz.mean(axis=0)
        q, _ = np.linalg.qr(np.array([[0.31, 0.72, 0.63], [0.84, -0.51, 0.19], [0.44, 0.47, -0.76]]))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        rotated = (xyz - center) @ q + center
        rr = call(export_calc, rotated, z, int(manifest["formal_charge"]))
        check_close(base_aim, array(rr, "aim"), f"{manifest['id']}: rotation invariance",
                    _receipt_for(a.output, manifest["id"], COUNTS["molecule_evaluations"]))

        permutation = np.random.default_rng(1729).permutation(len(z))
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(len(z))
        pr = call(export_calc, xyz[permutation], z[permutation], int(manifest["formal_charge"]))
        check_close(base_aim[heavy], array(pr, "aim")[inverse[heavy]],
                    f"{manifest['id']}: atom permutation equivariance",
                    _receipt_for(a.output, manifest["id"], COUNTS["molecule_evaluations"]))

    z_cat = np.concatenate([r[1] for r in base_results])
    xyz_cat = np.concatenate([r[2] for r in base_results])
    mol_idx = np.concatenate([np.full(len(r[1]), i, dtype=np.int32) for i, r in enumerate(base_results)])
    charges = [int(r[0]["formal_charge"]) for r in base_results]
    batch_data = {"coord": torch.as_tensor(xyz_cat, dtype=torch.float32, device=DEVICE),
                  "numbers": torch.as_tensor(z_cat, dtype=torch.int32, device=DEVICE),
                  "charge": torch.as_tensor(charges, dtype=torch.float32, device=DEVICE),
                  "mol_idx": torch.as_tensor(mol_idx, dtype=torch.int32, device=DEVICE)}
    batch = run_calc(export_calc, batch_data, molecule_evaluations=len(base_results))
    batch_aim = array(batch, "aim")
    batch_energy = array(batch, "energy")
    if batch_aim.shape != (len(z_cat), CHANNELS):
        raise ValueError("Batched AIM output does not restore the concatenated atom rows")
    expected_energy = np.concatenate([r[5].reshape(-1) for r in base_results])
    check_close(expected_energy, batch_energy.reshape(-1), "individual/batch energies",
                _receipt_for(a.output, manifests[0]["id"], COUNTS["molecule_evaluations"]), atol=1e-4, rtol=0.0)
    offset = 0
    for manifest, z, xyz, heavy, base_aim, export_energy in base_results:
        check_close(base_aim, batch_aim[offset:offset + len(z)],
                    f"{manifest['id']}: individual/batch agreement",
                    _receipt_for(a.output, manifest["id"], COUNTS["molecule_evaluations"]))
        offset += len(z)

    # Test dense batch padding. Zero rows represent padding and must not change outputs.
    max_n = max(len(row[1]) for row in base_results)
    dense_xyz = np.zeros((3, max_n, 3), dtype=np.float32)
    dense_z = np.zeros((3, max_n), dtype=np.int32)
    for i, (_, z, xyz, *_tail) in enumerate(base_results):
        dense_xyz[i, :len(z)] = xyz
        dense_z[i, :len(z)] = z
    dense_data = {"coord": torch.as_tensor(dense_xyz, dtype=torch.float32, device=DEVICE),
                  "numbers": torch.as_tensor(dense_z, dtype=torch.int32, device=DEVICE),
                  "charge": torch.as_tensor(charges, dtype=torch.float32, device=DEVICE)}
    dense = run_calc(export_calc, dense_data, molecule_evaluations=3)
    dense_aim = array(dense, "aim")
    if dense_aim.shape != (3, max_n, CHANNELS):
        raise ValueError("Dense padded output does not preserve expected batch and atom dimensions")
    dense_energy = array(dense, "energy").reshape(3, -1)
    for i, (manifest, z, xyz, heavy, base_aim, export_energy) in enumerate(base_results):
        pad_rows = dense_aim[i, len(z):]
        if pad_rows.size and not np.allclose(pad_rows, 0.0, atol=0.0, rtol=0.0):
            raise ValueError(f"{manifest['id']}: dense padding rows were not zeroed")
        check_close(base_aim, dense_aim[i, :len(z)], f"{manifest['id']}: padded batch features",
                    _receipt_for(a.output, manifest["id"], COUNTS["molecule_evaluations"]))
        check_close(np.asarray(export_energy).reshape(-1), dense_energy[i],
                    f"{manifest['id']}: padded batch energy",
                    _receipt_for(a.output, manifest["id"], COUNTS["molecule_evaluations"]), atol=1e-4, rtol=0.0)

    used = COUNTS["molecule_evaluations"]
    for manifest, *_ in base_results:
        receipt_path = a.output / str(manifest["id"]) / "smoke_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt.update({"status": "passed", "evaluations_used_total": used,
                       "padding_restoration": {"passed": True},
                       "force_gradient": {"passed": True, "parameters_frozen": True}})
        receipt.pop("receipt_path", None)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    global_receipt = json.loads(GLOBAL_RECEIPT.read_text(encoding="utf-8"))
    global_receipt.update(COUNTS)
    global_receipt["status"] = "passed"
    GLOBAL_RECEIPT.write_text(json.dumps(global_receipt, indent=2) + "\n", encoding="utf-8")
    print(f"3-ligand smoke passed with {used}/{MAX_EVALUATIONS} molecule evaluations")


def _receipt_for(outdir: Path, molecule_id: str, used: int) -> dict:
    path = outdir / str(molecule_id) / "smoke_receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["evaluations_used"] = used
    receipt["receipt_path"] = str(path)
    return receipt


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        out = None
        if "--output" in sys.argv:
            try:
                out = Path(sys.argv[sys.argv.index("--output") + 1])
            except (IndexError, ValueError):
                pass
        if out is not None and out.exists():
            global_path = out / "smoke_global_receipt.json"
            if global_path.is_file():
                data = json.loads(global_path.read_text(encoding="utf-8"))
                data["status"] = "failed"
                data["failure"] = f"{type(exc).__name__}: {exc}"
                global_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            for receipt_path in out.rglob("smoke_receipt.json"):
                data = json.loads(receipt_path.read_text(encoding="utf-8"))
                if data.get("status") == "running":
                    data["status"] = "failed"
                    data["failure"] = f"{type(exc).__name__}: {exc}"
                    data.pop("receipt_path", None)
                    receipt_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        raise
