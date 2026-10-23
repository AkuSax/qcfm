"""Validate AIM preparation inputs without chemistry minimization or model calls.

The default command checks paths, hashes, cohort IDs, and configuration only.
--prepare is a future action. It creates bound/free hydrogenated AIM manifests.
It never runs AIMNet2. Do not use --prepare until preparation is authorized.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "preparation" / "20261022T182444Z"
EXPORTS = PACKAGE / "arm1_atoms" / "safe_exports"
COVERAGE = PACKAGE / "arm2_chemistry" / "per_record_coverage.csv"
PROTOCOL = PACKAGE / "arm4_protocol" / "comparison_config.json"
ENV = PACKAGE / "arm3_environment"
OUT = Path(__file__).resolve().parent / "prepared_manifests"
HELD = {"660560", "3243170", "6375556"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inputs():
    for p in (EXPORTS, COVERAGE, PROTOCOL, ENV / "extract_aim_features.py", ENV / "dependency_receipt.json", ENV / "import_receipt.json"):
        if not p.exists():
            raise FileNotFoundError(p)
    with COVERAGE.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_id = {str(r["pubchem_cid"]): r for r in rows}
    if len(by_id) != 65:
        raise ValueError(f"Expected 65 unique chemistry rows, got {len(by_id)}")
    sidecars = sorted(EXPORTS.glob("*.json"))
    ids = {p.stem for p in sidecars}
    if len(ids) != 62 or ids & HELD:
        raise ValueError(f"Expected 62 compatible exports; got {len(ids)}")
    for cid in sorted(ids):
        meta = json.loads((EXPORTS / f"{cid}.json").read_text(encoding="utf-8"))
        npz = EXPORTS / f"{cid}.npz"
        if not npz.is_file() or sha256(npz) != meta["npz_sha256"]:
            raise ValueError(f"Export hash mismatch: {cid}")
        if cid not in by_id or meta["source_smiles"] != by_id[cid]["smiles"]:
            raise ValueError(f"Frozen source graph mismatch: {cid}")
        if not (meta["graph_match_count"] >= 1 and meta["shapes"]["x_pred_heavy_coords"][1] == 3):
            raise ValueError(f"Invalid atom-join sidecar: {cid}")
        with np.load(npz, allow_pickle=False) as data:
            required = {"source_atom_indices", "x_pred_heavy_coords", "atomic_numbers"}
            if not required.issubset(data.files):
                raise ValueError(f"Missing numeric export arrays: {cid}")
            n = len(data["source_atom_indices"])
            if data["x_pred_heavy_coords"].shape != (n, 3) or len(data["atomic_numbers"]) != n:
                raise ValueError(f"Array shape mismatch: {cid}")
            if not np.isfinite(data["x_pred_heavy_coords"]).all():
                raise ValueError(f"Non-finite coordinates: {cid}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["feature_manifest_required_before_fit"]["file"] != "aim_features.npz":
        raise ValueError("Arm4 feature bundle name changed")
    dep = json.loads((ENV / "dependency_receipt.json").read_text(encoding="utf-8"))
    imp = json.loads((ENV / "import_receipt.json").read_text(encoding="utf-8"))
    if dep.get("expected_final_aim_channels") != 256 or imp.get("model_initialized") or imp.get("weights_loaded"):
        raise ValueError("Environment receipt failed its safe-import contract")
    wr = json.loads((ENV / "weight_stage_receipt.json").read_text(encoding="utf-8"))
    weight = ENV / "weights" / "aimnet2_wb97m_d3_0.pt"
    if not weight.is_file() or weight.stat().st_size != 8836941 or sha256(weight) != "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28":
        raise ValueError("Pinned local weight does not match its expected receipt")
    if dep.get("weight", {}).get("hash_matches") is not True or dep.get("weight", {}).get("stage_receipt_verified") is not True or wr.get("sha256") != sha256(weight):
        raise ValueError("Weight staging receipt does not verify the pinned weight")
    return ids, rows, protocol


def prepare(ids, rows, outdir, include_free=False):
    # This future path performs fixed heavy-atom hydrogen placement and ETKDG/MMFF work.
    # It writes no model features and never loads AIMNet2.
    import sys
    sys.path.insert(0, str(PACKAGE / "orchestrator"))
    from chemistry_prepare import bound_geometry, free_geometry, source_graph
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    by_id = {str(r["pubchem_cid"]): r for r in rows}
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    elem_names = ["C", "N", "O", "S", "F", "Cl", "Br", "H"]
    for cid in sorted(ids, key=int):
        try:
            meta = json.loads((EXPORTS / f"{cid}.json").read_text(encoding="utf-8"))
            with np.load(EXPORTS / f"{cid}.npz", allow_pickle=False) as data:
                source_idx = data["source_atom_indices"].astype(int).tolist()
                xyz = data["x_pred_heavy_coords"].astype(float)
                zheavy = data["atomic_numbers"].astype(int)
            smiles = by_id[cid]["smiles"]
            bound = bound_geometry(smiles, source_idx, xyz)
            free = free_geometry(smiles) if include_free else None
            if bound.get("status") != "PASS":
                raise RuntimeError(f"Bound hydrogen placement did not converge: mmff_status={bound.get('mmff_status')}")
            if include_free and free.get("status") != "PASS":
                raise RuntimeError(f"Free-conformer MMFF94s did not converge: trials={free.get('trials')}")
            mol = source_graph(smiles)
            if int(Chem.GetFormalCharge(mol)) != int(by_id[cid]["total_formal_charge"]):
                raise ValueError("Formal charge differs from frozen chemistry audit")
            elem_counts = {el: int(sum(a.GetSymbol() == el for a in mol.GetAtoms())) for el in elem_names}
            elem_counts["H"] = int(sum(a.GetTotalNumHs() for a in mol.GetAtoms()))
            names_common = ["formal_charge", "heavy_atom_count", "molecular_weight", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds", "ring_count"] + [f"count_{el}" for el in elem_names]
            vals_common = [int(Chem.GetFormalCharge(mol)), mol.GetNumHeavyAtoms(), float(Descriptors.MolWt(mol)), float(Crippen.MolLogP(mol)), float(rdMolDescriptors.CalcTPSA(mol)), int(rdMolDescriptors.CalcNumHBD(mol)), int(rdMolDescriptors.CalcNumHBA(mol)), int(rdMolDescriptors.CalcNumRotatableBonds(mol)), int(rdMolDescriptors.CalcNumRings(mol))] + [elem_counts[e] for e in elem_names]
            if include_free:
                scalar_names = ["mmff94s_bound_minus_free_kcal_mol"] + names_common
                scalar_values = [float(bound["mmff_energy_kcal_mol"] - free["mmff_energy_kcal_mol"])] + vals_common
            else:
                scalar_names = ["mmff94s_bound_energy_kcal_mol"] + names_common
                scalar_values = [float(bound["mmff_energy_kcal_mol"])] + vals_common
            receipt_pair = {"status": "PASS", "cid": cid, "source_smiles": smiles,
                "stage": "bound_and_free" if include_free else "bound_hydrogens_only", "free_conformer_pending": not include_free,
                "formal_charge": int(Chem.GetFormalCharge(mol)), "hydrogenated_atom_count": int(len(bound["atomic_numbers"])),
                "simple_scalar_names": scalar_names, "simple_scalars": scalar_values, "element_counts": elem_counts,
                "bound": {k:v for k,v in bound.items() if k not in ("coordinates_angstrom", "atomic_numbers")},
                "free": ({k:v for k,v in free.items() if k not in ("coordinates_angstrom", "atomic_numbers")} if free is not None else None)}
            states = [("bound", bound)] + ([ ("free", free) ] if include_free else [])
            for state, rec in states:
                z = np.asarray(rec["atomic_numbers"], dtype=int)
                heavy = np.asarray(rec["heavy_atom_indices"], dtype=int)
                source_map = np.asarray(source_idx if state == "bound" else [i for i, v in enumerate(z) if v > 1], dtype=int)
                if len(source_map) != len(heavy) or not np.array_equal(z[heavy], zheavy):
                    raise ValueError(f"Heavy atom source map or element mismatch: {state}")
                if state == "bound" and not np.array_equal(np.asarray(rec["coordinates_angstrom"])[heavy], xyz):
                    raise ValueError("Bound heavy-atom coordinates changed")
                manifest = {"id": f"{cid}_{state}", "formal_charge": int(rec["formal_charge"]),
                    "atomic_numbers": z.tolist(), "coord_angstrom": np.asarray(rec["coordinates_angstrom"]).tolist(),
                    "heavy_atom_indices": heavy.tolist(), "source_atom_index": source_map.tolist(),
                    "parent_cid": cid, "state": state, "graph_sha256": rec["graph_sha256"],
                    "rdkit": rec["rdkit"], "seed": rec["seed"], "max_steps": rec["max_steps"], "coordinate_unit": "angstrom", "multiplicity": 1}
                mp = outdir / f"{cid}_{state}.json"
                mp.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
                receipt_pair[state]["manifest_sha256"] = sha256(mp)
                written.append(str(mp))
            (outdir / f"{cid}_preparation_receipt.json").write_text(json.dumps(receipt_pair, indent=2)+"\n", encoding="utf-8")
        except Exception as exc:
            failure = {"status":"BLOCKED", "cid":cid, "failure_code":type(exc).__name__, "reason":str(exc)}
            (outdir / f"{cid}_preparation_failure.json").write_text(json.dumps(failure,indent=2)+"\n",encoding="utf-8")
            raise
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Validate files and settings without minimizing molecules or loading models")
    ap.add_argument("--prepare", action="store_true", help="Prepare selected bound hydrogenated AIM input manifests; no AIM model calls")
    ap.add_argument("--cids", nargs="+", help="Explicit compatible CIDs. Defaults to the three lowest numeric compatible CIDs.")
    ap.add_argument("--all-compatible", action="store_true", help="Select all 62 compatible CIDs after the three-ligand smoke passes")
    ap.add_argument("--include-free", action="store_true", help="Also prepare free ETKDGv3/MMFF94s conformers; use only in a later bounded stage")
    ap.add_argument("--output-dir", type=Path, default=OUT)
    ap.add_argument("--smoke-receipt", type=Path, help="Require a passing three-CID smoke before full-cohort preparation")
    a = ap.parse_args()
    if not (a.dry_run or a.prepare):
        ap.error("Choose --dry-run or --prepare")
    ids, rows, _ = inputs()
    selected = sorted(ids, key=int)[:3] if not a.cids and not a.all_compatible else (set(a.cids) if a.cids else ids)
    if a.cids and a.all_compatible: ap.error("Choose --cids or --all-compatible")
    if not set(selected).issubset(ids): ap.error("Requested CID is not among 62 compatible exports")
    if a.prepare and (a.all_compatible or a.include_free):
        if not a.smoke_receipt or not a.smoke_receipt.is_file(): ap.error("Full or free-conformer preparation requires a passing --smoke-receipt")
        smoke=json.loads(a.smoke_receipt.read_text(encoding="utf-8"))
        if smoke.get("status")!="passed" or smoke.get("input_ids")!=["57341_bound","68261_bound","570059_bound"] or smoke.get("molecule_evaluations",999)>18 or smoke.get("api_calls",999)>14: ap.error("Smoke receipt does not pass the frozen three-CID gate")
    if a.prepare:
        out = prepare(selected, rows, a.output_dir, include_free=a.include_free)
        print(json.dumps({"status": "PASS", "prepared_records": len(selected), "manifests": len(out), "model_calls": 0, "files": out}, indent=2))
    else:
        print(json.dumps({"status": "PASS", "compatible_records": len(ids), "selected_smoke_cids": sorted(ids, key=int)[:3], "held_records": sorted(HELD), "selected_smoke_cids": sorted(ids, key=int)[:3], "expected_bound_manifests": len(sorted(ids, key=int)[:3]), "expected_free_manifests_now": 0, "model_calls": 0, "minimization": "not run", "feature_bundle": "aim_features.npz; arm4 schema", "next_action": "Run --prepare for the three smoke CIDs. This performs fixed-heavy-atom hydrogen placement only. Model extraction remains separately unauthorized."}, indent=2))

if __name__ == "__main__":
    main()


