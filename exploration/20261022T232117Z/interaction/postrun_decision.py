"""Evaluate the frozen fusion decision from saved predictions only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "exploration" / "20261022T232117Z"
PLAN = RUN / "FROZEN_PLAN.md"
ARM = ROOT / "preparation" / "20261022T182444Z" / "arm4_protocol"
PRIOR = ROOT / "preparation" / "20261022T182444Z" / "execution_full" / "max1000_20261022T203000Z" / "fit_retry01"
FEATURES = PRIOR.parent / "aim_features.npz"
DEFAULT_PREDICTIONS = RUN / "classical" / "predictions.csv"
BOOTSTRAP_SEED = 1729
BOOTSTRAP_DRAWS = 5000
MODEL_SEEDS = ("1729", "2718", "31415")
NEW_METHODS = (
    "boltz", "aim", "chem", "additive_q", "additive_chem", "kernel_q",
    "kernel_chem", "stack_q", "stack_chem", "lowrank_q", "lowrank_chem",
    "selected_fusion",
)
PRIOR_METHODS = {
    "native": "native",
    "prior_boltz_ridge": "boltz_ridge",
    "prior_ordinary_readout": "ordinary_readout",
    "prior_quantum_only": "quantum_only",
}
MANDATORY = (
    "prior_boltz_ridge", "prior_ordinary_readout", "prior_quantum_only",
    "boltz", "aim", "chem",
)
Q_CHEM_PAIRS = (("additive_q", "additive_chem"), ("kernel_q", "kernel_chem"),
                ("stack_q", "stack_chem"), ("lowrank_q", "lowrank_chem"))
EXPECTED_HASHES = {
    "aim_features.npz": "ee7f4c5e74f9bb3b1950c98395249948f0910348858fea0df3f4384e195fbd34",
    "record_manifest.csv": "e95fb518ecd2f52b7d65d8b4bc1a76519af5076cfda5e8c6931054763ed9ce23",
    "primary_pairs.csv": "0e18db377566cc8c65ef71c66cc24ebe301b3a4dacd4d78364bea7f2acf1cbe3",
    "sensitivity_pairs.csv": "d07ef017178fdc2baa167ba246ee560aa1de12645a2f63ddbd70611e309c2635",
    "prior_predictions.csv": "61727734edf75ec208284f0856a21a738d343bbe5b40a8f22ad980c52ef75eb3",
    "prior_fit_manifest.csv": "ed5fc30022a670fde2677149c2f4ff3eb9704c5616b62c7a30170bab0a4b3dd6",
}
EXPECTED_PLAN_SHA256 = "530a7ff532fbc56b5291e9a73e4ffb0a40689a0623fced5172656f897d1d502c"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm_dataset(value: str) -> str:
    value = value.strip().lower()
    return "sensitivity" if value in {"sensitivity", "hill_sensitivity", "fixed_hill"} else value


def is_true(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def finite(value: str | None) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def fmt(value: Any) -> str:
    return f"{float(value):.6f}" if value is not None and finite(str(value)) else "undefined"


def rank_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def spearman(y: np.ndarray, pred: np.ndarray) -> float:
    ry, rp = rank_average(y), rank_average(pred)
    if np.std(ry) == 0 or np.std(rp) == 0:
        return float("nan")
    return float(np.corrcoef(ry, rp)[0, 1])


def kendall_tau_b(y: np.ndarray, pred: np.ndarray) -> float:
    concordant = discordant = tie_y = tie_p = 0
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            sy, sp = np.sign(y[i] - y[j]), np.sign(pred[i] - pred[j])
            if sy == 0 and sp == 0:
                continue
            if sy == 0:
                tie_y += 1
            elif sp == 0:
                tie_p += 1
            elif sy == sp:
                concordant += 1
            else:
                discordant += 1
    den = math.sqrt((concordant + discordant + tie_p) * (concordant + discordant + tie_y))
    return float((concordant - discordant) / den) if den else float("nan")


def metrics(y_map: dict[str, float], p_map: dict[str, float],
            pairs: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, float]]:
    ids = sorted(set(y_map) & set(p_map))
    if not ids or set(ids) != set(y_map):
        raise ValueError(f"Prediction coverage mismatch: expected {len(y_map)}, got {len(ids)}")
    y = np.asarray([y_map[c] for c in ids], dtype=float)
    pred = np.asarray([p_map[c] for c in ids], dtype=float)
    group_errors: dict[str, list[float]] = {}
    pair_errors: list[float] = []
    order, exact, cliff = [], [], []
    for pair in pairs:
        a, b = pair["cid_a"], pair["cid_b"]
        dy = y_map[a] - y_map[b]
        dp = p_map[a] - p_map[b]
        err = dp - dy
        group_errors.setdefault(pair["chemical_group"], []).append(err)
        pair_errors.append(err)
        if dy != 0:
            order.append(1.0 if np.sign(dp) == np.sign(dy) else (0.5 if dp == 0 else 0.0))
        exact.append(float(np.sign(dp) == np.sign(dy)))
        if abs(dy) >= 0.3:
            cliff.append(float(np.sign(dp) == np.sign(dy)))
    if not group_errors:
        raise ValueError("No eligible pairs are available")
    group_mae = {g: float(np.mean(np.abs(e))) for g, e in group_errors.items()}
    values = np.asarray(pair_errors, dtype=float)
    out = {
        "n_records": len(ids), "n_groups_with_pairs": len(group_errors), "n_pairs": len(values),
        "macro_group_pair_mae": float(np.mean(list(group_mae.values()))),
        "pair_rmse": float(np.sqrt(np.mean(values ** 2))),
        "macro_group_pair_rmse": float(np.mean([np.sqrt(np.mean(np.square(e))) for e in group_errors.values()])),
        "ligand_mae": float(np.mean(np.abs(pred - y))),
        "ligand_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "spearman_rho": spearman(y, pred), "kendall_tau_b": kendall_tau_b(y, pred),
        "pair_ordering_accuracy": float(np.mean(order)) if order else float("nan"),
        "exact_sign_accuracy": float(np.mean(exact)) if exact else float("nan"),
        "sign_accuracy_abs_delta_ge_0_3": float(np.mean(cliff)) if cliff else float("nan"),
        "coverage_source_rows": 65, "coverage_predicted_cohort": len(ids),
    }
    return out, group_mae


def group_bootstrap(candidate: dict[str, float], comparator: dict[str, float],
                    draws: int = BOOTSTRAP_DRAWS) -> dict[str, Any]:
    groups = sorted(set(candidate) & set(comparator))
    if len(groups) != 8:
        raise ValueError(f"Expected 8 eligible groups for bootstrap, got {len(groups)}")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(groups), size=(draws, len(groups)))
    differences = np.asarray([
        np.mean([comparator[groups[j]] - candidate[groups[j]] for j in row])
        for row in idx
    ])
    return {
        "bootstrap_seed": BOOTSTRAP_SEED, "draws": draws,
        "gain_mean": float(np.mean([comparator[g] - candidate[g] for g in groups])),
        "gain_ci95": [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))],
        "interval_scope": "fixed-prediction paired chemical-group bootstrap; model-refit uncertainty omitted",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=RUN / "interaction" / "postrun")
    parser.add_argument("--arm-dir", type=Path, default=ARM)
    parser.add_argument("--prior-dir", type=Path, default=PRIOR)
    parser.add_argument("--features", type=Path, default=FEATURES)
    args = parser.parse_args()

    record_path = args.arm_dir / "record_manifest.csv"
    pair_paths = {"primary": args.arm_dir / "primary_pairs.csv",
                  "sensitivity": args.arm_dir / "sensitivity_pairs.csv"}
    prior_path = args.prior_dir / "predictions.csv"
    fit_manifest_path = args.prior_dir / "outer_fit_manifest.csv"
    hashes = {"record_manifest.csv": file_sha256(record_path),
              "primary_pairs.csv": file_sha256(pair_paths["primary"]),
              "sensitivity_pairs.csv": file_sha256(pair_paths["sensitivity"]),
              "prior_predictions.csv": file_sha256(prior_path),
              "prior_fit_manifest.csv": file_sha256(fit_manifest_path),
              "aim_features.npz": file_sha256(args.features)}
    if hashes != EXPECTED_HASHES:
        raise ValueError(f"Frozen input hash mismatch: {hashes}")
    plan_hash = file_sha256(PLAN)
    if plan_hash != EXPECTED_PLAN_SHA256:
        raise ValueError("Frozen plan hash changed")

    records = read_csv(record_path)
    record_by_id = {r["pubchem_cid"]: r for r in records}
    if len(records) != 65 or len(record_by_id) != 65:
        raise ValueError("Record manifest must contain 65 unique source IDs")
    with np.load(args.features, allow_pickle=False) as z:
        feature_cids = [str(x) for x in z["cids"].tolist()]
        feature_valid = z["feature_valid"].astype(bool)
        atom_counts = z["atom_mask"].astype(bool).sum(axis=1)
    if feature_cids != [r["pubchem_cid"] for r in records] or len(feature_valid) != 65:
        raise ValueError("Feature bundle IDs or rows differ from the frozen record manifest")
    valid_by_id = dict(zip(feature_cids, feature_valid.tolist()))
    if set(cid for cid, valid in valid_by_id.items() if not valid) != {"660560", "3243170", "6375556"}:
        raise ValueError("Feature-valid holds differ from the frozen three stereochemistry holds")
    if np.any(feature_valid & (atom_counts == 0)):
        raise ValueError("A feature-valid row has no active atoms")
    labels = {cid: float(r["label_log10_ic50_uM"]) for cid, r in record_by_id.items()
              if is_true(r["admitted"]) and valid_by_id[cid]}
    valid_primary = set(labels)
    valid_sensitivity = {cid for cid in valid_primary if not is_true(record_by_id[cid]["sensitivity_excluded"])}
    fixed_hill_excluded = valid_primary - valid_sensitivity
    if fixed_hill_excluded != {"899202", "976329"}:
        raise ValueError("Fixed-Hill exclusions differ from the frozen CIDs")
    datasets = {"primary": valid_primary, "sensitivity": valid_sensitivity}
    expected_counts = {"primary": (62, 136), "sensitivity": (60, 129)}
    pairs = {d: read_csv(p) for d, p in pair_paths.items()}
    for d in datasets:
        if len(datasets[d]) != expected_counts[d][0] or len(pairs[d]) != expected_counts[d][1]:
            raise ValueError(f"Frozen {d} cohort count mismatch")
        pair_ids = {p["cid_a"] for p in pairs[d]} | {p["cid_b"] for p in pairs[d]}
        if pair_ids - datasets[d]:
            raise ValueError(f"{d} pairs contain IDs outside the cohort")

    raw = read_csv(args.predictions)
    if not raw:
        raise ValueError("Prediction file is empty")
    required = {"dataset", "method", "pubchem_cid", "outer_fold", "y_true", "prediction",
                "status", "feature_valid", "cohort_valid", "failure_reason", "selected_family",
                "matched_chemistry_family"}
    if not required.issubset(raw[0]):
        raise ValueError(f"Prediction columns missing: {sorted(required - set(raw[0]))}")
    for r in raw:
        r["dataset"] = norm_dataset(r["dataset"])
        r["model_seed"] = str(r.get("model_seed") or r.get("seed") or "")
        if not finite(r["y_true"]):
            raise ValueError(f"Candidate label is missing for {r['pubchem_cid']}")
        expected_y = float(record_by_id[r["pubchem_cid"]]["label_log10_ic50_uM"])
        if abs(float(r["y_true"]) - expected_y) > 1e-12:
            raise ValueError("Candidate label differs from the frozen record manifest")
        if r.get("feature_valid") and is_true(r["feature_valid"]) != valid_by_id[r["pubchem_cid"]]:
            raise ValueError(f"Candidate feature_valid differs from the frozen bundle for {r['pubchem_cid']}")

    # Build candidate predictions with strict 65-row coverage per method unit.
    new_rows: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = {}
    for r in raw:
        if r["dataset"] not in datasets or r["method"] not in NEW_METHODS:
            continue
        key = (r["dataset"], r["model_seed"], r["method"], r["pubchem_cid"])
        if key in new_rows:
            raise ValueError(f"Duplicate candidate prediction row: {key}")
        new_rows[key] = r
    unit_rows: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for (d, seed, method, cid), row in new_rows.items():
        unit_rows.setdefault((d, seed, method), {})[cid] = row
    for dataset in datasets:
        for seed in MODEL_SEEDS:
            for method in NEW_METHODS:
                unit = unit_rows.get((dataset, seed, method), {})
                if len(unit) != 65 or set(unit) != set(record_by_id):
                    raise ValueError(f"{dataset}/{seed}/{method} must report all 65 unique IDs")
                for cid in datasets[dataset]:
                    row = unit[cid]
                    if row["status"].lower() != "predicted" or not finite(row["prediction"]):
                        raise ValueError(f"Missing valid cohort prediction: {dataset}/{seed}/{method}/{cid}")
                    if int(row["outer_fold"]) != int(record_by_id[cid]["outer_fold"]):
                        raise ValueError(f"Outer fold mismatch for {cid}")
                for cid, row in unit.items():
                    if int(row["outer_fold"]) != int(record_by_id[cid]["outer_fold"]):
                        raise ValueError(f"Outer fold mismatch for {dataset}/{seed}/{method}/{cid}")
                    if is_true(row["feature_valid"]) != valid_by_id[cid]:
                        raise ValueError(f"feature_valid differs from the frozen bundle for {cid}")
                    if not row["cohort_valid"].strip() or is_true(row["cohort_valid"]) != (cid in datasets[dataset]):
                        raise ValueError(f"cohort_valid differs from frozen cohort for {dataset}/{seed}/{method}/{cid}")
                    if cid not in datasets[dataset]:
                        expected_reason = "feature_invalid" if not valid_by_id[cid] else "fixed_hill_sensitivity_exclusion"
                        if row["status"].lower() != "missing" or row.get("failure_reason", "").strip() != expected_reason:
                            raise ValueError(f"Excluded candidate row has an invalid status or reason: {dataset}/{seed}/{method}/{cid}")

    # Build selected-fold matched chemistry predictions from the selected family metadata.
    selected_chem_units: dict[tuple[str, str], dict[str, float]] = {}
    quantum_family_units: dict[tuple[str, str], bool] = {}
    selected_family_units: dict[tuple[str, str], dict[str, str]] = {}
    for dataset in datasets:
        for seed in MODEL_SEEDS:
            selected = unit_rows[(dataset, seed, "selected_fusion")]
            chem_pred: dict[str, float] = {}
            families: dict[str, str] = {}
            quantum_only = True
            for cid in datasets[dataset]:
                row = selected[cid]
                family = (row.get("selected_family") or row.get("selected_method") or "").strip()
                if not family:
                    raise ValueError("selected_family is required on selected_fusion rows")
                families[cid] = family
                if family.endswith("_q"):
                    chem_family = family[:-2] + "_chem"
                elif family.endswith("_chem"):
                    chem_family = family
                    quantum_only = False
                else:
                    raise ValueError(f"Unknown selected family: {family}")
                if family not in {q for q, _ in Q_CHEM_PAIRS} | {c for _, c in Q_CHEM_PAIRS}:
                    raise ValueError(f"Selected family is outside the frozen fusion set: {family}")
                selected_method_row = unit_rows.get((dataset, seed, family), {}).get(cid)
                if selected_method_row is None or abs(float(selected_method_row["prediction"]) - float(row["prediction"])) > 1e-12:
                    raise ValueError(f"selected_fusion does not copy its selected family's score: {cid}/{family}")
                declared_chem = row.get("matched_chemistry_family", "").strip()
                if declared_chem and declared_chem != chem_family:
                    raise ValueError(f"Matched chemistry family metadata mismatch for {cid}")
                chem_row = unit_rows.get((dataset, seed, chem_family), {}).get(cid)
                if chem_row is None or int(chem_row["outer_fold"]) != int(row["outer_fold"]):
                    raise ValueError(f"Matched chemistry row is missing or has a fold mismatch: {cid}")
                if chem_row["status"].lower() != "predicted" or not finite(chem_row["prediction"]):
                    raise ValueError(f"Matched chemistry prediction is missing: {cid}")
                chem_pred[cid] = float(chem_row["prediction"])
            by_fold: dict[str, set[str]] = {}
            for cid, family in families.items():
                fold = str(record_by_id[cid]["outer_fold"])
                by_fold.setdefault(fold, set()).add(family)
            if any(len(fold_families) != 1 for fold_families in by_fold.values()):
                raise ValueError(f"Selected family changes within an outer fold: {dataset}/{seed}")
            key = (dataset, seed)
            selected_chem_units[key] = chem_pred
            selected_family_units[key] = families
            quantum_family_units[key] = quantum_only

    # Load and validate fixed prior OOF predictions.
    prior_raw = read_csv(prior_path)
    prior_units: dict[tuple[str, str, str], dict[str, float]] = {}
    prior_rows: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in prior_raw:
        d = norm_dataset(row["dataset"])
        method = row["method"]
        if method not in set(PRIOR_METHODS.values()) or d not in datasets:
            continue
        seed = str(row["seed"])
        key = (d, seed, method)
        prior_rows.setdefault(key, []).append(row)
        if row["status"] == "predicted" and finite(row["prediction"]):
            unit = prior_units.setdefault(key, {})
            if row["pubchem_cid"] in unit:
                raise ValueError(f"Duplicate prior prediction identity: {key}/{row['pubchem_cid']}")
            unit[row["pubchem_cid"]] = float(row["prediction"])
        if row["pubchem_cid"] in record_by_id and int(row["outer_fold"]) != int(record_by_id[row["pubchem_cid"]]["outer_fold"]):
            raise ValueError(f"Prior baseline fold mismatch for {row['pubchem_cid']}")
    for dataset in datasets:
        for seed in MODEL_SEEDS:
            for prior_method in PRIOR_METHODS.values():
                rows = prior_rows.get((dataset, seed, prior_method), [])
                expected_source_ids = set(record_by_id)
                if len(rows) != len(expected_source_ids) or {r["pubchem_cid"] for r in rows} != expected_source_ids:
                    raise ValueError(f"Prior {prior_method} source identity set mismatch for {dataset}/{seed}")
                predicted_ids = set(prior_units.get((dataset, seed, prior_method), {}))
                expected_predicted = (
                    set(record_by_id) if prior_method == "native" and dataset == "primary"
                    else (set(record_by_id) - fixed_hill_excluded) if prior_method == "native"
                    else datasets[dataset]
                )
                if predicted_ids != expected_predicted:
                    raise ValueError(f"Prior {prior_method} predicted identity set mismatch for {dataset}/{seed}")

    # Metric table: all new families and all prior fixed comparators.
    metric_rows: list[dict[str, Any]] = []
    group_mae_by_unit: dict[tuple[str, str, str], dict[str, float]] = {}
    predictions_by_unit: dict[tuple[str, str, str], dict[str, float]] = {}
    for dataset, cohort in datasets.items():
        y_map = {cid: labels[cid] for cid in cohort}
        for seed in MODEL_SEEDS:
            methods: dict[str, dict[str, float]] = {}
            for method in NEW_METHODS:
                unit = unit_rows[(dataset, seed, method)]
                methods[method] = {cid: float(unit[cid]["prediction"]) for cid in cohort}
            methods["matched_selected_chem"] = selected_chem_units[(dataset, seed)]
            for output_name, prior_method in PRIOR_METHODS.items():
                p_all = prior_units.get((dataset, seed, prior_method), {})
                if not cohort.issubset(p_all):
                    raise ValueError(f"Prior {prior_method} coverage mismatch for {dataset}/{seed}")
                p = {cid: p_all[cid] for cid in cohort}
                methods[output_name] = p
            for method, pred in methods.items():
                score, group_losses = metrics(y_map, pred, pairs[dataset])
                group_mae_by_unit[(dataset, seed, method)] = group_losses
                predictions_by_unit[(dataset, seed, method)] = pred
                metric_rows.append({"dataset": dataset, "model_seed": int(seed), "method": method,
                                    "bootstrap_seed": BOOTSTRAP_SEED,
                                    "per_group_macro_pair_mae": group_losses, **score})

    metric_means: list[dict[str, Any]] = []
    metric_names = ["macro_group_pair_mae", "pair_rmse", "macro_group_pair_rmse", "ligand_mae",
                    "ligand_rmse", "spearman_rho", "kendall_tau_b", "pair_ordering_accuracy",
                    "exact_sign_accuracy", "sign_accuracy_abs_delta_ge_0_3"]
    for dataset in datasets:
        methods = sorted({r["method"] for r in metric_rows if r["dataset"] == dataset})
        for method in methods:
            rows = [r for r in metric_rows if r["dataset"] == dataset and r["method"] == method]
            mean = {m: float(np.nanmean([r[m] for r in rows])) for m in metric_names}
            ranges = {m: [float(np.nanmin([r[m] for r in rows])), float(np.nanmax([r[m] for r in rows]))]
                      for m in metric_names}
            metric_means.append({"dataset": dataset, "method": method, "n_model_seeds": len(rows),
                                 "mean": mean, "seed_range_min_max": ranges})

    # Pair each selected prediction with mandatory comparators by model seed.
    comparison_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    gates_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        seed_comparisons: dict[str, list[dict[str, Any]]] = {m: [] for m in MANDATORY + ("matched_selected_chem",)}
        selected_seed_gains: dict[str, list[float]] = {m: [] for m in seed_comparisons}
        seed_lower: dict[str, list[float]] = {m: [] for m in seed_comparisons}
        seed_relative: dict[str, list[float]] = {m: [] for m in MANDATORY}
        seed_rho_gain: dict[str, list[float]] = {m: [] for m in MANDATORY}
        sensitivity_gains: dict[str, list[float]] = {m: [] for m in MANDATORY}
        for seed in MODEL_SEEDS:
            selected = predictions_by_unit[(dataset, seed, "selected_fusion")]
            selected_metrics, selected_group = metrics({c: labels[c] for c in datasets[dataset]}, selected, pairs[dataset])
            selected_families = selected_family_units[(dataset, seed)]
            all_q = all(f.endswith("_q") for f in selected_families.values())
            for comparator in MANDATORY + ("matched_selected_chem",):
                cmp_pred = predictions_by_unit[(dataset, seed, comparator)]
                cmp_metrics, cmp_group = metrics({c: labels[c] for c in datasets[dataset]}, cmp_pred, pairs[dataset])
                boot = group_bootstrap(selected_group, cmp_group)
                gain = cmp_metrics["macro_group_pair_mae"] - selected_metrics["macro_group_pair_mae"]
                row = {
                    "analysis_type": "selected_acceptance", "dataset": dataset,
                    "model_seed": int(seed), "method": "selected_fusion", "selected_method": "selected_fusion",
                    "selected_families_by_cid": selected_families,
                    "comparator": comparator,
                    "selected_primary_mae": selected_metrics["macro_group_pair_mae"],
                    "comparator_primary_mae": cmp_metrics["macro_group_pair_mae"],
                    "absolute_primary_gain": gain,
                    "relative_primary_gain": gain / cmp_metrics["macro_group_pair_mae"] if cmp_metrics["macro_group_pair_mae"] else float("nan"),
                    "spearman_gain": selected_metrics["spearman_rho"] - cmp_metrics["spearman_rho"],
                    "sensitivity_primary_gain": None,
                    "matched_chemistry_families": {cid: (f[:-2] + "_chem" if f.endswith("_q") else f)
                                                      for cid, f in selected_families.items()},
                    "quantum_family_selected_in_all_outer_folds_and_seeds": all_q,
                    **boot,
                }
                comparison_rows.append(row)
                bootstrap_rows.append({
                    "analysis_type": "selected_acceptance", "dataset": dataset,
                    "model_seed": int(seed), "bootstrap_seed": BOOTSTRAP_SEED,
                    "method": "selected_fusion", "comparator": comparator,
                    "group_count": 8, "draws": BOOTSTRAP_DRAWS,
                    "gain_mean": boot["gain_mean"], "ci95_lower": boot["gain_ci95"][0],
                    "ci95_upper": boot["gain_ci95"][1], "scope": boot["interval_scope"],
                })
                seed_comparisons[comparator].append(row)
                selected_seed_gains[comparator].append(gain)
                seed_lower[comparator].append(boot["gain_ci95"][0])
                if comparator in MANDATORY:
                    seed_relative[comparator].append(row["relative_primary_gain"])
                    seed_rho_gain[comparator].append(row["spearman_gain"])
            # These family comparisons are descriptive. They do not affect selection.
            for q_method, chem_method in Q_CHEM_PAIRS:
                q_pred = predictions_by_unit[(dataset, seed, q_method)]
                q_metrics, q_group = metrics({c: labels[c] for c in datasets[dataset]}, q_pred, pairs[dataset])
                for comparator in MANDATORY + (chem_method,):
                    cmp_pred = predictions_by_unit[(dataset, seed, comparator)]
                    cmp_metrics, cmp_group = metrics({c: labels[c] for c in datasets[dataset]}, cmp_pred, pairs[dataset])
                    boot = group_bootstrap(q_group, cmp_group)
                    gain = cmp_metrics["macro_group_pair_mae"] - q_metrics["macro_group_pair_mae"]
                    comparison_rows.append({
                        "analysis_type": "descriptive_family_comparison", "dataset": dataset,
                        "model_seed": int(seed), "method": q_method, "comparator": comparator,
                        "paired_chemistry_family": chem_method,
                        "method_primary_mae": q_metrics["macro_group_pair_mae"],
                        "comparator_primary_mae": cmp_metrics["macro_group_pair_mae"],
                        "absolute_primary_gain": gain,
                        "relative_primary_gain": gain / cmp_metrics["macro_group_pair_mae"] if cmp_metrics["macro_group_pair_mae"] else float("nan"),
                        "spearman_gain": q_metrics["spearman_rho"] - cmp_metrics["spearman_rho"],
                        "bootstrap_seed": BOOTSTRAP_SEED,
                        "bootstrap_draws": BOOTSTRAP_DRAWS,
                        "bootstrap_gain_mean": boot["gain_mean"],
                        "bootstrap_gain_ci95": boot["gain_ci95"],
                        "interval_scope": boot["interval_scope"],
                    })
                    bootstrap_rows.append({
                        "analysis_type": "descriptive_family_comparison", "dataset": dataset,
                        "model_seed": int(seed), "bootstrap_seed": BOOTSTRAP_SEED,
                        "method": q_method, "comparator": comparator,
                        "group_count": 8, "draws": BOOTSTRAP_DRAWS,
                        "gain_mean": boot["gain_mean"], "ci95_lower": boot["gain_ci95"][0],
                        "ci95_upper": boot["gain_ci95"][1], "scope": boot["interval_scope"],
                    })
        gates: dict[str, Any] = {}
        for comparator in MANDATORY:
            comp_mean = next(r["mean"]["macro_group_pair_mae"] for r in metric_means
                             if r["dataset"] == dataset and r["method"] == comparator)
            sel_mean = next(r["mean"]["macro_group_pair_mae"] for r in metric_means
                            if r["dataset"] == dataset and r["method"] == "selected_fusion")
            mean_gain = comp_mean - sel_mean
            rel_gain = mean_gain / comp_mean if comp_mean else float("nan")
            gates[comparator] = {
                "equal_coverage": True,
                "mean_absolute_gain_at_least_0_10": mean_gain >= 0.10,
                "mean_relative_gain_at_least_10_percent": rel_gain >= 0.10,
                "positive_gain_every_seed": all(x > 0 for x in selected_seed_gains[comparator]),
                "bootstrap_lower_bound_above_zero_every_seed": all(x > 0 for x in seed_lower[comparator]),
                "spearman_regression_no_more_than_0_05_every_seed": all(x >= -0.05 for x in seed_rho_gain[comparator]),
                "positive_sensitivity_mean_gain": None,
                "mean_absolute_gain": mean_gain, "mean_relative_gain": rel_gain,
                "seed_absolute_gains": selected_seed_gains[comparator],
                "seed_relative_gains": seed_relative[comparator],
                "seed_spearman_gains": seed_rho_gain[comparator],
                "seed_bootstrap_lower_bounds": seed_lower[comparator],
            }
        matched = "matched_selected_chem"
        matched_family_quantum = all(
            all(f.endswith("_q") for f in selected_family_units[(dataset, s)].values())
            for s in MODEL_SEEDS
        )
        gates[matched] = {
            "quantum_family_selected_in_all_outer_folds_and_seeds": matched_family_quantum,
            "positive_primary_gain_every_seed": all(x > 0 for x in selected_seed_gains[matched]),
            "bootstrap_lower_bound_above_zero_every_seed": all(x > 0 for x in seed_lower[matched]),
            "seed_absolute_gains": selected_seed_gains[matched],
            "seed_bootstrap_lower_bounds": seed_lower[matched],
            "mean_absolute_gain": float(np.mean(selected_seed_gains[matched])),
        }
        gates_by_dataset[dataset] = {
            "all_selected_families_are_quantum_families": matched_family_quantum,
            "quantum_family_selected_in_all_outer_folds": matched_family_quantum,
            "added_quantum_claim_allowed": False,
            "mandatory_comparators": gates,
            "acceptance_pass": dataset == "primary" and matched_family_quantum and all(
                all(gates[c][k] for k in ("mean_absolute_gain_at_least_0_10",
                                           "mean_relative_gain_at_least_10_percent",
                                           "positive_gain_every_seed",
                                           "bootstrap_lower_bound_above_zero_every_seed",
                                           "spearman_regression_no_more_than_0_05_every_seed"))
                for c in MANDATORY
            ) and gates[matched]["positive_primary_gain_every_seed"]
            and gates[matched]["bootstrap_lower_bound_above_zero_every_seed"],
        }

    # Add positive mean sensitivity gains after both cohort analyses are complete.
    if "sensitivity" in datasets:
        for comparator in MANDATORY:
            gain = next(r["mean"]["macro_group_pair_mae"] for r in metric_means
                        if r["dataset"] == "sensitivity" and r["method"] == comparator) - next(
                            r["mean"]["macro_group_pair_mae"] for r in metric_means
                            if r["dataset"] == "sensitivity" and r["method"] == "selected_fusion")
            gates_by_dataset["primary"]["mandatory_comparators"][comparator]["positive_sensitivity_mean_gain"] = gain > 0
            gates_by_dataset["primary"]["mandatory_comparators"][comparator]["sensitivity_mean_gain"] = gain
        gates_by_dataset["primary"]["acceptance_pass"] = gates_by_dataset["primary"]["acceptance_pass"] and all(
            gates_by_dataset["primary"]["mandatory_comparators"][c]["positive_sensitivity_mean_gain"]
            for c in MANDATORY
        )
        gates_by_dataset["primary"]["added_quantum_claim_allowed"] = (
            gates_by_dataset["primary"]["acceptance_pass"]
            and gates_by_dataset["primary"]["quantum_family_selected_in_all_outer_folds"]
        )
        gates_by_dataset["primary"]["claim_scope"] = (
            "exploratory predictive value only; no independent quantum physics or target-transfer claim"
        )

    output = {
        "status": "complete",
        "decision_source": "saved predictions only; no estimator fits",
        "prior_promotion_result": "NO_PROMOTION retained",
        "scientific_scope": "KLK5 development only; no target-transfer or independent-confirmation claim",
        "frozen_plan": str(PLAN), "frozen_plan_sha256": plan_hash,
        "input_paths": {"predictions": str(args.predictions), "prior_predictions": str(prior_path),
                        "record_manifest": str(record_path), "primary_pairs": str(pair_paths["primary"]),
                        "sensitivity_pairs": str(pair_paths["sensitivity"]), "features": str(args.features)},
        "input_sha256": {**hashes, "candidate_predictions.csv": file_sha256(args.predictions)},
        "bootstrap": {"bootstrap_seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS,
                      "model_seeds": [int(x) for x in MODEL_SEEDS],
                      "distinct_from_model_seed": True,
                      "scope": "fixed-prediction paired chemical-group bootstrap; refit uncertainty omitted"},
        "coverage": {"denominator": 65, "primary_records": 62, "primary_pairs": 136,
                     "sensitivity_records": 60, "sensitivity_pairs": 129,
                     "eligible_pair_groups": 8,
                     "stereochemistry_holds": ["660560", "3243170", "6375556"],
                     "fixed_hill_sensitivity_excluded_cids": sorted(fixed_hill_excluded),
                     "excluded_records": [
                         {"pubchem_cid": cid,
                          "reason": "stereochemistry_hold" if not valid_by_id[cid] else "fixed_hill_sensitivity_exclusion"}
                         for cid in sorted(set(record_by_id) - valid_sensitivity)
                     ]},
        "metric_rows": metric_rows,
        "undefined_rank_statistics": [
            {"dataset": row["dataset"], "model_seed": row["model_seed"], "method": row["method"],
             "undefined": [name for name in ("spearman_rho", "kendall_tau_b") if not math.isfinite(row[name])]}
            for row in metric_rows if not math.isfinite(row["spearman_rho"]) or not math.isfinite(row["kendall_tau_b"])
        ],
        "candidate_metric_means_and_seed_ranges": metric_means,
        "selected_fusion_comparisons": [
            r for r in comparison_rows if r.get("analysis_type") == "selected_acceptance"
        ],
        "paired_comparisons": comparison_rows,
        "descriptive_quantum_family_comparisons": [
            r for r in comparison_rows if r.get("analysis_type") == "descriptive_family_comparison"
        ],
        "frozen_acceptance": gates_by_dataset,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "decision_report.json"
    json_path.write_text(json.dumps(json_safe(output), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    bootstrap_path = args.output_dir / "paired_bootstrap.csv"
    bootstrap_fields = ["analysis_type", "dataset", "model_seed", "bootstrap_seed", "method",
                        "comparator", "group_count", "draws", "gain_mean", "ci95_lower",
                        "ci95_upper", "scope"]
    with bootstrap_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=bootstrap_fields)
        writer.writeheader()
        writer.writerows(bootstrap_rows)
    md_path = args.output_dir / "decision_report.md"
    lines = ["# Frozen fusion decision report", "",
             "This report uses saved predictions. It fits no estimator.", "",
             "Bootstrap intervals use 5,000 paired chemical-group draws with bootstrap seed 1729.",
             "They use fixed predictions. They omit model-refit uncertainty.", "",
             "## Candidate metrics", "",
             "| Dataset | Method | Pair MAE mean | Seed range | Pair RMSE mean | Ligand MAE mean | Spearman mean |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in metric_means:
        m, rg = r["mean"], r["seed_range_min_max"]["macro_group_pair_mae"]
        lines.append(f"| {r['dataset']} | {r['method']} | {fmt(m['macro_group_pair_mae'])} | {fmt(rg[0])}–{fmt(rg[1])} | {fmt(m['pair_rmse'])} | {fmt(m['ligand_mae'])} | {fmt(m['spearman_rho'])} |")
    lines += ["", "## Frozen acceptance", ""]
    for dataset, gate in gates_by_dataset.items():
        lines.append(f"{dataset}: acceptance pass = {gate['acceptance_pass']}; added quantum claim allowed = {gate['added_quantum_claim_allowed']}.")
        for cmp, checks in gate["mandatory_comparators"].items():
            if cmp == "matched_selected_chem":
                lines.append(f"Matched chemistry: mean gain {checks['mean_absolute_gain']:.6f}; all seed gains positive = {checks['positive_primary_gain_every_seed']}; all bootstrap lower bounds positive = {checks['bootstrap_lower_bound_above_zero_every_seed']}.")
            else:
                lines.append(f"{cmp}: absolute gain gate = {checks['mean_absolute_gain_at_least_0_10']}; relative gain gate = {checks['mean_relative_gain_at_least_10_percent']}; positive every seed = {checks['positive_gain_every_seed']}; bootstrap lower bound positive = {checks['bootstrap_lower_bound_above_zero_every_seed']}; Spearman gate = {checks['spearman_regression_no_more_than_0_05_every_seed']}.")
    lines += ["", "## Descriptive family comparisons", "",
              "The JSON report and paired_bootstrap.csv include each quantum family against all mandatory comparators and its matched chemistry family.",
              "These comparisons do not change the selected_fusion acceptance rule.", ""]
    lines.append(f"Paired bootstrap rows: {len(bootstrap_rows)}. Each row stores model_seed and bootstrap_seed.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "decision_json": str(json_path),
                      "decision_markdown": str(md_path), "paired_bootstrap_csv": str(bootstrap_path),
                      "methods": len(metric_means), "bootstrap_rows": len(bootstrap_rows)}, indent=2))


if __name__ == "__main__":
    main()
