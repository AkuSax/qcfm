"""Build the frozen, zero-fit schedule and validate all split boundaries."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import runner


def sha(path: Path) -> str:
    return runner.digest(path)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def stable_crossfit(groups: list[str], sizes: dict[str, int], seed: int,
                    dataset: str, outer: int, inner: int) -> dict:
    token = f"{seed}|{dataset}|{outer}|{inner}|crossfit-v1".encode()
    stable_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    rng = np.random.default_rng(stable_seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    # Greedy row-count balancing uses no labels.
    sides: list[list[str]] = [[], []]
    counts = [0, 0]
    for g in sorted(shuffled, key=lambda z: (-sizes[z], shuffled.index(z))):
        target = 0 if counts[0] <= counts[1] else 1
        sides[target].append(g); counts[target] += sizes[g]
    if not sides[0] or not sides[1]:
        raise ValueError("Cross-fit split has an empty side")
    return {"stable_seed": stable_seed, "groups": sorted(groups),
            "sides": [sorted(sides[0]), sorted(sides[1])],
            "row_counts": counts}


def pair_group_count(bundle: dict, idx: np.ndarray) -> tuple[int, int]:
    pairs = runner.pair_rows(idx, bundle["groups"])
    groups = {g for _, _, g in pairs}
    return len(groups), len(pairs)


def validate_splits(bundle: dict, cids: list[str], sensitivity: bool) -> dict:
    valid = runner.select_valid(bundle, sensitivity)
    rows = np.flatnonzero(valid)
    if len(rows) != (60 if sensitivity else 62):
        raise ValueError("Dataset record coverage does not match the frozen cohort")
    n_groups, n_pairs = pair_group_count(bundle, rows)
    if (n_groups, n_pairs) != (8, 129 if sensitivity else 136):
        raise ValueError(f"Unexpected cohort pair boundary: {n_groups} groups, {n_pairs} pairs")
    outer_groups: dict[str, set[int]] = {}
    fold_pair_counts = {}
    for i in rows:
        outer_groups.setdefault(str(bundle["groups"][i]), set()).add(int(bundle["outer"][i]))
    if any(len(folds) != 1 for folds in outer_groups.values()):
        raise ValueError("A chemical group crosses outer folds")
    for fold in range(5):
        test = rows[bundle["outer"][rows] == fold]
        train = rows[bundle["outer"][rows] != fold]
        groups_test = {str(bundle["groups"][i]) for i in test}
        groups_train = {str(bundle["groups"][i]) for i in train}
        if groups_test & groups_train:
            raise ValueError("A chemical group crosses outer train and test")
        _, n = pair_group_count(bundle, test)
        if n < 1:
            raise ValueError("An outer test fold has no eligible pair")
        fold_pair_counts[str(fold)] = n
        for inner_fold in (0, 1):
            train_outer = rows[bundle["outer"][rows] != fold]
            inner_validation = train_outer[bundle["inner"][fold, train_outer] == inner_fold]
            inner_train = train_outer[bundle["inner"][fold, train_outer] != inner_fold]
            if len(inner_validation) == 0 or len(inner_train) == 0:
                raise ValueError("An inner split has an empty role")
            gv = {str(bundle["groups"][i]) for i in inner_validation}
            gt = {str(bundle["groups"][i]) for i in inner_train}
            if gv & gt:
                raise ValueError("A chemical group crosses inner roles")
            if pair_group_count(bundle, inner_validation)[1] < 1:
                raise ValueError("An inner validation fold has no eligible pair")
            if pair_group_count(bundle, inner_train)[1] < 1:
                raise ValueError("An inner training fold has no eligible pair")
    return {"n_records": len(rows), "denominator": 65,
            "n_pair_groups": n_groups, "n_pairs": n_pairs,
            "outer_pair_counts": fold_pair_counts}


def validate_prior(bundle: dict, prior: dict) -> dict:
    cids = bundle["cids"]
    prior_rows = [r for r in runner.load_csv(runner.PRIOR_PREDICTIONS) if r["dataset"] == "primary"]
    manifest = json.loads((runner.PREVIOUS/"fit_retry01_output_hash_manifest.json").read_text(encoding="utf-8-sig")); expected_prediction_sha = next(x["sha256"] for x in manifest["files"] if x["path"] == "predictions.csv")
    if sha(runner.PRIOR_PREDICTIONS) != expected_prediction_sha:
        raise ValueError("Prior prediction file hash differs from its frozen manifest")
    summary = {}
    for method in ("native", "boltz_ridge", "ordinary_readout", "prior_quantum_only"):
        prior_key = "prior_quantum_only" if method == "prior_quantum_only" else method
        if prior_key not in prior.get("primary", {}):
            raise ValueError(f"Missing prior comparator: {method}")
        summary[method] = {}
        for seed in runner.SEEDS:
            prior_method = "quantum_only" if method == "prior_quantum_only" else method
            pred = prior.get("primary", {}).get(method, {}).get(seed, {})
            expected = set(cids) if method == "native" else {cids[i] for i in np.flatnonzero(bundle["valid"])}
            if set(pred) != expected:
                raise ValueError(f"Prior comparator identity coverage differs: {method}, {seed}")
            if any(not np.isfinite(v) for v in pred.values()):
                raise ValueError(f"Prior comparator has non-finite scores: {method}, {seed}")
            if method == "native":
                if any(abs(pred[c]-bundle["native"][i]) > 1e-8 for i, c in enumerate(cids)):
                    raise ValueError("Prior native scores differ from the frozen source scores")
            rows = [r for r in prior_rows if r["method"] == prior_method and int(r["seed"]) == seed]
            if len(rows) != 65:
                raise ValueError(f"Prior comparator row denominator differs: {method}, {seed}")
            for r in rows:
                i = cids.index(r["pubchem_cid"])
                if int(r["outer_fold"]) != int(bundle["outer"][i]):
                    raise ValueError("Prior comparator outer fold differs from the frozen split")
                status_should_be = "predicted" if r["pubchem_cid"] in expected else "missing"
                if r["status"] != status_should_be:
                    raise ValueError("Prior comparator validity status differs from the frozen feature cohort")
            summary[method][str(seed)] = {"n": len(pred),
                "prediction_sha256": hashlib.sha256(json.dumps(pred, sort_keys=True).encode()).hexdigest()}
    for dataset, sensitivity in (("primary", False), ("hill_sensitivity", True)):
        valid = runner.select_valid(bundle, sensitivity)
        expected = {cids[i] for i in np.flatnonzero(valid)}
        rows_ds = [r for r in runner.load_csv(runner.PRIOR_PREDICTIONS) if r["dataset"] == dataset]
        for method in ("boltz_ridge", "ordinary_readout", "quantum_only"):
            for seed in runner.SEEDS:
                values = prior.get(dataset, {}).get("prior_quantum_only" if method == "quantum_only" else method, {}).get(seed, {})
                if set(values) != expected:
                    raise ValueError(f"Prior comparator sensitivity identities differ: {dataset}, {method}, {seed}")
                method_rows = [r for r in rows_ds if r["method"] == method and int(r["seed"]) == seed]
                if len(method_rows) != 65:
                    raise ValueError(f"Prior comparator sensitivity denominator differs: {dataset}, {method}, {seed}")
                for r in method_rows:
                    i = cids.index(r["pubchem_cid"])
                    if int(r["outer_fold"]) != int(bundle["outer"][i]):
                        raise ValueError("Prior sensitivity outer fold differs from frozen split")
                    if r["status"] != ("predicted" if r["pubchem_cid"] in expected else "missing"):
                        raise ValueError("Prior sensitivity feature validity differs from frozen cohort")
    summary["source_prediction_sha256"] = expected_prediction_sha
    return summary


def validate_pair_manifest(bundle: dict, dataset: str, path: Path, expected_sha: str) -> dict:
    if sha(path) != expected_sha:
        raise ValueError(f"Frozen pair manifest hash changed: {dataset}")
    rows = runner.load_csv(path)
    if len(rows) != (136 if dataset == "primary" else 129):
        raise ValueError(f"Frozen pair manifest count changed: {dataset}")
    cids = bundle["cids"]
    y = {c: float(bundle["rows"][c]["label"]) for c in cids}
    valid = runner.select_valid(bundle, dataset == "hill_sensitivity")
    expected = set()
    for i, j, group in runner.pair_rows(np.flatnonzero(valid), bundle["groups"]):
        fold = int(bundle["outer"][i])
        a, b = sorted((cids[i], cids[j]))
        expected.add((fold, group, a, b))
    observed = set()
    for r in rows:
        ca, cb = r["cid_a"], r["cid_b"]
        if ca not in y or cb not in y or not valid[cids.index(ca)] or not valid[cids.index(cb)]:
            raise ValueError("Frozen pair manifest contains an ineligible identity")
        if abs(float(r["label_a_log10_ic50_uM"])-y[ca]) > 1e-10 or abs(float(r["label_b_log10_ic50_uM"])-y[cb]) > 1e-10:
            raise ValueError("Frozen pair label differs from the source label")
        delta = y[ca]-y[cb]
        if abs(float(r["delta_label_a_minus_b"])-delta) > 1e-10:
            raise ValueError("Frozen pair orientation or delta differs")
        observed.add((int(r["outer_fold"]), r["chemical_group"], *sorted((ca, cb))))
    if observed != expected:
        raise ValueError(f"Canonical pair IDs differ from the frozen {dataset} manifest")
    return {"path": str(path), "sha256": sha(path), "n_pairs": len(rows), "canonical_pair_set": "PASS"}


def build() -> tuple[dict, dict]:
    bundle = runner.load_bundle()
    cids = bundle["cids"]
    pair_audit = {
        "primary": validate_pair_manifest(bundle, "primary",
            runner.PACKAGE/"arm4_protocol"/"primary_pairs.csv",
            "0e18db377566cc8c65ef71c66cc24ebe301b3a4dacd4d78364bea7f2acf1cbe3"),
        "hill_sensitivity": validate_pair_manifest(bundle, "hill_sensitivity",
            runner.PACKAGE/"arm4_protocol"/"sensitivity_pairs.csv",
            "d07ef017178fdc2baa167ba246ee560aa1de12645a2f63ddbd70611e309c2635")}
    primary = validate_splits(bundle, cids, False)
    sensitivity = validate_splits(bundle, cids, True)
    prior = runner.load_frozen_oof(bundle)
    prior_summary = validate_prior(bundle, prior)

    cf_splits = []
    for ds, sens in (("primary", False), ("hill_sensitivity", True)):
        valid = runner.select_valid(bundle, sens)
        for seed in runner.SEEDS:
            for outer in range(5):
                outer_train = np.flatnonzero(valid & (bundle["outer"] != outer))
                for inner in (0, 1):
                    inner_train = outer_train[bundle["inner"][outer, outer_train] != inner]
                    groups = sorted({str(bundle["groups"][i]) for i in inner_train})
                    sizes = {g: int(sum(bundle["groups"][i] == g for i in inner_train)) for g in groups}
                    split = stable_crossfit(groups, sizes, seed, ds, outer, inner)
                    for held_side in (0, 1):
                        prediction_groups = set(split["sides"][held_side])
                        fit_groups = set(split["sides"][1-held_side])
                        fit_ids = [bundle["cids"][i] for i in inner_train if str(bundle["groups"][i]) in fit_groups]
                        predict_ids = [bundle["cids"][i] for i in inner_train if str(bundle["groups"][i]) in prediction_groups]
                        if not fit_ids or not predict_ids:
                            raise ValueError("Cross-fit side has no rows")
                        fit_rows = [i for i in inner_train if bundle["cids"][i] in set(fit_ids)]
                        if pair_group_count(bundle, np.asarray(fit_rows, int))[1] < 1:
                            raise ValueError("Cross-fit training side has no eligible pair")
                        cf_splits.append({"dataset": ds, "model_seed": seed,
                            "outer_fold": outer, "inner_fold": inner,
                            "held_side": held_side, "fit_groups": sorted(fit_groups),
                            "predict_groups": sorted(prediction_groups),
                            "fit_ids": fit_ids, "predict_ids": predict_ids,
                            "stable_seed": split["stable_seed"], "row_counts": split["row_counts"]})

    schedule = []
    ordinal = 0
    for ds in ("primary", "hill_sensitivity"):
        for seed in runner.SEEDS:
            for outer in range(5):
                unit = {"dataset": ds, "model_seed": seed, "outer_fold": outer}
                for inner in (0, 1):
                    # Three components, eight family fits, and two extra kernel weights.
                    for method in ("boltz", "aim", "chem", "additive_q", "additive_chem",
                                   "kernel_q_w0.25", "kernel_q_w0.75", "kernel_chem_w0.25",
                                   "kernel_chem_w0.75", "stack_q", "stack_chem", "lowrank_q", "lowrank_chem"):
                        ordinal += 1
                        schedule.append({**unit, "stage": "inner", "inner_fold": inner,
                                         "method": method, "attempt": ordinal})
                for method in runner.METHODS:
                    ordinal += 1
                    schedule.append({**unit, "stage": "outer", "inner_fold": "",
                                     "method": method, "attempt": ordinal})
                for item in cf_splits:
                    if item["dataset"] == ds and item["model_seed"] == seed and item["outer_fold"] == outer:
                        for base in ("boltz", "aim", "chem"):
                            ordinal += 1
                            schedule.append({**unit, "stage": "inner_stack_crossfit",
                                "inner_fold": item["inner_fold"], "method": base,
                                "held_side": item["held_side"], "attempt": ordinal})
    if ordinal != 1470:
        raise ValueError(f"Frozen fit schedule has {ordinal} attempts, not 1470")
    return {"bundle": bundle, "cf_splits": cf_splits, "schedule": schedule,
            "cohorts": {"primary": primary, "hill_sensitivity": sensitivity},
            "prior": prior_summary, "pair_manifests": pair_audit}


def write(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    journal = out/"fit_journal.jsonl"
    progress = out/"fit_progress.json"
    if journal.exists() and journal.stat().st_size:
        raise RuntimeError("Preflight cannot run after a fit journal exists")
    if progress.exists():
        old = json.loads(progress.read_text(encoding="utf-8-sig"))
        if int(old.get("attempted_fits", 0)) > 0 or old.get("started_unix") is not None:
            raise RuntimeError("Preflight cannot reset a started or attempted fitting run")
    built = build()
    write_csv(out / "fit_schedule.csv", built["schedule"],
        ["attempt", "dataset", "model_seed", "outer_fold", "stage", "inner_fold", "method", "held_side"])
    (out / "stack_crossfit_splits.json").write_text(
        json.dumps(built["cf_splits"], indent=2, sort_keys=True)+"\n", encoding="utf-8")
    (out / "cohort_preflight.json").write_text(
        json.dumps({"cohorts": built["cohorts"], "prior_comparators": built["prior"],
                    "pair_manifests": built["pair_manifests"]},
                   indent=2, sort_keys=True)+"\n", encoding="utf-8")
    plan_path = out.parent/"FROZEN_PLAN.md"
    plan_freeze = out.parent/"plan_freeze_receipt.json"
    freeze_record = json.loads(plan_freeze.read_text(encoding="utf-8-sig"))
    if sha(plan_path) != freeze_record.get("plan_sha256"):
        raise ValueError("Frozen plan hash differs from its freeze receipt")
    sources = [runner.FEATURES, runner.PREVIOUS/"aim_features.schema.json",
               runner.SOURCE/"records.csv", runner.SOURCE/"splits.csv", runner.PRIOR_PREDICTIONS,
               runner.PACKAGE/"arm4_protocol"/"primary_pairs.csv",
               runner.PACKAGE/"arm4_protocol"/"sensitivity_pairs.csv", plan_path, plan_freeze,
               runner.SUPPORT_PATH, Path(__file__), Path(runner.__file__),
               Path(runner.build_designs.__code__.co_filename),
               runner.HERE.parent/"REPORT_CONTRACT.md",
               out/"fit_schedule.csv", out/"stack_crossfit_splits.json"]
    code = {str(p.resolve()): sha(p) for p in sources}
    receipt = {"status": "PREFLIGHT_PASS_NO_FITS", "attempted_fits": 0,
        "schedule_attempts": len(built["schedule"]), "maximum_attempts": 1500,
        "reserved_attempts": 30, "maximum_seconds": 14400,
        "cohort_checks": built["cohorts"], "prior_comparator_checks": built["prior"],
        "feature_sha256": sha(runner.FEATURES),
        "schema_sha256": sha(runner.PREVIOUS/"aim_features.schema.json"),
        "frozen_plan_sha256": sha(plan_path), "plan_freeze_receipt_sha256": sha(plan_freeze),
        "support_import_path": str(Path(runner.support.__file__).resolve()),
        "support_sha256": sha(Path(runner.support.__file__).resolve()),
        "input_and_code_hashes": code,
        "journal_path": str(out/"fit_journal.jsonl"),
        "journal_exists_before_fit": (out/"fit_journal.jsonl").exists(),
        "pair_manifests": built["pair_manifests"]}
    (out / "preflight_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    (out / "fit_progress.json").write_text(json.dumps({"attempted_fits": 0,
        "maximum_fits": 1500, "reserve": 30, "maximum_seconds": 14400,
        "active_fit": None}, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    write(a.output)
