"""Shared nested runner for the frozen classical feature comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[2] / "preparation" / "20261022T182444Z"
PREVIOUS = PACKAGE / "execution_full" / "max1000_20261022T203000Z"
SOURCE = Path(r"C:\Users\Saxen\ppi_IG\results\representation_diagnosis\20261015\attempt02\gcp_a100_retry_01\collected")
FEATURES = PREVIOUS / "aim_features.npz"
PRIOR_PREDICTIONS = PREVIOUS / "fit_retry01" / "predictions.csv"
SUPPORT_PATH = HERE / "protocol_support.py"
sys.path.insert(0, str(HERE))
import protocol_support as support

sys.path.insert(0, str(HERE.parent / "interaction"))
from interaction import build_designs, FoldDesign, _raw_blocks

SEEDS = (1729, 2718, 31415)
FIT_CAP = 1500
WALL_SECONDS = 14400
ALPHA = 0.1
KERNEL_WEIGHTS = (0.25, 0.75)
METHODS = ("boltz", "aim", "chem", "additive_q", "additive_chem",
           "kernel_q", "kernel_chem", "stack_q", "stack_chem",
           "lowrank_q", "lowrank_chem")
NATIVE_METHODS = ("native", "boltz_ridge", "ordinary_readout", "quantum_only")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_bundle() -> dict[str, Any]:
    cids, x1, x2, record_rows = support.load_frozen_pooled_inputs()
    f = support.load_atom_features(FEATURES, cids)
    # The native score is saved unchanged in records.csv.
    rows = {r["pubchem_cid"]: r for r in record_rows}
    source_records = {r["pubchem_cid"]: r for r in load_csv(SOURCE / "records.csv")}
    if set(rows) != set(source_records):
        raise ValueError("Source and frozen records have different identities")
    y = np.asarray([float(rows[c]["label"]) for c in cids])
    native = np.asarray([float(source_records[c]["unchanged"]) for c in cids])
    group = np.asarray([rows[c]["chemical_group"] for c in cids], dtype=str)
    outer = np.full(len(cids), -1, dtype=int)
    inner = np.full((5, len(cids)), -1, dtype=int)
    row_for_cid = {c: i for i, c in enumerate(cids)}
    for r in support.load_fold_splits():
        i = row_for_cid[r["pubchem_cid"]]
        fold = int(r["outer_fold"])
        role = r["role"]
        if role == "test":
            outer[i] = fold
        elif role == "train":
            inner[fold, i] = int(r["inner_validation_fold"])
    if np.any(outer < 0):
        raise ValueError("An outer test assignment is missing")
    valid = np.asarray(f["feature_valid"], dtype=bool)
    if int(valid.sum()) != 62:
        raise ValueError("Feature coverage differs from the frozen 62 of 65 rows")
    return {"cids": cids, "rows": rows, "y": y, "native": native,
            "groups": group, "outer": outer, "inner": inner,
            "valid": valid, "X": f, "x1": x1, "x2": x2}


class FitJournal:
    """Append every estimator attempt before and after it runs."""
    def __init__(self, out: Path, cap: int = FIT_CAP, wall_seconds: int = WALL_SECONDS):
        self.out = out
        self.cap = cap
        self.wall_seconds = wall_seconds
        self.path = out / "fit_journal.jsonl"
        self.state = out / "fit_progress.json"
        old_state = json.loads(self.state.read_text(encoding="utf-8")) if self.state.exists() else {}
        self.started_unix = old_state.get("started_unix") or time.time()
        attempts = set()
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        if row.get("status") == "started":
                            attempts.add(int(row["attempt"]))
        self.count = max(int(old_state.get("attempted_fits", 0)), len(attempts))
        self.started = time.monotonic() - max(0.0, time.time()-self.started_unix)
        self._write_state(None)

    def _write_state(self, active: dict | None) -> None:
        self.state.write_text(json.dumps({"attempted_fits": self.count,
            "maximum_fits": self.cap, "started_unix": self.started_unix,
            "elapsed_seconds": time.time()-self.started_unix,
            "maximum_seconds": self.wall_seconds, "active_fit": active}, indent=2)+"\n",
            encoding="utf-8")

    def fit(self, method: str, params: dict, train_ids: list[str],
            predict_ids: list[str], seed: int, stage: str, fn):
        elapsed = time.time() - self.started_unix
        if self.count >= self.cap:
            raise RuntimeError("Fit cap reached before attempt")
        if elapsed >= self.wall_seconds:
            raise TimeoutError("Fit wall-clock limit reached before attempt")
        self.count += 1
        item = {"attempt": self.count, "method": method, "params": params,
                "parameter_sha256": hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest(),
                "train_ids": train_ids, "predict_ids": predict_ids,
                "seed": seed, "stage": stage, "status": "started",
                "started_unix": time.time(), "feature_sha256": digest(FEATURES),
                "runner_sha256": digest(Path(__file__).resolve()),
                "transform_module_sha256": digest(Path(build_designs.__code__.co_filename).resolve())}
        self._write_state(item)
        item["train_ids_sha256"] = hashlib.sha256("\n".join(train_ids).encode()).hexdigest()
        item["predict_ids_sha256"] = hashlib.sha256("\n".join(predict_ids).encode()).hexdigest()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, sort_keys=True)+"\n")
        try:
            result = fn()
            prediction = result.get("prediction") if isinstance(result, dict) else result
            if not np.isfinite(np.asarray(prediction, dtype=float)).all():
                raise FloatingPointError("Estimator output contains a non-finite value")
            item["coefficient_sha256"] = result.get("coefficient_sha256") if isinstance(result, dict) else None
            item["objective"] = result.get("objective") if isinstance(result, dict) else None
            if isinstance(result, dict) and result.get("artifact"):
                artifact_dir = self.out / "fit_artifacts"
                artifact_dir.mkdir(exist_ok=True)
                artifact_path = artifact_dir / f"fit_{self.count:04d}.npz"
                np.savez_compressed(artifact_path, **result["artifact"])
                item["artifact_path"] = str(artifact_path)
                item["artifact_sha256"] = digest(artifact_path)
            item.update(status="success", elapsed_seconds=time.monotonic()-self.started,
                        fit_seconds=time.time()-item["started_unix"])
            if isinstance(result, dict):
                result["fit_attempt"] = self.count
            return result
        except Exception as exc:
            item.update(status="failure", error_type=type(exc).__name__, error=str(exc),
                        traceback=traceback.format_exc(), fit_seconds=time.time()-item["started_unix"])
            raise
        finally:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, sort_keys=True)+"\n")
            self._write_state(None)


def pair_rows(indices: np.ndarray, groups: np.ndarray) -> list[tuple[int, int, str]]:
    by: dict[str, list[int]] = {}
    for i in indices:
        by.setdefault(str(groups[i]), []).append(int(i))
    pairs = []
    for g, rows in by.items():
        for j, a in enumerate(rows):
            for b in rows[j+1:]:
                pairs.append((a, b, g))
    return pairs


def fit_linear(X_train: np.ndarray, y_train: np.ndarray, ids: np.ndarray,
               groups: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Fit the frozen pair-MSE plus 0.1 ligand-anchor objective."""
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim != 2 or X.shape[0] != len(y) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Training matrix or labels are invalid")
    local = {int(row): j for j, row in enumerate(ids)}
    pairs = pair_rows(ids, groups)
    by: dict[str, list[tuple[int, int]]] = {}
    for a, b, g in pairs:
        by.setdefault(g, []).append((a, b))
    if not by:
        raise ValueError("Training split has no within-group pairs")
    n_features = X.shape[1]
    A, target, weight = [], [], []
    for g, gpairs in by.items():
        for a, b in gpairs:
            ia, ib = local[a], local[b]
            A.append(np.r_[X[ia]-X[ib], 0.0])
            target.append(y[ia]-y[ib])
            weight.append(1.0/(len(by)*len(gpairs)))
    for j in range(len(ids)):
        A.append(np.r_[X[j], 1.0]); target.append(y[j]); weight.append(0.1/len(ids))
    A = np.asarray(A); target = np.asarray(target); weight = np.asarray(weight)
    penalty = np.eye(n_features+1); penalty[-1, -1] = 0.0
    coef = np.linalg.solve(A.T@(weight[:, None]*A)+alpha*penalty, A.T@(weight*target))
    if not np.isfinite(coef).all():
        raise FloatingPointError("Ridge coefficients contain a non-finite value")
    return coef


def objective_value(X: np.ndarray, y: np.ndarray, ids: np.ndarray,
                    groups: np.ndarray, coef: np.ndarray, alpha: float = ALPHA) -> float:
    pairs = pair_rows(ids, groups)
    by: dict[str, list[float]] = {}
    lookup = {int(r): i for i, r in enumerate(ids)}
    pred = X@coef[:-1]+coef[-1]
    for a, b, group in pairs:
        by.setdefault(group, []).append(float(((pred[lookup[a]]-pred[lookup[b]])-(y[lookup[a]]-y[lookup[b]]))**2))
    pair = float(np.mean([np.mean(v) for v in by.values()]))
    anchor = float(np.mean((pred-y)**2))
    return pair+0.1*anchor+alpha*float(coef[:-1]@coef[:-1])


def raw_component(bundle: dict, method: str, train_idx: np.ndarray,
                  query_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    f = bundle["X"]
    mask = np.asarray(f["atom_mask"], bool)
    denom = mask.sum(1, keepdims=True)
    if np.any((denom == 0) & np.asarray(bundle["valid"], dtype=bool)[:, None]):
        raise ValueError("A feature-valid row has no active atoms")
    denom = np.maximum(denom, 1)
    bmean = np.where(mask[..., None], f["boltz_atoms"], 0.0).sum(1)/denom
    if method == "boltz":
        raw = np.concatenate([f["global_boltz"], bmean], axis=1)
    elif method == "aim":
        am = np.where(mask[..., None], f["aim_atoms"], 0.0).sum(1)/denom
        af = np.where(mask[..., None], f["aim_atoms_free"], 0.0).sum(1)/denom
        raw = np.concatenate([am, af, f["aim_scalars"], f["aim_scalars_free"], f["simple_scalars"]], axis=1)
    elif method == "chem":
        raw = np.asarray(f["simple_scalars"])
    else:
        raise ValueError(method)
    train = np.asarray(raw[train_idx], dtype=float)
    query = np.asarray(raw[query_idx], dtype=float)
    if not np.isfinite(train).all() or not np.isfinite(query).all():
        raise FloatingPointError("Component features contain a non-finite value")
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    z = (train-mean)/scale
    _, singular, vt = np.linalg.svd(z, full_matrices=False)
    rank = min(4, len(train)-1, train.shape[1], len(singular))
    if rank < 1:
        raise ValueError("Component PCA has no available rank")
    axes = vt[:rank].T.copy()
    for j in range(rank):
        pivot = int(np.argmax(np.abs(axes[:, j])))
        if axes[pivot, j] < 0:
            axes[:, j] *= -1
    score_scale = (z@axes).std(axis=0)
    score_scale[score_scale <= 1e-12] = 1.0
    block_scale = np.sqrt(rank)
    xtr = (z@axes)/score_scale/block_scale
    xq = (((query-mean)/scale)@axes)/score_scale/block_scale
    if not np.isfinite(xtr).all() or not np.isfinite(xq).all():
        raise FloatingPointError("Scaled design contains a non-finite value")
    return xtr, xq, {"mean": mean, "scale": scale, "axes": axes,
        "score_scale": score_scale, "block_scale": np.asarray([block_scale]),
        "rank": np.asarray([rank])}


def gaussian_features(xtrain: np.ndarray, xquery: np.ndarray,
                      alpha: float, weight: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return a train-fitted exact Gaussian kernel feature map."""
    dist = np.maximum(0.0, (xtrain*xtrain).sum(1)[:, None] + (xtrain*xtrain).sum(1)[None, :] - 2*xtrain@xtrain.T)
    kt = np.exp(-0.5*dist)
    eig, vec = np.linalg.eigh(kt)
    keep = eig > max(eig.max()*1e-12, 1e-12)
    if not keep.any():
        raise ValueError("Kernel has no positive eigenvalue")
    vec = vec.copy()
    for j in range(vec.shape[1]):
        pivot = int(np.argmax(np.abs(vec[:, j])))
        if vec[pivot, j] < 0:
            vec[:, j] *= -1
    basis = vec[:, keep] * np.sqrt(eig[keep])[None, :]
    cross_dist = np.maximum(0.0, (xquery*xquery).sum(1)[:, None] + (xtrain*xtrain).sum(1)[None, :] - 2*xquery@xtrain.T)
    kq = np.exp(-0.5*cross_dist)
    phi_q = kq @ (vec[:, keep] / np.sqrt(eig[keep])[None, :])
    scale = np.sqrt(weight)
    return scale*basis, scale*phi_q, {"eigenvalues": eig[keep],
        "eigenvectors": vec[:, keep], "sqrt_weight": np.asarray([scale]),
        "keep": np.flatnonzero(keep)}


def design(bundle: dict, method: str, train_idx: np.ndarray,
           query_idx: np.ndarray, params: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    params = dict(params or {})
    if method in ("boltz", "aim", "chem"):
        return raw_component(bundle, method, train_idx, query_idx)
    if method in ("additive_q", "additive_chem", "lowrank_q", "lowrank_chem"):
        family = "q" if method.endswith("_q") else "chem"
        additive = method.startswith("additive_")
        transform = FoldDesign.fit(bundle, train_idx, family, rank=4, additive=additive)
        if not additive:
            raw_a, raw_b = _raw_blocks(bundle, family)
            idx = np.asarray(train_idx, dtype=int)
            pa = transform.first.transform(raw_a[idx])
            pb = transform.second.transform(raw_b[idx])
            product = pa*pb
            transform.product_mean = product.mean(axis=0)
            transform.product_scale = product.std(axis=0)
            transform.product_scale[transform.product_scale <= 1e-12] = 1.0
        Xtr, Xq = transform.transform(bundle, train_idx), transform.transform(bundle, query_idx)
        state = {}
        for prefix, block in (("B", transform.first), ("S", transform.second)):
            for name in ("mean", "scale", "axes", "score_scale"):
                state[f"{prefix}_{name}"] = np.asarray(getattr(block, name))
            state[f"{prefix}_block_scale"] = np.asarray([block.block_scale])
        if not additive:
            state["product_mean"] = transform.product_mean
            state["product_scale"] = transform.product_scale
        state["rank"] = np.asarray([4])
        return Xtr, Xq, state
    if method in ("kernel_q", "kernel_chem"):
        fam = "q" if method.endswith("_q") else "chem"
        additive = "additive_"+fam
        # The component block definitions match the frozen additive design.
        full_train, full_query, prep = design(bundle, additive, train_idx, query_idx,
                                               {"alpha": ALPHA, "rank": 4})
        weight = float(params.get("kernel_weight", 0.25))
        if weight not in KERNEL_WEIGHTS:
            raise ValueError("Kernel weight is outside the frozen grid")
        k1tr, k1q, k1state = gaussian_features(full_train[:, :4], full_query[:, :4], ALPHA, weight)
        k2tr, k2q, k2state = gaussian_features(full_train[:, 4:8], full_query[:, 4:8], ALPHA, 1.0-weight)
        state = {**{f"block_{k}": v for k,v in prep.items()},
                 **{f"B_kernel_{k}": v for k,v in k1state.items()},
                 **{f"S_kernel_{k}": v for k,v in k2state.items()}}
        return np.concatenate([k1tr, k2tr], axis=1), np.concatenate([k1q, k2q], axis=1), state
    raise ValueError(method)


def load_frozen_oof(bundle: dict) -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    result: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for r in load_csv(PRIOR_PREDICTIONS):
        if r["method"] not in NATIVE_METHODS:
            continue
        method = "prior_quantum_only" if r["method"] == "quantum_only" else r["method"]
        seed = int(r["seed"])
        if r["status"] != "predicted":
            continue
        result.setdefault(r["dataset"], {}).setdefault(method, {}).setdefault(seed, {})[r["pubchem_cid"]] = float(r["prediction"])
    return result


def coef_hash(coef: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(coef, dtype="<f8").tobytes()).hexdigest()


def fitted_design(bundle: dict, method: str, train_idx: np.ndarray,
                  query_idx: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    return design(bundle, method, train_idx, query_idx, params)


def attempt_fit(bundle: dict, method: str, train_idx: np.ndarray, query_idx: np.ndarray,
                seed: int, stage: str, journal: FitJournal, params: dict | None = None):
    params = dict(params or {})
    train_ids = [bundle["cids"][i] for i in train_idx]
    query_ids = [bundle["cids"][i] for i in query_idx]
    cache_params = {**params, "stage": stage}
    key = cache_key(method, train_ids, query_ids, seed, cache_params)
    cache_path = journal.out/"base_prediction_cache"/f"{key}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        expected_prediction_sha = hashlib.sha256(np.asarray(cached.get("prediction", []), dtype="<f8").tobytes()).hexdigest()
        if (cached.get("cache_key") != key or cached.get("method") != method
            or cached.get("train_ids") != train_ids or cached.get("predict_ids") != query_ids
            or cached.get("model_seed") != seed or cached.get("feature_sha256") != digest(FEATURES)
            or cached.get("params") != cache_params
            or cached.get("runner_sha256") != digest(Path(__file__).resolve())
            or cached.get("transform_sha256") != digest(Path(build_designs.__code__.co_filename).resolve())
            or cached.get("prediction_sha256") != expected_prediction_sha
            or len(cached.get("prediction", [])) != len(query_ids)):
            raise ValueError("Prediction cache provenance differs")
        with (journal.out/"cache_reuse_journal.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"cache_key": key, "method": method, "stage": stage,
                "train_ids": train_ids, "predict_ids": query_ids, "model_seed": seed,
                "cache_path": str(cache_path), "status": "reused"}, sort_keys=True)+"\n")
        result = {"prediction": np.asarray(cached["prediction"], dtype=float),
                "coefficient_sha256": cached.get("coefficient_sha256"),
                "objective": cached.get("objective"), "fit_attempt": cached.get("fit_attempt"),
                "cached": True}
    else:
        def work():
            Xtr, Xq, prep_state = fitted_design(bundle, method, train_idx, query_idx, params)
            ytr = bundle["y"][train_idx]
            if not np.isfinite(ytr).all() or not np.isfinite(Xtr).all() or not np.isfinite(Xq).all():
                raise FloatingPointError("Fit input contains a non-finite value")
            coef = fit_linear(Xtr, ytr, train_idx, bundle["groups"], ALPHA)
            loss = objective_value(Xtr, ytr, train_idx, bundle["groups"], coef, ALPHA)
            if not np.isfinite(loss):
                raise FloatingPointError("Penalized anchored training objective is non-finite")
            prediction = Xq@coef[:-1]+coef[-1]
            if not np.isfinite(prediction).all():
                raise FloatingPointError("Fit prediction contains a non-finite value")
            artifact = {"X_train": Xtr, "X_query": Xq, "coef": coef,
                        "y_train": ytr, "train_indices": train_idx,
                        "query_indices": query_idx}
            artifact.update({f"prep_{k}": np.asarray(v) for k, v in prep_state.items()})
            return {"prediction": prediction, "coefficient_sha256": coef_hash(coef),
                    "objective": loss, "artifact": artifact}
        result = journal.fit(method, params, train_ids, query_ids, seed, stage, work)
        write_cache(journal.out, key, method, train_ids, query_ids, seed,
                    np.asarray(result["prediction"], dtype=float), cache_params,
                    result.get("coefficient_sha256"), result.get("objective"),
                    result.get("fit_attempt"))
    if method in ("boltz", "aim", "chem"):
        append_jsonl(journal.out/"base_fit_provenance.jsonl", {
            "method": method, "stage": stage, "model_seed": seed,
            "train_ids": train_ids, "predict_ids": query_ids,
            "fit_attempt": result.get("fit_attempt"), "cache_key": key,
            "coefficient_sha256": result.get("coefficient_sha256"),
            "prediction_sha256": hashlib.sha256(np.asarray(result["prediction"], dtype="<f8").tobytes()).hexdigest(),
            "cached": bool(result.get("cached", False)),
            "prediction_ids_sha256": hashlib.sha256("\n".join(query_ids).encode()).hexdigest()})
    return result


def score_by_group(bundle: dict, predictions: dict[str, float], ids: list[str]) -> tuple[float, dict[str, float]]:
    ys = {c: float(bundle["y"][bundle["cids"].index(c)]) for c in ids}
    pairs = []
    ix = {c: i for i, c in enumerate(bundle["cids"])}
    for a, b, group in pair_rows(np.asarray([ix[c] for c in ids]), bundle["groups"]):
        ca, cb = bundle["cids"][a], bundle["cids"][b]
        pairs.append((group, ca, cb))
    losses = support.group_pair_losses(ys, predictions, pairs)
    return float(np.mean(list(losses.values()))), losses


def fit_meta(bundle: dict, meta_train: np.ndarray, y_train: np.ndarray,
             train_idx: np.ndarray, meta_query: np.ndarray, query_idx: np.ndarray,
             method: str, seed: int, stage: str, journal: FitJournal):
    train_ids = [bundle["cids"][i] for i in train_idx]
    query_ids = [bundle["cids"][i] for i in query_idx]
    def work():
        train_mean = meta_train.mean(axis=0)
        train_scale = meta_train.std(axis=0)
        train_scale[train_scale <= 1e-12] = 1.0
        Xtr = (meta_train-train_mean)/train_scale
        Xq = (meta_query-train_mean)/train_scale
        if not np.isfinite(Xtr).all() or not np.isfinite(Xq).all():
            raise FloatingPointError("Meta score transform contains non-finite values")
        coef = fit_linear(Xtr, y_train, train_idx, bundle["groups"], ALPHA)
        loss = objective_value(Xtr, y_train, train_idx, bundle["groups"], coef, ALPHA)
        if not np.isfinite(loss):
            raise FloatingPointError("Meta objective is non-finite")
        prediction = Xq@coef[:-1]+coef[-1]
        state = {"mean": train_mean, "scale": train_scale, "coef": coef,
                 "X_train": Xtr, "X_query": Xq, "y_train": y_train,
                 "train_indices": train_idx, "query_indices": query_idx}
        return {"prediction": prediction, "coefficient_sha256": coef_hash(coef),
                "objective": loss, "artifact": state}
    return journal.fit(method, {"alpha": ALPHA, "meta_standardization": "train_oof_only"},
                       train_ids, query_ids, seed, stage, work)


def cache_key(method: str, train_ids: list[str], predict_ids: list[str], seed: int,
              params: dict) -> str:
    record = {"method": method, "train_ids": train_ids, "predict_ids": predict_ids,
              "model_seed": seed, "params": params, "feature_sha256": digest(FEATURES),
              "runner_sha256": digest(Path(__file__).resolve()),
              "transform_sha256": digest(Path(build_designs.__code__.co_filename).resolve())}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def write_cache(out: Path, key: str, method: str, train_ids: list[str],
                predict_ids: list[str], seed: int, predictions: np.ndarray,
                params: dict, coefficient_sha256: str | None = None,
                objective: float | None = None, fit_attempt: int | None = None) -> Path:
    cache_dir = out/"base_prediction_cache"
    cache_dir.mkdir(exist_ok=True)
    path = cache_dir/f"{key}.json"
    payload = {"cache_key": key, "method": method, "train_ids": train_ids,
        "predict_ids": predict_ids, "model_seed": seed, "params": params,
        "feature_sha256": digest(FEATURES), "prediction": [float(x) for x in predictions],
        "coefficient_sha256": coefficient_sha256, "objective": objective,
        "fit_attempt": fit_attempt,
        "prediction_sha256": hashlib.sha256(np.asarray(predictions, dtype="<f8").tobytes()).hexdigest(),
        "runner_sha256": digest(Path(__file__).resolve()),
        "transform_sha256": digest(Path(build_designs.__code__.co_filename).resolve())}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return path


def cached_base_fit(bundle: dict, method: str, train_idx: np.ndarray, query_idx: np.ndarray,
                    seed: int, stage: str, journal: FitJournal, out: Path) -> np.ndarray:
    params = {"alpha": ALPHA, "feature": method}
    fit = attempt_fit(bundle, method, train_idx, query_idx, seed, stage, journal, params)
    return np.asarray(fit["prediction"], dtype=float)


def map_predictions(bundle: dict, indices: np.ndarray, values: np.ndarray) -> dict[str, float]:
    if len(indices) != len(values):
        raise ValueError("Prediction length differs from requested identities")
    if not np.isfinite(values).all():
        raise FloatingPointError("Prediction contains a non-finite value")
    return {bundle["cids"][int(i)]: float(v) for i, v in zip(indices, values)}


def load_crossfit_splits(out: Path) -> list[dict]:
    path = out/"stack_crossfit_splits.json"
    if not path.exists():
        raise FileNotFoundError("Frozen cross-fit split file is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")


def remove_incomplete_unit(out: Path, dataset: str, seed: int, outer: int) -> None:
    partial = out/"partial_predictions.csv"
    if partial.exists():
        rows = load_csv(partial)
        kept = [r for r in rows if not (r["dataset"] == dataset
            and int(r["model_seed"]) == seed and int(r["outer_fold"]) == outer)]
        fields = list(rows[0]) if rows else []
        if fields:
            with partial.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(kept)
    for name in ("selection_manifest.jsonl", "inner_candidate_scores.jsonl"):
        path = out/name
        if path.exists():
            kept = []
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not (row.get("dataset") == dataset
                            and int(row.get("model_seed", -1)) == seed
                            and int(row.get("outer_fold", -1)) == outer):
                        kept.append(line)
            path.write_text("".join(kept), encoding="utf-8")


def run_unit(bundle: dict, dataset: str, sensitivity: bool, seed: int,
             outer_fold: int, journal: FitJournal, out: Path,
             prior: dict, crossfit_plan: list[dict], prediction_writer,
             selection_path: Path, score_path: Path) -> None:
    valid = select_valid(bundle, sensitivity)
    outer_train = np.flatnonzero(valid & (bundle["outer"] != outer_fold))
    outer_test = np.flatnonzero(valid & (bundle["outer"] == outer_fold))
    if len(outer_test) == 0:
        raise ValueError("Outer test fold is empty")
    inner_base: dict[str, dict[str, float]] = {m: {} for m in ("boltz", "aim", "chem")}
    inner_predictions: dict[str, dict[str, float]] = {m: {} for m in METHODS}
    inner_ids: list[str] = []
    inner_scores: dict[str, list[float]] = {}

    for inner_fold in (0, 1):
        train_idx = outer_train[bundle["inner"][outer_fold, outer_train] != inner_fold]
        val_idx = outer_train[bundle["inner"][outer_fold, outer_train] == inner_fold]
        if not len(train_idx) or not len(val_idx):
            raise ValueError("Inner fold is empty")
        if {bundle["groups"][i] for i in train_idx} & {bundle["groups"][i] for i in val_idx}:
            raise ValueError("Inner group leakage detected")
        inner_ids.extend(bundle["cids"][i] for i in val_idx)
        full_base: dict[str, dict[str, float]] = {}
        for base in ("boltz", "aim", "chem"):
            fit = attempt_fit(bundle, base, train_idx, val_idx, seed,
                              "inner", journal, {"alpha": ALPHA})
            mapped = map_predictions(bundle, val_idx, np.asarray(fit["prediction"]))
            full_base[base] = mapped
            inner_base[base].update(mapped)
            inner_predictions[base].update(mapped)

        for method in ("additive_q", "additive_chem", "lowrank_q", "lowrank_chem"):
            fit = attempt_fit(bundle, method, train_idx, val_idx, seed,
                              "inner", journal, {"alpha": ALPHA, "rank": 4})
            inner_predictions[method].update(map_predictions(bundle, val_idx, fit["prediction"]))
        for method in ("kernel_q", "kernel_chem"):
            for weight in KERNEL_WEIGHTS:
                fit = attempt_fit(bundle, method, train_idx, val_idx, seed,
                    "inner", journal, {"alpha": ALPHA, "kernel_weight": weight, "kernel_sigma": 1.0})
                key = f"{method}@{weight:.2f}"
                inner_predictions.setdefault(key, {}).update(map_predictions(bundle, val_idx, fit["prediction"]))

        # Cross-fit base predictions inside this exact inner-training partition.
        oof_base = {m: {} for m in ("boltz", "aim", "chem")}
        cf_rows = [r for r in crossfit_plan if r["dataset"] == dataset
                   and int(r["model_seed"]) == seed and int(r["outer_fold"]) == outer_fold
                   and int(r["inner_fold"]) == inner_fold]
        if len(cf_rows) != 2:
            raise ValueError("Frozen inner cross-fit split count differs")
        for split in cf_rows:
            train_cf = np.asarray([bundle["cids"].index(c) for c in split["fit_ids"]], dtype=int)
            predict_cf = np.asarray([bundle["cids"].index(c) for c in split["predict_ids"]], dtype=int)
            for base in ("boltz", "aim", "chem"):
                values = cached_base_fit(bundle, base, train_cf, predict_cf, seed,
                    "inner_stack_crossfit", journal, out)
                oof_base[base].update(map_predictions(bundle, predict_cf, values))
        train_ids = [bundle["cids"][i] for i in train_idx]
        if any(set(oof_base[m]) != set(train_ids) for m in oof_base):
            raise ValueError("Stack cross-fit predictions do not cover the inner training IDs")
        for stack, bases in (("stack_q", ("boltz", "aim", "chem")),
                             ("stack_chem", ("boltz", "chem"))):
            meta_train = np.asarray([[oof_base[b][c] for b in bases] for c in train_ids])
            meta_query = np.asarray([[full_base[b][c] for b in bases]
                                     for c in (bundle["cids"][i] for i in val_idx)])
            fit = fit_meta(bundle, meta_train, bundle["y"][train_idx], train_idx,
                meta_query, val_idx, stack, seed, "inner", journal)
            inner_predictions[stack].update(map_predictions(bundle, val_idx, fit["prediction"]))

    # Inner score pools all held-group pair errors, then averages groups equally.
    for method, pred in inner_predictions.items():
        if set(pred) != set(inner_ids):
            continue
        score, group_losses = score_by_group(bundle, pred, inner_ids)
        inner_scores[method] = [score]
        if not np.isfinite(score):
            raise FloatingPointError("Inner selection score is non-finite")
    kernel_choices = {}
    for method in ("kernel_q", "kernel_chem"):
        choices = [(inner_scores[f"{method}@{w:.2f}"][0], order, w)
                   for order, w in enumerate(KERNEL_WEIGHTS)]
        score, _, weight = min(choices)
        kernel_choices[method] = {"weight": weight, "score": score}
        inner_scores[method] = [score]
    fusion_order = ("additive_q", "additive_chem", "kernel_q", "kernel_chem",
                    "stack_q", "stack_chem", "lowrank_q", "lowrank_chem")
    selected_family = min(fusion_order, key=lambda m: inner_scores[m][0])
    score_row = {"dataset": dataset, "model_seed": seed, "outer_fold": outer_fold,
        "candidate_scores": inner_scores, "kernel_choices": kernel_choices,
        "selected_family": selected_family}
    append_jsonl(score_path, score_row)

    # Refit components once. The stacking meta models use inner OOF predictions.
    outer_base: dict[str, dict[str, float]] = {}
    for base in ("boltz", "aim", "chem"):
        fit = attempt_fit(bundle, base, outer_train, outer_test, seed,
                          "outer", journal, {"alpha": ALPHA})
        outer_base[base] = map_predictions(bundle, outer_test, fit["prediction"])
    outer_predictions: dict[str, dict[str, float]] = dict(outer_base)
    for method in ("additive_q", "additive_chem", "lowrank_q", "lowrank_chem"):
        fit = attempt_fit(bundle, method, outer_train, outer_test, seed,
                          "outer", journal, {"alpha": ALPHA, "rank": 4})
        outer_predictions[method] = map_predictions(bundle, outer_test, fit["prediction"])
    for method in ("kernel_q", "kernel_chem"):
        weight = kernel_choices[method]["weight"]
        fit = attempt_fit(bundle, method, outer_train, outer_test, seed, "outer",
            journal, {"alpha": ALPHA, "kernel_weight": weight, "kernel_sigma": 1.0})
        outer_predictions[method] = map_predictions(bundle, outer_test, fit["prediction"])
    train_ids = [bundle["cids"][i] for i in outer_train]
    for stack, bases in (("stack_q", ("boltz", "aim", "chem")),
                         ("stack_chem", ("boltz", "chem"))):
        meta_train = np.asarray([[inner_base[b][c] for b in bases] for c in train_ids])
        meta_query = np.asarray([[outer_base[b][bundle["cids"][i]] for b in bases]
                                 for i in outer_test])
        fit = fit_meta(bundle, meta_train, bundle["y"][outer_train], outer_train,
            meta_query, outer_test, stack, seed, "outer", journal)
        outer_predictions[stack] = map_predictions(bundle, outer_test, fit["prediction"])
    outer_predictions["selected_fusion"] = outer_predictions[selected_family]
    fold_info = {"dataset": dataset, "model_seed": seed, "outer_fold": outer_fold,
                 "selected_family": selected_family,
                 "selected_kernel_weight": kernel_choices.get(selected_family, {}).get("weight"),
                 "inner_score": inner_scores[selected_family]}
    append_jsonl(selection_path, fold_info)

    prior_method = {"prior_boltz_ridge": "boltz_ridge",
                    "prior_ordinary_readout": "ordinary_readout",
                    "prior_quantum_only": "prior_quantum_only"}
    for method, pred in outer_predictions.items():
        for cid, value in pred.items():
            prediction_writer({"dataset": dataset, "model_seed": seed, "method": method,
                "pubchem_cid": cid, "outer_fold": outer_fold,
                "y_true": bundle["y"][bundle["cids"].index(cid)], "prediction": value,
                "status": "predicted", "failure_reason": "", "feature_valid": True,
                "cohort_valid": True, "selected_family": selected_family if method == "selected_fusion" else "",
                "selected_kernel_weight": fold_info["selected_kernel_weight"] if method == "selected_fusion" else "",
                "matched_chemistry_family": (selected_family[:-2]+"_chem" if selected_family.endswith("_q") else selected_family) if method == "selected_fusion" else ""})
    for method, source_method in prior_method.items():
        prior_map = prior.get(dataset, {}).get(source_method, {}).get(seed, {})
        for i in outer_test:
            cid = bundle["cids"][i]
            if method == "prior_boltz_ridge" and cid in prior_map:
                value = prior_map[cid]
            elif method == "prior_ordinary_readout" and cid in prior_map:
                value = prior_map[cid]
            elif method == "prior_quantum_only" and cid in prior_map:
                value = prior_map[cid]
            else:
                continue
            prediction_writer({"dataset": dataset, "model_seed": seed, "method": method,
                "pubchem_cid": cid, "outer_fold": outer_fold, "y_true": bundle["y"][i],
                "prediction": value, "status": "predicted", "failure_reason": "",
                "feature_valid": bool(bundle["valid"][i]), "cohort_valid": True,
                "selected_family": "", "selected_kernel_weight": "", "matched_chemistry_family": ""})
    for i in outer_test:
        cid = bundle["cids"][i]
        prediction_writer({"dataset": dataset, "model_seed": seed, "method": "native",
            "pubchem_cid": cid, "outer_fold": outer_fold, "y_true": bundle["y"][i],
            "prediction": bundle["native"][i], "status": "predicted", "failure_reason": "",
            "feature_valid": bool(bundle["valid"][i]), "cohort_valid": True,
            "selected_family": "", "selected_kernel_weight": "", "matched_chemistry_family": ""})


def verify_frozen_start(out: Path, gate_path: Path) -> dict:
    receipt_path = out/"preflight_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if receipt.get("status") != "PREFLIGHT_PASS_NO_FITS" or receipt.get("schedule_attempts") != 1470:
        raise ValueError("Preflight receipt is incomplete or does not match the frozen schedule")
    if receipt.get("attempted_fits") != 0:
        raise ValueError("The preflight already records model fit attempts")
    gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
    required_gate = ("execution_authorized", "feature_manifest_frozen", "source_review_pass", "preflight_pass")
    if any(gate.get(k) is not True for k in required_gate):
        raise ValueError("Scientific gate receipt does not authorize this execution")
    if digest(receipt_path) != gate.get("preflight_receipt_sha256"):
        raise ValueError("Gate receipt does not bind the current preflight receipt")
    expected_gate_path = out.parent/"execution_gate.json"
    if gate_path.resolve() != expected_gate_path.resolve():
        raise ValueError("Use the frozen run-root execution_gate.json")
    freeze_path = out.parent/"plan_freeze_receipt.json"
    plan_path = out.parent/"FROZEN_PLAN.md"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
    if digest(plan_path) != freeze.get("plan_sha256") or digest(plan_path) != receipt.get("frozen_plan_sha256"):
        raise ValueError("Frozen plan hash differs from the freeze and preflight receipts")
    if digest(FEATURES) != receipt.get("feature_sha256"):
        raise ValueError("Feature bundle hash changed after preflight")
    if digest(PREVIOUS/"aim_features.schema.json") != receipt.get("schema_sha256"):
        raise ValueError("Feature schema hash changed after preflight")
    for raw_path, expected in receipt["input_and_code_hashes"].items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path.cwd()/path
        if not path.exists() or digest(path) != expected:
            raise ValueError(f"Frozen input or source code hash changed: {path}")
    return gate


def final_predictions(out: Path, bundle: dict, prior: dict) -> Path:
    partial = out/"partial_predictions.csv"
    partial_rows = load_csv(partial) if partial.exists() else []
    available = {}
    for r in partial_rows:
        key = (r["dataset"], int(r["model_seed"]), r["method"], r["pubchem_cid"])
        if key in available:
            raise ValueError(f"Duplicate partial prediction: {key}")
        available[key] = r
    methods = ("native", "prior_boltz_ridge", "prior_ordinary_readout", "prior_quantum_only",
               "boltz", "aim", "chem", "additive_q", "additive_chem", "kernel_q",
               "kernel_chem", "stack_q", "stack_chem", "lowrank_q", "lowrank_chem",
               "selected_fusion")
    prior_key = {"prior_boltz_ridge": "boltz_ridge", "prior_ordinary_readout": "ordinary_readout",
                 "prior_quantum_only": "prior_quantum_only"}
    fieldnames = ["dataset", "model_seed", "method", "pubchem_cid", "outer_fold", "y_true",
        "prediction", "status", "failure_reason", "feature_valid", "cohort_valid",
        "selected_family", "selected_kernel_weight", "matched_chemistry_family"]
    path = out/"predictions.csv"
    rows = []
    for dataset, sensitivity in (("primary", False), ("hill_sensitivity", True)):
        output_dataset = "sensitivity" if sensitivity else "primary"
        ds_key = dataset
        valid = select_valid(bundle, sensitivity)
        for seed in SEEDS:
            for method in methods:
                for i, cid in enumerate(bundle["cids"]):
                    cohort = bool(valid[i])
                    feature_valid = bool(bundle["valid"][i])
                    item = {"dataset": output_dataset, "model_seed": seed, "method": method,
                        "pubchem_cid": cid, "outer_fold": int(bundle["outer"][i]),
                        "y_true": float(bundle["y"][i]), "prediction": "", "status": "missing",
                        "failure_reason": "", "feature_valid": feature_valid,
                        "cohort_valid": cohort, "selected_family": "",
                        "selected_kernel_weight": "", "matched_chemistry_family": ""}
                    if method == "native":
                        if not sensitivity or cohort:
                            item.update(prediction=float(bundle["native"][i]), status="predicted")
                        else:
                            item["failure_reason"] = "fixed_hill_sensitivity_exclusion"
                    elif method in prior_key:
                        source_method = prior_key[method]
                        p = prior.get(ds_key, {}).get(source_method, {}).get(seed, {}).get(cid)
                        if p is not None:
                            item.update(prediction=p, status="predicted")
                        else:
                            item["failure_reason"] = "prior_feature_missing_or_fixed_sensitivity_exclusion"
                    else:
                        saved = available.get((ds_key, seed, method, cid))
                        if saved:
                            item.update(prediction=float(saved["prediction"]), status=saved["status"],
                                selected_family=saved.get("selected_family", ""),
                                selected_kernel_weight=saved.get("selected_kernel_weight", ""),
                                matched_chemistry_family=saved.get("matched_chemistry_family", ""))
                        else:
                            item["failure_reason"] = ("fixed_hill_sensitivity_exclusion" if sensitivity and cid in ("899202", "976329")
                                else "feature_invalid" if not feature_valid else "no_outer_test_prediction")
                    rows.append(item)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    return path


def execute(out: Path, gate_path: Path) -> None:
    verify_frozen_start(out, gate_path)
    out.mkdir(parents=True, exist_ok=True)
    if support.torch is not None:
        support.torch.set_num_threads(1)
        try:
            support.torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    bundle = load_bundle()
    prior = load_frozen_oof(bundle)
    crossfit_plan = load_crossfit_splits(out)
    journal = FitJournal(out)
    partial = out/"partial_predictions.csv"
    columns = ["dataset", "model_seed", "method", "pubchem_cid", "outer_fold", "y_true",
        "prediction", "status", "failure_reason", "feature_valid", "cohort_valid",
        "selected_family", "selected_kernel_weight", "matched_chemistry_family"]
    if not partial.exists():
        with partial.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=columns).writeheader()
    def prediction_writer(row: dict):
        with partial.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns); w.writerow(row)
    selection_path = out/"selection_manifest.jsonl"
    score_path = out/"inner_candidate_scores.jsonl"
    completed_path = out/"completed_units.jsonl"
    for dataset, sensitivity in (("primary", False), ("hill_sensitivity", True)):
        for seed in SEEDS:
            for outer in range(5):
                unit_key = f"{dataset}|{seed}|{outer}"
                completed = set()
                if completed_path.exists():
                    with completed_path.open(encoding="utf-8") as f:
                        completed = {json.loads(line)["unit"] for line in f if line.strip()}
                if unit_key in completed:
                    continue
                remove_incomplete_unit(out, dataset, seed, outer)
                run_unit(bundle, dataset, sensitivity, seed, outer, journal, out, prior,
                         crossfit_plan, prediction_writer, selection_path, score_path)
                append_jsonl(completed_path, {"unit": unit_key, "attempted_fits": journal.count,
                    "elapsed_seconds": time.time()-journal.started_unix})
                final_predictions(out, bundle, prior)
    final_path = final_predictions(out, bundle, prior)
    receipt = {"status": "COMPLETE", "attempted_fits": journal.count,
        "scheduled_fits": 1470, "fit_cap": 1500, "failure_reserve": 30,
        "elapsed_seconds": time.time()-journal.started_unix,
        "maximum_seconds": WALL_SECONDS, "feature_sha256": digest(FEATURES),
        "prediction_path": str(final_path), "prediction_sha256": digest(final_path),
        "fit_journal_sha256": digest(journal.path),
        "selection_manifest_sha256": digest(selection_path),
        "inner_candidate_scores_sha256": digest(score_path)}
    save_json(out/"classical_run_receipt.json", receipt)


def select_valid(bundle: dict, sensitivity: bool) -> np.ndarray:
    valid = bundle["valid"].copy()
    if sensitivity:
        for cid in ("899202", "976329"):
            valid[bundle["cids"].index(cid)] = False
    return valid


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--authorize-fitting", action="store_true")
    ap.add_argument("--gate-receipt", type=Path)
    args = ap.parse_args()
    if not args.authorize_fitting:
        ap.error("Loader mode only. Fitting needs explicit lead authorization.")
    if args.gate_receipt is None or not args.gate_receipt.is_file():
        ap.error("Fitting needs the frozen scientific gate receipt.")
    gate = json.loads(args.gate_receipt.read_text(encoding="utf-8-sig"))
    if gate.get("execution_authorized") is not True or gate.get("feature_manifest_frozen") is not True:
        ap.error("The gate receipt does not authorize the frozen feature bundle.")
    if not args.output.exists():
        ap.error("Run the zero-fit preflight first. The output directory must exist.")
    execute(args.output, args.gate_receipt)


if __name__ == "__main__":
    main()
