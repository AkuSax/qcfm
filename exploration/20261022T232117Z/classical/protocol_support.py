"""Leakage-safe loaders, metrics, and synthetic-test model shells for QGCF."""
from __future__ import annotations
import csv, hashlib
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np

ROOT = Path(r"C:\Users\Saxen\OneDrive\Documents\qcfm\preparation\20261022T182444Z\arm4_protocol")
SOURCE = Path(r"C:\Users\Saxen\ppi_IG\results\representation_diagnosis\20261015\attempt02\gcp_a100_retry_01\collected")
EXPECTED = {
    "records.csv": "b6c9473de5dd8d09af35f5b5e42983498ee468a15e96c19b0ba066c3cc20791c",
    "splits.csv": "137c76616c1940cdddabd021bb6d5d1f9b1ead0a78dddd450de73ac88dbb0888",
    "exact_inputs.npz": "5dcbd0011ed01995fe76a71c12f62073be5a7905c7cb6e16e9cee7402fea1540",
}



class FoldPreprocessor:
    """Fit median imputation and standard scaling on the supplied training rows only."""
    def fit(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or not len(x):
            raise ValueError("Training features must be a non-empty matrix")
        self.median_ = np.nanmedian(x, axis=0)
        if not np.isfinite(self.median_).all():
            raise ValueError("A feature is missing in every training row")
        filled = np.where(np.isfinite(x), x, self.median_)
        self.mean_ = filled.mean(axis=0)
        self.scale_ = filled.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError("Feature shape differs from the fitted training matrix")
        filled = np.where(np.isfinite(x), x, self.median_)
        return (filled - self.mean_) / self.scale_


def rankdata_average(x: Sequence[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
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


def spearman_rho(y: Sequence[float], pred: Sequence[float]) -> float:
    ry, rp = rankdata_average(y), rankdata_average(pred)
    if np.std(ry) == 0 or np.std(rp) == 0:
        return float("nan")
    return float(np.corrcoef(ry, rp)[0, 1])


def kendall_tau_b(y: Sequence[float], pred: Sequence[float]) -> float:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    c = d = tx = ty = 0
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            sy, sp = np.sign(y[i] - y[j]), np.sign(pred[i] - pred[j])
            if sy == 0 and sp == 0: continue
            if sy == 0: ty += 1
            elif sp == 0: tx += 1
            elif sy == sp: c += 1
            else: d += 1
    den = np.sqrt((c+d+tx) * (c+d+ty))
    return float((c-d)/den) if den else float("nan")


def evaluate_predictions(y: Mapping[str, float], pred: Mapping[str, float],
                         pairs: Sequence[tuple[str, str, str]]) -> dict:
    """Compute declared primary and secondary metrics on one common identity set."""
    ids = sorted(set(y) & set(pred))
    if not ids or len(ids) != len(set(y)):
        raise ValueError("Predictions must cover the declared common identity set")
    yt = np.asarray([y[c] for c in ids], float)
    yp = np.asarray([pred[c] for c in ids], float)
    per_group = group_pair_losses(y, pred, pairs)
    errors, order_scores, exact_signs, cliff_signs, group_rmse = [], [], [], [], {}
    for group, a, b in pairs:
        dy = float(y[a]) - float(y[b]); dp = float(pred[a]) - float(pred[b])
        e = dp - dy
        errors.append(e)
        if dy != 0:
            order_scores.append(1.0 if np.sign(dp) == np.sign(dy) else (0.5 if dp == 0 else 0.0))
        exact_signs.append(float(np.sign(dp) == np.sign(dy)))
        if abs(dy) >= .3: cliff_signs.append(float(np.sign(dp) == np.sign(dy)))
        group_rmse.setdefault(group, []).append(e*e)
    return {
        "n_records": len(ids), "n_groups_with_pairs": len(per_group), "n_pairs": len(pairs),
        "macro_group_pair_mae": float(np.mean(list(per_group.values()))),
        "macro_group_pair_rmse": float(np.mean([np.sqrt(np.mean(v)) for v in group_rmse.values()])),
        "pair_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "spearman_rho": spearman_rho(yt, yp), "kendall_tau_b": kendall_tau_b(yt, yp),
        "ligand_mae": float(np.mean(np.abs(yp-yt))),
        "ligand_rmse": float(np.sqrt(np.mean((yp-yt)**2))),
        "pair_ordering_accuracy": float(np.mean(order_scores)) if order_scores else float("nan"),
        "exact_label_sign_accuracy": float(np.mean(exact_signs)) if exact_signs else float("nan"),
        "sign_accuracy_abs_delta_ge_0_3": float(np.mean(cliff_signs)) if cliff_signs else float("nan"),
    }


def group_balanced_pair_mse(scores, labels, pairs):
    """Return pair MSE averaged within group, then across groups."""
    if torch is None:
        raise ImportError("PyTorch is required for pair training loss")
    by_group = {}
    for group, i, j in pairs:
        e = (scores[i] - scores[j]) - (labels[i] - labels[j])
        by_group.setdefault(group, []).append(e.square())
    if not by_group:
        raise ValueError("Training split has no eligible within-group pairs")
    return torch.stack([torch.stack(v).mean() for v in by_group.values()]).mean()


def anchored_group_pair_loss(scores, labels, pairs, anchor_weight: float = 0.1):
    """Add the frozen 0.1 ligand-score MSE anchor to group-balanced pair MSE."""
    pair = group_balanced_pair_mse(scores, labels, pairs)
    anchor = torch.mean((scores - labels).square())
    return pair + anchor_weight * anchor


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_frozen_pooled_inputs(source: Path = SOURCE) -> tuple[list[str], np.ndarray, np.ndarray, list[dict]]:
    """Load verified 384-d final-readout inputs in records.csv row order."""
    for name, expected in EXPECTED.items():
        path = source / name
        if sha256(path) != expected:
            raise ValueError(f"Source hash mismatch: {name}")
    with (source / "records.csv").open(newline="", encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))
    if len(records) != 65 or len({r["pubchem_cid"] for r in records}) != 65:
        raise ValueError("Unexpected record identities")
    with np.load(source / "exact_inputs.npz", allow_pickle=False) as z:
        x1, x2 = z["x1"].copy(), z["x2"].copy()
    if x1.shape != (65, 384) or x2.shape != (65, 384):
        raise ValueError("Unexpected final-readout input shape")
    if not np.isfinite(x1).all() or not np.isfinite(x2).all():
        raise ValueError("Non-finite frozen input")
    return [r["pubchem_cid"] for r in records], x1, x2, records


def load_atom_features(path: Path, expected_cids: Sequence[str]) -> dict[str, np.ndarray]:
    """Validate a row-aligned feature bundle without deserializing Python objects."""
    with np.load(path, allow_pickle=False) as z:
        required = {"cids", "boltz_atoms", "aim_atoms", "global_boltz", "atom_mask",
                    "allowed", "simple_scalars", "aim_scalars", "feature_valid",
                    "aim_atoms_free", "aim_scalars_free"}
        missing = required - set(z.files)
        if missing:
            raise ValueError(f"Missing feature arrays: {sorted(missing)}")
        out = {k: z[k].copy() for k in required}
    got = [str(x) for x in out["cids"].tolist()]
    if got != list(expected_cids):
        raise ValueError("Feature CID order differs from the frozen record order")
    n = len(got); valid = out["feature_valid"].astype(bool)
    if n != 65 or valid.shape != (n,):
        raise ValueError("Feature validity must align to all 65 source records")
    if out["boltz_atoms"].shape[:1] != (n,) or out["boltz_atoms"].shape[-1] != 128:
        raise ValueError("Boltz atom context must have 128 channels")
    if out["aim_atoms"].shape[:1] != (n,) or out["aim_atoms"].shape[-1] != 256:
        raise ValueError("AIM atom features must have 256 channels")
    if out["aim_atoms_free"].shape != out["aim_atoms"].shape:
        raise ValueError("Free AIM atom features must match the bound feature shape")
    if out["aim_scalars_free"].shape != out["aim_scalars"].shape:
        raise ValueError("Free AIM scalar features must match the bound scalar shape")
    if out["global_boltz"].shape != (n, 768):
        raise ValueError("Global Boltz inputs must have shape (65,768)")
    if out["atom_mask"].shape != out["boltz_atoms"].shape[:2]:
        raise ValueError("Atom mask shape does not match atom features")
    if out["allowed"].shape != (n, out["atom_mask"].shape[1], out["atom_mask"].shape[1]):
        raise ValueError("Graph mask shape does not match atom features")
    mask = out["atom_mask"]
    if mask.dtype != np.bool_ or mask.shape != (n, out["boltz_atoms"].shape[1]):
        raise ValueError("Atom mask must be boolean and match all 65 rows")
    if np.any(mask[:, 1:] & ~mask[:, :-1]):
        raise ValueError("Active atom rows must precede padded rows")
    for key in ("boltz_atoms", "aim_atoms", "aim_atoms_free"):
        if not np.isfinite(out[key][valid][mask[valid]]).all():
            raise ValueError(f"Valid active atoms contain non-finite values: {key}")
        if not np.isnan(out[key][~mask]).all():
            raise ValueError(f"Masked atom slots must contain NaN padding: {key}")
    for key in ("global_boltz", "simple_scalars", "aim_scalars", "aim_scalars_free"):
        if not np.isfinite(out[key][valid]).all():
            raise ValueError(f"Valid rows contain non-finite values: {key}")
    return out


def load_fold_splits(path: Path = SOURCE / "splits.csv") -> list[dict]:
    """Load saved splits after checking the source manifest hash."""
    if sha256(path) != EXPECTED["splits.csv"]:
        raise ValueError("Split source hash mismatch")
    return list(csv.DictReader(path.open(newline="", encoding="utf-8-sig")))


def require_common_coverage(predictions: Mapping[str, Mapping[str, float]],
                            eligible_cids: Sequence[str],
                            pairs: Sequence[tuple[int, str, str, str]],
                            min_fraction: float = 0.90) -> tuple[list[str], list[tuple[int, str, str, str]]]:
    """Return the common score set, or fail a declared coverage gate."""
    ids = list(eligible_cids)
    if len(ids) != 65:
        raise ValueError("Coverage denominator must include all 65 frozen source records")
    if len(ids) != len(set(ids)):
        raise ValueError("Eligible identities are not unique")
    common = [cid for cid in ids if all(cid in p and np.isfinite(p[cid]) for p in predictions.values())]
    if len(common) / len(ids) < min_fraction:
        raise ValueError("Common prediction coverage is below the declared gate")
    covered = set(common)
    common_pairs = [p for p in pairs if p[2] in covered and p[3] in covered]
    by_fold = {p[0] for p in common_pairs}
    expected_folds = {p[0] for p in pairs}
    if by_fold != expected_folds:
        missing = sorted(expected_folds - by_fold)
        raise ValueError(f"Outer fold(s) have no common eligible pairs: {missing}")
    return common, common_pairs


def group_pair_losses(y: Mapping[str, float], pred: Mapping[str, float],
                      pairs: Sequence[tuple[str, str, str]]) -> dict[str, float]:
    """Return one MAE per chemical group from oriented ligand-score differences."""
    err: dict[str, list[float]] = {}
    for group, a, b in pairs:
        if a not in y or b not in y or a not in pred or b not in pred:
            raise ValueError("Pair member lacks a label or prediction")
        err.setdefault(group, []).append(abs((pred[a] - pred[b]) - (y[a] - y[b])))
    return {g: float(np.mean(v)) for g, v in err.items()}


def macro_group_mae(y: Mapping[str, float], pred: Mapping[str, float],
                    pairs: Sequence[tuple[str, str, str]]) -> float:
    losses = group_pair_losses(y, pred, pairs)
    if not losses:
        raise ValueError("No eligible within-group pairs")
    return float(np.mean(list(losses.values())))


def paired_group_bootstrap_reduction(y: Mapping[str, float], candidate: Mapping[str, float],
                                     comparator: Mapping[str, float],
                                     pairs: Sequence[tuple[str, str, str]],
                                     n_resamples: int = 5000, seed: int = 1729) -> dict:
    """Bootstrap group-level MAE reduction; never resample pair rows."""
    ca, cb = group_pair_losses(y, comparator, pairs), group_pair_losses(y, candidate, pairs)
    groups = sorted(set(ca) & set(cb))
    if len(groups) < 2:
        raise ValueError("Need at least 2 paired groups")
    values = np.asarray([ca[g] - cb[g] for g in groups], dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(n_resamples, len(values)))].mean(axis=1)
    return {"groups": len(groups), "mean_reduction": float(values.mean()),
            "ci95_low": float(np.quantile(draws, .025)),
            "ci95_high": float(np.quantile(draws, .975)),
            "n_resamples": n_resamples, "seed": seed,
            "limit": "Fixed out-of-fold predictions; does not include model-refit uncertainty."}


def coverage_fraction(pred: Mapping[str, float], source_cids: Sequence[str]) -> float:
    if len(source_cids) != 65 or len(set(source_cids)) != 65:
        raise ValueError("Feature coverage denominator must be all 65 source records")
    return sum(c in pred and np.isfinite(pred[c]) for c in source_cids) / 65


def require_feature_coverage(feature_valid: Sequence[bool], min_fraction: float = 0.90) -> float:
    valid = np.asarray(feature_valid, dtype=bool)
    if valid.shape != (65,):
        raise ValueError("Feature coverage must report all 65 frozen source rows")
    fraction = float(valid.mean())
    if fraction < min_fraction:
        raise ValueError("Feature coverage is below the declared 90% gate")
    return fraction


try:
    import torch
    from torch import nn
except ImportError:  # Keep manifest and metrics tools usable without torch.
    torch = None
    nn = None


if nn is not None:
    class _AtomCore(nn.Module):
        def __init__(self, use_quantum: bool, *, boltz_channels: int = 128,
                     quantum_channels: int = 256, projection: int = 32,
                     heads: int = 4, ff: int = 64, dropout: float = 0.1):
            super().__init__()
            self.use_quantum = use_quantum
            self.bq = nn.Linear(boltz_channels, projection)
            self.key = nn.Linear(quantum_channels if use_quantum else boltz_channels, projection)
            self.value = nn.Linear(quantum_channels if use_quantum else boltz_channels, projection)
            self.attn = nn.MultiheadAttention(projection, heads, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(projection)
            self.ffn = nn.Sequential(nn.Linear(projection, ff), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(ff, projection))
            self.norm2 = nn.LayerNorm(projection)
            self.global_proj = nn.Linear(768, projection)
            self.readout = nn.Sequential(nn.Linear(2 * projection, ff), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(ff, 1))

        def forward(self, boltz_atoms, quantum_atoms, global_boltz, allowed, atom_mask=None):
            q = self.bq(boltz_atoms)
            src = quantum_atoms if self.use_quantum else boltz_atoms
            k, v = self.key(src), self.value(src)
            batch, n, _ = q.shape
            # MultiheadAttention uses True for a blocked edge.
            valid = (atom_mask.bool() if atom_mask is not None else
                     torch.ones((batch, n), dtype=torch.bool, device=q.device))
            allowed = allowed.bool() & valid[:, None, :] & valid[:, :, None]
            safe_padded_diagonal = (~valid)[:, :, None] & torch.eye(n, dtype=torch.bool, device=q.device)[None, :, :]
            allowed = allowed | safe_padded_diagonal
            blocked = (~allowed).repeat_interleave(self.attn.num_heads, dim=0)
            attended, _ = self.attn(q, k, v, attn_mask=blocked, need_weights=False)
            h = self.norm1(q + attended)
            h = self.norm2(h + self.ffn(h))
            if atom_mask is None:
                atom_mask = torch.ones((batch, n), dtype=h.dtype, device=h.device)
            weight = atom_mask.to(h.dtype).unsqueeze(-1)
            h = h * weight
            pooled = (h * weight).sum(1) / weight.sum(1).clamp_min(1.0)
            g = self.global_proj(global_boltz)
            return self.readout(torch.cat([pooled, g], dim=-1)).squeeze(-1)

    class QuantumCrossAttention(_AtomCore):
        """QGCF: Boltz atom queries attend to AIM atoms within the fixed graph mask."""
        def __init__(self, **kwargs):
            super().__init__(True, **kwargs)

    class BoltzOnlyAtomAttention(_AtomCore):
        """Matched-capacity control: Boltz atom queries attend to Boltz atoms."""
        def __init__(self, **kwargs):
            kwargs.setdefault("ff", 128)
            super().__init__(False, **kwargs)

    class PooledFusionReadout(nn.Module):
        def __init__(self, boltz_channels=128, quantum_channels=256, global_channels=768, hidden=192):
            super().__init__()
            self.bp = nn.Linear(boltz_channels, 32)
            self.ap = nn.Linear(quantum_channels, 32)
            self.gp = nn.Linear(global_channels, 32)
            self.net = nn.Sequential(nn.Linear(96, hidden), nn.GELU(), nn.Linear(hidden, 1))
        def forward(self, boltz_atoms, quantum_atoms, global_boltz, atom_mask=None):
            if atom_mask is None:
                atom_mask = torch.ones(boltz_atoms.shape[:2], device=boltz_atoms.device)
            w = atom_mask.to(boltz_atoms.dtype).unsqueeze(-1)
            b = (boltz_atoms * w).sum(1) / w.sum(1).clamp_min(1.0)
            a = (quantum_atoms * w).sum(1) / w.sum(1).clamp_min(1.0)
            return self.net(torch.cat([self.bp(b), self.ap(a), self.gp(global_boltz)], dim=-1)).squeeze(-1)

    class AtomConcatReadout(nn.Module):
        def __init__(self, boltz_channels=128, quantum_channels=256, global_channels=768,
                     hidden=280, pooled_channels=32):
            super().__init__()
            self.atom = nn.Linear(boltz_channels + quantum_channels, pooled_channels)
            self.global_proj = nn.Linear(global_channels, 32)
            self.net = nn.Sequential(nn.Linear(pooled_channels + 32, hidden),
                                     nn.GELU(), nn.Linear(hidden, 1))
        def forward(self, boltz_atoms, quantum_atoms, global_boltz, atom_mask=None):
            h = torch.nn.functional.gelu(self.atom(torch.cat([boltz_atoms, quantum_atoms], dim=-1)))
            if atom_mask is None:
                atom_mask = torch.ones(h.shape[:2], device=h.device)
            w = atom_mask.to(h.dtype).unsqueeze(-1)
            h = (h * w).sum(1) / w.sum(1).clamp_min(1.0)
            return self.net(torch.cat([h, self.global_proj(global_boltz)], dim=-1)).squeeze(-1)

    class FinalReadoutEnsemble(nn.Module):
        """Adapt only the 2 saved final predictors. Do not unfreeze upstream features."""
        def __init__(self):
            super().__init__()
            def head():
                return nn.Sequential(nn.Linear(384, 384), nn.ReLU(),
                                     nn.Linear(384, 384), nn.ReLU(), nn.Linear(384, 1))
            self.m1, self.m2 = head(), head()

        @classmethod
        def from_npz(cls, path):
            """Load ordinary final-head weights from a non-pickled NPZ archive."""
            model = cls()
            with np.load(path, allow_pickle=False) as archive:
                state = {}
                for name, key in (("m1", "m1"), ("m2", "m2")):
                    head = getattr(model, name)
                    for layer_idx, key_idx in ((0, 0), (2, 2), (4, 4)):
                        head[layer_idx].weight.data.copy_(torch.as_tensor(archive[f"{key}::{key_idx}.weight"]))
                        head[layer_idx].bias.data.copy_(torch.as_tensor(archive[f"{key}::{key_idx}.bias"]))
            return model

        def forward(self, x1, x2):
            return 0.5 * (self.m1(x1).squeeze(-1) + self.m2(x2).squeeze(-1))

    class DenseReadout(nn.Module):
        """Regularized scalar or pooled-feature readout; caller defines input columns."""
        def __init__(self, input_channels: int, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(input_channels, hidden), nn.GELU(), nn.Linear(hidden, 1))
        def forward(self, x):
            return self.net(x).squeeze(-1)
else:
    class QuantumCrossAttention:  # pragma: no cover - exercised only without torch
        def __init__(self, *args, **kwargs): raise ImportError("PyTorch is required")
    BoltzOnlyAtomAttention = QuantumCrossAttention
    PooledFusionReadout = QuantumCrossAttention
    AtomConcatReadout = QuantumCrossAttention
    DenseReadout = QuantumCrossAttention
    FinalReadoutEnsemble = QuantumCrossAttention
