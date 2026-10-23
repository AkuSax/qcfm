"""Fold-local low-rank feature maps for the frozen fusion comparison.

This module builds design matrices only. The shared runner owns all estimator
fits, score journals, and attempt accounting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


RANK_CAP = 4
METHODS = ("additive_q", "additive_chem", "lowrank_q", "lowrank_chem")


def _raw_blocks(bundle: Mapping[str, Any], family: str) -> tuple[np.ndarray, np.ndarray]:
    """Return Boltz and second-modality blocks from bundle['X'] arrays."""
    x = bundle["X"]
    mask = np.asarray(x["atom_mask"], dtype=bool)
    denom = mask.sum(axis=1, keepdims=True)
    denom = np.maximum(denom, 1)

    boltz_global = np.asarray(x["global_boltz"], dtype=np.float64)
    boltz_atoms = np.asarray(x["boltz_atoms"], dtype=np.float64)
    boltz_mean = np.where(mask[..., None], boltz_atoms, 0.0).sum(axis=1) / denom
    boltz = np.concatenate((boltz_global, boltz_mean), axis=1)

    if family == "q":
        aim_bound = np.asarray(x["aim_atoms"], dtype=np.float64)
        aim_free = np.asarray(x["aim_atoms_free"], dtype=np.float64)
        aim_bound_mean = np.where(mask[..., None], aim_bound, 0.0).sum(axis=1) / denom
        aim_free_mean = np.where(mask[..., None], aim_free, 0.0).sum(axis=1) / denom
        scalar = np.asarray(x["aim_scalars"], dtype=np.float64)
        scalar_free = np.asarray(x["aim_scalars_free"], dtype=np.float64)
        simple = np.asarray(x["simple_scalars"], dtype=np.float64)
        second = np.concatenate((aim_bound_mean, aim_free_mean, scalar, scalar_free, simple), axis=1)
    elif family == "chem":
        second = np.asarray(x["simple_scalars"], dtype=np.float64)
    else:
        raise ValueError(f"Unknown feature family: {family}")
    return boltz, second


@dataclass
class _BlockProjector:
    rank: int
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    axes: np.ndarray | None = None
    score_scale: np.ndarray | None = None
    block_scale: float = 1.0

    def fit(self, values: np.ndarray) -> "_BlockProjector":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("A feature block needs at least two training rows")
        if not np.isfinite(values).all():
            raise ValueError("Training feature block contains non-finite values")
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0)
        self.scale[self.scale <= 1e-12] = 1.0
        z = (values - self.mean) / self.scale
        _, singular, vt = np.linalg.svd(z, full_matrices=False)
        rank = min(self.rank, values.shape[0] - 1, values.shape[1], len(singular))
        if rank < 1:
            raise ValueError("No PCA component is available")
        axes = vt[:rank].T.copy()
        # Fix the arbitrary SVD sign: the largest absolute loading is positive.
        for component in range(rank):
            pivot = int(np.argmax(np.abs(axes[:, component])))
            if axes[pivot, component] < 0:
                axes[:, component] *= -1.0
        self.axes = axes
        scores = z @ self.axes
        self.score_scale = scores.std(axis=0)
        self.score_scale[self.score_scale <= 1e-12] = 1.0
        self.block_scale = float(np.sqrt(rank))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.axes is None or self.scale is None or self.score_scale is None:
            raise RuntimeError("Projector is not fitted")
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.mean):
            raise ValueError("Feature block shape differs from its training block")
        if not np.isfinite(values).all():
            raise ValueError("Feature block contains non-finite values")
        out = ((values - self.mean) / self.scale) @ self.axes
        out = out / self.score_scale / self.block_scale
        if not np.isfinite(out).all():
            raise ValueError("Projected feature block contains non-finite values")
        return out


@dataclass
class FoldDesign:
    family: str
    additive: bool
    first: _BlockProjector
    second: _BlockProjector

    @classmethod
    def fit(cls, bundle: Mapping[str, Any], train_idx: Sequence[int], family: str,
            rank: int = RANK_CAP, additive: bool = False) -> "FoldDesign":
        first, second = _raw_blocks(bundle, family)
        idx = np.asarray(train_idx, dtype=int)
        if idx.ndim != 1 or len(idx) < 2 or np.any(idx < 0) or np.any(idx >= len(first)):
            raise ValueError("Training indices are invalid")
        valid = np.asarray(bundle["valid"], dtype=bool)
        if not valid[idx].all():
            raise ValueError("Training indices include invalid feature rows")
        if np.any(np.asarray(bundle["X"]["atom_mask"], dtype=bool)[idx].sum(axis=1) == 0):
            raise ValueError("A valid training row has no active atoms")
        return cls(family, additive,
                   _BlockProjector(rank).fit(first[idx]),
                   _BlockProjector(rank).fit(second[idx]))

    def transform(self, bundle: Mapping[str, Any], indices: Sequence[int]) -> np.ndarray:
        first, second = _raw_blocks(bundle, self.family)
        idx = np.asarray(indices, dtype=int)
        if idx.ndim != 1 or np.any(idx < 0) or np.any(idx >= len(first)):
            raise ValueError("Query indices are invalid")
        valid = np.asarray(bundle["valid"], dtype=bool)
        if not valid[idx].all():
            raise ValueError("Query indices include invalid feature rows")
        if np.any(np.asarray(bundle["X"]["atom_mask"], dtype=bool)[idx].sum(axis=1) == 0):
            raise ValueError("A valid query row has no active atoms")
        a, b = self.first.transform(first[idx]), self.second.transform(second[idx])
        blocks = [a, b]
        if not self.additive:
            if a.shape[1] != b.shape[1]:
                raise ValueError("Paired blocks must have equal projected rank")
            product = a * b
            if not hasattr(self, "product_scale"):
                raise RuntimeError("Interaction design lacks its training product scale")
            blocks.append((product - self.product_mean) / self.product_scale / np.sqrt(product.shape[1]))
        out = np.concatenate(blocks, axis=1)
        if not np.isfinite(out).all():
            raise ValueError("Design matrix contains non-finite values")
        return out


def build_designs(bundle: Mapping[str, Any], train_idx: Sequence[int],
                  query_idx: Sequence[int], method: str,
                  params: Mapping[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build train/query matrices with fold-local scaling and PCA.

    q pairs Boltz global plus mean atom features with AIM bound/free means and
    AIM scalars. chem pairs the same Boltz block with simple chemistry scalars.
    Each block uses train-only standardization and PCA. Each projected block
    has unit train-row RMS. The interaction uses matched component axes and
    has unit train-row RMS. Main effects are always included.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    params = dict(params or {})
    rank = int(params.get("rank", RANK_CAP))
    if rank != RANK_CAP:
        raise ValueError(f"The frozen rank cap is {RANK_CAP}")
    alpha = float(params.get("alpha", 0.1))
    if alpha != 0.1:
        raise ValueError("The frozen ridge alpha is 0.1")
    family = "q" if method.endswith("_q") else "chem"
    additive = method.startswith("additive_")
    design = FoldDesign.fit(bundle, train_idx, family, rank=rank, additive=additive)
    if not additive:
        train_a = design.first.transform(_raw_blocks(bundle, family)[0][np.asarray(train_idx, dtype=int)])
        train_b = design.second.transform(_raw_blocks(bundle, family)[1][np.asarray(train_idx, dtype=int)])
        train_product = train_a * train_b
        design.product_mean = train_product.mean(axis=0)
        design.product_scale = train_product.std(axis=0)
        design.product_scale[design.product_scale <= 1e-12] = 1.0
        if not np.isfinite(design.product_scale).all():
            raise ValueError("Training interaction block has invalid scale")
    return design.transform(bundle, train_idx), design.transform(bundle, query_idx)
