"""Synthetic transform checks. This script fits no predictor."""
from __future__ import annotations

import numpy as np

from interaction import METHODS, build_designs


def main() -> None:
    rng = np.random.default_rng(1729)
    n, atoms = 18, 7
    mask = np.ones((n, atoms), dtype=bool)
    mask[:, -1] = False

    def atom_block(channels: int) -> np.ndarray:
        out = rng.normal(size=(n, atoms, channels))
        out[:, -1, :] = np.nan
        return out

    x = {
        "global_boltz": rng.normal(size=(n, 768)),
        "boltz_atoms": atom_block(128),
        "atom_mask": mask,
        "aim_atoms": atom_block(256),
        "aim_atoms_free": atom_block(256),
        "aim_scalars": rng.normal(size=(n, 19)),
        "aim_scalars_free": rng.normal(size=(n, 19)),
        "simple_scalars": rng.normal(size=(n, 18)),
    }
    bundle = {"X": x, "valid": np.ones(n, dtype=bool)}
    train = np.arange(12)
    query = np.arange(12, n)
    for method in METHODS:
        train_design, query_design = build_designs(bundle, train, query, method,
                                                   {"rank": 4, "alpha": 0.1})
        expected_width = 8 if method.startswith("additive_") else 12
        assert train_design.shape == (len(train), expected_width)
        assert query_design.shape == (len(query), expected_width)
        assert np.isfinite(train_design).all() and np.isfinite(query_design).all()

    # Query changes cannot alter the fitted train matrix or train transform.
    base_train, base_query = build_designs(bundle, train, query, "lowrank_q")
    changed = {"X": dict(x), "valid": bundle["valid"].copy()}
    changed["X"]["global_boltz"] = x["global_boltz"].copy()
    changed["X"]["global_boltz"][query] += 1000
    changed_train, changed_query = build_designs(changed, train, query, "lowrank_q")
    np.testing.assert_allclose(base_train, changed_train, rtol=0, atol=1e-12)
    assert not np.allclose(base_query, changed_query)
    print("PASS: four synthetic designs are finite and fold-local; no estimator fit")


if __name__ == "__main__":
    main()
