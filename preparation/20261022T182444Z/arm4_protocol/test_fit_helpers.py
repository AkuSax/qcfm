import sys
import unittest
import shutil
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_entrypoint as fit


class FitHelperTests(unittest.TestCase):
    def test_pair_index_mapping_and_metric_cids_are_distinct(self):
        records = {"a": {"chemical_group": "g"}, "b": {"chemical_group": "g"},
                   "c": {"chemical_group": "h"}}
        self.assertEqual(fit.pairs_for_ids(["c", "b", "a"], records), [("g", 1, 2)])
        self.assertEqual(fit.metric_pairs_for_ids(["c", "b", "a"], records), [("g", "b", "a")])

    def test_ridge_pair_rows_have_zero_intercept_anchor_rows_have_one(self):
        records = {"a": {"chemical_group": "g"}, "b": {"chemical_group": "g"}}
        A, b, w = fit.build_ridge_system(["a", "b"], {"a": 2., "b": 1.},
                                         {"a": 0., "b": 0.}, np.array([[1., 3.], [4., 8.]]),
                                         {"a": 0, "b": 1}, records)
        self.assertEqual(A.shape, (3, 3))
        self.assertEqual(A[0, -1], 0.)
        self.assertEqual(A[1, -1], 1.)
        self.assertEqual(A[2, -1], 1.)

    def test_quantum_only_score_ignores_boltz_and_native_inputs(self):
        rng = np.random.default_rng(4)
        data = {"boltz_atoms": rng.normal(size=(2, 3, 128)).astype("f"),
                "aim_atoms": rng.normal(size=(2, 3, 256)).astype("f"),
                "global_boltz": rng.normal(size=(2, 768)).astype("f"),
                "atom_mask": np.ones((2, 3), dtype=bool),
                "allowed": np.ones((2, 3, 3), dtype=bool),
                "simple_scalars": rng.normal(size=(2, 5)).astype("f"),
                "aim_scalars": rng.normal(size=(2, 7)).astype("f")}
        model = fit.new_model("quantum_only", None, 11, torch.device("cpu"), data).eval()
        with torch.no_grad():
            p1 = fit.model_scores("quantum_only", model, data, [0, 1], np.zeros((2, 384)),
                                  np.zeros((2, 384)), np.array([100., -100.]), torch.device("cpu"))
            data2 = dict(data); data2["boltz_atoms"] = data["boltz_atoms"] * 1000
            data2["global_boltz"] = data["global_boltz"] * -1000
            p2 = fit.model_scores("quantum_only", model, data2, [0, 1], np.ones((2, 384)),
                                  np.ones((2, 384)), np.array([-100., 100.]), torch.device("cpu"))
        self.assertTrue(torch.equal(p1, p2))

    def test_fit_budget_is_exact_and_within_cap(self):
        plan = fit.planned_fit_count()
        self.assertEqual(plan, {"full_primary": 735, "fixed_sensitivity": 735,
                                "no_retuning_primary_controls": 30, "total": 1500})
        self.assertEqual(plan["total"], fit.MAX_FITS)

    def test_fit_attempt_is_persisted_before_model_operation(self):
        out = HERE / "_synthetic_fit_progress_test"
        out.mkdir(exist_ok=True)
        try:
            state = {"fit_count": 0}
            count = fit.reserve_fit(state, out, "primary", 1729, 0, "ridge", {"alpha": 1.})
            self.assertEqual(count, 1)
            receipt = __import__("json").loads((out/"fit_progress.json").read_text())
            self.assertEqual(receipt["active_fit"]["method"], "ridge")
            state["fit_count"] = fit.MAX_FITS
            with self.assertRaisesRegex(RuntimeError, "before fit attempt"):
                fit.reserve_fit(state, out, "primary", 1729, 0, "ridge", {})
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
