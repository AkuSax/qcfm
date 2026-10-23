import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol_support import (QuantumCrossAttention, BoltzOnlyAtomAttention,
    PooledFusionReadout, AtomConcatReadout, group_pair_losses,
    macro_group_mae, paired_group_bootstrap_reduction,
    require_common_coverage, require_feature_coverage, coverage_fraction,
    FoldPreprocessor, evaluate_predictions, anchored_group_pair_loss)


class SyntheticProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)
        np.random.seed(1729)

    def test_parameter_budgets_and_capacity_matches(self):
        models = [QuantumCrossAttention(), BoltzOnlyAtomAttention(),
                  PooledFusionReadout(), AtomConcatReadout()]
        counts = [sum(p.numel() for p in m.parameters()) for m in models]
        self.assertLessEqual(counts[0], 60000)
        for count in counts[1:]:
            self.assertLessEqual(abs(count-counts[0])/counts[0], .10)

    def test_attention_handles_padding_and_batch_matches_single(self):
        model = QuantumCrossAttention().eval()
        boltz = torch.randn(2, 5, 128)
        aim = torch.randn(2, 5, 256)
        glob = torch.randn(2, 768)
        atom_mask = torch.tensor([[1,1,1,0,0],[1,1,1,1,1]], dtype=torch.bool)
        allowed = atom_mask[:,:,None] & atom_mask[:,None,:]
        # Padded query rows have no allowed key. The implementation must make them safe.
        with torch.no_grad():
            batched = model(boltz, aim, glob, allowed, atom_mask)
            one = model(boltz[:1,:3], aim[:1,:3], glob[:1], allowed[:1,:3,:3], atom_mask[:1,:3])
            two = model(boltz[1:], aim[1:], glob[1:], allowed[1:], atom_mask[1:])
        self.assertTrue(torch.isfinite(batched).all())
        self.assertTrue(torch.allclose(batched[:1], one, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(batched[1:], two, atol=1e-6, rtol=1e-6))
        # The other fusion readouts must accept the same synthetic padded batch.
        pooled = PooledFusionReadout().eval()
        concat = AtomConcatReadout().eval()
        with torch.no_grad():
            self.assertTrue(torch.isfinite(pooled(boltz, aim, glob, atom_mask)).all())
            self.assertTrue(torch.isfinite(concat(boltz, aim, glob, atom_mask)).all())

    def test_boltz_control_handles_padded_attention(self):
        model = BoltzOnlyAtomAttention().eval()
        b = torch.randn(2, 4, 128); a = torch.randn(2, 4, 256); g = torch.randn(2, 768)
        mask = torch.tensor([[1,1,0,0],[1,1,1,1]],dtype=torch.bool)
        allowed = mask[:,:,None] & mask[:,None,:]
        with torch.no_grad(): out = model(b,a,g,allowed,mask)
        self.assertEqual(tuple(out.shape),(2,))
        self.assertTrue(torch.isfinite(out).all())

    def test_group_macro_mae_does_not_weight_dense_group_more(self):
        y={"a":"0", "b":"0", "c":"0", "d":"0", "e":"0", "f":"0"}
        y={k:float(v) for k,v in y.items()}
        p={"a":1.,"b":0.,"c":0.,"d":0.,"e":0.,"f":0.}
        pairs=[("g1","a","b")]+[("g2",a,b) for a,b in [("c","d"),("c","e"),("c","f"),("d","e"),("d","f"),("e","f")]]
        losses=group_pair_losses(y,p,pairs)
        self.assertAlmostEqual(losses["g1"],1.0)
        self.assertAlmostEqual(losses["g2"],0.0)
        self.assertAlmostEqual(macro_group_mae(y,p,pairs),0.5)

    def test_group_pair_loss_balances_groups_and_anchors_singletons(self):
        labels=torch.tensor([0.,1.,0.,0.,3.])
        scores=torch.tensor([0.,0.,0.,0.,1.],requires_grad=True)
        pairs=[("small",0,1),("large",2,3)]
        loss=anchored_group_pair_loss(scores,labels,pairs)
        # Pair MSE is (1 + 0) / 2. The anchor uses every score, including singleton index 4.
        expected_pair=0.5
        expected_anchor=float(torch.mean((scores.detach()-labels)**2))
        self.assertAlmostEqual(float(loss.detach()),expected_pair+0.1*expected_anchor,places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertNotEqual(float(scores.grad[4]),0.0)

    def test_preprocessing_fits_training_rows_only(self):
        train=np.array([[1.0],[3.0],[np.nan]])
        pre=FoldPreprocessor().fit(train)
        out=pre.transform(np.array([[1000.0]]))
        self.assertAlmostEqual(float(pre.mean_[0]),2.0)
        self.assertAlmostEqual(float(out[0,0]),998.0/(2.0/3.0)**0.5)

    def test_secondary_metrics_run_on_synthetic_predictions(self):
        y={"a":0.,"b":1.,"c":2.,"d":3.}
        pred={"a":0.,"b":1.,"c":3.,"d":3.}
        pairs=[("g1","a","b"),("g2","c","d")]
        m=evaluate_predictions(y,pred,pairs)
        self.assertEqual(m["n_records"],4)
        self.assertEqual(m["n_pairs"],2)
        self.assertAlmostEqual(m["ligand_mae"],0.25)
        self.assertAlmostEqual(m["spearman_rho"],0.9486832980505138)
        self.assertAlmostEqual(m["kendall_tau_b"],0.9128709291752769)

    def test_group_bootstrap_is_deterministic(self):
        y={"a":0.,"b":1.,"c":0.,"d":1.,"e":0.,"f":1.}
        q={"a":0.,"b":0.,"c":0.,"d":0.,"e":0.,"f":0.}
        p={"a":0.,"b":1.,"c":0.,"d":1.,"e":0.,"f":1.}
        pairs=[("g1","a","b"),("g2","c","d"),("g3","e","f")]
        a=paired_group_bootstrap_reduction(y,p,q,pairs,n_resamples=500,seed=1729)
        b=paired_group_bootstrap_reduction(y,p,q,pairs,n_resamples=500,seed=1729)
        self.assertEqual(a,b)
        self.assertEqual(a["groups"],3)
        self.assertGreater(a["mean_reduction"],0)

    def test_feature_coverage_keeps_all_65_in_denominator(self):
        valid = np.ones(65, dtype=bool); valid[:3] = False
        self.assertAlmostEqual(require_feature_coverage(valid), 62/65)
        preds = {f"c{i:02d}": 0. for i in range(65)}
        for i in range(3): preds.pop(f"c{i:02d}")
        self.assertAlmostEqual(coverage_fraction(preds, [f"c{i:02d}" for i in range(65)]), 62/65)
        valid[3:8] = False
        with self.assertRaisesRegex(ValueError, "90%"):
            require_feature_coverage(valid)

    def test_common_coverage_uses_all_65_and_requires_each_fold_pair(self):
        ids=[f"c{i:02d}" for i in range(65)]
        preds={"a":{c:0. for c in ids},"b":{c:0. for c in ids}}
        pairs=[]
        for fold in range(5): pairs.append((fold,f"g{fold}",ids[2*fold],ids[2*fold+1]))
        common, cp=require_common_coverage(preds,ids,pairs)
        self.assertEqual(len(common),65); self.assertEqual(len(cp),5)
        with self.assertRaisesRegex(ValueError,"all 65"):
            require_common_coverage(preds,ids[:64],pairs)
        del preds["b"][ids[8]]
        with self.assertRaisesRegex(ValueError,"coverage|fold"):
            require_common_coverage(preds,ids,pairs)

if __name__ == "__main__":
    unittest.main(verbosity=2)
