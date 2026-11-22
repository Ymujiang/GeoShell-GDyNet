"""Unit tests for physical invariances, parameter accounting, ablations, and checkpoint inference."""

import os
import sys
import unittest
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from geoshell_gdynet.model import VARIANTS, build_variant_model
from evaluate import evaluate_checkpoint

SAMPLE_NPZ = os.path.join(REPO_ROOT, "data", "sample_peo_traj.npz")
CKPT_PATH = os.path.join(REPO_ROOT, "checkpoints", "geoshell_gdynet_peo_seed0.pt")


class TestGeoShellGDyNet(unittest.TestCase):
    def test_parameter_counts_match_manuscript(self):
        p_canon = sum(p.numel() for p in build_variant_model("canonical_h16").parameters() if p.requires_grad)
        p_pm = sum(p.numel() for p in build_variant_model("canonical_pm_h46").parameters() if p.requires_grad)
        p_geo16 = sum(p.numel() for p in build_variant_model("geoshell_h16").parameters() if p.requires_grad)
        p_geo32 = sum(p.numel() for p in build_variant_model("geoshell_h32").parameters() if p.requires_grad)

        self.assertEqual(p_canon, 5651)
        self.assertEqual(p_pm, 25691)
        self.assertEqual(p_geo16, 8377)
        self.assertEqual(p_geo32, 25561)

    def test_atom_permutation_and_so3_invariance(self):
        torch.manual_seed(0)
        model = build_variant_model("geoshell_h32")
        model.eval()
        B, N, M, N0 = 2, 40, 20, 6
        at = torch.randint(1, 17, (B, N))
        nl = torch.randint(0, N, (B, N, M))
        nv = torch.randn(B, N, M, 3)
        nd = nv.norm(dim=-1)
        ti = torch.arange(N0).unsqueeze(0).expand(B, -1)

        with torch.no_grad():
            ref = model(at, nd, nl, ti, nv)

            # Atom-index permutation
            perm = torch.randperm(N)
            inv_perm = torch.argsort(perm)
            out_perm = model(at[:, perm], nd[:, perm, :], inv_perm[nl[:, perm, :]], inv_perm[ti], nv[:, perm, :, :])
            self.assertLess((ref - out_perm).abs().max().item(), 1e-5)

            # SO(3) rotation
            q, _ = torch.linalg.qr(torch.randn(3, 3))
            if torch.det(q) < 0:
                q[:, 0] = -q[:, 0]
            out_rot = model(at, nd, nl, ti, nv @ q.t())
            self.assertLess((ref - out_rot).abs().max().item(), 1e-5)

            # O(3) reflection
            refl = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
            out_refl = model(at, nd, nl, ti, nv @ refl.t())
            self.assertLess((ref - out_refl).abs().max().item(), 1e-5)

    def test_all_ablation_variants_forward(self):
        B, N, M, N0 = 2, 30, 12, 4
        at = torch.randint(1, 17, (B, N))
        nl = torch.randint(0, N, (B, N, M))
        nv = torch.randn(B, N, M, 3)
        nd = nv.norm(dim=-1)
        ti = torch.arange(N0).unsqueeze(0).expand(B, -1)
        for name in VARIANTS:
            m = build_variant_model(name)
            m.eval()
            with torch.no_grad():
                out = m(at, nd, nl, ti, nv)
            self.assertEqual(out.shape, (B, N0, 4), f"Shape mismatch for {name}")
            self.assertTrue(torch.isfinite(out).all(), f"Non-finite output for {name}")
            self.assertTrue(torch.allclose(out.sum(dim=-1), torch.ones(B, N0), atol=1e-5))

    def test_pretrained_checkpoint_load_and_infer(self):
        if not os.path.exists(CKPT_PATH) or not os.path.exists(SAMPLE_NPZ):
            self.skipTest("Sample data or checkpoint missing")
        v2, preds = evaluate_checkpoint(CKPT_PATH, SAMPLE_NPZ, variant="geoshell_h32", tau=20, batch_size=8, gpu=-1)
        self.assertEqual(preds.shape[1:], (50, 4))
        self.assertTrue(np.isfinite(v2))
        self.assertGreater(v2, 1.0)


if __name__ == "__main__":
    unittest.main()
