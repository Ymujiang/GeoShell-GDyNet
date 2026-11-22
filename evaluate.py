"""Zero-leakage checkpoint evaluation script on trajectory graph caches."""

import argparse
import os
import sys
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from geoshell_gdynet.data import TrajectoryGraphCache
from geoshell_gdynet.model import build_variant_model
from geoshell_gdynet.vamp import vamp2_score, compute_ck_test, compute_its


def evaluate_checkpoint(ckpt_path: str, cache_path: str, variant: str = "geoshell_h32", tau: int = 20, batch_size: int = 32, gpu: int = -1):
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")
    cache = TrajectoryGraphCache(cache_path)
    model = build_variant_model(variant=variant, n_classes=4).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    at_dev = cache.atom_types.to(device)
    ti_dev = cache.target_index.to(device)
    preds = []
    with torch.no_grad():
        for i in range(0, cache.num_frames, batch_size):
            j = min(i + batch_size, cache.num_frames)
            B = j - i
            nv = cache.nbr_vecs[i:j].to(device) if cache.nbr_vecs is not None else None
            preds.append(
                model(
                    at_dev.unsqueeze(0).expand(B, -1),
                    cache.nbr_dists[i:j].to(device),
                    cache.nbr_lists[i:j].to(device),
                    ti_dev.unsqueeze(0).expand(B, -1),
                    nv,
                ).cpu()
            )
    tp = torch.cat(preds, dim=0)
    if tp.shape[0] > tau:
        v2 = float(vamp2_score(tp[:-tau], tp[tau:], include_constant=True).item())
    else:
        v2 = float("nan")
    return v2, tp.numpy()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a GeoShell-GDyNet checkpoint.")
    parser.add_argument("--checkpoint", default=os.path.join(REPO_ROOT, "checkpoints", "geoshell_gdynet_peo_seed0.pt"))
    parser.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "sample_peo_traj.npz"))
    parser.add_argument("--variant", default="geoshell_h32")
    parser.add_argument("--tau", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    v2, tp_np = evaluate_checkpoint(args.checkpoint, args.data, args.variant, args.tau, args.batch_size, args.gpu)
    pops = tp_np.mean(axis=(0, 1))
    print(f"Evaluated frames: {tp_np.shape[0]} | Targets: {tp_np.shape[1]} | States: {tp_np.shape[2]}")
    print(f"VAMP-2 (tau={args.tau}): {v2:.4f}")
    print(f"State populations: {np.round(pops, 4).tolist()}")


if __name__ == "__main__":
    main()
