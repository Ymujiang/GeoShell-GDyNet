"""GPU latency, throughput, and peak VRAM profiler for GeoShell-GDyNet."""

import argparse
import os
import sys
import time
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from geoshell_gdynet.model import build_variant_model
from geoshell_gdynet.vamp import VampLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="geoshell_h32")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_atoms", type=int, default=3820)
    parser.add_argument("--n_nbrs", type=int, default=20)
    parser.add_argument("--n_targets", type=int, default=50)
    parser.add_argument("--n_warmup", type=int, default=20)
    parser.add_argument("--n_iters", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_variant_model(args.variant).to(device)
    B, N, M, N0 = args.batch_size, args.n_atoms, args.n_nbrs, args.n_targets
    at = torch.randint(1, 17, (B, N), device=device)
    nl = torch.randint(0, N, (B, N, M), device=device)
    nv = torch.randn(B, N, M, 3, device=device)
    nd = nv.norm(dim=-1)
    ti = torch.arange(N0, device=device).unsqueeze(0).expand(B, -1)

    model.eval()
    with torch.no_grad():
        for _ in range(args.n_warmup):
            _ = model(at, nd, nl, ti, nv)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.n_iters):
            _ = model(at, nd, nl, ti, nv)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) * 1000.0 / args.n_iters
    print(f"{args.variant}: infer={dt:.3f} ms/batch ({B*1000.0/dt:.1f} frames/s)")


if __name__ == "__main__":
    main()
