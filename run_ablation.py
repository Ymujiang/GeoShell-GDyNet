"""Controlled ablation variant builder and trainable parameter counter."""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from geoshell_gdynet.model import VARIANTS, build_variant_model


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="Inspect GeoShell-GDyNet ablation variants.")
    parser.add_argument("--list_params", action="store_true", help="Print trainable parameter counts for all variants.")
    parser.add_argument("--n_classes", type=int, default=4)
    args = parser.parse_args()

    if args.list_params:
        for name in VARIANTS:
            m = build_variant_model(name, n_classes=args.n_classes)
            print(f"{name:28s}: {count_trainable_params(m):7d} trainable parameters")


if __name__ == "__main__":
    main()
