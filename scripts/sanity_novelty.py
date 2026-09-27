"""Sanity-check helper for newly-created novelty modules."""
import importlib
import sys

import torch


def main(module_ids):
    ok = True
    for mid in module_ids:
        m = importlib.import_module(f"models.novelty_attempts.{mid}")
        mdl = m.Model(**m.KWARGS).eval()
        n_params = sum(p.numel() for p in mdl.parameters())
        x = torch.randn(40, 1280)
        with torch.no_grad():
            y, attn, _ = mdl(x, return_attention=True)
        y_val = float(y.item())
        attn_shape = tuple(attn.shape) if attn is not None else None
        in_range = 0.0 <= y_val <= 3.0
        print(
            f"{mid}: y_shape={tuple(y.shape)} y={y_val:.4f} "
            f"attn={attn_shape} params={n_params:,} in_range={in_range}"
        )
        if not in_range:
            ok = False
        if tuple(y.shape) != (1,):
            ok = False
    print("SANITY:", "OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1:])

