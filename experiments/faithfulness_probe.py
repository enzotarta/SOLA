"""Cosine similarity of the residual stream to baseline at every block boundary."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))


class ActivationCapturer:
    """Forward hook on each `model.blocks[i]` capturing post-block residual."""
    def __init__(self, model):
        self.handles = []
        self.acts: List[Tuple[int, torch.Tensor]] = []
        for i, blk in enumerate(model.blocks):
            def hook(m, inp, out, idx=i):
                self.acts.append((idx, out.detach().to("cpu")))
            self.handles.append(blk.register_forward_hook(hook))

    def reset(self):
        self.acts = []

    def remove(self):
        for h in self.handles:
            h.remove()


def per_block_cos_sim(base_acts, sub_acts):
    """Compute mean cos-sim per block. Returns a list of length n_blocks."""
    assert len(base_acts) == len(sub_acts)
    out = []
    base_acts = sorted(base_acts, key=lambda x: x[0])
    sub_acts = sorted(sub_acts, key=lambda x: x[0])
    for (i, ab), (j, sb) in zip(base_acts, sub_acts):
        assert i == j, f"block index mismatch {i} vs {j}"
        # ab, sb: (B, N, D)
        sim = F.cosine_similarity(ab.float(), sb.float(),
                                    dim=-1).mean().item()
        out.append(float(sim))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--diag", required=True,
                    help="fixed_target_*.json with r_abs_mean per head")
    ap.add_argument("--r-thresh", type=float, default=0.4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-batches", type=int, default=4,
                    help="number of calibration batches; activations "
                         "are concatenated and averaged. 4 batches at "
                         "bs=64 = 256 images; usually sufficient")
    ap.add_argument("--imagenet-val-dir", default="data/imagenet/val")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--shift-cats", default="delta_shift,soft_shift")
    ap.add_argument("--rank-k", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import timm
    from shift_substitution import (
        build_imagenet_val_eval_loader, detect_model_config,
    )
    from kernel_substitution import (
        kernel_from_diagnostics_mean,
        install_kernel_substitutions,
        restore_forwards as restore_kernel,
    )
    from attn_matrix_substitution import (
        install_broadcast_substitutions,
        install_rank_k_substitutions,
        restore_forwards as restore_attn,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device} ...")
    model = timm.create_model(args.model, pretrained=True,
                                 img_size=args.img_size).eval().to(device)
    cfg = detect_model_config(model, image_size=args.img_size)

    loader, _ = build_imagenet_val_eval_loader(
        args.batch_size, image_size=args.img_size,
        val_dir=args.imagenet_val_dir, num_workers=2)
    # Collect calibration batch
    xs = []
    for bi, batch in enumerate(loader):
        if bi >= args.n_batches: break
        xs.append(batch[0])
    x = torch.cat(xs, dim=0).to(device)
    print(f"  calibration batch: {x.shape}")

    tax = json.loads(Path(args.taxonomy).read_text())
    rows = tax["results"]
    res_by_lh = {(r["layer"], r["head"]): r for r in rows}
    diag = json.loads(Path(args.diag).read_text())
    diag_by_lh = {(r["layer"], r["head"]): r["r_abs_mean"]
                  for r in diag["per_head"]}

    shift_cats = set(args.shift_cats.split(","))
    shift_heads = [(r["layer"], r["head"]) for r in rows
                   if r["category"] in shift_cats]
    ft_heads = [(r["layer"], r["head"]) for r in rows
                if r["category"] == "fixed_target"]
    ng_heads = [(r["layer"], r["head"]) for r in rows
                if r["category"] == "near_global"]
    ft_low_r = [lh for lh in ft_heads
                if diag_by_lh.get(lh, 1.0) < args.r_thresh]
    ng_low_r = [lh for lh in ng_heads
                if diag_by_lh.get(lh, 1.0) < args.r_thresh]
    print(f"  shifts: {len(shift_heads)}")
    print(f"  FT all: {len(ft_heads)}, low_r: {len(ft_low_r)}: {ft_low_r}")
    print(f"  NG all: {len(ng_heads)}, low_r: {len(ng_low_r)}: {ng_low_r}")

    cap = ActivationCapturer(model)

    # baseline
    cap.reset()
    with torch.no_grad():
        _ = model(x)
    baseline_acts = list(cap.acts)
    print(f"  baseline blocks captured: {len(baseline_acts)}")

    out = {
        "model": args.model,
        "img_size": args.img_size,
        "batch_size": int(x.shape[0]),
        "n_blocks": len(baseline_acts),
        "shift_heads": [list(h) for h in shift_heads],
        "ft_all": [list(h) for h in ft_heads],
        "ft_low_r": [list(h) for h in ft_low_r],
        "ng_all": [list(h) for h in ng_heads],
        "ng_low_r": [list(h) for h in ng_low_r],
        "r_thresh": args.r_thresh,
        "rank_k": args.rank_k,
        "conditions": {},
    }

    def run_condition(name, install_fn, restore_fn):
        cap.reset()
        saved = install_fn()
        with torch.no_grad():
            _ = model(x)
        sub_acts = list(cap.acts)
        restore_fn(model, saved)
        sims = per_block_cos_sim(baseline_acts, sub_acts)
        last_sim = sims[-1]
        min_sim = min(sims)
        print(f"  [{name}]  per-block cos-sim:  "
              f"first={sims[0]:.4f}  min={min_sim:.4f}  last={last_sim:.4f}")
        return {
            "per_block_cos_sim": sims,
            "min_cos_sim": min_sim,
            "last_cos_sim": last_sim,
        }

    # shifts
    if shift_heads:
        kernel_map = {(li, hi): kernel_from_diagnostics_mean(
                          np.asarray(res_by_lh[(li, hi)]["mean_kernel"]))
                       for (li, hi) in shift_heads}
        out["conditions"]["shifts"] = run_condition(
            "shifts",
            install_fn=lambda: install_kernel_substitutions(
                model, kernel_map, cfg["grid"], cfg["n_extra_tokens"]),
            restore_fn=restore_kernel,
        )

    # broadcast: low_r and all
    if ft_low_r:
        out["conditions"]["broadcast_low_r"] = run_condition(
            "broadcast_low_r",
            install_fn=lambda: install_broadcast_substitutions(model, ft_low_r),
            restore_fn=restore_attn,
        )
    if ft_heads:
        out["conditions"]["broadcast_all"] = run_condition(
            "broadcast_all",
            install_fn=lambda: install_broadcast_substitutions(model, ft_heads),
            restore_fn=restore_attn,
        )

    # rank-10: low_r and all
    if ng_low_r:
        out["conditions"]["rank10_low_r"] = run_condition(
            "rank10_low_r",
            install_fn=lambda: install_rank_k_substitutions(
                model, ng_low_r, k_rank=args.rank_k),
            restore_fn=restore_attn,
        )
    if ng_heads:
        out["conditions"]["rank10_all"] = run_condition(
            "rank10_all",
            install_fn=lambda: install_rank_k_substitutions(
                model, ng_heads, k_rank=args.rank_k),
            restore_fn=restore_attn,
        )

    cap.remove()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    # Summary
    print("\nper-block cos-sim summary (rows: condition; cols: block)")
    if out["conditions"]:
        n_blk = out["n_blocks"]
        header = "condition         " + "".join(f"  L{i:>2d}" for i in range(n_blk))
        print(header)
        for name, r in out["conditions"].items():
            row = name.ljust(18) + "".join(
                f"  {s:5.3f}" for s in r["per_block_cos_sim"])
            print(row)


if __name__ == "__main__":
    main()
