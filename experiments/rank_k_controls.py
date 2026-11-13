"""Per-image rank-k substitution at the heads of a category, with controls.

Same three conditions as broadcast_controls.py.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from shift_substitution import (
    detect_model_config, evaluate,
    build_tiny_imagenet_eval_loader, build_imagenet_val_eval_loader,
)
from attn_matrix_substitution import (
    install_rank_k_substitutions, restore_forwards,
)
from kernel_substitution_controls import (
    compute_cls_features, evaluate_cls_cos_sim,
)
from broadcast_controls import (
    select_low_r_in_category, select_all_in_category,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--r-thresh", type=float, default=0.4)
    ap.add_argument("--k-rank", type=int, default=1)
    ap.add_argument("--n-eval-batches", type=int, default=156)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--source", default="imagenet-val",
                    choices=("tiny-imagenet", "imagenet-val"))
    ap.add_argument("--imagenet-val-dir", default="data/imagenet/val")
    ap.add_argument("--metric", default="top1", choices=("top1", "cls-cos-sim"))
    ap.add_argument("--out", default="results/rank_k.json")
    ap.add_argument("--n-random-seeds", type=int, default=3)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--category", default="near_global")
    ap.add_argument("--layer-min", type=int, default=None)
    ap.add_argument("--layer-max", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import timm
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device} ...")
    model = timm.create_model(args.model, pretrained=True,
                                 img_size=args.img_size).eval().to(device)
    cfg = detect_model_config(model, image_size=args.img_size)

    if args.source == "imagenet-val":
        eval_loader, _ = build_imagenet_val_eval_loader(
            args.eval_batch_size, image_size=args.img_size,
            val_dir=args.imagenet_val_dir, num_workers=4)
    else:
        eval_loader, _ = build_tiny_imagenet_eval_loader(
            args.eval_batch_size, args.img_size)

    low_heads = select_low_r_in_category(args.taxonomy, args.diag,
                                           args.r_thresh, args.category)
    all_heads = select_all_in_category(args.taxonomy, args.category)
    if args.layer_min is not None or args.layer_max is not None:
        lmin = args.layer_min if args.layer_min is not None else -1
        lmax = args.layer_max if args.layer_max is not None else 10**6
        low_heads = [(li, hi) for (li, hi) in low_heads if lmin <= li <= lmax]
        all_heads = [(li, hi) for (li, hi) in all_heads if lmin <= li <= lmax]
        print(f"Layer range filter: [{lmin}, {lmax}]")
    print(f"Category: {args.category}, k_rank={args.k_rank}")
    print(f"Low-r_abs (<{args.r_thresh}) {args.category}: n={len(low_heads)}")
    print(f"All {args.category}: n={len(all_heads)}")

    # Baseline
    if args.metric == "top1":
        b1, b5, n_eval = evaluate(model, eval_loader, args.n_eval_batches, device)
        ci = 1.96 * (b1*(1-b1)/max(n_eval,1))**0.5 * 100
        print(f"Baseline (n={n_eval}): top-1={b1*100:.2f}%  CI+/-{ci:.2f}")
        baseline_feats = None
    else:
        baseline_feats = compute_cls_features(model, eval_loader,
                                                args.n_eval_batches, device)
        n_eval = baseline_feats.shape[0]
        b1 = ci = None
        print(f"Baseline CLS features (n={n_eval}).")

    def run_eval():
        if args.metric == "top1":
            t1, t5, _ = evaluate(model, eval_loader, args.n_eval_batches, device)
            return {"top1": t1, "top5": t5,
                    "delta_top1": (t1-b1)*100, "delta_top5": (t5-b5)*100}
        else:
            cos, _ = evaluate_cls_cos_sim(model, eval_loader,
                                            args.n_eval_batches,
                                            device, baseline_feats)
            return {"cls_cos": cos}

    results = {"model": args.model, "metric": args.metric,
               "category": args.category, "k_rank": args.k_rank,
               "low_r_heads": [list(h) for h in low_heads],
               "n_all": len(all_heads),
               "r_thresh": args.r_thresh,
               "n_eval_batches": args.n_eval_batches}

    # A: low-r heads, rank-k
    print(f"A: rank-{args.k_rank} on low-r {args.category} (n={len(low_heads)})")
    saved = install_rank_k_substitutions(model, low_heads, k_rank=args.k_rank)
    a_res = run_eval()
    restore_forwards(model, saved)
    print(f"  result: {a_res}")
    results["A_low_r"] = a_res

    # B: random non-target heads
    print(f"B: rank-{args.k_rank} on random non-{args.category} heads (n={len(low_heads)})")
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.num_heads
    target_set = set(all_heads)
    avail = [(L, h) for L in range(n_layers) for h in range(n_heads)
             if (L, h) not in target_set]
    n_to_pick = min(len(low_heads), len(avail))
    if n_to_pick < len(low_heads):
        print(f"  WARNING: only {len(avail)} non-target heads available; "
              f"capping to {n_to_pick}")
    trials = []
    for seed in range(args.n_random_seeds):
        rng = random.Random(seed)
        chosen = rng.sample(avail, n_to_pick)
        saved = install_rank_k_substitutions(model, chosen, k_rank=args.k_rank)
        r = run_eval(); r["heads"] = [list(h) for h in chosen]; r["seed"] = seed
        trials.append(r)
        restore_forwards(model, saved)
        print(f"  seed {seed}: {r}")
    results["B_random"] = {"trials": trials}

    # C: ALL target heads
    print(f"C: rank-{args.k_rank} on ALL {args.category} (n={len(all_heads)})")
    saved = install_rank_k_substitutions(model, all_heads, k_rank=args.k_rank)
    c_res = run_eval()
    restore_forwards(model, saved)
    print(f"  result: {c_res}")
    results["C_all"] = c_res

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    print("\nSUMMARY")
    if args.metric == "top1":
        deltas = [t["delta_top1"] for t in trials]
        print(f"  A (n={len(low_heads)}):  delta top1={a_res['delta_top1']:+.2f}  (CI+/-{ci:.2f})")
        print(f"  B (n={n_to_pick}, {args.n_random_seeds} seeds):  mean delta top1={np.mean(deltas):+.2f} +/-{np.std(deltas):.2f}")
        print(f"  C (n={len(all_heads)}):  delta top1={c_res['delta_top1']:+.2f}")
    else:
        coses = [t["cls_cos"] for t in trials]
        print(f"  A (n={len(low_heads)}):  cos={a_res['cls_cos']:.4f}")
        print(f"  B (n={n_to_pick}, {args.n_random_seeds} seeds):  mean cos={np.mean(coses):.4f} +/-{np.std(coses):.4f}")
        print(f"  C (n={len(all_heads)}):  cos={c_res['cls_cos']:.4f}")


if __name__ == "__main__":
    main()
