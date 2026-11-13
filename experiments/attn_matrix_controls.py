"""Image-mean substitution: replace the attention of a head with its average
over images. Shows that an image-independent matrix is not enough.
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
    detect_model_config, evaluate, parse_heads,
    build_tiny_imagenet_eval_loader, build_imagenet_val_eval_loader,
)
from attn_matrix_substitution import (
    install_matrix_substitutions, restore_forwards,
    record_mean_attention, scramble_matrix_row_perm,
)
from kernel_substitution_controls import (
    compute_cls_features, evaluate_cls_cos_sim,
)


def select_heads(taxonomy_json, ft_diag_json, r_thresh: float):
    tax = json.loads(Path(taxonomy_json).read_text())
    diag = json.loads(Path(ft_diag_json).read_text())
    diag_by_lh = {(r["layer"], r["head"]): r["r_abs_mean"]
                  for r in diag["per_head"]}
    selected = []
    for r in tax["results"]:
        if r["category"] != "fixed_target":
            continue
        lh = (r["layer"], r["head"])
        if diag_by_lh.get(lh, 1.0) < r_thresh:
            selected.append(lh)
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--diag", required=True,
                    help="fixed_target diagnostic JSON")
    ap.add_argument("--r-thresh", type=float, default=0.4)
    ap.add_argument("--calib-batches", type=int, default=4)
    ap.add_argument("--calib-batch-size", type=int, default=32)
    ap.add_argument("--n-eval-batches", type=int, default=156,
                    help="156 batches at batch=64 = 9984 images")
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--source", default="imagenet-val",
                    choices=("tiny-imagenet", "imagenet-val"))
    ap.add_argument("--imagenet-val-dir", default="data/imagenet/val")
    ap.add_argument("--metric", default="top1", choices=("top1", "cls-cos-sim"))
    ap.add_argument("--out", default="results/attn_matrix_controls.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-random-seeds", type=int, default=3)
    args = ap.parse_args()

    import timm
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device} ...")
    model = timm.create_model(args.model, pretrained=True).eval().to(device)
    cfg = detect_model_config(model, image_size=224)

    # Build loaders. Use the same loader for calib + eval (different
    # batches via shuffling=False -> deterministic head/tail split).
    if args.source == "imagenet-val":
        eval_loader, _ = build_imagenet_val_eval_loader(
            args.eval_batch_size, image_size=224,
            val_dir=args.imagenet_val_dir, num_workers=4)
        calib_loader, _ = build_imagenet_val_eval_loader(
            args.calib_batch_size, image_size=224,
            val_dir=args.imagenet_val_dir, num_workers=4)
    else:
        eval_loader, _ = build_tiny_imagenet_eval_loader(
            args.eval_batch_size, 224)
        calib_loader, _ = build_tiny_imagenet_eval_loader(
            args.calib_batch_size, 224)

    # Selection
    heads = select_heads(args.taxonomy, args.diag, args.r_thresh)
    print(f"Selected {len(heads)} fixed_target heads with r_abs<{args.r_thresh}: {heads}")
    if not heads:
        raise RuntimeError("No heads selected; lower --r-thresh.")

    # Record A_bar
    print(f"Recording A_bar on {args.calib_batches} calib batches...")
    Abar_map = record_mean_attention(model, calib_loader, device,
                                      args.calib_batches, heads)
    for lh in heads:
        A = Abar_map[lh]
        # Sanity: each row should sum to ~1 (it's a mean of row-stochastic
        # matrices -> still row-stochastic).
        rs = A.sum(axis=-1)
        assert (rs.min() > 0.99) and (rs.max() < 1.01), (lh, rs.min(), rs.max())

    # Choose evaluator
    if args.metric == "top1":
        run_eval = lambda: evaluate(model, eval_loader, args.n_eval_batches, device)
        baseline_top1, baseline_top5, n_eval = run_eval()
        print(f"Baseline (n={n_eval}): top-1={baseline_top1*100:.2f}%  "
              f"top-5={baseline_top5*100:.2f}%")
        ci_half = 1.96 * (baseline_top1*(1-baseline_top1)/max(n_eval,1))**0.5 * 100
    else:
        # cls-cos-sim: pre-compute baseline features
        baseline_feats = compute_cls_features(model, eval_loader,
                                                args.n_eval_batches, device)
        baseline_top1 = None
        n_eval = baseline_feats.shape[0]
        print(f"Baseline CLS features computed (n={n_eval}).")

    results = {
        "model": args.model, "metric": args.metric,
        "selected_heads": [list(h) for h in heads],
        "r_thresh": args.r_thresh,
        "calib_batches": args.calib_batches,
        "n_eval_batches": args.n_eval_batches,
    }

    # Control A: real heads + real A_bar
    print(f"CONTROL A: real heads, real A_bar (n={len(heads)})")
    saved = install_matrix_substitutions(model, Abar_map)
    if args.metric == "top1":
        t1, t5, _ = evaluate(model, eval_loader, args.n_eval_batches, device)
        d1 = (t1 - baseline_top1) * 100
        d5 = (t5 - baseline_top5) * 100
        print(f"  top-1={t1*100:.2f}% (delta {d1:+.2f}, CI+/-{ci_half:.2f})  top-5 delta {d5:+.2f}")
        results["control_A"] = {"baseline_top1": baseline_top1,
                                  "subbed_top1": t1, "delta_top1": d1,
                                  "delta_top5": d5, "n_eval": n_eval,
                                  "ci_half_width": ci_half}
    else:
        cos, _ = evaluate_cls_cos_sim(model, eval_loader, args.n_eval_batches,
                                        device, baseline_feats)
        print(f"  CLS cos-sim to baseline = {cos:.4f}")
        results["control_A"] = {"subbed_cls_cos": cos, "n_eval": n_eval}
    restore_forwards(model, saved)

    # Control B: random non-overlapping heads, A_bars from selected
    print("CONTROL B: random heads, A_bar drawn from selected pool")
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.num_heads
    real_set = set(heads)
    trials = []
    for seed in range(args.n_random_seeds):
        rng = random.Random(seed)
        avail = [(L, h) for L in range(n_layers) for h in range(n_heads)
                 if (L, h) not in real_set]
        chosen = rng.sample(avail, len(heads))
        # Each chosen head gets an A_bar randomly drawn from selected pool
        Abar_pool = list(Abar_map.values())
        rng.shuffle(Abar_pool)
        ctl_map = {lh: Abar_pool[i % len(Abar_pool)]
                   for i, lh in enumerate(chosen)}
        saved = install_matrix_substitutions(model, ctl_map)
        if args.metric == "top1":
            t1, _, _ = evaluate(model, eval_loader, args.n_eval_batches, device)
            d1 = (t1 - baseline_top1) * 100
            trials.append({"seed": seed, "subs": [list(p) for p in chosen],
                           "delta_top1": d1})
            print(f"  seed {seed}: heads={chosen}  delta top1={d1:+.2f}")
        else:
            cos, _ = evaluate_cls_cos_sim(model, eval_loader, args.n_eval_batches,
                                            device, baseline_feats)
            trials.append({"seed": seed, "subs": [list(p) for p in chosen],
                           "subbed_cls_cos": cos})
            print(f"  seed {seed}: heads={chosen}  cos={cos:.4f}")
        restore_forwards(model, saved)
    if args.metric == "top1":
        arr = np.array([t["delta_top1"] for t in trials])
        results["control_B"] = {"trials": trials,
                                  "mean_delta_top1": float(arr.mean()),
                                  "std_delta_top1": float(arr.std())}
    else:
        arr = np.array([t["subbed_cls_cos"] for t in trials])
        results["control_B"] = {"trials": trials,
                                  "mean_cls_cos": float(arr.mean()),
                                  "std_cls_cos": float(arr.std())}

    # Control C: real heads, row-permuted A_bar
    print("CONTROL C: real heads, row-permuted A_bar")
    scrambled = {lh: scramble_matrix_row_perm(A, seed=hash(lh) & 0xFFFF)
                 for lh, A in Abar_map.items()}
    saved = install_matrix_substitutions(model, scrambled)
    if args.metric == "top1":
        t1, t5, _ = evaluate(model, eval_loader, args.n_eval_batches, device)
        d1 = (t1 - baseline_top1) * 100
        d5 = (t5 - baseline_top5) * 100
        print(f"  top-1={t1*100:.2f}% (delta {d1:+.2f})  top-5 delta {d5:+.2f}")
        results["control_C"] = {"subbed_top1": t1, "delta_top1": d1,
                                  "delta_top5": d5}
    else:
        cos, _ = evaluate_cls_cos_sim(model, eval_loader, args.n_eval_batches,
                                        device, baseline_feats)
        print(f"  CLS cos-sim to baseline = {cos:.4f}")
        results["control_C"] = {"subbed_cls_cos": cos}
    restore_forwards(model, saved)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    print("\nSUMMARY")
    if args.metric == "top1":
        print(f"  A. Real:           delta top1={results['control_A']['delta_top1']:+.2f}  "
              f"(CI+/-{ci_half:.2f})")
        print(f"  B. Random heads:   mean delta top1={results['control_B']['mean_delta_top1']:+.2f}  "
              f"(+/-{results['control_B']['std_delta_top1']:.2f})")
        print(f"  C. Permuted A_bar: delta top1={results['control_C']['delta_top1']:+.2f}")
    else:
        print(f"  A. Real:           cos={results['control_A']['subbed_cls_cos']:.4f}")
        print(f"  B. Random heads:   mean cos={results['control_B']['mean_cls_cos']:.4f}  "
              f"(+/-{results['control_B']['std_cls_cos']:.4f})")
        print(f"  C. Permuted A_bar: cos={results['control_C']['subbed_cls_cos']:.4f}")


if __name__ == "__main__":
    main()
