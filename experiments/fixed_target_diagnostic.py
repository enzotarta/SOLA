"""Across-image residual r_abs_mean of each head: the L1 distance of A(x) from its
mean over images, divided by two and averaged over queries and images.

This is the quantity compared with --r-thresh in the broadcast and rank-k
experiments. It is different from the within-image residual that appears in the
bound for the broadcast substitution, see sola/diagnostics.py.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from operator_taxonomy import make_attention_recorder_class
from escape_mass_sweep import detect_model_config


def compute_image_indep_residual(attn_h: np.ndarray) -> dict:
    """
    attn_h : (n_img, n_q, n_k) softmax outputs for ONE head, with
             extra tokens stripped by the caller if wanted.
    Returns dict with mean / std / per-image array.
    """
    A = attn_h.astype(np.float64)
    A_bar = A.mean(axis=0)                       # (n_q, n_k)
    diffs = np.abs(A - A_bar[None])              # (n_img, n_q, n_k)
    per_img_per_q = diffs.sum(axis=-1) / 2.0     # (n_img, n_q) in [0, 1]
    per_img = per_img_per_q.mean(axis=-1)        # (n_img,)
    return {
        "r_abs_mean": float(per_img.mean()),
        "r_abs_std":  float(per_img.std()),
        "r_abs_min":  float(per_img.min()),
        "r_abs_max":  float(per_img.max()),
        "n_img":      int(per_img.shape[0]),
    }


def sweep(model, eval_loader, device, n_batches: int):
    AttentionRecorder = make_attention_recorder_class()
    cfg = detect_model_config(model, image_size=224)
    n_extra = cfg["n_extra_tokens"]

    rec = AttentionRecorder(model)
    with torch.no_grad():
        for bi, batch in enumerate(eval_loader):
            if bi >= n_batches:
                break
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device)
            _ = model(batch)
    attn_by_layer = rec.collect()
    rec.remove()

    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.num_heads
    out = []
    for li in range(n_layers):
        a = attn_by_layer[li].numpy()                 # (B', n_h, N, N)
        a_sp = a[:, :, n_extra:, n_extra:]            # spatial only
        a_sp = a_sp / a_sp.sum(axis=-1, keepdims=True).clip(min=1e-12)
        for hi in range(n_heads):
            stats = compute_image_indep_residual(a_sp[:, hi])
            out.append({"layer": li, "head": hi, **stats})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch16_224")
    ap.add_argument("--taxonomy", required=True,
                    help="Existing taxonomy_*.json with per-head categories")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="results/fixed_target_diag.json")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import timm
    from shift_substitution import build_tiny_imagenet_eval_loader

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device} ...")
    model = timm.create_model(args.model, pretrained=True,
                                 img_size=args.img_size).eval().to(device)
    eval_loader, _ = build_tiny_imagenet_eval_loader(args.batch_size, args.img_size)

    print(f"Recording attention on {args.batches} batches ...")
    per_head = sweep(model, eval_loader, device, n_batches=args.batches)

    # Join with taxonomy categories
    tax = json.loads(Path(args.taxonomy).read_text())
    cat_by_lh = {(r["layer"], r["head"]): r["category"] for r in tax["results"]}
    for r in per_head:
        r["category"] = cat_by_lh.get((r["layer"], r["head"]), "?")

    # Per-category summary
    by_cat = {}
    for r in per_head:
        by_cat.setdefault(r["category"], []).append(r["r_abs_mean"])
    summary = {
        cat: {
            "n_heads": len(vals),
            "mean":   float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std":    float(np.std(vals)),
            "min":    float(np.min(vals)),
            "max":    float(np.max(vals)),
        }
        for cat, vals in by_cat.items()
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": args.model,
        "taxonomy": str(args.taxonomy),
        "n_batches": args.batches,
        "summary_by_category": summary,
        "per_head": per_head,
    }, indent=2))
    print(f"Wrote {out_path}")

    # Pretty-print summary
    print(f"\nPer-category image-indep residual r_abs_mean")
    for cat in sorted(summary, key=lambda c: summary[c]["median"]):
        s = summary[cat]
        print(f"  {cat:14s}  n={s['n_heads']:3d}  median={s['median']:.3f}  "
              f"mean={s['mean']:.3f}  range=[{s['min']:.3f}, {s['max']:.3f}]")


if __name__ == "__main__":
    main()
