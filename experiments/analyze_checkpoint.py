"""Taxonomy, r_abs and effective rank for one training checkpoint."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))


def compute_eff_rank(attn_h, q_levels=(0.90, 0.99)):
    """Effective rank @ q for one head's attention matrices.
    attn_h : (n_img, N, N).  Returns dict {q -> mean eff_rank across
    images}."""
    A = attn_h.astype(np.float64)
    n_img = A.shape[0]
    out = {q: [] for q in q_levels}
    for i in range(n_img):
        s = np.linalg.svd(A[i], compute_uv=False)
        e = s ** 2
        e = e / e.sum().clip(min=1e-12)
        c = np.cumsum(np.sort(e)[::-1])
        for q in q_levels:
            r = int(np.searchsorted(c, q) + 1)
            out[q].append(r)
    return {f"eff_rank_{int(q*100)}": float(np.mean(out[q])) for q in q_levels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="path to .pth checkpoint produced by train_deit_tiny.py")
    ap.add_argument("--imagenet-val-dir",
                    default="data/imagenet/val")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-batches", type=int, default=4)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    import timm
    from operator_taxonomy import (
        sweep_model_heads, make_attention_recorder_class,
    )
    from fixed_target_diagnostic import compute_image_indep_residual
    from shift_substitution import (
        build_imagenet_val_eval_loader, detect_model_config,
    )

    print(f"Loading {args.model} (untrained) on {device} ...")
    model = timm.create_model(
        args.model, pretrained=False, num_classes=1000,
        img_size=args.img_size,
    ).eval().to(device)

    ck = torch.load(args.checkpoint, map_location="cpu")
    if "model" in ck:
        sd = ck["model"]
        epoch = ck.get("epoch", -1)
        val_top1 = ck.get("val_top1", None)
    else:
        sd = ck
        epoch = -1
        val_top1 = None
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARN: {len(missing)} missing keys (likely trainable-only "
              f"e.g. fc/head); first: {missing[:3]}")
    if unexpected:
        print(f"  WARN: {len(unexpected)} unexpected keys; first: {unexpected[:3]}")
    print(f"  loaded epoch={epoch} val_top1={val_top1}")

    cfg = detect_model_config(model, image_size=args.img_size)
    loader, _ = build_imagenet_val_eval_loader(
        args.batch_size, image_size=args.img_size,
        val_dir=args.imagenet_val_dir, num_workers=4)

    # operator taxonomy + per-head TE residual + diagnostics
    print(f"  Sweeping heads (n_batches={args.n_batches}, K={args.K}) ...")
    tax_results = sweep_model_heads(model, loader, device,
                                     n_batches=args.n_batches, K=args.K)
    cat_counts: dict[str, int] = {}
    for r in tax_results:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1

    # r_abs (image-mean residual) and eff_rank using full attention
    AttentionRecorder = make_attention_recorder_class()
    rec = AttentionRecorder(model)
    n_iter = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= args.n_batches:
                break
            x = batch[0].to(device)
            _ = model(x)
            n_iter += 1
    attn_by_layer = rec.collect()
    rec.remove()

    n_extra = cfg["n_extra_tokens"]
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.num_heads
    rabs_by_lh = {}
    erank_by_lh = {}
    for li in range(n_layers):
        a = attn_by_layer[li].numpy()
        a_sp = a[:, :, n_extra:, n_extra:]
        a_sp = a_sp / a_sp.sum(axis=-1, keepdims=True).clip(min=1e-12)
        for hi in range(n_heads):
            rabs_by_lh[(li, hi)] = compute_image_indep_residual(a_sp[:, hi])
            # eff_rank: only need on a small subset of images (n_img ~ 16)
            erank_by_lh[(li, hi)] = compute_eff_rank(
                a_sp[:min(a_sp.shape[0], 16), hi])

    # Merge all per-head info
    per_head = []
    for r in tax_results:
        lh = (r["layer"], r["head"])
        d = dict(r["diag"])
        d.update(rabs_by_lh[lh])
        d.update(erank_by_lh[lh])
        per_head.append({
            "layer": r["layer"],
            "head":  r["head"],
            "category": r["category"],
            "diag": d,
            "mean_kernel": r["mean_kernel"],
        })

    out = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "epoch": epoch,
        "val_top1": val_top1,
        "K": args.K,
        "n_batches": args.n_batches,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "category_counts": cat_counts,
        "per_head": per_head,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    print(f"\nCategory counts (epoch {epoch}, top1 {val_top1}):")
    from operator_taxonomy import CATEGORIES
    for cat in CATEGORIES:
        if cat in cat_counts:
            print(f"  {cat:18s}  {cat_counts[cat]:3d}")


if __name__ == "__main__":
    main()
