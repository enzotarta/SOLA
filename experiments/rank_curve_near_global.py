"""Sweep k in the per-image rank-k substitution at all heads of one category."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from shift_substitution import image_embedding  # noqa: E402
from attn_matrix_substitution import (
    install_rank_k_substitutions, restore_forwards,
)
from shift_substitution import (
    build_imagenet_val_eval_loader, evaluate, detect_model_config,
)
from broadcast_controls import select_all_in_category


@torch.no_grad()
def compute_cls_features(model, loader, n_batches, device):
    feats = []
    for i, batch in enumerate(loader):
        if i >= n_batches: break
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True)
        feats.append(image_embedding(model, x).float().cpu())
    return torch.cat(feats, 0)


def cls_cos_mean(a, b):
    a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (a * b).sum(dim=-1).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--imagenet-val-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--metric", default="top1", choices=("top1", "cls-cos"))
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20, 50])
    ap.add_argument("--n-batches", type=int, default=782)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--category", default="near_global")
    ap.add_argument("--image-size", type=int, default=None,
                     help="defaults to the model's native patch_embed.img_size, else 224")
    args = ap.parse_args()

    import timm
    print(f"Loading {args.model} ...", flush=True)
    nclasses = 1000 if args.metric == "top1" else 0
    model = timm.create_model(args.model, pretrained=True,
                                 num_classes=nclasses).eval().to(args.device)

    image_size = args.image_size
    if image_size is None:
        native = getattr(model.patch_embed, "img_size", None)
        image_size = int(native[0]) if native is not None else 224

    heads = select_all_in_category(args.taxonomy, args.category)
    print(f"category={args.category}: {len(heads)} heads, image_size={image_size}", flush=True)

    loader, _ = build_imagenet_val_eval_loader(
        args.batch_size, image_size=image_size,
        val_dir=args.imagenet_val_dir, num_workers=4)

    if args.metric == "top1":
        b1, b5, n = evaluate(model, loader, args.n_batches, args.device)
        baseline_feats = None
        ci = 1.96 * (b1*(1-b1)/max(n,1))**0.5 * 100
        print(f"Baseline top-1 = {b1*100:.2f}%  CI +/-{ci:.2f}  (n={n})", flush=True)
    else:
        baseline_feats = compute_cls_features(model, loader, args.n_batches, args.device)
        n = baseline_feats.shape[0]
        b1 = b5 = None
        print(f"Baseline CLS feats: {tuple(baseline_feats.shape)}", flush=True)

    out = {"model": args.model, "category": args.category,
            "n_heads": len(heads), "metric": args.metric,
            "n_eval": n, "baseline_top1": b1, "ks": []}

    for k in args.ks:
        saved = install_rank_k_substitutions(model, heads, k_rank=k)
        if args.metric == "top1":
            t1, t5, _ = evaluate(model, loader, args.n_batches, args.device)
            res = {"k": k, "top1": t1, "delta_top1": (t1-b1)*100}
            print(f"  k={k:>3}: top1={t1*100:.2f}%  delta ={(t1-b1)*100:+.2f}%", flush=True)
        else:
            f = compute_cls_features(model, loader, args.n_batches, args.device)
            cos = cls_cos_mean(f, baseline_feats)
            res = {"k": k, "cls_cos": cos}
            print(f"  k={k:>3}: cls_cos = {cos:.4f}", flush=True)
        out["ks"].append(res)
        restore_forwards(model, saved)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
