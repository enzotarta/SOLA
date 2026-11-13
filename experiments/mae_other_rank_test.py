"""Rank-k sweep at the Stage-1 `other` heads of the MAE-Tiny model trained from
scratch.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from shift_substitution import image_embedding  # noqa: E402
from attn_matrix_substitution import (
    install_rank_k_substitutions, restore_forwards,
)
from shift_substitution import build_imagenet_val_eval_loader, detect_model_config


@torch.no_grad()
def compute_cls_features(model, loader, n_batches, device):
    feats = []
    for i, batch in enumerate(loader):
        if i >= n_batches: break
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True)
        feats.append(image_embedding(model, x).float().cpu())
    return torch.cat(feats, 0)


def cls_cos(a, b):
    a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (a * b).sum(dim=-1).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_tiny_patch16_224")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--analysis-json", required=True)
    ap.add_argument("--imagenet-val-dir", required=True)
    ap.add_argument("--n-batches", type=int, default=156)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ranks", type=int, nargs="+",
                    default=[1, 10, 30, 50, 100])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    import timm
    device = args.device
    print(f"Loading {args.model} on {device} ...")
    model = timm.create_model(
        args.model, pretrained=False, num_classes=0, img_size=224,
    ).eval().to(device)

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0: print(f"    missing[:3]={missing[:3]}")

    analysis = json.load(open(args.analysis_json))
    n_layers = analysis["n_layers"]; n_heads = analysis["n_heads"]
    other_heads = [(h["layer"], h["head"]) for h in analysis["per_head"]
                   if h["category"] == "other"]
    non_other = [(h["layer"], h["head"]) for h in analysis["per_head"]
                 if h["category"] != "other"]
    er90_other = sorted([h["diag"]["eff_rank_90"] for h in analysis["per_head"]
                         if h["category"] == "other"])
    print(f"\n'other' heads: n={len(other_heads)} of {n_layers*n_heads}")
    print(f"  eff_rank_90 median={np.median(er90_other):.0f}, "
          f"max={max(er90_other):.0f}, range=[{min(er90_other):.0f}, {max(er90_other):.0f}]")
    er_max = int(max(er90_other))

    # Eval loader
    print("Building eval loader ...")
    loader, _ = build_imagenet_val_eval_loader(
        args.batch_size, image_size=224,
        val_dir=args.imagenet_val_dir, num_workers=4)

    # Baseline
    print("Computing baseline CLS features ...")
    baseline = compute_cls_features(model, loader, args.n_batches, device)
    n_eval = baseline.shape[0]
    print(f"  baseline shape: {tuple(baseline.shape)}, n={n_eval}")

    results = {
        "model": args.model, "checkpoint": args.ckpt,
        "n_other_heads": len(other_heads),
        "other_heads": other_heads,
        "er90_other": er90_other,
        "n_eval": n_eval,
        "ranks_tested": [],
    }

    # A: rank-k at all 'other' heads
    print(f"A: rank-k at all {len(other_heads)} 'other' heads")
    for k in args.ranks + [er_max]:
        if k <= 0: continue
        saved = install_rank_k_substitutions(model, other_heads, k_rank=k)
        feats = compute_cls_features(model, loader, args.n_batches, device)
        cos = cls_cos(feats, baseline)
        restore_forwards(model, saved)
        tag = f"adaptive_max_er90_{k}" if k == er_max else f"rank_{k}"
        results["ranks_tested"].append({"k": k, "tag": tag, "cls_cos": cos,
                                          "subset": "all_other"})
        print(f"  k={k:>4} ({tag:<25}): cls_cos = {cos:.4f}")

    # B: rank-k at the same number of random NON-'other' heads
    print(f"B: rank-k at {len(other_heads)} random non-'other' heads")
    rng = random.Random(args.seed)
    rand_heads = rng.sample(non_other, len(other_heads))
    print(f"  random subset: {rand_heads[:5]}{'...' if len(rand_heads)>5 else ''}")
    for k in [10, 50]:
        saved = install_rank_k_substitutions(model, rand_heads, k_rank=k)
        feats = compute_cls_features(model, loader, args.n_batches, device)
        cos = cls_cos(feats, baseline)
        restore_forwards(model, saved)
        results["ranks_tested"].append({"k": k, "tag": f"rank_{k}_B_random",
                                          "cls_cos": cos, "subset": "random_non_other",
                                          "heads": rand_heads})
        print(f"  k={k:>4} (B random): cls_cos = {cos:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
