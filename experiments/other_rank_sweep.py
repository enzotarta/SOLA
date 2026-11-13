"""Rank-30 and rank-50 at all heads of the Stage-1 `other` class, per backbone,
and at as many random heads outside it.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from shift_substitution import image_embedding  # noqa: E402
from attn_matrix_substitution import (
    install_rank_k_substitutions, restore_forwards,
)
from shift_substitution import (
    build_imagenet_val_eval_loader, build_tiny_imagenet_eval_loader,
    detect_model_config, evaluate,
)
from broadcast_controls import select_all_in_category


# (taxonomy_file, timm_model_name, metric). The sweep needs the Stage-1 labels: the
# final taxonomy files have no other heads left after the Stage-2 redistribution.
BACKBONES = [
    ("taxonomy_deit_with_other.json",            "deit_tiny_patch16_224",                          "top1"),
    ("taxonomy_deit_s_with_other.json",          "deit_small_patch16_224",                         "top1"),
    ("taxonomy_vits_with_other.json",            "vit_small_patch16_224.augreg_in21k_ft_in1k",     "top1"),
    ("taxonomy_vitb_in1k_pure_with_other.json",  "vit_base_patch16_224.augreg_in1k",               "top1"),
    ("taxonomy_vitb_in21k_with_other.json",      "vit_base_patch16_224.augreg2_in21k_ft_in1k",     "top1"),
    ("taxonomy_mae_with_other.json",             "vit_base_patch16_224.mae",                       "cls-cos"),
    ("taxonomy_vitb_dino_with_other.json",       "vit_base_patch16_224.dino",                      "cls-cos"),
    ("taxonomy_dinov2_with_other.json",          "vit_base_patch14_dinov2.lvd142m",                "cls-cos"),
    ("taxonomy_clip_with_other.json",            "vit_base_patch16_clip_224.openai",               "cls-cos"),
    ("taxonomy_siglip_with_other.json",          "vit_base_patch16_siglip_224.webli",                    "cls-cos"),
]


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


def draw_random_heads(taxonomy_file: Path, rng: random.Random) -> list:
    """Random-head control: as many heads outside the other class as there are in it."""
    rows = json.load(open(taxonomy_file))["results"]
    n_other = sum(r["category"] == "other" for r in rows)
    non_other = [(r["layer"], r["head"]) for r in rows if r["category"] != "other"]
    return rng.sample(non_other, min(n_other, len(non_other)))


def run_backbone(model_name: str, taxonomy_file: Path, metric: str,
                 ks: List[int], rand_heads: list,
                 args) -> dict:
    import timm
    device = args.device
    print(f"\n[{model_name}] taxonomy={taxonomy_file.name} metric={metric}",
          flush=True)

    # Image size: DINOv2 has 14x14 patches; keep 224 to match the
    # taxonomy that was computed at 224.
    img_size = 224

    print(f"Loading {model_name} ...")
    model = timm.create_model(
        model_name, pretrained=True, num_classes=(1000 if metric == "top1" else 0),
        img_size=img_size,
    ).eval().to(device)
    cfg = detect_model_config(model, image_size=img_size)

    # Heads
    other_heads = select_all_in_category(str(taxonomy_file), "other")
    tax_data = json.load(open(taxonomy_file))
    n_total = len(tax_data["results"])
    print(f"  'other' heads: {len(other_heads)} / {n_total}")
    if not other_heads:
        return {"model": model_name, "skip": "no_other_heads",
                "n_total": n_total, "n_other": 0}

    # Eval loader (use full IN val for top-1, 10k subset for cls-cos)
    print(f"Building eval loader ...")
    if metric == "top1":
        # Full IN val for top-1 deltas
        loader, _ = build_imagenet_val_eval_loader(
            args.eval_batch_size, image_size=img_size,
            val_dir=args.imagenet_val_dir, num_workers=4)
        n_eval_batches = args.n_eval_batches_top1
    else:
        loader, _ = build_imagenet_val_eval_loader(
            args.eval_batch_size, image_size=img_size,
            val_dir=args.imagenet_val_dir, num_workers=4)
        n_eval_batches = args.n_eval_batches_cos

    # Baseline
    print(f"Computing baseline ...")
    if metric == "top1":
        b1, b5, n_eval = evaluate(model, loader, n_eval_batches, device)
        base_top1 = b1; base_top5 = b5
        baseline_feats = None
        ci = 1.96 * (b1 * (1 - b1) / max(n_eval, 1)) ** 0.5 * 100
        print(f"  baseline top-1 = {b1*100:.2f}%  CI +/-{ci:.2f}  (n={n_eval})")
    else:
        baseline_feats = compute_cls_features(model, loader, n_eval_batches, device)
        n_eval = baseline_feats.shape[0]
        base_top1 = base_top5 = None
        print(f"  baseline CLS shape = {tuple(baseline_feats.shape)}")

    print(f"  B-control random heads ({len(rand_heads)}): {rand_heads[:5]}{'...' if len(rand_heads)>5 else ''}")

    out = {
        "model": model_name, "taxonomy": str(taxonomy_file),
        "metric": metric, "n_total_heads": n_total,
        "n_other": len(other_heads), "other_heads": other_heads,
        "n_random_B": len(rand_heads), "random_B_heads": rand_heads,
        "n_eval": n_eval, "baseline_top1": base_top1, "baseline_top5": base_top5,
        "ks": [],
    }

    def run_eval():
        if metric == "top1":
            t1, t5, _ = evaluate(model, loader, n_eval_batches, device)
            return {"top1": t1, "top5": t5,
                    "delta_top1": (t1 - base_top1) * 100,
                    "delta_top5": (t5 - base_top5) * 100}
        else:
            feats = compute_cls_features(model, loader, n_eval_batches, device)
            return {"cls_cos": cls_cos_mean(feats, baseline_feats)}

    for k in ks:
        # A: at all 'other' heads
        saved = install_rank_k_substitutions(model, other_heads, k_rank=k)
        a_res = run_eval()
        restore_forwards(model, saved)
        # B: at random non-other heads
        if rand_heads:
            saved = install_rank_k_substitutions(model, rand_heads, k_rank=k)
            b_res = run_eval()
            restore_forwards(model, saved)
        else:
            b_res = None
        out["ks"].append({"k": k, "A_other": a_res, "B_random": b_res})
        if metric == "top1":
            print(f"  k={k:>3}: A_other delta top1 = {a_res['delta_top1']:+.2f}%   "
                  f"B_random delta top1 = {b_res['delta_top1']:+.2f}%" if b_res else f"  k={k}: A_other delta top1 = {a_res['delta_top1']:+.2f}%")
        else:
            print(f"  k={k:>3}: A_other cls_cos = {a_res['cls_cos']:.4f}   "
                  f"B_random cls_cos = {b_res['cls_cos']:.4f}" if b_res else f"  k={k}: A_other cls_cos = {a_res['cls_cos']:.4f}")

    # release model
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-dir", default="results",
                    help="Directory containing taxonomy_*.json files.")
    ap.add_argument("--imagenet-val-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--n-eval-batches-top1", type=int, default=782,
                    help="full ImageNet-1k val at 64 bs = ~782 batches")
    ap.add_argument("--n-eval-batches-cos", type=int, default=156,
                    help="10k subset for cls-cos (=156 batches @ 64 bs)")
    ap.add_argument("--ks", type=int, nargs="+", default=[30, 50])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-models", type=str, default=None,
                    help="comma-sep substring filter on timm model name")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    exp_dir = Path(args.experiments_dir)
    backbones = []
    for tax_file, model_name, metric in BACKBONES:
        p = exp_dir / tax_file
        if not p.exists():
            print(f"WARNING: {p} missing - skipping {model_name}")
            continue
        # drawn for every backbone, so that --only-models gives the control of the full run
        rand_heads = draw_random_heads(p, rng)
        if args.only_models:
            if not any(s in model_name for s in args.only_models.split(",")):
                continue
        backbones.append((p, model_name, metric, rand_heads))
    print(f"Will sweep {len(backbones)} backbones x {len(args.ks)} k values\n")

    all_results = {"ks_tested": args.ks, "backbones": []}
    for tax_file, model_name, metric, rand_heads in backbones:
        try:
            res = run_backbone(model_name, tax_file, metric, args.ks, rand_heads, args)
        except Exception as e:
            print(f"  ERROR on {model_name}: {type(e).__name__}: {e}")
            res = {"model": model_name, "error": f"{type(e).__name__}: {e}"}
        all_results["backbones"].append(res)
        # write incrementally
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [saved partial -> {args.out}]")

    print(f"\nDONE. Wrote {args.out}")


if __name__ == "__main__":
    main()
