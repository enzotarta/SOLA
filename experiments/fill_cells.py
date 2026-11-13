"""Standalone group tests for the table of verified heads.

Substitutes jointly every head that the Stage-1 rules assign to a category on a
calibration batch, without an r_abs gate.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm

SOLA_CODE = Path(__file__).resolve().parent.parent / "sola"
sys.path.insert(0, str(SOLA_CODE))
from diagnostics import head_diagnostics, classify  # noqa: E402
from substitutions import (  # noqa: E402
    install, restore, kernel_from_mean,
    make_kernel_forward, make_broadcast_forward, adaptive_layer_forwards,
)
from data import imagenet_val_loader, detect_grid_and_extra, capture_attention  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shift_substitution import image_embedding  # noqa: E402

SHIFT_CATS = {"delta_shift", "soft_shift", "self_only"}


def build_taxonomy(model, cal_x, grid, n_extra):
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.num_heads
    attns = capture_attention(model, cal_x, n_extra)
    kappa_bar, diag, cat = {}, {}, {}
    for li in range(n_layers):
        a = attns[li].numpy()
        for hi in range(n_heads):
            ker, d = head_diagnostics(a[:, hi], *grid, K=5)
            kappa_bar[(li, hi)] = ker
            diag[(li, hi)] = d
            cat[(li, hi)] = classify(d)
    return kappa_bar, diag, cat


@torch.no_grad()
def top1(model, loader, n_max, device):
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
        if total >= n_max:
            break
    return correct / max(total, 1), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--op", choices=["shift", "broadcast_diffuse", "rank10_ng"], required=True)
    ap.add_argument("--metric", choices=["top1", "cossim"], required=True)
    ap.add_argument("--imagenet-val-dir", required=True)
    ap.add_argument("--n-cal", type=int, default=32)
    ap.add_argument("--n-eval", type=int, default=9984)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--k-rank10", type=int, default=10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--image-size", type=int, default=None,
                     help="defaults to the model's native patch_embed.img_size, else 224")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device}")
    model = timm.create_model(args.model, pretrained=True).eval().to(device)
    image_size = args.image_size
    if image_size is None:
        native = getattr(model.patch_embed, "img_size", None)
        image_size = int(native[0]) if native is not None else 224
    grid, n_extra = detect_grid_and_extra(model, image_size=image_size)
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.num_heads
    print(f"  image_size {image_size}, grid {grid}, {n_layers} layers, {n_heads} heads/layer")

    cal_loader = imagenet_val_loader(args.imagenet_val_dir, args.n_cal, image_size=image_size)
    eval_loader = imagenet_val_loader(args.imagenet_val_dir, args.batch_size, image_size=image_size)
    cal_x, _ = next(iter(cal_loader))
    cal_x = cal_x.to(device)

    print("Building taxonomy from calibration batch...")
    kappa_bar, diag, cat = build_taxonomy(model, cal_x, grid, n_extra)

    if args.op == "shift":
        heads = [(l, h) for (l, h), c in cat.items() if c in SHIFT_CATS]
        per_layer = {}
        for (l, h) in heads:
            per_layer.setdefault(l, {})[h] = torch.as_tensor(
                kernel_from_mean(kappa_bar[(l, h)]), dtype=torch.float32).to(device)
        layer_fwds = {l: make_kernel_forward(model.blocks[l].attn, kers, grid, n_extra)
                      for l, kers in per_layer.items()}
    elif args.op == "broadcast_diffuse":
        heads = [(l, h) for (l, h), c in cat.items() if c == "diffuse_local"]
        per_layer = {}
        for l, h in heads:
            per_layer.setdefault(l, []).append(h)
        layer_fwds = {l: make_broadcast_forward(model.blocks[l].attn, hs)
                      for l, hs in per_layer.items()}
    else:  # rank10_ng
        heads = [(l, h) for (l, h), c in cat.items() if c == "near_global"]
        head_to_k = {(l, h): args.k_rank10 for (l, h) in heads}
        layer_fwds = adaptive_layer_forwards(
            model, kernel_map={}, head_to_k_map=head_to_k,
            grid=grid, n_extra=n_extra, k_floor=args.k_rank10)

    n_heads_tested = len(heads)
    print(f"op={args.op}: {n_heads_tested} heads selected")

    result = {"model": args.model, "op": args.op, "metric": args.metric,
              "n_heads_tested": n_heads_tested, "n_eval": args.n_eval}

    if n_heads_tested == 0:
        print("No heads in this category for this backbone -- nothing to test.")
        result["note"] = "no heads in category"
    elif args.metric == "top1":
        base_acc, n_base = top1(model, eval_loader, args.n_eval, device)
        saved = install(model, layer_fwds)
        sub_acc, n_sub = top1(model, eval_loader, args.n_eval, device)
        restore(model, saved)
        delta = 100 * (sub_acc - base_acc)
        print(f"baseline top-1: {100*base_acc:.2f}% (n={n_base})")
        print(f"substituted top-1: {100*sub_acc:.2f}% (n={n_sub})")
        print(f"delta top-1: {delta:+.3f}%")
        result.update({"baseline_top1": base_acc, "sub_top1": sub_acc, "delta_top1": delta})
    else:  # cossim
        total_cos, total_n = 0.0, 0
        with torch.no_grad():
            for x, _ in eval_loader:
                x = x.to(device)
                base = image_embedding(model, x)
                saved = install(model, layer_fwds)
                sub = image_embedding(model, x)
                restore(model, saved)
                cos = F.cosine_similarity(base, sub, dim=-1)
                total_cos += cos.sum().item()
                total_n += cos.numel()
                if total_n >= args.n_eval:
                    break
        mean_cos = total_cos / max(total_n, 1)
        print(f"CLS cos-sim to baseline: {mean_cos:.4f} (n={total_n})")
        result["cls_cos_sim"] = mean_cos

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
