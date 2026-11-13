"""Run the SOLA substitutions on one timm ViT and print the change in top-1.

By default the heads of ViT-B/16 ImageNet-21k FT are read from results/ and the
evaluation uses the first 9,984 val images, which reproduces the numbers of the
experiments. With --fresh the taxonomy is recomputed on --n-cal random val images.

    python reproduce.py --val-dir data/imagenet/val
    python reproduce.py --val-dir data/imagenet/val --fresh --model deit_tiny_patch16_224
    python reproduce.py --val-dir data/imagenet/val --eval-order random
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch

from sola.data import capture_attention, detect_grid_and_extra, imagenet_val_loader
from sola.diagnostics import classify, head_diagnostics
from sola.substitutions import (adaptive_layer_forwards, install, kernel_from_mean,
                                make_broadcast_forward, make_kernel_forward, restore)

RESULTS = Path(__file__).resolve().parent / "results"
SHIFT_CATS = {"delta_shift", "soft_shift", "self_only"}
DEFAULT_MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"


def stored_heads():
    def load(name):
        return json.loads((RESULTS / f"{name}.json").read_text())
    rows = load("taxonomy_vitb_in21k")["results"]
    adaptive = load("adaptive_rank_vitb_in21k")
    return dict(
        kernels={(r["layer"], r["head"]): np.asarray(r["mean_kernel"]) for r in rows},
        shift=[tuple(h) for h in load("kernel_vitb_delta_inet")["selected_heads"]],
        broadcast=[tuple(h) for h in load("broadcast_vitb")["low_r_heads"]],
        rank10=[tuple(h) for h in load("rank10_vitb_ng")["low_r_heads"]],
        rank10_all=[(r["layer"], r["head"]) for r in load("taxonomy_vitb_in21k_with_other")["results"]
                    if r["category"] == "near_global"],
        adaptive_shift=[tuple(h) for h in adaptive["shift_heads"]],
        adaptive_k={(l, h): k for l, h, k in adaptive["rank_heads_with_k"]},
    )


def fresh_heads(model, x, grid, n_extra, gate):
    attn = capture_attention(model, x, n_extra, renormalize=True)
    kernels, diag, cat = {}, {}, {}
    for li, a in enumerate(attn):
        a = a.numpy()
        for hi in range(a.shape[1]):
            kernels[(li, hi)], diag[(li, hi)] = head_diagnostics(a[:, hi], *grid, K=5)
            cat[(li, hi)] = classify(diag[(li, hi)])
    print("Taxonomy:", dict(Counter(cat.values()).most_common()))
    shift = [h for h, c in cat.items() if c in SHIFT_CATS]
    return dict(
        kernels=kernels,
        shift=[h for h, c in cat.items() if c == "delta_shift"],
        broadcast=[h for h, c in cat.items() if c == "fixed_target" and diag[h][gate] < 0.4],
        rank10=[h for h, c in cat.items() if c == "near_global" and diag[h][gate] < 0.4],
        rank10_all=[h for h, c in cat.items() if c == "near_global"],
        adaptive_shift=shift,
        adaptive_k={h: max(10, math.ceil(d["eff_rank_90"])) for h, d in diag.items() if h not in shift},
    )


@torch.no_grad()
def top1(model, loader, device):
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device, non_blocking=True)).argmax(-1).cpu()
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100 * correct / total, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--val-dir", required=True, help="ImageNet val laid out as val/<wnid>/*.JPEG")
    ap.add_argument("--fresh", action="store_true", help="recompute the taxonomy instead of reading results/")
    ap.add_argument("--n-cal", type=int, default=128)
    ap.add_argument("--gate", choices=["r_img", "r_row"], default="r_img",
                    help="residual used to select heads with --fresh; r_img is what the experiments use")
    ap.add_argument("--n-eval", type=int, default=9984)
    ap.add_argument("--eval-order", choices=["class", "random"], default="class")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--exact-svd", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.fresh and args.model != DEFAULT_MODEL:
        ap.error("stored head lists are for the default model; pass --fresh for other models")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(args.model, pretrained=True).eval().to(device)
    grid, n_extra = detect_grid_and_extra(model)

    if args.fresh:
        cal = imagenet_val_loader(args.val_dir, args.n_cal, n=args.n_cal, order="random", seed=args.seed)
        heads = fresh_heads(model, next(iter(cal))[0].to(device), grid, n_extra, args.gate)
    else:
        heads = stored_heads()

    loader = imagenet_val_loader(args.val_dir, args.batch_size, n=args.n_eval, order=args.eval_order, seed=args.seed)
    base, n = top1(model, loader, device)
    print(f"Baseline top-1: {base:.2f}% on {n} images ({args.eval_order} order)")

    def report(label, n_heads, forwards):
        saved = install(model, forwards)
        acc, _ = top1(model, loader, device)
        restore(model, saved)
        print(f"{label:<34s} {n_heads:>4d} heads   delta top-1 {acc - base:+.2f}")

    def kernel_forwards(hs):
        per_layer = {}
        for l, h in hs:
            per_layer.setdefault(l, {})[h] = torch.as_tensor(
                kernel_from_mean(heads["kernels"][(l, h)]), dtype=torch.float32, device=device)
        return {l: make_kernel_forward(model.blocks[l].attn, k, grid, n_extra) for l, k in per_layer.items()}

    def broadcast_forwards(hs):
        per_layer = {}
        for l, h in hs:
            per_layer.setdefault(l, []).append(h)
        return {l: make_broadcast_forward(model.blocks[l].attn, v) for l, v in per_layer.items()}

    def rank_forwards(head_to_k, kernels=None):
        return adaptive_layer_forwards(model, kernels or {}, head_to_k, grid, n_extra, k_floor=1,
                                       exact_svd=args.exact_svd)

    report("shift kernel", len(heads["shift"]), kernel_forwards(heads["shift"]))
    report("broadcast, low-residual fixed-target", len(heads["broadcast"]), broadcast_forwards(heads["broadcast"]))
    report("rank-10, low-residual near-global", len(heads["rank10"]), rank_forwards({h: 10 for h in heads["rank10"]}))
    report("rank-10, all near-global", len(heads["rank10_all"]), rank_forwards({h: 10 for h in heads["rank10_all"]}))
    adaptive_kernels = {h: kernel_from_mean(heads["kernels"][h]) for h in heads["adaptive_shift"]}
    report("all heads (Cor. 3.1, tau=0.9)", len(heads["adaptive_shift"]) + len(heads["adaptive_k"]),
           rank_forwards(heads["adaptive_k"], adaptive_kernels))


if __name__ == "__main__":
    main()
