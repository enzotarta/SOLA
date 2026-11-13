"""Kernel substitution at the heads of chosen categories, with two controls.

A: targeted heads with their own mean kernel. B: random heads with random unit
shifts. C: targeted heads with the kernel reflected through the origin. The
metric is top-1 for classifiers and CLS cosine similarity to baseline otherwise.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from shift_substitution import (
    detect_model_config, evaluate, parse_heads,
    build_tiny_imagenet_eval_loader, build_imagenet_val_eval_loader, image_embedding,
)


# CLS-feature cosine-similarity eval (for encoder-only models without logits)

def _extract_cls(model, x):
    """Image-level feature for a batch x, see shift_substitution.image_embedding."""
    return image_embedding(model, x)


def compute_cls_features(model, dataloader, n_batches, device):
    """Run the (unmodified or hooked) model and collect CLS features.
    Returns a CPU tensor of shape (N, D)."""
    import torch
    model.eval()
    out = []
    n = 0
    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            if bi >= n_batches:
                break
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device, non_blocking=True)
            cls = _extract_cls(model, x)
            out.append(cls.detach().cpu())
            n += x.size(0)
    return torch.cat(out, dim=0)


def evaluate_cls_cos_sim(model, dataloader, n_batches, device,
                          baseline_feats):
    """
    Compute the mean cosine similarity between this model's CLS features
    (presumably with substitution installed) and `baseline_feats`
    (computed beforehand with the model un-substituted).
    Returns (mean_cos, n_eval).
    """
    import torch
    import torch.nn.functional as F
    feats = compute_cls_features(model, dataloader, n_batches, device)
    n = feats.shape[0]
    base = baseline_feats[:n]
    a = F.normalize(feats, dim=-1)
    b = F.normalize(base, dim=-1)
    cos = (a * b).sum(dim=-1)
    return float(cos.mean().item()), n
from kernel_substitution import (
    install_kernel_substitutions, restore_forwards,
    make_delta_kernel, kernel_from_diagnostics_mean,
)


# Helpers

def kernels_from_taxonomy(tax_results, heads):
    """For a list of (layer, head), look up the mean kernel from a
    taxonomy JSON's `results` list."""
    by_lh = {(r["layer"], r["head"]): r for r in tax_results}
    out = {}
    for (li, hi) in heads:
        r = by_lh[(li, hi)]
        kernel = np.asarray(r["mean_kernel"])
        out[(li, hi)] = kernel_from_diagnostics_mean(kernel)
    return out


def heads_in_categories(tax_results, categories):
    out = []
    for r in tax_results:
        if r["category"] in categories:
            out.append((r["layer"], r["head"]))
    return out


def scramble_kernel(kernel: np.ndarray) -> np.ndarray:
    """Spatial point-reflection of a kernel about its center.

    A delta at offset (dy, dx) becomes a delta at (-dy, -dx), a shift in the
    wrong direction. A kernel symmetric about the origin (e.g. a centered
    Gaussian) is almost unchanged.
    """
    return kernel[::-1, ::-1].copy()


# Controls

def control_A_real_kernels(model, eval_loader, cfg, device, n_eval_batches,
                            real_kernel_map):
    print(f"CONTROL A: real heads, real kernels (n={len(real_kernel_map)})")
    base_top1, base_top5, n_eval = evaluate(model, eval_loader, n_eval_batches, device)
    print(f"baseline (n={n_eval}): top-1={base_top1*100:.2f}%  top-5={base_top5*100:.2f}%")
    saved = install_kernel_substitutions(model, real_kernel_map,
                                           cfg["grid"], cfg["n_extra_tokens"])
    top1, top5, _ = evaluate(model, eval_loader, n_eval_batches, device)
    restore_forwards(model, saved)
    d1 = (top1 - base_top1) * 100
    d5 = (top5 - base_top5) * 100
    half_width = 1.96 * (base_top1 * (1 - base_top1) / max(n_eval, 1)) ** 0.5 * 100
    print(f"after substitution: top-1={top1*100:.2f}% (delta {d1:+.2f})  "
          f"top-5={top5*100:.2f}% (delta {d5:+.2f})")
    print(f"  ~95% CI half-width on top-1 (n={n_eval}): +/-{half_width:.2f} pts")
    return {"baseline_top1": base_top1, "subbed_top1": top1,
            "delta_top1": d1, "delta_top5": d5,
            "n_eval": n_eval, "ci_half_width": half_width}


def control_B_random_heads_random_shifts(model, eval_loader, cfg, device,
                                            n_eval_batches, real_heads,
                                            n_random_seeds=3, K_search=5):
    print("CONTROL B: random heads, random unit shifts")
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.num_heads
    unit_shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    base_top1, base_top5, _ = evaluate(model, eval_loader, n_eval_batches, device)
    print(f"baseline: top-1={base_top1*100:.2f}%  top-5={base_top5*100:.2f}%")

    n_to_pick = len(real_heads)
    deltas = []
    for seed in range(n_random_seeds):
        rng = random.Random(seed)
        all_avail = [(L, h) for L in range(n_layers)
                     for h in range(n_heads) if (L, h) not in real_heads]
        chosen_heads = rng.sample(all_avail, n_to_pick)
        kernel_map = {}
        for (L, h) in chosen_heads:
            dy, dx = rng.choice(unit_shifts)
            kernel_map[(L, h)] = make_delta_kernel(K_search, dy, dx)

        saved = install_kernel_substitutions(model, kernel_map,
                                               cfg["grid"], cfg["n_extra_tokens"])
        top1, top5, _ = evaluate(model, eval_loader, n_eval_batches, device)
        restore_forwards(model, saved)
        d1 = (top1 - base_top1) * 100
        deltas.append({"seed": seed, "subs": [list(p) for p in chosen_heads],
                       "delta_top1": d1})
        print(f"  seed {seed}: heads={chosen_heads}  delta top1={d1:+.2f}")

    arr = np.array([d["delta_top1"] for d in deltas])
    return {"baseline_top1": base_top1, "trials": deltas,
            "mean_delta_top1": float(arr.mean()),
            "std_delta_top1": float(arr.std())}


def control_A_real_kernels_cossim(model, eval_loader, cfg, device,
                                     n_eval_batches, real_kernel_map,
                                     baseline_feats):
    print(f"CONTROL A (cos-sim): real heads, real kernels (n={len(real_kernel_map)})")
    saved = install_kernel_substitutions(model, real_kernel_map,
                                           cfg["grid"], cfg["n_extra_tokens"])
    cos, n = evaluate_cls_cos_sim(model, eval_loader, n_eval_batches,
                                    device, baseline_feats)
    restore_forwards(model, saved)
    print(f"  CLS cos-sim to baseline = {cos:.4f}  (n={n})")
    return {"subbed_cls_cos": cos, "n_eval": n}


def control_B_random_heads_random_shifts_cossim(model, eval_loader, cfg, device,
                                                   n_eval_batches, real_heads,
                                                   baseline_feats,
                                                   n_random_seeds=3, K_search=5):
    print("CONTROL B (cos-sim): random heads, random unit shifts")
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.num_heads
    unit_shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    n_to_pick = len(real_heads)
    trials = []
    for seed in range(n_random_seeds):
        rng = random.Random(seed)
        all_avail = [(L, h) for L in range(n_layers)
                     for h in range(n_heads) if (L, h) not in real_heads]
        chosen_heads = rng.sample(all_avail, n_to_pick)
        kernel_map = {(L, h): make_delta_kernel(K_search,
                                                 *rng.choice(unit_shifts))
                       for (L, h) in chosen_heads}
        saved = install_kernel_substitutions(model, kernel_map,
                                               cfg["grid"], cfg["n_extra_tokens"])
        cos, _ = evaluate_cls_cos_sim(model, eval_loader, n_eval_batches,
                                        device, baseline_feats)
        restore_forwards(model, saved)
        trials.append({"seed": seed, "subs": [list(p) for p in chosen_heads],
                        "subbed_cls_cos": cos})
        print(f"  seed {seed}: heads={chosen_heads}  cos={cos:.4f}")
    arr = np.array([t["subbed_cls_cos"] for t in trials])
    return {"trials": trials,
            "mean_cls_cos": float(arr.mean()),
            "std_cls_cos": float(arr.std())}


def control_C_scrambled_kernels_cossim(model, eval_loader, cfg, device,
                                          n_eval_batches, real_kernel_map,
                                          baseline_feats):
    print("CONTROL C (cos-sim): real heads, scrambled kernels")
    scrambled = {(li, hi): scramble_kernel(k) for (li, hi), k in real_kernel_map.items()}
    saved = install_kernel_substitutions(model, scrambled,
                                           cfg["grid"], cfg["n_extra_tokens"])
    cos, _ = evaluate_cls_cos_sim(model, eval_loader, n_eval_batches,
                                    device, baseline_feats)
    restore_forwards(model, saved)
    print(f"  CLS cos-sim to baseline = {cos:.4f}")
    return {"subbed_cls_cos": cos}


def control_C_scrambled_kernels(model, eval_loader, cfg, device, n_eval_batches,
                                  real_kernel_map):
    print("CONTROL C: real heads, spatially scrambled kernels (point-reflected)")
    base_top1, base_top5, _ = evaluate(model, eval_loader, n_eval_batches, device)
    print(f"baseline: top-1={base_top1*100:.2f}%  top-5={base_top5*100:.2f}%")

    scrambled = {(li, hi): scramble_kernel(k) for (li, hi), k in real_kernel_map.items()}
    saved = install_kernel_substitutions(model, scrambled,
                                           cfg["grid"], cfg["n_extra_tokens"])
    top1, top5, _ = evaluate(model, eval_loader, n_eval_batches, device)
    restore_forwards(model, saved)
    d1 = (top1 - base_top1) * 100
    d5 = (top5 - base_top5) * 100
    print(f"  delta top1={d1:+.2f}  delta top5={d5:+.2f}")
    return {"baseline_top1": base_top1, "subbed_top1": top1,
            "delta_top1": d1, "delta_top5": d5}


# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--taxonomy", required=True,
                    help="Path to operator_taxonomy.py output JSON")
    ap.add_argument("--categories", nargs="+", required=True,
                    help=("Categories of heads to substitute. Examples: "
                          "delta_shift, soft_shift, diffuse_local, "
                          "gaussian_local, multi_delta. "
                          "Multiple categories will be substituted "
                          "simultaneously."))
    ap.add_argument("--source", default="tiny-imagenet",
                    choices=["tiny-imagenet", "imagenet-val"])
    ap.add_argument("--imagenet-val-dir", default="data/imagenet/val")
    ap.add_argument("--batches", type=int, default=156)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--n-random-seeds", type=int, default=3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="kernel_controls.json")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--metric", default="top1",
                    choices=["top1", "cls-cos-sim"],
                    help=("Evaluation metric. 'top1' uses ImageNet-1k "
                          "logits (works for supervised models). "
                          "'cls-cos-sim' computes cosine similarity of "
                          "the substituted CLS feature against the "
                          "unmodified CLS, for encoder-only models "
                          "(DINO, CLIP, MAE)."))
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tax = json.loads(Path(args.taxonomy).read_text())
    if tax.get("model") != args.model:
        print(f"WARNING: taxonomy model = {tax.get('model')} but "
              f"--model = {args.model}")
    selected_heads = heads_in_categories(tax["results"], args.categories)
    if not selected_heads:
        raise SystemExit(f"No heads in categories {args.categories}")
    real_kernel_map = kernels_from_taxonomy(tax["results"], selected_heads)
    print(f"Selected {len(selected_heads)} heads in categories "
          f"{args.categories}: {selected_heads}")

    import timm
    model = timm.create_model(args.model, pretrained=True,
                                 img_size=args.image_size).eval().to(device)
    cfg = detect_model_config(model, args.image_size)

    if args.source == "tiny-imagenet":
        eval_loader, _ = build_tiny_imagenet_eval_loader(
            args.batch_size, args.image_size)
    else:
        eval_loader, _ = build_imagenet_val_eval_loader(
            args.batch_size, args.image_size,
            val_dir=args.imagenet_val_dir, num_workers=args.num_workers)

    results = {
        "model": args.model,
        "categories": args.categories,
        "selected_heads": [list(p) for p in selected_heads],
        "source": args.source,
        "n_batches": args.batches,
        "metric": args.metric,
    }

    if args.metric == "top1":
        results["control_A"] = control_A_real_kernels(
            model, eval_loader, cfg, device, args.batches, real_kernel_map)
        results["control_B"] = control_B_random_heads_random_shifts(
            model, eval_loader, cfg, device, args.batches,
            set(selected_heads), n_random_seeds=args.n_random_seeds)
        results["control_C"] = control_C_scrambled_kernels(
            model, eval_loader, cfg, device, args.batches, real_kernel_map)

        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.out}")
        print("\nSUMMARY (top-1)")
        print(f"  A. Real kernels:      delta top1={results['control_A']['delta_top1']:+.2f}  "
              f"(95% CI +/-{results['control_A']['ci_half_width']:.2f})")
        print(f"  B. Random heads/shifts: delta top1={results['control_B']['mean_delta_top1']:+.2f}  "
              f"(std +/-{results['control_B']['std_delta_top1']:.2f})")
        print(f"  C. Scrambled kernels: delta top1={results['control_C']['delta_top1']:+.2f}")
    else:
        # cls-cos-sim mode
        print("Computing baseline CLS features ...")
        baseline_feats = compute_cls_features(
            model, eval_loader, args.batches, device)
        print(f"  collected {baseline_feats.shape[0]} baseline CLS vectors")
        results["baseline_cls_cos"] = 1.0       # by definition
        results["control_A"] = control_A_real_kernels_cossim(
            model, eval_loader, cfg, device, args.batches,
            real_kernel_map, baseline_feats)
        results["control_B"] = control_B_random_heads_random_shifts_cossim(
            model, eval_loader, cfg, device, args.batches,
            set(selected_heads), baseline_feats,
            n_random_seeds=args.n_random_seeds)
        results["control_C"] = control_C_scrambled_kernels_cossim(
            model, eval_loader, cfg, device, args.batches,
            real_kernel_map, baseline_feats)

        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.out}")
        print("\nSUMMARY (CLS cos-sim to baseline)")
        print(f"  A. Real kernels:      cos={results['control_A']['subbed_cls_cos']:.4f}")
        print(f"  B. Random heads/shifts: cos={results['control_B']['mean_cls_cos']:.4f}  "
              f"(std +/-{results['control_B']['std_cls_cos']:.4f})")
        print(f"  C. Scrambled kernels: cos={results['control_C']['subbed_cls_cos']:.4f}")


if __name__ == "__main__":
    main()
