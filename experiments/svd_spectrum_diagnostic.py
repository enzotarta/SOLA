"""Per-head singular value spectrum of softmax(QK^T) on a calibration batch.

Reports sigma_1 / ||A||_F and the effective rank at 90% and 99% of the energy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from operator_taxonomy import make_attention_recorder_class
from escape_mass_sweep import detect_model_config


def spectrum_stats_per_head(attn_h: np.ndarray) -> dict:
    """
    attn_h : (n_img, n_q, n_k), the attention of one head.
    Returns mean spectrum across images.
    """
    A = torch.from_numpy(attn_h.astype(np.float32))
    # Per-image SVD; use float32 single-precision SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)        # S: (n_img, k)
    # Energy: sum of squared singular values
    E = (S ** 2).sum(dim=-1)                                    # (n_img,)
    # Cumulative
    cum = (S ** 2).cumsum(dim=-1) / E.unsqueeze(-1)             # (n_img, k)
    # Effective rank for 90% / 99% energy: smallest r such that cum[r-1] >= thr
    def eff_rank(thr):
        # First index where cumulative reaches threshold
        ranks = []
        for i in range(cum.shape[0]):
            r = int((cum[i] >= thr).nonzero()[0].item()) + 1
            ranks.append(r)
        return float(np.mean(ranks))
    s1_frac = float((S[:, 0] / torch.sqrt(E)).mean().item())
    s2_over_s1 = float((S[:, 1] / S[:, 0]).mean().item()) if S.shape[-1] > 1 else float("nan")
    return {
        "sigma1_frac":          s1_frac,
        "sigma2_over_sigma1":   s2_over_s1,
        "eff_rank_90":          eff_rank(0.90),
        "eff_rank_99":          eff_rank(0.99),
        "n_img":                int(attn_h.shape[0]),
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
        a = attn_by_layer[li].numpy()                  # (B', n_h, N, N)
        # Strip extras + renormalize to spatial
        a_sp = a[:, :, n_extra:, n_extra:]
        a_sp = a_sp / a_sp.sum(axis=-1, keepdims=True).clip(min=1e-12)
        for hi in range(n_heads):
            stats = spectrum_stats_per_head(a_sp[:, hi])
            out.append({"layer": li, "head": hi, **stats})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out", default="results/svd_spectrum.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--imagenet-val-dir",
                    default="data/imagenet/val")
    ap.add_argument("--source", default="imagenet-val",
                    choices=("imagenet-val", "tiny-imagenet"),
                    help="calibration images: the first ones of ImageNet val "
                         "in class order, or of the tiny-ImageNet val split")
    args = ap.parse_args()

    import timm
    from shift_substitution import (build_imagenet_val_eval_loader,
                                    build_tiny_imagenet_eval_loader)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device} ...")
    model = timm.create_model(args.model, pretrained=True,
                                 img_size=args.img_size).eval().to(device)
    if args.source == "tiny-imagenet":
        eval_loader, _ = build_tiny_imagenet_eval_loader(args.batch_size, args.img_size)
    else:
        eval_loader, _ = build_imagenet_val_eval_loader(
            args.batch_size, image_size=args.img_size,
            val_dir=args.imagenet_val_dir, num_workers=2)

    print(f"Recording attention on {args.batches} batches ...")
    per_head = sweep(model, eval_loader, device, n_batches=args.batches)

    tax = json.loads(Path(args.taxonomy).read_text())
    cat = {(r["layer"], r["head"]): r["category"] for r in tax["results"]}
    for r in per_head:
        r["category"] = cat.get((r["layer"], r["head"]), "?")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": args.model,
                                       "n_batches": args.batches,
                                       "per_head": per_head}, indent=2))
    print(f"Wrote {out_path}")

    # Per-category summary
    by_cat = {}
    for r in per_head:
        by_cat.setdefault(r["category"], []).append(r)
    print(f"\nPer-category SVD spectrum summary")
    print(f"{'cat':<14}  {'n':>4}  {'sigma1_frac':>11}  {'eff_r90':>8}  {'eff_r99':>8}")
    for c in sorted(by_cat, key=lambda c: -np.median([r['sigma1_frac'] for r in by_cat[c]])):
        rs = by_cat[c]
        s1 = np.median([r['sigma1_frac'] for r in rs])
        r90 = np.median([r['eff_rank_90'] for r in rs])
        r99 = np.median([r['eff_rank_99'] for r in rs])
        print(f"{c:<14}  {len(rs):>4}  {s1:>11.3f}  {r90:>8.1f}  {r99:>8.1f}")


if __name__ == "__main__":
    main()
