"""Check for heads whose attention is a sharp shift with an image-dependent offset."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from operator_taxonomy import (
    make_attention_recorder_class, recentered_kernel_stack,
)
from escape_mass_sweep import detect_model_config


def per_image_shift_stats(attn_h: np.ndarray, H: int, W: int, K: int = 5) -> dict:
    """attn_h : (n_img, n_q, n_k) for one head."""
    n_img, n_q, _ = attn_h.shape
    # Interior query mask: only queries with full window in-grid
    interior_idx = []
    for qi in range(n_q):
        qy, qx = qi // W, qi % W
        if K <= qy < H - K and K <= qx < W - K:
            interior_idx.append(qi)

    per_img_top1 = []
    peaks = []
    for i in range(n_img):
        stack = recentered_kernel_stack(attn_h[i:i+1], H, W, K)  # (n_q, 2K+1, 2K+1)
        if not interior_idx:
            continue
        mean_i = stack[interior_idx].mean(axis=0)
        Z = mean_i.sum()
        if Z <= 0:
            continue
        mean_i_n = mean_i / Z
        flat = mean_i_n.flatten()
        top1 = float(flat.max())
        peak_flat = int(flat.argmax())
        py = peak_flat // (2 * K + 1) - K
        px = peak_flat % (2 * K + 1) - K
        per_img_top1.append(top1)
        peaks.append((py, px))

    if not per_img_top1:
        return {"per_img_top1_mean": float("nan"), "peak_y_std": float("nan"),
                "peak_x_std": float("nan"), "peak_unique": 0}
    py_arr = np.array([p[0] for p in peaks])
    px_arr = np.array([p[1] for p in peaks])
    return {
        "per_img_top1_mean": float(np.mean(per_img_top1)),
        "per_img_top1_min":  float(np.min(per_img_top1)),
        "peak_y_std":        float(np.std(py_arr)),
        "peak_x_std":        float(np.std(px_arr)),
        "peak_unique":       int(len({tuple(p) for p in peaks})),
        "n_img":             len(peaks),
    }


def sweep(model, eval_loader, device, n_batches: int, K: int = 5):
    AttentionRecorder = make_attention_recorder_class()
    cfg = detect_model_config(model, image_size=224)
    H, W = cfg["grid"]
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
        a_sp = a[:, :, n_extra:, n_extra:]
        a_sp = a_sp / a_sp.sum(axis=-1, keepdims=True).clip(min=1e-12)
        for hi in range(n_heads):
            stats = per_image_shift_stats(a_sp[:, hi], H, W, K=K)
            out.append({"layer": li, "head": hi, **stats})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--taxonomy", required=True,
                    help="taxonomy_*.json (used for category labels)")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out", default="results/per_image_shift_diag.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import timm
    from shift_substitution import build_tiny_imagenet_eval_loader

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device} ...")
    model = timm.create_model(args.model, pretrained=True,
                                 img_size=args.img_size).eval().to(device)
    eval_loader, _ = build_tiny_imagenet_eval_loader(args.batch_size, args.img_size)

    print(f"Recording attention on {args.batches} calibration batches ...")
    per_head = sweep(model, eval_loader, device, n_batches=args.batches, K=args.K)

    # Join with taxonomy
    tax = json.loads(Path(args.taxonomy).read_text())
    cat = {(r["layer"], r["head"]): r["category"] for r in tax["results"]}
    for r in per_head:
        r["category"] = cat.get((r["layer"], r["head"]), "?")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": args.model,
                                       "n_batches": args.batches,
                                       "K": args.K,
                                       "per_head": per_head}, indent=2))
    print(f"Wrote {out_path}")

    # Pretty-print: sort by per_img_top1, show top 30 with peak_std
    print("\nPer-image-shift candidates (top per_img_top1, all heads)")
    sortable = [r for r in per_head if not np.isnan(r.get("per_img_top1_mean", float("nan")))]
    sortable.sort(key=lambda r: -r["per_img_top1_mean"])
    print(f"{'L':>3} {'H':>3} {'cat':<14} {'top1_pi':>7} {'pk_std_y':>8} {'pk_std_x':>8} {'pk_uniq':>7}")
    for r in sortable[:40]:
        print(f"{r['layer']:>3} {r['head']:>3} {r['category']:<14} "
              f"{r['per_img_top1_mean']:>7.3f} "
              f"{r['peak_y_std']:>8.2f} {r['peak_x_std']:>8.2f} "
              f"{r['peak_unique']:>7d}")


if __name__ == "__main__":
    main()
