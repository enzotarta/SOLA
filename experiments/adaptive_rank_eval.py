"""Full-coverage substitution.

Kernels at shift-class heads and per-image rank-k elsewhere, with
k = max(--k-floor, ceil(eff_rank_90)) and --k-floor 10 by default.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from kernel_substitution import (
    apply_kernel_to_values, kernel_from_diagnostics_mean,
)
from shift_substitution import image_embedding


def make_adaptive_rank_attention_forward(
    attn,
    shifts: Dict[int, torch.Tensor],
    head_to_k: Dict[int, int],
    grid: Tuple[int, int],
    n_extra_tokens: int,
):
    """Per-layer forward with kernel sub at `shifts` and per-head
    rank-k truncation at every head listed in `head_to_k`."""
    # Group rank heads by k for batched SVD
    k_groups: Dict[int, List[int]] = {}
    for h, k in head_to_k.items():
        k_groups.setdefault(int(k), []).append(int(h))
    k_groups_sorted = sorted(k_groups.items())

    def new_forward(x, *args, **kwargs):
        B, N, C = x.shape
        H = attn.num_heads
        Dh = C // H
        qkv = attn.qkv(x).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            q = attn.q_norm(q); k = attn.k_norm(k)
        a = (q @ k.transpose(-2, -1)) * attn.scale
        a = a.softmax(dim=-1)
        if hasattr(attn, "attn_drop"):
            a = attn.attn_drop(a)

        # Per-head rank-k SVD truncation (grouped by k for batching)
        for k_val, heads in k_groups_sorted:
            if k_val >= N:
                continue                              # nothing to truncate
            idx = torch.tensor(heads, device=x.device, dtype=torch.long)
            A_sub = a.index_select(1, idx)           # (B, ns, N, N)
            B0, ns, N0, _ = A_sub.shape
            A_flat = A_sub.reshape(B0 * ns, N0, N0)
            q_proj = min(k_val + 4, N0)
            U, S, V = torch.svd_lowrank(A_flat, q=q_proj, niter=2)
            U_k = U[..., :k_val]
            S_k = S[..., :k_val]
            V_k = V[..., :k_val]
            A_rk = (U_k @ torch.diag_embed(S_k)
                     @ V_k.transpose(-2, -1)).reshape(B0, ns, N0, N0)
            for j, h in enumerate(heads):
                a[:, h] = A_rk[:, j]

        attn_out = a @ v                              # (B, H, N, Dh)

        # Kernel substitution at shift heads (last, since it overrides
        # attn_out at those heads)
        for h, kernel in shifts.items():
            v_h = v[:, h]
            attn_out[:, h] = apply_kernel_to_values(
                v_h, kernel, grid, n_extra_tokens)

        out = attn_out.transpose(1, 2).reshape(B, N, C)
        out = attn.proj(out)
        if hasattr(attn, "proj_drop") and attn.proj_drop is not None:
            out = attn.proj_drop(out)
        return out

    return new_forward


def install_adaptive_rank(
    model: nn.Module,
    kernel_map: Dict[Tuple[int, int], np.ndarray],
    head_to_k: Dict[Tuple[int, int], int],
    grid: Tuple[int, int],
    n_extra_tokens: int,
):
    """kernel_map: shift heads (kernel sub).
    head_to_k: every other head, with its rank target."""
    shifts_by_layer: Dict[int, Dict[int, torch.Tensor]] = {}
    for (li, hi), ker in kernel_map.items():
        ker_t = torch.as_tensor(ker, dtype=torch.float32)
        shifts_by_layer.setdefault(li, {})[hi] = ker_t
    rank_by_layer: Dict[int, Dict[int, int]] = {}
    for (li, hi), k in head_to_k.items():
        rank_by_layer.setdefault(li, {})[hi] = int(k)

    affected = set(shifts_by_layer) | set(rank_by_layer)
    saved = {}
    for li in sorted(affected):
        attn = model.blocks[li].attn
        device = next(attn.parameters()).device
        shifts_li = {h: ker.to(device)
                     for h, ker in shifts_by_layer.get(li, {}).items()}
        saved[li] = attn.forward
        attn.forward = make_adaptive_rank_attention_forward(
            attn,
            shifts=shifts_li,
            head_to_k=rank_by_layer.get(li, {}),
            grid=grid,
            n_extra_tokens=n_extra_tokens,
        )
    return saved


def restore_forwards(model, saved):
    for li, fwd in saved.items():
        model.blocks[li].attn.forward = fwd


def evaluate_top1(model, loader, n_eval_max, device):
    correct = 0; total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            pred = model(x).argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
            if total >= n_eval_max: break
    return correct / max(total, 1), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--svd-spectrum", required=True,
                    help="svd_spectrum_*.json with per-head eff_rank_90")
    ap.add_argument("--shift-cats", default="delta_shift,soft_shift,"
                                              "multi_delta,gaussian_local,"
                                              "self_only")
    ap.add_argument("--k-floor", type=int, default=10,
                    help="minimum k for adaptive rank-k")
    ap.add_argument("--k-ceil", type=int, default=200,
                    help="maximum k; heads above are skipped (~baseline)")
    ap.add_argument("--n-eval", type=int, default=50000)
    ap.add_argument("--cls-cos-n", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--imagenet-val-dir",
                    default="data/imagenet/val")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-top1", action="store_true",
                    help="skip top-1 (encoder-only models)")
    args = ap.parse_args()

    import timm
    from shift_substitution import (
        build_imagenet_val_eval_loader, detect_model_config,
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device} ...", flush=True)
    model = timm.create_model(args.model, pretrained=True,
                                 img_size=args.img_size).eval().to(device)
    cfg = detect_model_config(model, image_size=args.img_size)

    tax = json.loads(Path(args.taxonomy).read_text())
    rows = tax["results"]
    res_by_lh = {(r["layer"], r["head"]): r for r in rows}
    sd = json.loads(Path(args.svd_spectrum).read_text())
    er90_by_lh = {(r["layer"], r["head"]): r["eff_rank_90"]
                  for r in sd["per_head"]}

    shift_cats = set(args.shift_cats.split(","))
    shift_heads = [(r["layer"], r["head"]) for r in rows
                   if r["category"] in shift_cats
                   and res_by_lh[(r["layer"], r["head"])].get("mean_kernel")
                   is not None]

    rank_heads_with_k: Dict[Tuple[int, int], int] = {}
    rank_dist: Dict[int, int] = {}
    for r in rows:
        lh = (r["layer"], r["head"])
        if lh in set(shift_heads):
            continue
        er = er90_by_lh.get(lh)
        if er is None:
            continue
        k = max(args.k_floor, int(math.ceil(er)))
        if k > args.k_ceil:
            continue
        rank_heads_with_k[lh] = k
        rank_dist[k] = rank_dist.get(k, 0) + 1

    print(f"Shifts: n={len(shift_heads)}", flush=True)
    print(f"Adaptive rank-k: n={len(rank_heads_with_k)}; "
          f"k floor={args.k_floor}, ceil={args.k_ceil}", flush=True)
    print(f"  k distribution: "
          f"{sorted(rank_dist.items())[:8]}{'...' if len(rank_dist)>8 else ''}",
          flush=True)
    n_total_subs = len(shift_heads) + len(rank_heads_with_k)
    n_total_heads = len(rows)
    print(f"  coverage: {n_total_subs}/{n_total_heads} = "
          f"{100*n_total_subs/n_total_heads:.1f}%", flush=True)

    kernel_map = {(li, hi): kernel_from_diagnostics_mean(
                       np.asarray(res_by_lh[(li, hi)]["mean_kernel"]))
                   for (li, hi) in shift_heads}

    out = {
        "model": args.model,
        "img_size": args.img_size,
        "k_floor": args.k_floor,
        "k_ceil": args.k_ceil,
        "n_shifts": len(shift_heads),
        "n_rank_subs": len(rank_heads_with_k),
        "n_total_heads": n_total_heads,
        "coverage_frac": n_total_subs / n_total_heads,
        "rank_distribution": dict(sorted(rank_dist.items())),
        "shift_heads": [list(h) for h in shift_heads],
        "rank_heads_with_k": [[li, hi, k] for (li, hi), k
                               in rank_heads_with_k.items()],
    }

    if not args.no_top1:
        loader, _ = build_imagenet_val_eval_loader(
            args.batch_size, image_size=args.img_size,
            val_dir=args.imagenet_val_dir, num_workers=4)
        print("\n[baseline] top-1 ...", flush=True)
        top1_b, n_b = evaluate_top1(model, loader, args.n_eval, device)
        print(f"  baseline: top1={top1_b*100:.3f}%  n={n_b}", flush=True)
        out["baseline_top1"] = top1_b
        out["baseline_n"] = n_b

        loader, _ = build_imagenet_val_eval_loader(
            args.batch_size, image_size=args.img_size,
            val_dir=args.imagenet_val_dir, num_workers=4)
        print("\n[adaptive SOLA] top-1 ...", flush=True)
        saved = install_adaptive_rank(model, kernel_map, rank_heads_with_k,
                                        cfg["grid"], cfg["n_extra_tokens"])
        top1_s, n_s = evaluate_top1(model, loader, args.n_eval, device)
        restore_forwards(model, saved)
        print(f"  adaptive: top1={top1_s*100:.3f}%  n={n_s}  "
              f"delta ={100*(top1_s-top1_b):+.3f}%", flush=True)
        out["adaptive_top1"] = top1_s
        out["adaptive_n"] = n_s
        out["delta_top1_pct"] = 100 * (top1_s - top1_b)

    # CLS cos-sim (batched)
    print(f"\n[CLS cos-sim] target n={args.cls_cos_n}", flush=True)

    def cls_features(m, n_max):
        loader_local, _ = build_imagenet_val_eval_loader(
            args.batch_size, image_size=args.img_size,
            val_dir=args.imagenet_val_dir, num_workers=4)
        feats = []; n_seen = 0
        with torch.no_grad():
            for batch in loader_local:
                x = batch[0].to(device)
                feats.append(image_embedding(m, x).detach().cpu())
                n_seen += x.shape[0]
                if n_seen >= n_max: break
        return torch.cat(feats, dim=0)[:n_max]

    f_base = cls_features(model, args.cls_cos_n)
    saved = install_adaptive_rank(model, kernel_map, rank_heads_with_k,
                                    cfg["grid"], cfg["n_extra_tokens"])
    f_sub = cls_features(model, args.cls_cos_n)
    restore_forwards(model, saved)

    cos = torch.cosine_similarity(f_base.float(), f_sub.float(), dim=-1)
    out["cls_cos_mean"] = cos.mean().item()
    out["cls_cos_p10"] = float(torch.quantile(cos.float(), 0.10).item())
    out["cls_cos_p90"] = float(torch.quantile(cos.float(), 0.90).item())
    out["cls_cos_n"] = int(f_base.shape[0])
    print(f"  CLS cos-sim: mean={out['cls_cos_mean']:.4f}  "
          f"p10={out['cls_cos_p10']:.4f}  p90={out['cls_cos_p90']:.4f}  "
          f"n={f_base.shape[0]}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
