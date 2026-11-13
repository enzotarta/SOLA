"""Per-class top-1 on full ImageNet val under each operator substitution."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))


def evaluate_per_class(model, loader, n_batches, device, n_classes=1000):
    """Run inference and return per-class (correct1, total) arrays."""
    model.eval()
    correct = torch.zeros(n_classes, dtype=torch.long)
    total = torch.zeros(n_classes, dtype=torch.long)
    aggregate_top1 = aggregate_count = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= n_batches:
                break
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            ok = (pred == y)
            valid = (y >= 0)
            ok = ok & valid
            y_cpu = y.cpu().clamp(min=0, max=n_classes-1)
            ok_cpu = ok.cpu().long()
            mask_cpu = valid.cpu()
            # Only update for valid labels
            y_valid = y_cpu[mask_cpu]
            ok_valid = ok_cpu[mask_cpu]
            total.scatter_add_(0, y_valid, torch.ones_like(y_valid))
            correct.scatter_add_(0, y_valid, ok_valid)
            aggregate_top1 += ok.sum().item()
            aggregate_count += valid.sum().item()
    return (correct.numpy().tolist(),
            total.numpy().tolist(),
            aggregate_top1 / max(aggregate_count, 1),
            aggregate_count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--diag", default=None,
                    help="r_abs diagnostic JSON for low_r broadcast subset")
    ap.add_argument("--r-thresh", type=float, default=0.4)
    ap.add_argument("--shift-cats", default="delta_shift,soft_shift")
    ap.add_argument("--n-eval-batches", type=int, default=782)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--imagenet-val-dir", default="data/imagenet/val")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--rank-k", type=int, default=10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions",
                    default="baseline,shifts,broadcast_ft,rank10_ng",
                    help="comma-separated list of conditions to run")
    args = ap.parse_args()

    import timm
    from shift_substitution import (
        build_imagenet_val_eval_loader, detect_model_config,
    )
    from kernel_substitution import (
        kernel_from_diagnostics_mean,
        install_kernel_substitutions,
        restore_forwards as restore_kernel,
    )
    from attn_matrix_substitution import (
        install_broadcast_substitutions,
        install_rank_k_substitutions,
        restore_forwards as restore_attn,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device} (img_size={args.img_size}) ...")
    model = timm.create_model(args.model, pretrained=True,
                                 img_size=args.img_size).eval().to(device)
    cfg = detect_model_config(model, image_size=args.img_size)

    loader, _ = build_imagenet_val_eval_loader(
        args.eval_batch_size, image_size=args.img_size,
        val_dir=args.imagenet_val_dir, num_workers=4)

    tax = json.loads(Path(args.taxonomy).read_text())
    rows = tax["results"]
    res_by_lh = {(r["layer"], r["head"]): r for r in rows}

    shift_cats = set(args.shift_cats.split(","))
    shift_heads = [(r["layer"], r["head"]) for r in rows
                   if r["category"] in shift_cats]
    ft_heads = [(r["layer"], r["head"]) for r in rows
                if r["category"] == "fixed_target"]
    ng_heads = [(r["layer"], r["head"]) for r in rows
                if r["category"] == "near_global"]

    diag_by_lh = None
    if args.diag is not None:
        diag = json.loads(Path(args.diag).read_text())
        diag_by_lh = {(r["layer"], r["head"]): r["r_abs_mean"]
                      for r in diag["per_head"]}

    print(f"Shifts: {len(shift_heads)}  FT: {len(ft_heads)}  "
          f"NG: {len(ng_heads)}")

    conditions = [c.strip() for c in args.conditions.split(",")]
    out = {
        "model": args.model,
        "img_size": args.img_size,
        "n_classes": 1000,
        "conditions": conditions,
        "shift_heads": [list(h) for h in shift_heads],
        "ft_heads": [list(h) for h in ft_heads],
        "ng_heads": [list(h) for h in ng_heads],
        "results": {},
    }

    def run_condition(name, install_fn=None, restore_fn=None):
        print(f"\n[{name}] running ...")
        saved = install_fn() if install_fn else None
        correct, total, agg_top1, n = evaluate_per_class(
            model, loader, args.n_eval_batches, device)
        if saved is not None:
            restore_fn(model, saved)
        print(f"  {name}: top1={agg_top1*100:.2f}%  n={n}")
        return {
            "per_class_correct": correct,
            "per_class_total": total,
            "aggregate_top1": agg_top1,
            "n_eval": n,
        }

    if "baseline" in conditions:
        out["results"]["baseline"] = run_condition("baseline")

    if "shifts" in conditions and shift_heads:
        kernel_map = {(li, hi): kernel_from_diagnostics_mean(
                          np.asarray(res_by_lh[(li, hi)]["mean_kernel"]))
                       for (li, hi) in shift_heads}
        out["results"]["shifts"] = run_condition(
            "shifts",
            install_fn=lambda: install_kernel_substitutions(
                model, kernel_map, cfg["grid"], cfg["n_extra_tokens"]),
            restore_fn=restore_kernel,
        )

    if "broadcast_ft" in conditions and ft_heads:
        out["results"]["broadcast_ft"] = run_condition(
            "broadcast_ft",
            install_fn=lambda: install_broadcast_substitutions(model, ft_heads),
            restore_fn=restore_attn,
        )

    if "broadcast_ft_low_r" in conditions and ft_heads and diag_by_lh:
        low_r_ft = [(li, hi) for (li, hi) in ft_heads
                    if diag_by_lh.get((li, hi), 1.0) < args.r_thresh]
        print(f"\n  low-r FT subset: n={len(low_r_ft)}: {low_r_ft}")
        out["low_r_ft_heads"] = [list(h) for h in low_r_ft]
        if low_r_ft:
            out["results"]["broadcast_ft_low_r"] = run_condition(
                "broadcast_ft_low_r",
                install_fn=lambda: install_broadcast_substitutions(
                    model, low_r_ft),
                restore_fn=restore_attn,
            )

    if "rank10_ng" in conditions and ng_heads:
        out["results"]["rank10_ng"] = run_condition(
            f"rank{args.rank_k}_ng",
            install_fn=lambda: install_rank_k_substitutions(
                model, ng_heads, k_rank=args.rank_k),
            restore_fn=restore_attn,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    # Quick summary: aggregate top-1 + #classes that moved >= 1 pp
    if "baseline" in out["results"]:
        baseline = np.array(out["results"]["baseline"]["per_class_correct"]) \
                   / np.maximum(np.array(out["results"]["baseline"]["per_class_total"]), 1)
        print("\naggregate top-1 / #classes shifted >= 5 pp")
        for name, r in out["results"].items():
            cur = np.array(r["per_class_correct"]) \
                  / np.maximum(np.array(r["per_class_total"]), 1)
            delta = cur - baseline
            n_down = int((delta <= -0.05).sum())
            n_up = int((delta >= 0.05).sum())
            print(f"  {name:<24}  top1={r['aggregate_top1']*100:.2f}%  "
                  f"down>=5pp:{n_down:>4d}  up>=5pp:{n_up:>4d}")


if __name__ == "__main__":
    main()
