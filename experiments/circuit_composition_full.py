"""Path patching for broadcast and rank-10 senders."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from circuit_composition import (
    HeadOutputCapturer, install_mean_ablations, restore_forwards,
    per_head_cossim,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--diag", default=None,
                    help="r_abs_mean diagnostic JSON (needed for "
                         "broadcast and rank10 senders to pick low-r subset)")
    ap.add_argument("--operator", required=True,
                    choices=["shift", "broadcast", "rank10"])
    ap.add_argument("--r-thresh", type=float, default=0.4,
                    help="r_abs threshold for low-r subset")
    ap.add_argument("--rank-k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-batches", type=int, default=4)
    ap.add_argument("--imagenet-val-dir",
                    default="data/imagenet/val")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-senders", type=int, default=None)
    args = ap.parse_args()

    import timm
    from shift_substitution import (
        build_imagenet_val_eval_loader, detect_model_config,
    )
    from attn_matrix_substitution import (
        install_broadcast_substitutions,
        install_rank_k_substitutions,
    )
    from kernel_substitution import (
        kernel_from_diagnostics_mean,
        install_kernel_substitutions,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device} ...", flush=True)
    model = timm.create_model(args.model, pretrained=True,
                                 img_size=args.img_size).eval().to(device)
    cfg = detect_model_config(model, image_size=args.img_size)

    loader, _ = build_imagenet_val_eval_loader(
        args.batch_size, image_size=args.img_size,
        val_dir=args.imagenet_val_dir, num_workers=2)
    xs = []
    for bi, batch in enumerate(loader):
        if bi >= args.n_batches: break
        xs.append(batch[0])
    x = torch.cat(xs, dim=0).to(device)
    print(f"  calibration batch: {x.shape}", flush=True)

    tax = json.loads(Path(args.taxonomy).read_text())
    rows = tax["results"]
    res_by_lh = {(r["layer"], r["head"]): r for r in rows}
    n_blocks = max(r["layer"] for r in rows) + 1
    n_heads = max(r["head"] for r in rows) + 1

    # Pick senders by operator
    if args.operator == "shift":
        cats = {"delta_shift", "soft_shift"}
        senders = sorted([(r["layer"], r["head"]) for r in rows
                          if r["category"] in cats],
                         key=lambda t: (t[0], t[1]))
    else:
        # need r_abs from --diag
        assert args.diag, f"--diag required for operator={args.operator}"
        diag = json.loads(Path(args.diag).read_text())
        rabs_by_lh = {(r["layer"], r["head"]): r["r_abs_mean"]
                      for r in diag["per_head"]}
        target_cat = ("fixed_target" if args.operator == "broadcast"
                      else "near_global")
        senders = sorted(
            [(r["layer"], r["head"]) for r in rows
             if r["category"] == target_cat
             and rabs_by_lh.get((r["layer"], r["head"]), 1.0) < args.r_thresh],
            key=lambda t: (t[0], t[1]))

    if args.max_senders:
        senders = senders[:args.max_senders]
    print(f"  {args.operator} senders: {len(senders)} of {n_blocks}x{n_heads}",
          flush=True)
    by_layer = {}
    for (l, h) in senders:
        by_layer.setdefault(l, []).append(h)
    print("  by layer: " + ", ".join(f"L{l}={len(hs)}"
                                       for l, hs in sorted(by_layer.items())),
          flush=True)

    cap = HeadOutputCapturer(model)

    # Baseline
    cap.reset()
    with torch.no_grad():
        _ = model(x)
    baseline_acts = dict(cap.acts)
    print(f"  baseline heads captured: {len(baseline_acts)}", flush=True)

    # Precompute per-sender kernel for shifts
    if args.operator == "shift":
        kernel_full = {lh: kernel_from_diagnostics_mean(
                         np.asarray(res_by_lh[lh]["mean_kernel"]))
                       for lh in senders}

    out = {
        "model": args.model,
        "operator": args.operator,
        "r_thresh": args.r_thresh,
        "rank_k": args.rank_k,
        "img_size": args.img_size,
        "batch_size": int(x.shape[0]),
        "n_blocks": n_blocks,
        "n_heads_per_block": n_heads,
        "senders": [list(h) for h in senders],
        "per_sender": {},
    }

    def run_with(install_fn):
        cap.reset()
        saved = install_fn()
        with torch.no_grad():
            _ = model(x)
        sub_acts = dict(cap.acts)
        restore_forwards(model, saved)
        return sub_acts

    for si, sender in enumerate(senders):
        le, he = sender
        print(f"\n[{si+1}/{len(senders)}] sender L{le} h{he} ({args.operator})",
              flush=True)

        if args.operator == "shift":
            kmap = {sender: kernel_full[sender]}
            install_op = lambda: install_kernel_substitutions(
                model, kmap, cfg["grid"], cfg["n_extra_tokens"])
        elif args.operator == "broadcast":
            install_op = lambda: install_broadcast_substitutions(
                model, [sender])
        elif args.operator == "rank10":
            install_op = lambda: install_rank_k_substitutions(
                model, [sender], k_rank=args.rank_k)

        sub_acts = run_with(install_op)
        sims_op = per_head_cossim(baseline_acts, sub_acts)
        recv_sims_op = {f"{l},{h}": s for (l, h), s in sims_op.items()
                         if l > le}
        own_sim_op = sims_op.get(sender, None)

        sub_acts = run_with(lambda: install_mean_ablations(model, [sender]))
        sims_abl = per_head_cossim(baseline_acts, sub_acts)
        recv_sims_abl = {f"{l},{h}": s for (l, h), s in sims_abl.items()
                          if l > le}
        own_sim_abl = sims_abl.get(sender, None)

        def summary(d):
            if not d: return None
            v = np.asarray(list(d.values()), dtype=np.float64)
            return {"median": float(np.median(v)),
                    "mean":   float(v.mean()),
                    "min":    float(v.min()),
                    "p10":    float(np.percentile(v, 10)),
                    "p90":    float(np.percentile(v, 90)),
                    "n":      int(v.size)}

        out["per_sender"][f"{le},{he}"] = {
            "sender_layer": le, "sender_head": he,
            "operator": args.operator,
            "own_sim_op": own_sim_op,
            "own_sim_ablate": own_sim_abl,
            "recv_op": recv_sims_op,
            "recv_ablate": recv_sims_abl,
            "op_summary": summary(recv_sims_op),
            "ablate_summary": summary(recv_sims_abl),
        }
        op_s = out["per_sender"][f"{le},{he}"]["op_summary"]
        ab_s = out["per_sender"][f"{le},{he}"]["ablate_summary"]
        if op_s and ab_s:
            print(f"  receivers (n={op_s['n']}): "
                  f"op-med={op_s['median']:.4f}, abl-med={ab_s['median']:.4f}, "
                  f"gap={op_s['median']-ab_s['median']:.4f}",
                  flush=True)
    cap.remove()

    # Aggregate
    op_meds = [s["op_summary"]["median"] for s in out["per_sender"].values()
                if s["op_summary"]]
    ab_meds = [s["ablate_summary"]["median"] for s in out["per_sender"].values()
                if s["ablate_summary"]]
    out["aggregate"] = {
        "n_senders_with_receivers": len(op_meds),
        "op_median_of_medians":     float(np.median(op_meds)) if op_meds else None,
        "ablate_median_of_medians": float(np.median(ab_meds)) if ab_meds else None,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)
    print("aggregate")
    for k, v in out["aggregate"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
