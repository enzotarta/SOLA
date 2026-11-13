"""Path patching for shift senders: substitute or mean-ablate one sender and compare
every downstream head's attention output with baseline.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))


# Per-head output capturer

class HeadOutputCapturer:
    """Forward pre-hook on every `attn.proj` to capture the concatenated
    per-head SDPA output (shape (B, N, n_heads*head_dim)) and split it
    into (layer, head) -> (B, N, head_dim) tensors on CPU.

    Works for both the unmodified Attention.forward and for the
    custom forward installed by install_kernel_substitutions.
    """
    def __init__(self, model):
        self.handles = []
        self.acts: Dict[Tuple[int, int], torch.Tensor] = {}
        for li, blk in enumerate(model.blocks):
            attn = blk.attn
            n_heads = attn.num_heads
            head_dim = attn.head_dim

            def pre_hook(module, inputs, layer_idx=li, nh=n_heads, hd=head_dim):
                x = inputs[0]                                # (B, N, nh*hd)
                B, N, C = x.shape
                # split per head: (nh, B, N, hd)
                xh = x.detach().reshape(B, N, nh, hd).permute(2, 0, 1, 3)
                for h in range(nh):
                    self.acts[(layer_idx, h)] = xh[h].cpu()
            self.handles.append(attn.proj.register_forward_pre_hook(pre_hook))

    def reset(self):
        self.acts = {}

    def remove(self):
        for h in self.handles:
            h.remove()


# Mean ablation: replace one head's SDPA output with its batch-mean

def make_mean_ablated_attention_forward(attn, ablated_heads):
    """ablated_heads: set of head indices in this layer to mean-ablate."""

    def new_forward(x, *args, **kwargs):
        B, N, C = x.shape
        Dh = C // attn.num_heads
        qkv = attn.qkv(x).reshape(
            B, N, 3, attn.num_heads, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            q = attn.q_norm(q); k = attn.k_norm(k)
        a = (q @ k.transpose(-2, -1)) * attn.scale
        a = a.softmax(dim=-1)
        attn_out = a @ v                                # (B, heads, N, Dh)

        for h in ablated_heads:
            # batch-mean across images, broadcast back
            mean_h = attn_out[:, h].mean(dim=0, keepdim=True)         # (1, N, Dh)
            attn_out[:, h] = mean_h.expand_as(attn_out[:, h]).clone()

        out = attn_out.transpose(1, 2).reshape(B, N, C)
        out = attn.proj(out)
        if hasattr(attn, "proj_drop") and attn.proj_drop is not None:
            out = attn.proj_drop(out)
        return out

    return new_forward


def install_mean_ablations(model, heads):
    """heads: list of (layer, head). Returns a saved dict for restore_forwards."""
    by_layer: Dict[int, set] = {}
    for (li, hi) in heads:
        by_layer.setdefault(li, set()).add(hi)
    saved = {}
    for li, hs in by_layer.items():
        attn = model.blocks[li].attn
        saved[li] = attn.forward
        attn.forward = make_mean_ablated_attention_forward(attn, hs)
    return saved


def restore_forwards(model, saved):
    for li, fwd in saved.items():
        model.blocks[li].attn.forward = fwd


# Cos-sim aggregation

def per_head_cossim(base_acts, sub_acts):
    """For every (l, h) shared by both dicts: mean cos-sim across (B, N)."""
    out = {}
    for k in base_acts:
        if k not in sub_acts: continue
        a, b = base_acts[k].float(), sub_acts[k].float()
        sim = F.cosine_similarity(a, b, dim=-1).mean().item()
        out[k] = float(sim)
    return out


# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-batches", type=int, default=4)
    ap.add_argument("--imagenet-val-dir", default="data/imagenet/val")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--shift-cats", default="delta_shift,soft_shift")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-senders", type=int, default=None,
                    help="cap number of shift senders (for quick tests)")
    args = ap.parse_args()

    import timm
    from shift_substitution import (
        build_imagenet_val_eval_loader, detect_model_config,
    )
    from kernel_substitution import (
        kernel_from_diagnostics_mean,
        install_kernel_substitutions,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device} ...")
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
    print(f"  calibration batch: {x.shape}")

    tax = json.loads(Path(args.taxonomy).read_text())
    rows = tax["results"]
    res_by_lh = {(r["layer"], r["head"]): r for r in rows}
    shift_cats = set(args.shift_cats.split(","))
    shift_heads: List[Tuple[int, int]] = sorted(
        [(r["layer"], r["head"]) for r in rows
         if r["category"] in shift_cats],
        key=lambda t: (t[0], t[1]))
    if args.max_senders:
        shift_heads = shift_heads[:args.max_senders]
    n_blocks = max(r["layer"] for r in rows) + 1
    n_heads = max(r["head"] for r in rows) + 1
    print(f"  shift senders: {len(shift_heads)} of {n_blocks}x{n_heads} = {n_blocks*n_heads}")
    print(f"  by layer: " + ", ".join(
        f"L{li}={sum(1 for s in shift_heads if s[0]==li)}"
        for li in range(n_blocks) if any(s[0]==li for s in shift_heads)))

    cap = HeadOutputCapturer(model)

    # baseline
    cap.reset()
    with torch.no_grad():
        _ = model(x)
    baseline_acts = dict(cap.acts)
    print(f"  baseline heads captured: {len(baseline_acts)}")

    # Precompute per-sender kernel
    kernel_full = {lh: kernel_from_diagnostics_mean(
                       np.asarray(res_by_lh[lh]["mean_kernel"]))
                   for lh in shift_heads}

    out = {
        "model": args.model,
        "img_size": args.img_size,
        "batch_size": int(x.shape[0]),
        "n_blocks": n_blocks,
        "n_heads_per_block": n_heads,
        "shift_senders": [list(h) for h in shift_heads],
        # per-sender results: {sender_str: {"shift": {"(l,h)": sim, ...},
        #                                   "ablate": {"(l,h)": sim, ...}}}
        "per_sender": {},
    }

    def run_sub(install_fn):
        cap.reset()
        saved = install_fn()
        with torch.no_grad():
            _ = model(x)
        sub_acts = dict(cap.acts)
        restore_forwards(model, saved)
        return sub_acts

    for si, sender in enumerate(shift_heads):
        le, he = sender
        print(f"\n[{si+1}/{len(shift_heads)}] sender L{le} h{he}")

        # Faithful substitution: own kernel only on this head
        kmap = {sender: kernel_full[sender]}
        sub_acts = run_sub(lambda: install_kernel_substitutions(
            model, kmap, cfg["grid"], cfg["n_extra_tokens"]))
        sims_shift = per_head_cossim(baseline_acts, sub_acts)
        # Receivers = downstream heads (l_l > l_e)
        recv_sims_shift = {f"{l},{h}": s for (l, h), s in sims_shift.items()
                            if l > le}
        # Sender's own perturbation (l == le, h == he): how much its own output
        # changed under substitution.
        own_sim_shift = sims_shift.get(sender, None)

        # Mean ablation
        sub_acts = run_sub(lambda: install_mean_ablations(model, [sender]))
        sims_abl = per_head_cossim(baseline_acts, sub_acts)
        recv_sims_abl = {f"{l},{h}": s for (l, h), s in sims_abl.items()
                          if l > le}
        own_sim_abl = sims_abl.get(sender, None)

        # Summary stats
        def summary(d):
            if not d: return None
            v = np.asarray(list(d.values()), dtype=np.float64)
            return {
                "median": float(np.median(v)),
                "mean":   float(v.mean()),
                "min":    float(v.min()),
                "p10":    float(np.percentile(v, 10)),
                "p90":    float(np.percentile(v, 90)),
                "n":      int(v.size),
            }

        out["per_sender"][f"{le},{he}"] = {
            "sender_layer": le,
            "sender_head": he,
            "own_sim_shift": own_sim_shift,
            "own_sim_ablate": own_sim_abl,
            "recv_shift": recv_sims_shift,
            "recv_ablate": recv_sims_abl,
            "shift_summary": summary(recv_sims_shift),
            "ablate_summary": summary(recv_sims_abl),
        }

        ss = out["per_sender"][f"{le},{he}"]["shift_summary"]
        sa = out["per_sender"][f"{le},{he}"]["ablate_summary"]
        if ss and sa:
            print(f"  receivers (n={ss['n']}):")
            print(f"    shift:    median={ss['median']:.4f}  min={ss['min']:.4f}  p10={ss['p10']:.4f}")
            print(f"    ablate:   median={sa['median']:.4f}  min={sa['min']:.4f}  p10={sa['p10']:.4f}")
            print(f"    gap (shift - ablate) median: {ss['median']-sa['median']:.4f}")
        else:
            print("  no downstream receivers (sender at last layer)")

    cap.remove()

    # Aggregate across senders
    shift_meds = [s["shift_summary"]["median"] for s in out["per_sender"].values()
                   if s["shift_summary"]]
    abl_meds   = [s["ablate_summary"]["median"] for s in out["per_sender"].values()
                   if s["ablate_summary"]]
    out["aggregate"] = {
        "n_senders_with_receivers": len(shift_meds),
        "shift_median_of_medians":  float(np.median(shift_meds)) if shift_meds else None,
        "ablate_median_of_medians": float(np.median(abl_meds)) if abl_meds else None,
        "shift_min_of_medians":     float(np.min(shift_meds)) if shift_meds else None,
        "ablate_min_of_medians":    float(np.min(abl_meds)) if abl_meds else None,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    print("\naggregate")
    for k, v in out["aggregate"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
