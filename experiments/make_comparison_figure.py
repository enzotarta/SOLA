"""Operator inventory, shift residual and near-global rank over training, for the
supervised and the MAE runs.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CATEGORY_ORDER = (
    "delta_shift", "soft_shift", "multi_delta",
    "gaussian_local", "self_only", "diffuse_local",
    "fixed_target", "near_global",
)
PRETTY = {
    "delta_shift":    "delta-shift",   "soft_shift":     "soft-shift",
    "multi_delta":    "multi-delta",   "gaussian_local": "Gaussian-local",
    "self_only":      "self-only",     "diffuse_local":  "diffuse-local",
    "fixed_target":   "fixed-target",  "near_global":    "near-global",
}
COLORS = {
    "delta_shift":    "#d62728", "soft_shift":     "#ff9896",
    "multi_delta":    "#9467bd", "gaussian_local": "#8c564b",
    "self_only":      "#e377c2", "diffuse_local":  "#1f77b4",
    "fixed_target":   "#2ca02c", "near_global":    "#7f7f7f",
}

ER90_THRESHOLD = 40   # eff_rank_90 boundary for redistributing 'other'


def _redistribute_other(rows):
    """In place: every head with category 'other' is reassigned to
    'near_global' (if eff_rank_90 <= ER90_THRESHOLD) or 'diffuse_local'
    (otherwise). Category counts are recomputed."""
    for r in rows:
        cc_to_ng = cc_to_dl = 0
        for h in r["per_head"]:
            if h["category"] != "other":
                continue
            er = h.get("diag", {}).get("eff_rank_90")
            if er is None:
                continue
            if er <= ER90_THRESHOLD:
                h["category"] = "near_global"; cc_to_ng += 1
            else:
                h["category"] = "diffuse_local"; cc_to_dl += 1
        cc = r.setdefault("category_counts", {})
        if "other" in cc:
            cc["near_global"] = cc.get("near_global", 0) + cc_to_ng
            cc["diffuse_local"] = cc.get("diffuse_local", 0) + cc_to_dl
            cc.pop("other", None)
    return rows


def load_all(analysis_dir: Path):
    rows = [json.loads(p.read_text()) for p in sorted(analysis_dir.glob("*.json"))]
    rows.sort(key=lambda d: d.get("epoch", -1))
    return _redistribute_other(rows)


def median_diag(rows, category, key):
    out = []
    for r in rows:
        vals = [h["diag"][key] for h in r["per_head"]
                if h["category"] == category and key in h["diag"]
                and not (isinstance(h["diag"][key], float)
                         and (np.isnan(h["diag"][key]) or np.isinf(h["diag"][key])))]
        out.append(float(np.median(vals)) if vals else np.nan)
    return np.array(out)


def stack_counts(rows, cats):
    counts = np.zeros((len(cats), len(rows)), dtype=float)
    for j, r in enumerate(rows):
        cc = r.get("category_counts", {})
        for i, c in enumerate(cats):
            counts[i, j] = cc.get(c, 0)
    return counts


def plot_inventory(ax, rows, cats, title, n_heads_total):
    epochs = np.array([r.get("epoch", i) for i, r in enumerate(rows)])
    counts = stack_counts(rows, cats)
    ax.stackplot(
        epochs, counts,
        labels=[PRETTY[c] for c in cats],
        colors=[COLORS[c] for c in cats],
        alpha=0.85,
    )
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel("# heads", fontsize=9)
    ax.set_ylim(0, n_heads_total)
    ax.set_title(title, fontsize=10, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sup-dir", required=True)
    ap.add_argument("--mae-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sup_rows = load_all(Path(args.sup_dir))
    mae_rows = load_all(Path(args.mae_dir))
    if not sup_rows or not mae_rows:
        raise SystemExit("missing analyses")

    sup_n_heads = sum(sup_rows[0].get("category_counts", {}).values())
    mae_n_heads = sum(mae_rows[0].get("category_counts", {}).values())
    print(f"sup: {len(sup_rows)} ckpts, {sup_n_heads} heads/model")
    print(f"mae: {len(mae_rows)} ckpts, {mae_n_heads} heads/model")

    # Categories that ever appear in either trajectory
    cats_present = [
        c for c in CATEGORY_ORDER
        if any(r.get("category_counts", {}).get(c, 0) > 0
               for r in sup_rows + mae_rows)
    ]

    fig = plt.figure(figsize=(8.5, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.32,
                          width_ratios=[1, 1], height_ratios=[1, 1])

    # (a) supervised inventory
    ax_a = fig.add_subplot(gs[0, 0])
    plot_inventory(ax_a, sup_rows, cats_present,
                   "(a) supervised DeiT-Tiny: operator inventory",
                   sup_n_heads)

    # (b) MAE inventory
    ax_b = fig.add_subplot(gs[0, 1])
    plot_inventory(ax_b, mae_rows, cats_present,
                   "(b) MAE ViT-Tiny: operator inventory",
                   mae_n_heads)
    # Single shared legend on (b)
    ax_b.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                fontsize=7, frameon=False, title="category",
                title_fontsize=8)

    # (c) shift TE residual
    ax_c = fig.add_subplot(gs[1, 0])
    sup_eps = np.array([r.get("epoch", i) for i, r in enumerate(sup_rows)])
    mae_eps = np.array([r.get("epoch", i) for i, r in enumerate(mae_rows)])
    sup_r = median_diag(sup_rows, "delta_shift", "kernel_l1_residual_mean")
    mae_r = median_diag(mae_rows, "delta_shift", "kernel_l1_residual_mean")
    sup_has = np.isfinite(sup_r).any()
    if sup_has:
        ax_c.plot(sup_eps, sup_r, "o-", color="#1f77b4", markersize=3,
                  linewidth=1.2, label="supervised")
    ax_c.plot(mae_eps, mae_r, "s--", color="#d62728", markersize=3,
              linewidth=1.2, label="MAE")
    if not sup_has:
        ax_c.text(0.97, 0.92,
                  "supervised: 0 delta-shift heads at 100 ep",
                  transform=ax_c.transAxes, ha="right", va="top",
                  fontsize=7, color="#1f77b4", style="italic")
    ax_c.set_xlabel("epoch", fontsize=9)
    ax_c.set_ylabel(r"$r$ (delta-shift TE residual, median)", fontsize=9)
    ax_c.set_title(r"(c) shift sharpening: $r \to 0$",
                   fontsize=10, loc="left")
    ax_c.grid(alpha=0.3, linewidth=0.4)
    ax_c.legend(fontsize=8, frameon=False)
    for sp in ("top", "right"):
        ax_c.spines[sp].set_visible(False)

    # (d) eff_rank_90 evolution: near-global AND diffuse-local
    ax_d = fig.add_subplot(gs[1, 1])
    sup_ng = median_diag(sup_rows, "near_global", "eff_rank_90")
    mae_ng = median_diag(mae_rows, "near_global", "eff_rank_90")
    sup_dl = median_diag(sup_rows, "diffuse_local", "eff_rank_90")
    mae_dl = median_diag(mae_rows, "diffuse_local", "eff_rank_90")
    ax_d.plot(sup_eps, sup_ng, "o-", color="#1f77b4", markersize=3,
              linewidth=1.2, label="sup: near-global")
    ax_d.plot(mae_eps, mae_ng, "s--", color="#d62728", markersize=3,
              linewidth=1.2, label="MAE: near-global")
    ax_d.plot(sup_eps, sup_dl, "o:", color="#1f77b4", markersize=3,
              linewidth=1.2, alpha=0.55, label="sup: diffuse-local")
    ax_d.plot(mae_eps, mae_dl, "s:", color="#d62728", markersize=3,
              linewidth=1.2, alpha=0.55, label="MAE: diffuse-local")
    ax_d.axhline(10, color="grey", linewidth=0.6, linestyle=":", alpha=0.5)
    ax_d.text(2, 11.5, r"fixed $k{=}10$ (Prop.\,3)", fontsize=6.5,
              color="grey", alpha=0.85)
    ax_d.set_xlabel("epoch", fontsize=9)
    ax_d.set_ylabel(r"eff\_rank$_{90}$ (median)", fontsize=9)
    ax_d.set_title("(d) rank evolution: near-global vs. diffuse-local",
                   fontsize=10, loc="left")
    ax_d.grid(alpha=0.3, linewidth=0.4)
    ax_d.legend(fontsize=6.5, frameon=False, loc="upper left", ncol=2)
    for sp in ("top", "right"):
        ax_d.spines[sp].set_visible(False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")

    # Print final-epoch comparison table
    def at_last(rows, cats):
        cc = rows[-1].get("category_counts", {})
        return {c: cc.get(c, 0) for c in cats}
    print("\nFinal-epoch operator inventory (epoch 99):")
    print(f"  {'category':<14s}  sup  MAE")
    for c in cats_present:
        a = at_last(sup_rows, [c])[c]
        b = at_last(mae_rows, [c])[c]
        print(f"  {PRETTY[c]:<14s}  {a:>3d}  {b:>3d}")


if __name__ == "__main__":
    main()
