"""Per-(layer, head) map of categories and of the heads that pass a substitution test."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np


CATEGORY_ORDER = (
    "delta_shift",
    "soft_shift",
    "multi_delta",
    "gaussian_local",
    "self_only",
    "diffuse_local",
    "fixed_target",
    "near_global",
    "other",
)

PRETTY = {
    "delta_shift":    "delta-shift",
    "soft_shift":     "soft-shift",
    "multi_delta":    "multi-delta",
    "gaussian_local": "Gaussian-local",
    "self_only":      "self-only",
    "diffuse_local":  "diffuse-local",
    "fixed_target":   "fixed-target",
    "near_global":    "near-global",
    "other":          "other",
}

COLORS = {
    "delta_shift":    "#d62728",
    "soft_shift":     "#ff9896",
    "multi_delta":    "#9467bd",
    "gaussian_local": "#8c564b",
    "self_only":      "#e377c2",
    "diffuse_local":  "#1f77b4",
    "fixed_target":   "#2ca02c",
    "near_global":    "#7f7f7f",
    "other":          "#bcbd22",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--diag", required=True,
                    help="fixed_target_*.json with r_abs_mean per head")
    ap.add_argument("--r-thresh", type=float, default=0.4)
    ap.add_argument("--shift-cats", default="delta_shift,soft_shift")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tax = json.loads(Path(args.taxonomy).read_text())
    rows = tax["results"]
    diag = json.loads(Path(args.diag).read_text())
    diag_by_lh = {(r["layer"], r["head"]): r["r_abs_mean"]
                  for r in diag["per_head"]}

    n_layers = max(r["layer"] for r in rows) + 1
    n_heads = max(r["head"] for r in rows) + 1

    cat_grid = np.full((n_heads, n_layers), "?", dtype=object)
    for r in rows:
        cat_grid[r["head"], r["layer"]] = r["category"]

    shift_cats = set(args.shift_cats.split(","))
    shifts = {(r["layer"], r["head"]) for r in rows
              if r["category"] in shift_cats}
    ft_low_r = {(r["layer"], r["head"]) for r in rows
                if r["category"] == "fixed_target"
                and diag_by_lh.get((r["layer"], r["head"]), 1.0) < args.r_thresh}
    dl_low_r = {(r["layer"], r["head"]) for r in rows
                if r["category"] == "diffuse_local"
                and diag_by_lh.get((r["layer"], r["head"]), 1.0) < args.r_thresh}
    ng_low_r = {(r["layer"], r["head"]) for r in rows
                if r["category"] == "near_global"
                and diag_by_lh.get((r["layer"], r["head"]), 1.0) < args.r_thresh}

    fig, ax = plt.subplots(figsize=(0.55 * n_layers + 2, 0.42 * n_heads + 1.5))
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(-0.5, n_heads - 0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    # Cell background by category
    for L in range(n_layers):
        for h in range(n_heads):
            cat = cat_grid[h, L]
            color = COLORS.get(cat, "#ffffff")
            ax.add_patch(mpatches.Rectangle(
                (L - 0.5, h - 0.5), 1.0, 1.0,
                facecolor=color, edgecolor="white", linewidth=0.6))

    # Substitution annotations
    sz = 0.18                                 # marker half-side as cell fraction
    for (L, h) in shifts:
        # upper-right square
        ax.add_patch(mpatches.Rectangle(
            (L + 0.5 - 2*sz, h - 0.5), 2*sz, 2*sz,
            facecolor="black", edgecolor="white", linewidth=0.4))
    for (L, h) in ft_low_r:
        # lower-right triangle
        x0, y0 = L + 0.5 - 2*sz, h + 0.5
        ax.add_patch(mpatches.Polygon(
            [(x0, y0), (x0 + 2*sz, y0), (x0 + 2*sz, y0 - 2*sz)],
            facecolor="black", edgecolor="white", linewidth=0.4))
    for (L, h) in ng_low_r:
        # lower-left circle
        ax.add_patch(mpatches.Circle(
            (L - 0.5 + sz, h + 0.5 - sz), sz,
            facecolor="black", edgecolor="white", linewidth=0.4))
    for (L, h) in dl_low_r:
        # upper-left diamond
        cx, cy = L - 0.5 + sz, h - 0.5 + sz
        ax.add_patch(mpatches.Polygon(
            [(cx - sz, cy), (cx, cy - sz),
             (cx + sz, cy), (cx, cy + sz)],
            facecolor="black", edgecolor="white", linewidth=0.4))

    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)], fontsize=8)
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f"h{i}" for i in range(n_heads)], fontsize=8)
    ax.tick_params(length=0)
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)

    if args.title:
        ax.set_title(args.title, fontsize=11)

    # Legend: categories
    cat_handles = [mpatches.Patch(color=COLORS[c], label=PRETTY[c])
                   for c in CATEGORY_ORDER if (cat_grid == c).any()]
    # Legend: substitution markers (match the glyphs drawn in cells)
    sub_handles = [
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor="black", markeredgecolor="white",
               markersize=9, markeredgewidth=0.6,
               label="shift substituted (UR)"),
        Line2D([0], [0], marker="v", color="none",
               markerfacecolor="black", markeredgecolor="white",
               markersize=10, markeredgewidth=0.6,
               label=r"broadcast on low-$r_{\mathrm{abs}}$ FT (BR)"),
        Line2D([0], [0], marker="D", color="none",
               markerfacecolor="black", markeredgecolor="white",
               markersize=8, markeredgewidth=0.6,
               label=r"broadcast on low-$r_{\mathrm{abs}}$ DL (UL)"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="black", markeredgecolor="white",
               markersize=9, markeredgewidth=0.6,
               label=r"rank-10 on low-$r_{\mathrm{abs}}$ NG (BL)"),
    ]
    leg1 = ax.legend(handles=cat_handles,
                     loc="upper left", bbox_to_anchor=(1.02, 1.0),
                     fontsize=8, frameon=False, title="category",
                     title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=sub_handles,
              loc="upper left", bbox_to_anchor=(1.02, 0.50),
              fontsize=8, frameon=False, title="substituted",
              title_fontsize=9)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}  ({n_layers} layers x {n_heads} heads, "
          f"{len(shifts)} shifts, {len(ft_low_r)} low-r FT, "
          f"{len(dl_low_r)} low-r DL, {len(ng_low_r)} low-r NG)")


if __name__ == "__main__":
    main()
