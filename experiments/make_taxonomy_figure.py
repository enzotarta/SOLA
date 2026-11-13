"""Stacked bars of head counts per category, one row per backbone."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
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
    "delta_shift":    "#d62728",   # red
    "soft_shift":     "#ff9896",   # pink
    "multi_delta":    "#9467bd",   # purple
    "gaussian_local": "#8c564b",   # brown
    "self_only":      "#e377c2",   # magenta
    "diffuse_local":  "#1f77b4",   # blue
    "fixed_target":   "#2ca02c",   # green
    "near_global":    "#7f7f7f",   # grey
    "other":          "#bcbd22",   # olive
}

PRETTY_MODEL = {
    "deit_tiny_patch16_224":                       "DeiT-Tiny (IN-1k)",
    "deit_small_patch16_224":                      "DeiT-Small (IN-1k)",
    "vit_small_patch16_224.augreg_in21k_ft_in1k":  "ViT-S/16 (IN-21k FT)",
    "vit_base_patch16_224.augreg_in1k":            "ViT-B/16 (IN-1k)",
    "vit_base_patch16_224":                        "ViT-B/16 (IN-21k FT)",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k":  "ViT-B/16 (IN-21k FT)",
    "vit_base_patch16_224.mae":                    "ViT-B/16 (MAE)",
    "vit_base_patch16_224.dino":                   "ViT-B/16 (DINO)",
    "vit_base_patch14_dinov2.lvd142m":             "ViT-B/14 (DINOv2)",
    "vit_base_patch16_clip_224.openai":            "CLIP ViT-B/16",
    "vit_base_patch16_siglip_224":                 "SigLIP ViT-B/16",
    "vit_base_patch16_siglip_224.webli":           "SigLIP ViT-B/16",
    "vit_base_patch16_siglip_224.v2_webli":        "SigLIP ViT-B/16",
    "blip_crossattn_base":                         "BLIP cross-attn",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="taxonomy JSONs from operator_taxonomy.py")
    ap.add_argument("--out", default="figures/taxonomy.pdf")
    args = ap.parse_args()

    model_counts = []
    for inp in args.inputs:
        d = json.loads(Path(inp).read_text())
        cats = [r["category"] for r in d["results"]]
        c = Counter(cats)
        total = sum(c.values())
        model_counts.append((d["model"], c, total))

    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(model_counts) + 1.5))
    n_models = len(model_counts)
    y_pos = np.arange(n_models)

    left = np.zeros(n_models)
    for cat in CATEGORY_ORDER:
        widths = []
        for _, c, total in model_counts:
            widths.append(c.get(cat, 0) / total * 100)
        widths = np.array(widths)
        if widths.sum() == 0:
            continue
        bars = ax.barh(y_pos, widths, left=left,
                       color=COLORS[cat], edgecolor="white",
                       linewidth=0.5, label=PRETTY[cat])
        # Label segments large enough to fit text
        for yi, (w, l) in enumerate(zip(widths, left)):
            if w >= 4:
                _, c, total = model_counts[yi]
                count = c.get(cat, 0)
                ax.text(l + w / 2, yi, f"{count}", ha="center", va="center",
                        color="white" if cat in {"near_global", "fixed_target",
                                                   "diffuse_local"} else "black",
                        fontsize=8, fontweight="bold")
        left += widths

    ax.set_yticks(y_pos)
    labels = [f"{PRETTY_MODEL.get(name, name)}\n({total} heads)"
              for name, _, total in model_counts]
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("fraction of heads (%)")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
               ncol=5, fontsize=8, frameon=False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
