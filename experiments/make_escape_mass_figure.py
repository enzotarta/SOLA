"""Escape-mass heatmaps per (layer, head), side by side across models."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_matrix(model_result: dict, K: int) -> np.ndarray:
    """Return (n_layers, n_heads) matrix of escape mass at the given K."""
    escape = model_result["escape"]
    n_layers = model_result["n_layers"]
    K_str = str(K)
    rows = []
    for li in range(n_layers):
        rows.append(escape[str(li)][K_str])
    return np.asarray(rows)


PRETTY_NAMES = {
    "deit_tiny_patch16_224": "DeiT-Tiny",
    "vit_base_patch16_224": "ViT-B/16 (supervised)",
    "vit_base_patch16_clip_224.openai": "CLIP ViT-B/16",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="escape_comparison.json")
    ap.add_argument("--out", default="figures/escape_mass_grid.pdf")
    ap.add_argument("--kernel", type=int, default=3,
                    help="Window size K to plot (must be in the swept set).")
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text())
    model_names = data["models"]
    results = data["results"]

    matrices = [load_matrix(results[m], args.kernel) for m in model_names]

    vmin = 0.0
    vmax = max(M.max() for M in matrices)

    n_panels = len(matrices)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(3.5 * n_panels + 0.6, 4.0),
        gridspec_kw={"width_ratios": [m.shape[1] for m in matrices]})
    if n_panels == 1:
        axes = [axes]

    im = None
    for ax, name, M in zip(axes, model_names, matrices):
        im = ax.imshow(M, aspect="auto", vmin=vmin, vmax=vmax,
                       cmap="viridis_r", interpolation="nearest")
        ax.set_title(PRETTY_NAMES.get(name, name), fontsize=10)
        ax.set_xlabel("head index")
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(np.arange(M.shape[1]), fontsize=7)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(np.arange(M.shape[0]), fontsize=7)

    axes[0].set_ylabel("layer index")

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(f"escape mass at $K={args.kernel}$  (lower = more local)",
                   fontsize=9)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"Wrote {out_path}")

    # Also report per-model lowest-escape-mass heads at the chosen K
    print("\nLowest-escape-mass heads per model:")
    for name, M in zip(model_names, matrices):
        flat = [(M[li, hi], li, hi)
                for li in range(M.shape[0]) for hi in range(M.shape[1])]
        flat.sort(key=lambda t: t[0])
        print(f"  {PRETTY_NAMES.get(name, name)}:")
        for em, li, hi in flat[:8]:
            print(f"    L{li}H{hi:<2}  escape={em:.3f}")


if __name__ == "__main__":
    main()
