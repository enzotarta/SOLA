"""Render representative attention matrices used in the overview figure."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from escape_mass_sweep import AttentionRecorder
from shift_substitution import build_imagenet_val_eval_loader


def save_matrix(A, out_stem, cmap, vmax_pct=99,
                  facecolor=None, edgecolor=None):
    """Render a 2D matrix as a square heatmap (no axes,
    no margins) at print-quality. Saves both PNG and PDF."""
    vmax = float(np.percentile(A, vmax_pct))
    fig = plt.figure(figsize=(3.0, 3.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(A, cmap=cmap, aspect="equal", vmin=0, vmax=vmax,
                interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if facecolor:
        fig.patch.set_facecolor(facecolor)
    for sp in ax.spines.values():
        sp.set_visible(edgecolor is not None)
        if edgecolor is not None:
            sp.set_linewidth(1.0); sp.set_edgecolor(edgecolor)
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight",
                  pad_inches=0)
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_shift_kernel(out_stem, cmap="Reds"):
    K = 5
    kernel = np.zeros((K, K))
    kernel[1, 3] = 1.0
    fig = plt.figure(figsize=(3.0, 3.0), dpi=300)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.imshow(kernel, cmap=cmap, vmin=0, vmax=1.2,
                extent=[-K/2, K/2, K/2, -K/2])
    ax.plot(0, 0, "x", color="black", markersize=20, markeredgewidth=3)
    ax.annotate("", xy=(1.5, -1.5), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color="black",
                                    lw=2.5, mutation_scale=20))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-K/2 - 0.2, K/2 + 0.2)
    ax.set_ylim(K/2 + 0.2, -K/2 - 0.2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight",
                  pad_inches=0)
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_broadcast_row(out_stem, cmap="Greens"):
    N = 30
    row = np.zeros(N)
    row[7] = 1.0; row[8] = 0.4; row[6] = 0.2
    A = np.tile(row, (N, 1))
    save_matrix(A, out_stem, cmap=cmap, vmax_pct=100)


def save_rankk_factors(out_stem):
    fig = plt.figure(figsize=(4.5, 2.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.5, 12); ax.set_ylim(-0.5, 6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    grey = "#7f8c8d"
    yellow = "#bcbd22"
    accent = "#5d6d7e"
    ax.add_patch(Rectangle((0.4, 0.5), 1.4, 4.5,
                              facecolor=grey, alpha=0.65,
                              edgecolor=accent, linewidth=1.2))
    ax.text(1.1, 0.0, r"$U_k$", ha="center", va="top",
              fontsize=14, color=accent)
    ax.add_patch(Rectangle((2.4, 1.8), 1.4, 1.4,
                              facecolor=yellow, alpha=0.65,
                              edgecolor=accent, linewidth=1.2))
    for i in range(3):
        ax.add_patch(Rectangle((2.6 + 0.35 * i, 2.85 - 0.35 * i),
                                  0.30, 0.30,
                                  facecolor="#7d7d12", edgecolor="none"))
    ax.text(3.1, 1.55, r"$\Sigma_k$", ha="center", va="top",
              fontsize=14, color=accent)
    ax.add_patch(Rectangle((4.4, 1.8), 4.4, 1.4,
                              facecolor=grey, alpha=0.65,
                              edgecolor=accent, linewidth=1.2))
    ax.text(6.6, 1.55, r"$V_k^{\top}$", ha="center", va="top",
              fontsize=14, color=accent)
    ax.text(2.1, 2.5, r"$\cdot$", ha="center", va="center", fontsize=20)
    ax.text(4.1, 2.5, r"$\cdot$", ha="center", va="center", fontsize=20)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight",
                  pad_inches=0)
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


@torch.no_grad()
def capture_attn(model, x, layers):
    rec = AttentionRecorder(model)
    _ = model(x)
    rec.remove()
    return {li: rec.stacked()[li] for li in layers if li in rec.stacked()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="vit_base_patch16_224.augreg2_in21k_ft_in1k")
    ap.add_argument("--imagenet-val-dir",
                    default="data/imagenet/val")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--shift-head", type=int, nargs=2, default=(1, 10))
    ap.add_argument("--ft-head",    type=int, nargs=2, default=(6, 10))
    ap.add_argument("--ng-head",    type=int, nargs=2, default=(1, 5))
    ap.add_argument("--out-dir",
                    default="figures/teaser_assets")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import timm
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device} ...", flush=True)
    model = timm.create_model(args.model, pretrained=True).eval().to(device)

    loader, _ = build_imagenet_val_eval_loader(
        args.batch_size, image_size=224,
        val_dir=args.imagenet_val_dir, num_workers=2)
    x = next(iter(loader))[0].to(device)

    layers_needed = sorted({args.shift_head[0], args.ft_head[0],
                              args.ng_head[0]})
    attn = capture_attn(model, x, layers_needed)

    def avg(li, hi):
        A = attn[li][:, hi].float().cpu().numpy()
        return A.mean(axis=0)

    A_shift = avg(*args.shift_head)
    A_ft    = avg(*args.ft_head)
    A_ng    = avg(*args.ng_head)
    print(f"Captured. Shapes {A_shift.shape}.", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Real attention matrices (N x N, averaged over batch)
    save_matrix(A_shift, str(out_dir / "shift_attn"), cmap="Reds")
    save_matrix(A_ft,    str(out_dir / "broadcast_attn"), cmap="Greens")
    save_matrix(A_ng,    str(out_dir / "rankk_attn"),  cmap="Greys")

    # Schematic substitutions
    save_shift_kernel(str(out_dir / "shift_kernel"))
    save_broadcast_row(str(out_dir / "broadcast_row"))
    save_rankk_factors(str(out_dir / "rankk_factors"))

    # Save raw attention arrays for any further use
    np.savez(out_dir / "attention_matrices.npz",
              shift=A_shift, broadcast=A_ft, rankk=A_ng)

    print(f"Wrote {len(list(out_dir.glob('*')))} files to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
