"""Per-sender distributions of downstream cosine similarity."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_data(path):
    d = json.loads(Path(path).read_text())
    senders = list(d["per_sender"].keys())
    rows = []
    for s in senders:
        block = d["per_sender"][s]
        shift_vals = list(block["recv_shift"].values())
        abl_vals   = list(block["recv_ablate"].values())
        rows.append({
            "sender": s,
            "label": f"L{block['sender_layer']}.h{block['sender_head']}",
            "shift": np.asarray(shift_vals),
            "ablate": np.asarray(abl_vals),
        })
    return d["model"], rows


def plot_panel(ax, rows, title):
    n = len(rows)
    width = 0.36
    xs = np.arange(n)
    labels = [r["label"] for r in rows]

    # box plots (no fliers; we'll overlay strip)
    bp_s = ax.boxplot([r["shift"] for r in rows],
                       positions=xs - width/2, widths=width*0.85,
                       patch_artist=True, showfliers=False, manage_ticks=False,
                       boxprops=dict(facecolor="#1f77b4", alpha=0.5,
                                     edgecolor="#1f77b4"),
                       medianprops=dict(color="black", linewidth=1.2),
                       whiskerprops=dict(color="#1f77b4"),
                       capprops=dict(color="#1f77b4"))
    bp_a = ax.boxplot([r["ablate"] for r in rows],
                       positions=xs + width/2, widths=width*0.85,
                       patch_artist=True, showfliers=False, manage_ticks=False,
                       boxprops=dict(facecolor="#d62728", alpha=0.5,
                                     edgecolor="#d62728"),
                       medianprops=dict(color="black", linewidth=1.2),
                       whiskerprops=dict(color="#d62728"),
                       capprops=dict(color="#d62728"))

    # strip overlay
    rng = np.random.default_rng(0)
    for i, r in enumerate(rows):
        s_jit = (rng.random(len(r["shift"])) - 0.5) * width * 0.55
        a_jit = (rng.random(len(r["ablate"])) - 0.5) * width * 0.55
        ax.scatter(xs[i] - width/2 + s_jit, r["shift"],
                    s=8, color="#1f77b4", alpha=0.45, edgecolor="none")
        ax.scatter(xs[i] + width/2 + a_jit, r["ablate"],
                    s=8, color="#d62728", alpha=0.45, edgecolor="none")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("downstream head cos-sim vs baseline", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.7)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vitb", required=True)
    ap.add_argument("--deit", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_v, rows_v = load_data(args.vitb)
    model_d, rows_d = load_data(args.deit)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2),
                              gridspec_kw=dict(width_ratios=[len(rows_v),
                                                              len(rows_d)]))
    plot_panel(axes[0], rows_v, f"(a) ViT-B/16 ({model_v})")
    plot_panel(axes[1], rows_d, f"(b) DeiT-Tiny ({model_d})")

    # Custom legend
    handles = [
        plt.Rectangle((0,0), 1, 1, facecolor="#1f77b4", alpha=0.5,
                       edgecolor="#1f77b4", label="own-kernel substitution"),
        plt.Rectangle((0,0), 1, 1, facecolor="#d62728", alpha=0.5,
                       edgecolor="#d62728", label="mean ablation"),
    ]
    fig.legend(handles=handles, loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=2, fontsize=9, frameon=False)
    fig.suptitle("", y=1.0)

    fig.text(0.5, -0.02,
              "Each pair: distribution over all downstream heads (l$_l$ > l$_e$, all h) "
              "of cos-sim with baseline.",
              ha="center", fontsize=8, style="italic")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")

    # Also print a small summary table
    print("\nsummary for paper")
    for label, rows in [("ViT-B", rows_v), ("DeiT-Tiny", rows_d)]:
        s_meds = [float(np.median(r["shift"])) for r in rows]
        a_meds = [float(np.median(r["ablate"])) for r in rows]
        print(f"  {label}: shift med-of-meds={np.median(s_meds):.4f} "
              f"ablate med-of-meds={np.median(a_meds):.4f}  "
              f"gap={np.median(s_meds)-np.median(a_meds):.4f}")


if __name__ == "__main__":
    main()
