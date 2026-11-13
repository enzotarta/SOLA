"""Re-run the Stage-1 cascade on stored diagnostics with each threshold moved by
+/-0.05, and count the heads that change category.
"""
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))


def classify(d, T):
    """Same cascade as operator_taxonomy.classify_head, with thresholds
    drawn from T = (fixed_target, near_global_escape, self_only,
    delta_top1, multi_top3, soft_r, gauss_res, diffuse_escape)."""
    if d["argmax_concentration"] > T[0]:
        return "fixed_target"
    if d["escape_mass_K3"] > T[1]:
        return "near_global"
    if d["self_mass"] > T[2] and tuple(d["peak_offset"]) == (0, 0):
        return "self_only"
    if (d["top1_mass"] > T[3] and tuple(d["peak_offset"]) != (0, 0)
            and d["modality_count"] <= 2):
        return "delta_shift"
    if (d["top3_mass"] > T[4] and d["modality_count"] >= 2
            and d["fingerprint_entropy"] < 2.0):
        return "multi_delta"
    if (tuple(d["peak_offset"]) != (0, 0) and d["kernel_l1_residual_mean"] < T[5]
            and d["modality_count"] <= 3 and d["fingerprint_entropy"] < 2.5):
        return "soft_shift"
    if (d["gaussian_fit_residual"] < T[6] and d["modality_count"] == 1
            and d["escape_mass_K3"] < 0.5):
        return "gaussian_local"
    if d["escape_mass_K3"] < T[7]:
        return "diffuse_local"
    return "other"


T_REF = (0.5, 0.7, 0.7, 0.5, 0.7, 0.40, 0.15, 0.5)
NAMES = ("FT_argmax", "NG_escape", "self_mass", "delta_top1",
         "multi_top3", "soft_r", "gauss_res", "DL_escape")
DELTA = 0.05


def count_flips(rows, T1, T2):
    n = 0
    for r in rows:
        c1 = classify(r["diag"], T1)
        c2 = classify(r["diag"], T2)
        if c1 != c2:
            n += 1
    return n


def main():
    exp = Path(__file__).resolve().parent.parent / "results"
    pairs = [
        ("DeiT-Tiny", "taxonomy_deit.json"),
        ("DeiT-Small", "taxonomy_deit_s.json"),
        ("ViT-S/16 IN-21k", "taxonomy_vits.json"),
        ("ViT-B/16 IN-1k", "taxonomy_vitb_in1k_pure.json"),
        ("ViT-B/16 IN-21k", "taxonomy_vitb_in21k.json"),
        ("ViT-B/16 MAE", "taxonomy_mae.json"),
        ("ViT-B/16 DINO", "taxonomy_vitb_dino.json"),
        ("ViT-B/14 DINOv2", "taxonomy_dinov2.json"),
        ("CLIP ViT-B/16", "taxonomy_clip.json"),
        ("SigLIP ViT-B/16", "taxonomy_siglip.json"),
    ]

    print(f"{'backbone':22s}  {'n':>4s}  ", end="")
    for n in NAMES:
        print(f"{n:>10s}", end="")
    print(f"  {'mean':>5s}")
    all_flip_pcts = []
    per_threshold = [[] for _ in NAMES]
    for label, fname in pairs:
        path = exp / fname
        if not path.exists():
            print(f"{label:22s}  [missing {fname}]")
            continue
        rows = json.loads(path.read_text())["results"]
        N = len(rows)
        print(f"{label:22s}  {N:>4d}  ", end="")
        flips = []
        for k in range(len(T_REF)):
            T_minus = list(T_REF); T_minus[k] -= DELTA
            T_plus  = list(T_REF); T_plus[k]  += DELTA
            flips_pm = max(count_flips(rows, T_REF, tuple(T_minus)),
                           count_flips(rows, T_REF, tuple(T_plus)))
            pct = 100.0 * flips_pm / N
            flips.append(pct)
            per_threshold[k].append(pct)
            print(f"{pct:>9.1f}%", end="")
        mean = sum(flips) / len(flips)
        all_flip_pcts.append(mean)
        print(f"  {mean:>4.1f}%")
    print()
    print("Per-threshold worst-case flip pct across backbones:")
    for n, lst in zip(NAMES, per_threshold):
        if lst:
            print(f"  {n:12s}: max={max(lst):.1f}%  mean={sum(lst)/len(lst):.1f}%")
    if all_flip_pcts:
        print(f"\nMean-over-backbones, mean-over-thresholds: {sum(all_flip_pcts)/len(all_flip_pcts):.1f}%")


if __name__ == "__main__":
    main()
