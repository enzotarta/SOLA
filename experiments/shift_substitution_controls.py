"""Controls for the shift substitution: random heads and wrong offsets."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import torch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))

from shift_substitution import (
    detect_model_config, extract_offset, install_substitutions,
    restore_forwards, evaluate, build_tiny_imagenet_eval_loader,
    parse_heads,
)


def control_A_random_heads(model, eval_loader, cfg, device, n_eval_batches,
                            real_shift_heads, n_random_seeds=3):
    """
    Pick random (layer, head) pairs UNIFORMLY from all non-shift heads in the
    model; assign each a random unit-shift offset; measure accuracy.
    Repeat over n_random_seeds different choices to get noise floor.

    Sampling globally (rather than restricting to the layers of the real
    shift heads) keeps this control well-defined for small models where a
    layer can saturate, and treats architectures with different head counts
    per block in the same way.
    """
    print("CONTROL A: uniformly-random heads with random unit shifts")
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.num_heads
    n_to_pick = len(real_shift_heads)

    unit_shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    base_top1, base_top5, _ = evaluate(model, eval_loader, n_eval_batches, device)
    print(f"baseline: top-1={base_top1*100:.2f}%  top-5={base_top5*100:.2f}%")

    deltas = []
    for seed in range(n_random_seeds):
        rng = random.Random(seed)
        all_avail = [(L, h) for L in range(n_layers)
                     for h in range(n_heads)
                     if (L, h) not in real_shift_heads]
        chosen_heads = rng.sample(all_avail, n_to_pick)
        sub_map = {chosen: rng.choice(unit_shifts) for chosen in chosen_heads}

        saved = install_substitutions(
            model, sub_map, cfg["grid"], cfg["n_extra_tokens"], "head_output")
        top1, top5, _ = evaluate(model, eval_loader, n_eval_batches, device)
        restore_forwards(model, saved)
        d1 = (top1 - base_top1) * 100
        d5 = (top5 - base_top5) * 100
        head_strs = [f"L{l}H{h}({dy:+d},{dx:+d})"
                     for (l, h), (dy, dx) in sub_map.items()]
        print(f"  seed {seed}: {head_strs}  delta top1={d1:+.2f}  delta top5={d5:+.2f}")
        deltas.append({"seed": seed, "subs": list(sub_map.items()),
                       "delta_top1": d1, "delta_top5": d5,
                       "top1": top1, "top5": top5})

    mean_d1 = float(np.mean([d["delta_top1"] for d in deltas]))
    std_d1 = float(np.std([d["delta_top1"] for d in deltas]))
    print(f"\n  CONTROL A mean delta top1 = {mean_d1:+.2f}  (std {std_d1:.2f}) over {n_random_seeds} seeds")
    return {"baseline_top1": base_top1, "trials": deltas,
            "mean_delta_top1": mean_d1, "std_delta_top1": std_d1}


def control_B_wrong_offsets(model, eval_loader, cfg, device, n_eval_batches,
                              real_offsets, n_random_seeds=3):
    """
    real_offsets: dict {(layer, head): (dy, dx)} of the ACTUAL shift heads.
    For each seed, replace each head with a RANDOM unit shift different from
    its real offset. Measure the cost of getting the shift wrong.
    """
    print("CONTROL B: real shift heads with WRONG offsets")
    unit_shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    base_top1, base_top5, _ = evaluate(model, eval_loader, n_eval_batches, device)
    print(f"baseline: top-1={base_top1*100:.2f}%  top-5={base_top5*100:.2f}%")

    deltas = []
    for seed in range(n_random_seeds):
        rng = random.Random(seed + 100)
        sub_map = {}
        for (li, hi), real_off in real_offsets.items():
            wrong_options = [s for s in unit_shifts if s != real_off]
            sub_map[(li, hi)] = rng.choice(wrong_options)

        saved = install_substitutions(
            model, sub_map, cfg["grid"], cfg["n_extra_tokens"], "head_output")
        top1, top5, _ = evaluate(model, eval_loader, n_eval_batches, device)
        restore_forwards(model, saved)
        d1 = (top1 - base_top1) * 100
        d5 = (top5 - base_top5) * 100
        head_strs = [f"L{l}H{h}: real{real_offsets[(l,h)]} -> {sub_map[(l,h)]}"
                     for (l, h) in sub_map]
        print(f"  seed {seed}:  {head_strs[0]}")
        for hs in head_strs[1:]:
            print(f"             {hs}")
        print(f"            delta top1={d1:+.2f}  delta top5={d5:+.2f}")
        deltas.append({"seed": seed, "subs": list(sub_map.items()),
                       "delta_top1": d1, "delta_top5": d5,
                       "top1": top1, "top5": top5})

    mean_d1 = float(np.mean([d["delta_top1"] for d in deltas]))
    std_d1 = float(np.std([d["delta_top1"] for d in deltas]))
    print(f"\n  CONTROL B mean delta top1 = {mean_d1:+.2f}  (std {std_d1:.2f}) over {n_random_seeds} seeds")
    return {"baseline_top1": base_top1, "trials": deltas,
            "mean_delta_top1": mean_d1, "std_delta_top1": std_d1}


def control_C_full_eval(model, eval_loader, cfg, device, n_eval_batches,
                          real_offsets):
    """
    Real shift heads with their correct offsets, on the full eval set, for
    tighter error bars.
    """
    print("CONTROL C: real shift heads with correct offsets (full eval)")

    base_top1, base_top5, n_eval = evaluate(model, eval_loader, n_eval_batches, device)
    print(f"baseline (n={n_eval}): top-1={base_top1*100:.2f}%  top-5={base_top5*100:.2f}%")

    saved = install_substitutions(
        model, real_offsets, cfg["grid"], cfg["n_extra_tokens"], "head_output")
    top1, top5, _ = evaluate(model, eval_loader, n_eval_batches, device)
    restore_forwards(model, saved)
    d1 = (top1 - base_top1) * 100
    d5 = (top5 - base_top5) * 100

    # Wilson 95% CI half-width for a binomial at this n
    half_width_top1 = 1.96 * (base_top1 * (1 - base_top1) / max(n_eval, 1)) ** 0.5 * 100
    print(f"After substitution:  top-1={top1*100:.2f}% (delta {d1:+.2f})  top-5={top5*100:.2f}% (delta {d5:+.2f})")
    print(f"  ~95% CI half-width on top-1 (n={n_eval}): +/-{half_width_top1:.2f} pts")

    return {"baseline_top1": base_top1, "subbed_top1": top1,
            "delta_top1": d1, "delta_top5": d5,
            "n_eval": n_eval, "ci_half_width": half_width_top1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch16_224")
    ap.add_argument("--shift-heads", nargs="+", required=True,
                    help="layer,head pairs of identified shift heads")
    ap.add_argument("--controls", nargs="+", default=["A", "B", "C"],
                    choices=["A", "B", "C"])
    ap.add_argument("--batches", type=int, default=156,
                    help="156 batches at batch-size 64 = full tiny-imagenet val")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-extract-batches", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--n-random-seeds", type=int, default=3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="shift_controls_result.json")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    real_shift_heads = parse_heads(args.shift_heads)

    import timm
    model = timm.create_model(args.model, pretrained=True).eval().to(device)
    cfg = detect_model_config(model, args.image_size)
    eval_loader, _ = build_tiny_imagenet_eval_loader(
        args.batch_size, args.image_size)

    # First: extract real offsets for the shift heads
    print(f"\nExtracting offsets for the {len(real_shift_heads)} identified shift heads...")
    real_offsets = {}
    for (li, hi) in real_shift_heads:
        info = extract_offset(model, li, hi, eval_loader,
                              n_batches=args.n_extract_batches,
                              config=cfg, device=device)
        real_offsets[(li, hi)] = info["offset"]
        print(f"  L{li}H{hi}: offset={info['offset']}  "
              f"consistency={info['argmax_consistency']:.3f}")

    results = {"model": args.model, "real_offsets": {f"L{l}H{h}": list(o) for (l, h), o in real_offsets.items()},
               "n_eval_batches": args.batches}

    if "A" in args.controls:
        results["control_A"] = control_A_random_heads(
            model, eval_loader, cfg, device, args.batches,
            real_shift_heads, n_random_seeds=args.n_random_seeds)
    if "B" in args.controls:
        results["control_B"] = control_B_wrong_offsets(
            model, eval_loader, cfg, device, args.batches,
            real_offsets, n_random_seeds=args.n_random_seeds)
    if "C" in args.controls:
        results["control_C"] = control_C_full_eval(
            model, eval_loader, cfg, device, args.batches, real_offsets)

    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {args.out}")

    # Final summary
    print("SUMMARY - what the controls tell us")
    if "C" in args.controls:
        c = results["control_C"]
        print(f"  Real shifts, full eval:  delta top1 = {c['delta_top1']:+.2f}  "
              f"(n={c['n_eval']}, 95% CI +/-{c['ci_half_width']:.2f})")
    if "A" in args.controls:
        a = results["control_A"]
        print(f"  Random heads + random shifts: delta top1 = {a['mean_delta_top1']:+.2f} +/- {a['std_delta_top1']:.2f}")
    if "B" in args.controls:
        b = results["control_B"]
        print(f"  Real shift heads, WRONG shifts: delta top1 = {b['mean_delta_top1']:+.2f} +/- {b['std_delta_top1']:.2f}")
    print()
    print("Interpretation:")
    print("  Strong claim if:")
    print("    Control C delta ~ 0  AND")
    print("    Control A delta < (much more negative)  AND")
    print("    Control B delta < (much more negative)")
    print("  Weak claim if:")
    print("    Controls A and B aren't much worse than the real substitution.")


if __name__ == "__main__":
    main()
