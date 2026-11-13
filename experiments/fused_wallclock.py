"""Wall-clock of the Triton path against the unmodified model."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))


def cuda_time_callable(fn, n_warmup=20, n_runs=100, device="cuda"):
    """Time a no-arg callable on CUDA. Returns (mean_ms, std_ms)."""
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = fn()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        timings = []
        for _ in range(n_runs):
            start.record()
            _ = fn()
            end.record()
            torch.cuda.synchronize(device)
            timings.append(start.elapsed_time(end))
    arr = np.array(timings)
    return float(arr.mean()), float(arr.std())


def quick_top1(model, loader, n_batches, device, dtype=torch.float32):
    correct = total = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= n_batches: break
            x, y = batch[0].to(device, dtype=dtype), batch[1].to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def find_top_shift_layer(shift_heads_lh):
    """Return (layer_idx, count) of the layer with the most shifts."""
    from collections import Counter
    by_layer = Counter([li for li, _ in shift_heads_lh])
    if not by_layer:
        return None, 0
    li, n = by_layer.most_common(1)[0]
    return li, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--img-sizes", default="native",
                    help="comma-separated list of input sizes, "
                         "or 'native' for the model's native size")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-warmup", type=int, default=20)
    ap.add_argument("--n-runs", type=int, default=100)
    ap.add_argument("--shift-cats", default="delta_shift,soft_shift")
    ap.add_argument("--broadcast-json", default=None,
                    help="path to a broadcast experiment JSON whose "
                         "low_r_heads field lists validated broadcast "
                         "heads to add to the Triton substitution")
    ap.add_argument("--source", default="imagenet-val",
                    choices=("imagenet-val", "tiny-imagenet", "synthetic"))
    ap.add_argument("--imagenet-val-dir", default="data/imagenet/val")
    ap.add_argument("--n-acc-batches", type=int, default=10,
                    help="number of batches for accuracy sanity check; "
                         "0 disables")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="fp32",
                    choices=("fp32", "fp16"),
                    help="model + input dtype for the benchmark; fp16 "
                         "engages Tensor Cores in both SDPA and Triton")
    args = ap.parse_args()

    import timm
    from shift_substitution import (
        detect_model_config,
        build_imagenet_val_eval_loader,
        build_tiny_imagenet_eval_loader,
    )
    from kernel_substitution import (
        kernel_from_diagnostics_mean,
        install_kernel_substitutions,
        restore_forwards as restore_kernel,
    )
    from fused_substitutions import (
        install_fused_shift_substitutions,
        restore_forwards as restore_fused,
    )
    from fused_substitutions_triton import (
        install_triton_fused_substitutions,
        restore_forwards as restore_triton,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device != "cuda":
        raise SystemExit("CUDA required for fused wall-clock benchmark.")

    # Native size lookup: query the model's default cfg
    if args.img_sizes == "native":
        tmp = timm.create_model(args.model, pretrained=False)
        native = tmp.default_cfg.get("input_size", (3, 224, 224))[-1]
        del tmp
        sizes = [native]
    else:
        sizes = [int(s) for s in args.img_sizes.split(",")]
    print(f"Image sizes to benchmark: {sizes}")

    tax = json.loads(Path(args.taxonomy).read_text())
    shift_cats = set(args.shift_cats.split(","))
    res_by_lh = {(r["layer"], r["head"]): r for r in tax["results"]}

    broadcast_heads_global = []
    if args.broadcast_json is not None:
        bc = json.loads(Path(args.broadcast_json).read_text())
        broadcast_heads_global = [tuple(h) for h in bc.get("low_r_heads", [])]
        print(f"Broadcast heads (low_r): n={len(broadcast_heads_global)}: "
              f"{broadcast_heads_global}")
    shift_heads = [(r["layer"], r["head"]) for r in tax["results"]
                   if r["category"] in shift_cats]
    print(f"Shift heads ({sorted(shift_cats)}): n={len(shift_heads)} "
          f"{shift_heads}")
    if not shift_heads:
        raise SystemExit("No shift heads in taxonomy. Nothing to benchmark.")

    top_layer, top_n = find_top_shift_layer(shift_heads)
    print(f"Most-shift-dense layer: L{top_layer} with {top_n} shifts")

    out = {
        "model": args.model,
        "shift_heads": [list(h) for h in shift_heads],
        "n_shift_heads": len(shift_heads),
        "top_shift_layer": top_layer,
        "top_shift_layer_count": top_n,
        "sizes": {},
    }

    for size in sizes:
        print(f"\nSize: {size}")
        torch.cuda.empty_cache()
        model = timm.create_model(args.model, pretrained=True,
                                    img_size=size).eval().to(device)
        if args.dtype == "fp16":
            model = model.half()
        cfg = detect_model_config(model, image_size=size)
        n_heads = model.blocks[0].attn.num_heads
        torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
        x = torch.randn(args.batch_size, 3, size, size, device=device,
                          dtype=torch_dtype)

        kernel_map = {
            (li, hi): kernel_from_diagnostics_mean(
                np.asarray(res_by_lh[(li, hi)]["mean_kernel"]))
            for (li, hi) in shift_heads
        }

        # end-to-end baseline
        print("[end-to-end] baseline ...")
        m_ms, s_ms = cuda_time_callable(lambda: model(x),
                                         args.n_warmup, args.n_runs)
        baseline_e2e = (m_ms, s_ms)
        print(f"  baseline e2e: {m_ms:.3f} +/- {s_ms:.3f} ms")

        # per-attention-block baseline
        print(f"[per-block] baseline (layer {top_layer}) ...")
        block = model.blocks[top_layer]
        # Drive the attn module with a representative input matching
        # the block's interface: same dtype, residual stream tensor.
        ref_in = torch.randn(args.batch_size, cfg["n_extra_tokens"]
                              + cfg["grid"][0]*cfg["grid"][1],
                              model.embed_dim, device=device,
                              dtype=torch_dtype)
        m_ms, s_ms = cuda_time_callable(
            lambda: block.attn(block.norm1(ref_in)),
            args.n_warmup, args.n_runs)
        baseline_block = (m_ms, s_ms)
        print(f"  baseline block: {m_ms:.3f} +/- {s_ms:.3f} ms")

        # install conv2d-fused, time both
        print("[conv2d-fused] installing ...")
        saved = install_fused_shift_substitutions(
            model, kernel_map, cfg["grid"], cfg["n_extra_tokens"])

        print("[end-to-end] conv2d-fused ...")
        m_ms, s_ms = cuda_time_callable(lambda: model(x),
                                         args.n_warmup, args.n_runs)
        fused_e2e = (m_ms, s_ms)
        print(f"  conv2d-fused e2e: {m_ms:.3f} +/- {s_ms:.3f} ms")

        print(f"[per-block] conv2d-fused (layer {top_layer}) ...")
        m_ms, s_ms = cuda_time_callable(
            lambda: block.attn(block.norm1(ref_in)),
            args.n_warmup, args.n_runs)
        fused_block = (m_ms, s_ms)
        print(f"  conv2d-fused block: {m_ms:.3f} +/- {s_ms:.3f} ms")

        restore_fused(model, saved)

        # install Triton-fused (shift only), time both
        print("[triton-shift] installing ...")
        saved_t = install_triton_fused_substitutions(
            model, kernel_map=kernel_map,
            broadcast_heads=None,
            grid=cfg["grid"], n_extra_tokens=cfg["n_extra_tokens"])

        print("[end-to-end] triton-shift ...")
        m_ms, s_ms = cuda_time_callable(lambda: model(x),
                                         args.n_warmup, args.n_runs)
        triton_e2e = (m_ms, s_ms)
        print(f"  triton-shift e2e: {m_ms:.3f} +/- {s_ms:.3f} ms")

        print(f"[per-block] triton-shift (layer {top_layer}) ...")
        m_ms, s_ms = cuda_time_callable(
            lambda: block.attn(block.norm1(ref_in)),
            args.n_warmup, args.n_runs)
        triton_block = (m_ms, s_ms)
        print(f"  triton-shift block: {m_ms:.3f} +/- {s_ms:.3f} ms")

        restore_triton(model, saved_t)

        # install combined Triton (shift + broadcast), time both
        triton_sb_e2e = triton_sb_block = (None, None)
        if broadcast_heads_global:
            print("[triton-S+B] installing ...")
            saved_sb = install_triton_fused_substitutions(
                model, kernel_map=kernel_map,
                broadcast_heads=broadcast_heads_global,
                grid=cfg["grid"], n_extra_tokens=cfg["n_extra_tokens"])

            print("[end-to-end] triton-S+B ...")
            m_ms, s_ms = cuda_time_callable(lambda: model(x),
                                             args.n_warmup, args.n_runs)
            triton_sb_e2e = (m_ms, s_ms)
            print(f"  triton-S+B e2e: {m_ms:.3f} +/- {s_ms:.3f} ms")

            # Pick a "best" layer for per-block: the one with most
            # combined substitutions.
            from collections import Counter
            shift_layers = Counter(li for (li, _) in shift_heads)
            bcast_layers = Counter(li for (li, _) in broadcast_heads_global)
            combined_density = {li: shift_layers[li] + bcast_layers[li]
                                for li in (set(shift_layers) | set(bcast_layers))}
            sb_layer = max(combined_density, key=combined_density.get)
            sb_block = model.blocks[sb_layer]
            sb_count = combined_density[sb_layer]
            print(f"[per-block] triton-S+B (layer {sb_layer}, "
                  f"density={sb_count}) ...")
            m_ms, s_ms = cuda_time_callable(
                lambda: sb_block.attn(sb_block.norm1(ref_in)),
                args.n_warmup, args.n_runs)
            triton_sb_block = (m_ms, s_ms)
            print(f"  triton-S+B block: {m_ms:.3f} +/- {s_ms:.3f} ms")
            # Also re-time baseline at the same layer for fair ratio
            restore_triton(model, saved_sb)
            print(f"[per-block] baseline (layer {sb_layer}) for S+B ratio ...")
            m_ms_b, s_ms_b = cuda_time_callable(
                lambda: sb_block.attn(sb_block.norm1(ref_in)),
                args.n_warmup, args.n_runs)
            sb_baseline_block = (m_ms_b, s_ms_b)
            print(f"  baseline block @L{sb_layer}: {m_ms_b:.3f} +/- {s_ms_b:.3f} ms")
        else:
            sb_layer = None
            sb_count = 0
            sb_baseline_block = (None, None)

        # install FULL Triton (shift + broadcast + Triton attention
        # replacing SDPA on every other head). Installed on every
        # layer, so all attention runs through Triton.
        print("[triton-FULL] installing ...")
        saved_full = install_triton_fused_substitutions(
            model, kernel_map=kernel_map,
            broadcast_heads=(broadcast_heads_global or None),
            grid=cfg["grid"], n_extra_tokens=cfg["n_extra_tokens"],
            use_triton_for_sdpa=True)

        print("[end-to-end] triton-FULL ...")
        m_ms, s_ms = cuda_time_callable(lambda: model(x),
                                         args.n_warmup, args.n_runs)
        triton_full_e2e = (m_ms, s_ms)
        print(f"  triton-FULL e2e: {m_ms:.3f} +/- {s_ms:.3f} ms")

        # Per-block at the same layer used for S+B comparison
        if sb_layer is not None:
            full_block_at = sb_layer
        else:
            full_block_at = top_layer
        print(f"[per-block] triton-FULL (layer {full_block_at}) ...")
        full_block_module = model.blocks[full_block_at]
        m_ms, s_ms = cuda_time_callable(
            lambda: full_block_module.attn(full_block_module.norm1(ref_in)),
            args.n_warmup, args.n_runs)
        triton_full_block = (m_ms, s_ms)
        print(f"  triton-FULL block: {m_ms:.3f} +/- {s_ms:.3f} ms")

        restore_triton(model, saved_full)

        # accuracy sanity (only at native, only if loader available)
        acc_base = acc_fused = acc_kernel = acc_triton = None
        acc_triton_sb = acc_triton_full = None
        if args.n_acc_batches > 0 and args.source != "synthetic":
            try:
                if args.source == "imagenet-val":
                    loader, _ = build_imagenet_val_eval_loader(
                        args.batch_size, image_size=size,
                        val_dir=args.imagenet_val_dir, num_workers=2)
                else:
                    loader, _ = build_tiny_imagenet_eval_loader(
                        args.batch_size, size)
                print("[acc] baseline ...")
                acc_base = quick_top1(model, loader, args.n_acc_batches, device, dtype=torch_dtype)
                print(f"  baseline top-1: {acc_base*100:.2f}%")

                print("[acc] fused ...")
                saved = install_fused_shift_substitutions(
                    model, kernel_map, cfg["grid"], cfg["n_extra_tokens"])
                acc_fused = quick_top1(model, loader, args.n_acc_batches, device, dtype=torch_dtype)
                restore_fused(model, saved)
                print(f"  fused top-1: {acc_fused*100:.2f}%")

                # Cross-check vs the structural kernel-substitution path
                print("[acc] kernel-substitution (structural reference) ...")
                saved_k = install_kernel_substitutions(
                    model, kernel_map, cfg["grid"], cfg["n_extra_tokens"])
                acc_kernel = quick_top1(model, loader, args.n_acc_batches, device, dtype=torch_dtype)
                restore_kernel(model, saved_k)
                print(f"  kernel-sub top-1: {acc_kernel*100:.2f}%")

                print("[acc] triton-shift ...")
                saved_t2 = install_triton_fused_substitutions(
                    model, kernel_map=kernel_map,
                    broadcast_heads=None,
                    grid=cfg["grid"], n_extra_tokens=cfg["n_extra_tokens"])
                acc_triton = quick_top1(model, loader, args.n_acc_batches, device, dtype=torch_dtype)
                restore_triton(model, saved_t2)
                print(f"  triton-shift top-1: {acc_triton*100:.2f}%")

                if broadcast_heads_global:
                    print("[acc] triton-S+B ...")
                    saved_sb_acc = install_triton_fused_substitutions(
                        model, kernel_map=kernel_map,
                        broadcast_heads=broadcast_heads_global,
                        grid=cfg["grid"],
                        n_extra_tokens=cfg["n_extra_tokens"])
                    acc_triton_sb = quick_top1(model, loader,
                                                  args.n_acc_batches, device)
                    restore_triton(model, saved_sb_acc)
                    print(f"  triton-S+B top-1: {acc_triton_sb*100:.2f}%")

                print("[acc] triton-FULL ...")
                saved_full_acc = install_triton_fused_substitutions(
                    model, kernel_map=kernel_map,
                    broadcast_heads=(broadcast_heads_global or None),
                    grid=cfg["grid"],
                    n_extra_tokens=cfg["n_extra_tokens"],
                    use_triton_for_sdpa=True)
                acc_triton_full = quick_top1(model, loader,
                                                args.n_acc_batches, device)
                restore_triton(model, saved_full_acc)
                print(f"  triton-FULL top-1: {acc_triton_full*100:.2f}%")
            except Exception as e:
                print(f"  WARN: accuracy check skipped: {e}")

        out["sizes"][str(size)] = {
            "n_tokens": cfg["grid"][0]*cfg["grid"][1] + cfg["n_extra_tokens"],
            "n_heads": n_heads,
            "baseline_e2e_ms":     {"mean": baseline_e2e[0], "std": baseline_e2e[1]},
            "fused_e2e_ms":        {"mean": fused_e2e[0],    "std": fused_e2e[1]},
            "triton_e2e_ms":       {"mean": triton_e2e[0],   "std": triton_e2e[1]},
            "triton_sb_e2e_ms":    {"mean": triton_sb_e2e[0], "std": triton_sb_e2e[1]} if triton_sb_e2e[0] is not None else None,
            "baseline_block_ms":   {"mean": baseline_block[0], "std": baseline_block[1]},
            "fused_block_ms":      {"mean": fused_block[0],    "std": fused_block[1]},
            "triton_block_ms":     {"mean": triton_block[0],   "std": triton_block[1]},
            "triton_sb_block_ms":  {"mean": triton_sb_block[0], "std": triton_sb_block[1]} if triton_sb_block[0] is not None else None,
            "sb_baseline_block_ms": {"mean": sb_baseline_block[0], "std": sb_baseline_block[1]} if sb_baseline_block[0] is not None else None,
            "sb_layer":            sb_layer,
            "sb_density":          sb_count,
            "speedup_e2e_fused":   baseline_e2e[0] / max(fused_e2e[0], 1e-12),
            "speedup_e2e_triton":  baseline_e2e[0] / max(triton_e2e[0], 1e-12),
            "speedup_e2e_triton_sb": (baseline_e2e[0] / triton_sb_e2e[0]) if triton_sb_e2e[0] else None,
            "speedup_block_fused":  baseline_block[0] / max(fused_block[0], 1e-12),
            "speedup_block_triton": baseline_block[0] / max(triton_block[0], 1e-12),
            "speedup_block_triton_sb": (sb_baseline_block[0] / triton_sb_block[0]) if (sb_baseline_block[0] is not None and triton_sb_block[0]) else None,
            "triton_full_e2e_ms":  {"mean": triton_full_e2e[0], "std": triton_full_e2e[1]},
            "triton_full_block_ms":{"mean": triton_full_block[0], "std": triton_full_block[1]},
            "speedup_e2e_triton_full":   baseline_e2e[0] / max(triton_full_e2e[0], 1e-12),
            "speedup_block_triton_full": (sb_baseline_block[0] if sb_baseline_block[0] is not None else baseline_block[0]) / max(triton_full_block[0], 1e-12),
            "top1_baseline":     acc_base,
            "top1_fused":        acc_fused,
            "top1_kernel_sub":   acc_kernel,
            "top1_triton":       acc_triton,
            "top1_triton_sb":    acc_triton_sb,
            "top1_triton_full":  acc_triton_full,
        }

        del model
        torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    # Summary table (compact, end-to-end-focused)
    print("\nSUMMARY (e2e: baseline/Triton-S/Triton-S+B/Triton-FULL)")
    print(f"{'size':>5}  {'N':>5}  "
          f"{'e2e-base':>9}  {'e2e-S':>8}  {'x':>5}  "
          f"{'e2e-S+B':>8}  {'x':>5}  "
          f"{'e2e-FULL':>9}  {'x':>5}  "
          f"{'top1: base/S/S+B/F':<25}")
    for sz, d in out["sizes"].items():
        eb = d["baseline_e2e_ms"]["mean"]
        et = d["triton_e2e_ms"]["mean"]
        bsb_e2e = d.get("triton_sb_e2e_ms")
        full_e2e = d.get("triton_full_e2e_ms")
        bsb_e = bsb_e2e["mean"] if bsb_e2e else float("nan")
        full_e = full_e2e["mean"] if full_e2e else float("nan")
        rt = eb/max(et,1e-9)
        rsb = eb/max(bsb_e,1e-9) if bsb_e2e else float("nan")
        rfull = eb/max(full_e,1e-9) if full_e2e else float("nan")
        t1s = []
        for k in ("top1_baseline", "top1_triton",
                   "top1_triton_sb", "top1_triton_full"):
            v = d.get(k)
            t1s.append(f"{v*100:.2f}" if v is not None else "-")
        t1_str = "/".join(t1s)
        print(f"{sz:>5}  {d['n_tokens']:>5}  "
              f"{eb:>8.3f}  {et:>7.3f}  {rt:>4.3f}x  "
              f"{bsb_e:>7.3f}  {rsb:>4.3f}x  "
              f"{full_e:>8.3f}  {rfull:>4.3f}x  "
              f"{t1_str:<25}")
    print("\nSUMMARY (per-block at densest layer)")
    print(f"{'size':>5}  {'sb-L':>4} {'dens':>4}  "
          f"{'blk-base':>9}  {'blk-S+B':>8}  {'x':>5}  "
          f"{'blk-FULL':>9}  {'x':>5}")
    for sz, d in out["sizes"].items():
        sb_base = d.get("sb_baseline_block_ms")
        bsb_block = d.get("triton_sb_block_ms")
        full_block = d.get("triton_full_block_ms")
        if sb_base is None:
            sb_bb = d["baseline_block_ms"]["mean"]
        else:
            sb_bb = sb_base["mean"]
        bsb_b = bsb_block["mean"] if bsb_block else float("nan")
        full_b = full_block["mean"] if full_block else float("nan")
        sb_layer = d.get("sb_layer", "-")
        sb_dens = d.get("sb_density", 0)
        rb = sb_bb/max(bsb_b,1e-9) if bsb_block else float("nan")
        rf = sb_bb/max(full_b,1e-9) if full_block else float("nan")
        print(f"{sz:>5}  {str(sb_layer):>4} {sb_dens:>4}  "
              f"{sb_bb:>8.3f}  {bsb_b:>7.3f}  {rb:>4.3f}x  "
              f"{full_b:>8.3f}  {rf:>4.3f}x")


if __name__ == "__main__":
    main()
