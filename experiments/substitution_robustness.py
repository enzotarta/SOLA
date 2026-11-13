"""Substitution deltas under eval-time corruptions: noise, blur and contrast."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


sys.path.insert(0, str(Path(__file__).parent))


# Corruptions

def make_corruption(spec: str):
    """Return a callable Tensor -> Tensor that applies the corruption.
    Operates on already-normalized tensors. spec examples:
      clean
      gauss_noise_<sigma>     sigma in pixel-fraction units (e.g. 0.1)
      gauss_blur_<sigma>      sigma of Gaussian blur kernel
      contrast_<f>            multiply contrast by f (e.g. 0.5)
      jpeg_<q>                JPEG round-trip at quality q (1..95)
    """
    if spec == "clean":
        return lambda x: x

    # spec parses as "<name_with_underscores>_<value>"; split on last "_"
    name, _, val_s = spec.rpartition("_")
    val = float(val_s) if val_s else 0.0

    # All corruptions act on the un-normalized image. Our val pipeline
    # already normalizes; we de-normalize -> corrupt -> re-normalize.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def denorm(x): return (x.cpu() * std + mean).clamp(0, 1)
    def renorm(x): return (x - mean) / std

    if name == "gauss_noise":
        sigma = val
        def f(x):
            img = denorm(x)
            noise = torch.randn_like(img) * sigma
            img = (img + noise).clamp(0, 1)
            return renorm(img).to(x.device)
        return f
    if name == "gauss_blur":
        sigma = val
        ksize = max(3, int(round(sigma * 4)) | 1)
        blur = T.GaussianBlur(kernel_size=ksize, sigma=sigma)
        def f(x):
            img = denorm(x)
            img = blur(img)
            return renorm(img).to(x.device)
        return f
    if name == "contrast":
        factor = val
        def f(x):
            img = denorm(x)
            img = TF.adjust_contrast(img, factor)
            return renorm(img).to(x.device)
        return f
    if name == "jpeg":
        from io import BytesIO
        from PIL import Image
        q = int(val)
        def f(x):
            img = denorm(x)
            out = []
            for i in range(img.shape[0]):
                pil = TF.to_pil_image(img[i])
                buf = BytesIO()
                pil.save(buf, format="JPEG", quality=q)
                buf.seek(0)
                pil2 = Image.open(buf).convert("RGB")
                out.append(TF.to_tensor(pil2))
            img2 = torch.stack(out)
            return renorm(img2).to(x.device)
        return f

    raise ValueError(f"unknown corruption spec: {spec}")


# Eval

@torch.no_grad()
def eval_top1(model, loader, device, corruption_fn, max_batches=None):
    correct = 0; total = 0
    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches: break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = corruption_fn(x)
        logits = model(x)
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1), total


# Substitution config builders

def build_shift_sub(model, tax_results, cfg, shift_cats):
    from kernel_substitution import (
        kernel_from_diagnostics_mean, install_kernel_substitutions,
    )
    res_by_lh = {(r["layer"], r["head"]): r for r in tax_results}
    shift_heads = [(r["layer"], r["head"]) for r in tax_results
                   if r["category"] in shift_cats]
    if not shift_heads:
        return None, []
    kmap = {(li, hi): kernel_from_diagnostics_mean(
                np.asarray(res_by_lh[(li, hi)]["mean_kernel"]))
            for (li, hi) in shift_heads}
    def install():
        return install_kernel_substitutions(model, kmap, cfg["grid"],
                                              cfg["n_extra_tokens"])
    return install, shift_heads


def build_broadcast_sub(model, tax_results, diag_by_lh, r_thresh):
    from attn_matrix_substitution import install_broadcast_substitutions
    ft_low_r = [(r["layer"], r["head"]) for r in tax_results
                if r["category"] == "fixed_target"
                and diag_by_lh.get((r["layer"], r["head"]), 1.0) < r_thresh]
    if not ft_low_r:
        return None, []
    def install():
        return install_broadcast_substitutions(model, ft_low_r)
    return install, ft_low_r


def build_rank10_sub(model, tax_results, diag_by_lh, r_thresh, k_rank=10):
    from attn_matrix_substitution import install_rank_k_substitutions
    ng_low_r = [(r["layer"], r["head"]) for r in tax_results
                if r["category"] == "near_global"
                and diag_by_lh.get((r["layer"], r["head"]), 1.0) < r_thresh]
    if not ng_low_r:
        return None, []
    def install():
        return install_rank_k_substitutions(model, ng_low_r, k_rank=k_rank)
    return install, ng_low_r


# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--diag", default=None,
                    help="fixed_target_*.json with r_abs_mean per head")
    ap.add_argument("--r-thresh", type=float, default=0.4)
    ap.add_argument("--rank-k", type=int, default=10)
    ap.add_argument("--substitutions",
                    default="shift,broadcast_low_r,rank10_low_r",
                    help="comma-separated subset of "
                         "{shift, broadcast_low_r, rank10_low_r}")
    ap.add_argument("--corruptions",
                    default="clean,gauss_noise_0.05,gauss_blur_2,contrast_0.5,jpeg_25",
                    help="comma-separated list (clean is recommended first)")
    ap.add_argument("--shift-cats", default="delta_shift,soft_shift")
    ap.add_argument("--imagenet-val-dir", default="data/imagenet/val")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="cap batches (None = full val)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    import timm
    from shift_substitution import (
        build_imagenet_val_eval_loader, detect_model_config,
    )
    from kernel_substitution import restore_forwards as restore_kernel
    from attn_matrix_substitution import restore_forwards as restore_attn

    print(f"Loading {args.model} on {device} ...")
    model = timm.create_model(args.model, pretrained=True,
                                 img_size=args.img_size).eval().to(device)
    cfg = detect_model_config(model, image_size=args.img_size)

    loader, _ = build_imagenet_val_eval_loader(
        args.batch_size, image_size=args.img_size,
        val_dir=args.imagenet_val_dir, num_workers=4)

    # Substitution configs
    tax = json.loads(Path(args.taxonomy).read_text())["results"]
    diag_by_lh = {}
    if args.diag and Path(args.diag).exists():
        diag = json.loads(Path(args.diag).read_text())
        diag_by_lh = {(r["layer"], r["head"]): r["r_abs_mean"]
                       for r in diag["per_head"]}

    shift_cats = set(args.shift_cats.split(","))
    sub_specs = [s.strip() for s in args.substitutions.split(",") if s.strip()]
    builders = {
        "shift": lambda: build_shift_sub(model, tax, cfg, shift_cats),
        "broadcast_low_r": lambda: build_broadcast_sub(
            model, tax, diag_by_lh, args.r_thresh),
        "rank10_low_r": lambda: build_rank10_sub(
            model, tax, diag_by_lh, args.r_thresh, args.rank_k),
    }
    sub_configs = {}
    for s in sub_specs:
        b = builders[s]()
        sub_configs[s] = b  # (install_fn, heads)

    corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()]
    print(f"Substitutions to test: {sub_specs}")
    print(f"Corruptions to test:   {corruptions}")
    for s, (inst, heads) in sub_configs.items():
        print(f"  {s}: {len(heads)} heads "
              f"({'OK' if inst is not None else 'NO HEADS, will skip'})")

    # restore picker
    def restore_for(s):
        if s == "shift":
            return restore_kernel
        return restore_attn

    out = {
        "model": args.model,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "r_thresh": args.r_thresh,
        "rank_k": args.rank_k,
        "shift_cats": list(shift_cats),
        "results": {},     # corruption -> {baseline, sub_name -> {top1, n, delta}}
    }

    for c in corruptions:
        cfn = make_corruption(c)
        print(f"\ncorruption={c}")
        # baseline
        b_top1, b_n = eval_top1(model, loader, device, cfn, args.max_batches)
        print(f"  baseline:                 top1={b_top1*100:.2f}%  n={b_n}")
        out["results"][c] = {"baseline": {"top1": b_top1, "n": b_n}}
        for s, (inst, heads) in sub_configs.items():
            if inst is None:
                continue
            saved = inst()
            try:
                s_top1, s_n = eval_top1(model, loader, device, cfn,
                                          args.max_batches)
            finally:
                restore_for(s)(model, saved)
            delta = s_top1 - b_top1
            print(f"  sub {s:18s} top1={s_top1*100:.2f}%  delta={delta*100:+.2f}pp")
            out["results"][c][s] = {
                "top1": s_top1, "n": s_n, "delta_pp": delta * 100,
                "n_heads": len(heads),
            }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    # Cross-corruption summary
    print(f"\ncross-corruption deltas (sub - baseline, in pp)")
    header = "sub".ljust(20) + "  ".join(f"{c:>14s}" for c in corruptions)
    print(header)
    for s in sub_specs:
        row = s.ljust(20)
        for c in corruptions:
            r = out["results"][c].get(s)
            cell = "--" if r is None else f"{r['delta_pp']:+.2f}pp"
            row += f"  {cell:>14s}"
        print(row)


if __name__ == "__main__":
    main()
