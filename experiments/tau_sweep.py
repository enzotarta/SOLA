"""Sweep tau in the rank rule of the full-coverage substitution and report fidelity and mean k / d_h."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

SOLA_CODE = Path(__file__).resolve().parent.parent / "sola"
sys.path.insert(0, str(SOLA_CODE))
from diagnostics import head_diagnostics, classify  # noqa: E402
from substitutions import (  # noqa: E402
    install, restore, kernel_from_mean, adaptive_layer_forwards,
)
from data import detect_grid_and_extra, capture_attention  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shift_substitution import image_embedding  # noqa: E402

SHIFT_CATS = {"delta_shift", "soft_shift", "self_only"}


def build_loader(val_dir, batch_size, siglip_norm, image_size=224):
    try:
        bicubic = T.InterpolationMode.BICUBIC
    except AttributeError:
        bicubic = 3
    if siglip_norm:
        tf = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=bicubic),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        tf = T.Compose(
            [
                T.Resize(256, interpolation=bicubic),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    ds = ImageFolder(val_dir, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                       pin_memory=True), ds.classes


def per_head_eff_rank_curve(attn_batch):
    """attn_batch: (n_img, N, N) numpy. Returns per-image cumulative energy
    curves (cumsum(s^2)/total) so eff_rank at any tau is a cheap searchsorted."""
    curves = []
    for i in range(attn_batch.shape[0]):
        s = np.linalg.svd(attn_batch[i], compute_uv=False)
        cum = np.cumsum(s ** 2)
        curves.append(cum / cum[-1])
    return curves  # list of 1D arrays, length N each


def eff_rank_at_tau(curves, tau):
    ks = [int(np.searchsorted(c, tau)) + 1 for c in curves]
    return float(np.mean(ks))


def build_taxonomy(model, cal_x, grid, n_extra):
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.num_heads
    attns = capture_attention(model, cal_x, n_extra)
    kappa_bar, cat, curves = {}, {}, {}
    for li in range(n_layers):
        a = attns[li].numpy()
        for hi in range(n_heads):
            ker, d = head_diagnostics(a[:, hi], *grid, K=5)
            kappa_bar[(li, hi)] = ker
            cat[(li, hi)] = classify(d)
            curves[(li, hi)] = per_head_eff_rank_curve(a[:, hi])
    return kappa_bar, cat, curves, n_layers, n_heads


def run_one_tau(model, kappa_bar, cat, curves, tau, grid, n_extra, k_floor, d_h):
    shift_heads = [(l, h) for (l, h), c in cat.items() if c in SHIFT_CATS]
    kernel_map = {(l, h): kernel_from_mean(kappa_bar[(l, h)]) for (l, h) in shift_heads}
    head_to_k = {}
    ratios = []
    for (l, h), c in cat.items():
        if (l, h) in shift_heads:
            continue
        k = max(k_floor, int(math.ceil(eff_rank_at_tau(curves[(l, h)], tau))))
        head_to_k[(l, h)] = k
        ratios.append(min(k, d_h) / d_h if k < d_h else 1.0)
    layer_fwds = adaptive_layer_forwards(
        model, kernel_map=kernel_map, head_to_k_map=head_to_k,
        grid=grid, n_extra=n_extra, k_floor=k_floor,
    )
    mean_ratio = float(np.mean(ratios)) if ratios else float("nan")
    return layer_fwds, mean_ratio


@torch.no_grad()
def cls_cos_sim(model, loader, layer_fwds, n_max, device):
    total_cos, total_n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        base = image_embedding(model, x)
        saved = install(model, layer_fwds)
        sub = image_embedding(model, x)
        restore(model, saved)
        cos = F.cosine_similarity(base, sub, dim=-1)
        total_cos += cos.sum().item()
        total_n += cos.numel()
        if total_n >= n_max:
            break
    return total_cos / max(total_n, 1)


@torch.no_grad()
def siglip_zeroshot_top1(model, loader, text_embeds, logit_scale, logit_bias,
                          layer_fwds, n_max, device):
    if layer_fwds is not None:
        saved = install(model, layer_fwds)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        emb = model(x)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        logits = emb @ text_embeds.T * logit_scale + logit_bias
        pred = logits.argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
        if total >= n_max:
            break
    if layer_fwds is not None:
        restore(model, saved)
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dinov2", "siglip"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--imagenet-val-dir", required=True)
    ap.add_argument("--n-cal", type=int, default=32)
    ap.add_argument("--n-eval", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--k-floor", type=int, default=10)
    ap.add_argument("--d-h", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=None,
                     help="defaults to 224, or the model's native patch_embed.img_size if larger")
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.90, 0.93, 0.95, 0.97, 0.99])
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--class-names", default="results/imagenet_classes.txt",
                     help="human-readable class names in the same sorted-wnid order as "
                          "ImageFolder, for SigLIP zero-shot prompts")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device} (mode={args.mode})")
    model = timm.create_model(args.model, pretrained=True).eval().to(device)
    image_size = args.image_size
    if image_size is None:
        native = getattr(model.patch_embed, "img_size", None)
        image_size = int(native[0]) if native is not None else 224
    grid, n_extra = detect_grid_and_extra(model, image_size=image_size)
    print(f"  image_size {image_size}, grid {grid}, n_extra {n_extra}")

    siglip_norm = args.mode == "siglip"
    cal_loader, classes = build_loader(args.imagenet_val_dir, args.n_cal, siglip_norm, image_size)
    eval_loader, _ = build_loader(args.imagenet_val_dir, args.batch_size, siglip_norm, image_size)

    cal_x, _ = next(iter(cal_loader))
    cal_x = cal_x.to(device)
    print("Building taxonomy + per-head eff-rank curves from calibration batch...")
    kappa_bar, cat, curves, n_layers, n_heads = build_taxonomy(model, cal_x, grid, n_extra)
    print(f"  {n_layers} layers x {n_heads} heads, "
          f"{sum(1 for c in cat.values() if c in SHIFT_CATS)} shift heads")

    text_embeds = logit_scale = logit_bias = None
    if args.mode == "siglip":
        from transformers import AutoModel, AutoTokenizer
        print("Loading HF SigLIP text tower...")
        hf = AutoModel.from_pretrained("google/siglip-base-patch16-224").eval().to(device)
        tok = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
        logit_scale = float(hf.logit_scale.exp().item())
        logit_bias = float(hf.logit_bias.item())
        with open(args.class_names) as f:
            human_names = [line.strip() for line in f if line.strip()]
        assert len(human_names) == len(classes), (
            f"class-names file has {len(human_names)} entries, "
            f"but ImageFolder found {len(classes)} classes")
        prompts = [f"This is a photo of a {c}." for c in human_names]
        with torch.no_grad():
            enc = tok(prompts, padding="max_length", return_tensors="pt").to(device)
            text_embeds = hf.text_model(**enc).pooler_output
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        base_metric = siglip_zeroshot_top1(
            model, eval_loader, text_embeds, logit_scale, logit_bias, None,
            args.n_eval, device)
        print(f"Baseline zero-shot top-1: {100*base_metric:.2f}%")
    else:
        base_metric = 1.0  # cos-sim to self is trivially 1.0, not needed
        print("Baseline CLS cos-sim to itself is 1.0 by definition.")

    results = []
    for tau in args.taus:
        layer_fwds, mean_ratio = run_one_tau(
            model, kappa_bar, cat, curves, tau, grid, n_extra, args.k_floor, args.d_h)
        if args.mode == "siglip":
            metric = siglip_zeroshot_top1(
                model, eval_loader, text_embeds, logit_scale, logit_bias,
                layer_fwds, args.n_eval, device)
            print(f"tau={tau:.2f}  zero-shot top-1={100*metric:.2f}%  "
                  f"(delta {100*(metric-base_metric):+.2f}%)  "
                  f"mean k/d_h={mean_ratio:.3f}")
        else:
            metric = cls_cos_sim(model, eval_loader, layer_fwds, args.n_eval, device)
            print(f"tau={tau:.2f}  CLS cos-sim={metric:.4f}  "
                  f"mean k/d_h={mean_ratio:.3f}")
        results.append({"tau": tau, "metric": metric, "mean_k_over_dh": mean_ratio})

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "mode": args.mode,
                       "baseline_metric": base_metric, "sweep": results}, f, indent=2)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
