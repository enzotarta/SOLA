"""Zero-shot ImageNet-1k with SigLIP under the full-coverage substitution."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from kernel_substitution import kernel_from_diagnostics_mean
from adaptive_rank_eval import (
    install_adaptive_rank, restore_forwards,
)


def build_text_embeddings(class_names, hf_text_model, hf_tokenizer,
                            device,
                            prompt_template="This is a photo of a {}."):
    # Single prompt template, the form used on the HF model card.
    prompts = [prompt_template.format(c) for c in class_names]
    enc = hf_tokenizer(prompts, padding="max_length",
                          return_tensors="pt").to(device)
    with torch.no_grad():
        out = hf_text_model(**enc)
        text_embeds = out.pooler_output
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    return text_embeds


@torch.no_grad()
def zeroshot_hits(vision_model, loader, text_embeds,
                   logit_scale, logit_bias, n_max, device):
    """Boolean vector, one entry per image: is the zero-shot prediction correct."""
    hits = []
    total = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        emb = vision_model(x)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        logits = emb @ text_embeds.T * logit_scale + logit_bias
        hits.append((logits.argmax(-1) == y).cpu().numpy())
        total += y.numel()
        if total >= n_max:
            break
    return np.concatenate(hits)


def paired_delta(base, sub):
    """Accuracy change in points with the standard error of the paired difference."""
    n = len(base)
    lost = int((base & ~sub).sum())
    gained = int((~base & sub).sum())
    se = np.sqrt(max(lost + gained - (lost - gained) ** 2 / n, 0.0)) / n * 100
    return {"delta_top1_pct": 100 * float(sub.mean() - base.mean()),
            "lost": lost, "gained": gained, "paired_se_pct": float(se)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy",
                    default="results/taxonomy_siglip.json")
    ap.add_argument("--svd-spectrum",
                    default="results/svd_spectrum_siglip.json")
    ap.add_argument("--class-names",
                    default="results/imagenet_classes.txt")
    ap.add_argument("--imagenet-val-dir",
                    default="data/imagenet/val")
    ap.add_argument("--n-eval", type=int, default=10000)
    ap.add_argument("--order", choices=["class", "random"], default="class",
                    help="class: the first images in class order; random: a seeded uniform draw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--k-floor", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.9,
                    help="Spectral capture threshold; selects "
                         "eff_rank_<int(100*tau)> from svd_spectrum.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import timm
    import math
    from transformers import AutoModel, AutoTokenizer
    from shift_substitution import detect_model_config

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # SigLIP preprocessing: resize directly to 224x224 with mean=std=0.5, what
    # the model expects. The ImageNet loader of the other scripts differs.
    def build_siglip_val_loader(batch_size, val_dir, num_workers=4):
        import torchvision.transforms as T
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader, Subset
        try:
            from torchvision.transforms import InterpolationMode
            bicubic = InterpolationMode.BICUBIC
        except Exception:
            bicubic = 3
        tf = T.Compose([
            T.Resize((224, 224), interpolation=bicubic),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        ds = ImageFolder(val_dir, transform=tf)
        n_classes = len(ds.classes)
        if args.order == "random":
            idx = np.random.RandomState(args.seed).permutation(len(ds))[:args.n_eval]
            ds = Subset(ds, sorted(idx.tolist()))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
        return loader, n_classes

    # text encoder (HF)
    print("Loading HF SigLIP for text + logit-scale ...", flush=True)
    hf = AutoModel.from_pretrained(
        "google/siglip-base-patch16-224").eval().to(device)
    tok = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
    logit_scale = float(hf.logit_scale.exp().item())
    logit_bias = float(hf.logit_bias.item())
    print(f"  logit_scale={logit_scale:.4f}, "
          f"logit_bias={logit_bias:.4f}", flush=True)

    # vision encoder (timm; same weights, supports our substitutions)
    print("Loading timm SigLIP vision encoder ...", flush=True)
    vis = timm.create_model(
        "vit_base_patch16_siglip_224.webli", pretrained=True).eval().to(device)
    cfg = detect_model_config(vis, image_size=224)

    # numerical agreement check on a real preprocessed batch
    print("Verifying timm <-> HF vision agreement ...", flush=True)
    _check_loader, _ = build_siglip_val_loader(
        4, args.imagenet_val_dir, num_workers=0)
    _check_x, _ = next(iter(_check_loader))
    _check_x = _check_x.to(device)
    with torch.no_grad():
        timm_out = vis(_check_x)
        hf_out = hf.vision_model(_check_x).pooler_output
    diff = (timm_out - hf_out).abs().max().item()
    print(f"  timm vs HF max abs diff: {diff:.4e}", flush=True)
    # the two encoders differ by 8e-3 to 1.1e-2 on this preprocessing; other
    # weights under the same model name (SigLIP 2) differ by more than 1
    if diff > 5e-2:
        print("  WARNING: timm and HF encoders disagree.", flush=True)

    # text embeddings (1000 class prompts)
    class_names = [l.strip() for l in
                    Path(args.class_names).read_text().splitlines()
                    if l.strip()]
    assert len(class_names) == 1000, \
        f"got {len(class_names)} classes, expected 1000"
    print(f"Building text embeddings for {len(class_names)} classes ...",
          flush=True)
    text_embeds = build_text_embeddings(
        class_names, hf.text_model, tok, device)

    # Free HF memory we no longer need
    del hf
    torch.cuda.empty_cache()

    # baseline zero-shot
    print(f"Baseline zero-shot top-1 (n_max={args.n_eval}) ...",
          flush=True)
    loader, _ = build_siglip_val_loader(
        args.batch_size, args.imagenet_val_dir, num_workers=4)
    base_hits = zeroshot_hits(
        vis, loader, text_embeds, logit_scale, logit_bias,
        args.n_eval, device)
    base_top1, n_b = float(base_hits.mean()), len(base_hits)
    print(f"  baseline: top1={base_top1*100:.3f}%  n={n_b}", flush=True)

    # adaptive SOLA
    print("Adaptive SOLA zero-shot top-1 ...", flush=True)
    tax = json.loads(Path(args.taxonomy).read_text())
    rows = tax["results"]
    res_by_lh = {(r["layer"], r["head"]): r for r in rows}
    sd = json.loads(Path(args.svd_spectrum).read_text())
    er_field = f"eff_rank_{int(round(100 * args.tau))}"
    print(f"  Using {er_field} (tau={args.tau})", flush=True)
    if er_field not in sd["per_head"][0]:
        raise KeyError(
            f"{er_field} not in svd_spectrum; available: "
            f"{[k for k in sd['per_head'][0] if k.startswith('eff_rank_')]}")
    er_by_lh = {(r["layer"], r["head"]): r[er_field]
                 for r in sd["per_head"]}

    SHIFT_CATS = {"delta_shift", "soft_shift", "self_only"}
    shift_heads = [(r["layer"], r["head"]) for r in rows
                   if r["category"] in SHIFT_CATS
                   and res_by_lh[(r["layer"], r["head"])].get("mean_kernel")
                   is not None]
    rank_heads_with_k = {}
    for r in rows:
        lh = (r["layer"], r["head"])
        if lh in set(shift_heads):
            continue
        er = er_by_lh.get(lh)
        if er is None:
            continue
        rank_heads_with_k[lh] = max(args.k_floor, int(math.ceil(er)))
    print(f"  shifts={len(shift_heads)}, rank-k={len(rank_heads_with_k)} "
          f"(total={len(shift_heads)+len(rank_heads_with_k)}/{len(rows)})",
          flush=True)

    kernel_map = {(li, hi): kernel_from_diagnostics_mean(
                       np.asarray(res_by_lh[(li, hi)]["mean_kernel"]))
                   for (li, hi) in shift_heads}

    loader, _ = build_siglip_val_loader(
        args.batch_size, args.imagenet_val_dir, num_workers=4)
    saved = install_adaptive_rank(
        vis, kernel_map, rank_heads_with_k,
        cfg["grid"], cfg["n_extra_tokens"])
    sub_hits = zeroshot_hits(
        vis, loader, text_embeds, logit_scale, logit_bias,
        args.n_eval, device)
    restore_forwards(vis, saved)
    sub_top1, n_s = float(sub_hits.mean()), len(sub_hits)
    paired = paired_delta(base_hits, sub_hits)
    print(f"  adaptive: top1={sub_top1*100:.3f}%  n={n_s}  "
          f"delta ={paired['delta_top1_pct']:+.3f} +/- {paired['paired_se_pct']:.3f}%",
          flush=True)

    k_vals = list(rank_heads_with_k.values())
    out = {
        "model": "vit_base_patch16_siglip_224.webli",
        "tau": args.tau,
        "k_floor": args.k_floor,
        "order": args.order,
        "seed": args.seed if args.order == "random" else None,
        "taxonomy": args.taxonomy,
        "svd_spectrum": args.svd_spectrum,
        "n_eval": n_b,
        "logit_scale": logit_scale,
        "logit_bias": logit_bias,
        "timm_hf_max_diff": diff,
        "n_shifts": len(shift_heads),
        "n_rank_subs": len(rank_heads_with_k),
        "n_total_heads": len(rows),
        "k_min": min(k_vals) if k_vals else None,
        "k_median": sorted(k_vals)[len(k_vals) // 2] if k_vals else None,
        "k_max": max(k_vals) if k_vals else None,
        "k_mean": sum(k_vals) / len(k_vals) if k_vals else None,
        "baseline_top1": base_top1,
        "adaptive_top1": sub_top1,
        **paired,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
