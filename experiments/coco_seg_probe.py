"""Linear segmentation probe on COCO-Stuff, on frozen patch tokens, evaluated with
and without the full-coverage substitution.
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

SOLA_CODE = Path(__file__).resolve().parent.parent / "sola"
sys.path.insert(0, str(SOLA_CODE))
from diagnostics import head_diagnostics, classify  # noqa: E402
from substitutions import (  # noqa: E402
    install, restore, kernel_from_mean, adaptive_layer_forwards,
)
from data import detect_grid_and_extra, capture_attention  # noqa: E402

IGNORE_INDEX = 255
N_CLASSES = 184  # COCO-Stuff raw label ids observed 0-183, 255 = unlabeled (ignored)


class CocoStuffSeg(Dataset):
    def __init__(self, img_dir, mask_dir, image_size=224, n_max=None, seed=0):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        stems = sorted(p.stem for p in self.mask_dir.glob("*.png"))
        rng = np.random.RandomState(seed)
        rng.shuffle(stems)
        if n_max is not None:
            stems = stems[:n_max]
        self.stems = stems
        self.image_size = image_size
        try:
            bicubic = T.InterpolationMode.BICUBIC
        except AttributeError:
            bicubic = 3
        self.img_tf = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=bicubic),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        img = Image.open(self.img_dir / f"{stem}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{stem}.png")
        img_t = self.img_tf(img)
        mask_t = torch.from_numpy(
            np.array(mask.resize((self.image_size, self.image_size), Image.NEAREST))
        ).long()
        return img_t, mask_t


def patchify_labels(mask, patch=16):
    """Majority-vote label per patch. mask: (B, H, W) -> (B, H/patch, W/patch)."""
    B, H, W = mask.shape
    gh, gw = H // patch, W // patch
    out = torch.full((B, gh, gw), IGNORE_INDEX, dtype=torch.long)
    for b in range(B):
        for py in range(gh):
            for px in range(gw):
                block = mask[b, py * patch:(py + 1) * patch, px * patch:(px + 1) * patch]
                vals, counts = torch.unique(block, return_counts=True)
                out[b, py, px] = vals[counts.argmax()]
    return out


@torch.no_grad()
def extract_patch_tokens(model, x, n_extra):
    feats = model.forward_features(x)
    if feats.dim() == 3:
        return feats[:, n_extra:]
    raise RuntimeError(f"unexpected feature shape {feats.shape}")


def train_probe(model, loader, n_extra, grid, device, n_classes, epochs=3, lr=1e-3):
    H, W = grid
    probe = nn.Linear(model.embed_dim if hasattr(model, "embed_dim") else model.num_features,
                       n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for ep in range(epochs):
        tot_loss, nb = 0.0, 0
        for x, mask in loader:
            x = x.to(device)
            plabels = patchify_labels(mask, patch=224 // H).to(device)  # (B, H, W)
            with torch.no_grad():
                feats = extract_patch_tokens(model, x, n_extra)  # (B, H*W, C)
            logits = probe(feats).transpose(1, 2).reshape(-1, n_classes, H, W)
            loss = F.cross_entropy(logits, plabels, ignore_index=IGNORE_INDEX)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            nb += 1
        print(f"  probe epoch {ep}: loss {tot_loss / max(nb,1):.4f}")
    return probe


@torch.no_grad()
def eval_miou(model, probe, loader, n_extra, grid, device, n_classes):
    H, W = grid
    inter = torch.zeros(n_classes)
    union = torch.zeros(n_classes)
    for x, mask in loader:
        x = x.to(device)
        plabels = patchify_labels(mask, patch=224 // H).to(device)
        feats = extract_patch_tokens(model, x, n_extra)
        logits = probe(feats).transpose(1, 2).reshape(-1, n_classes, H, W)
        pred = logits.argmax(1)
        valid = plabels != IGNORE_INDEX
        for c in range(n_classes):
            p = (pred == c) & valid
            g = (plabels == c) & valid
            inter[c] += (p & g).sum().item()
            union[c] += (p | g).sum().item()
    present = union > 0
    iou = inter[present] / union[present].clamp(min=1)
    return iou.mean().item(), int(present.sum().item())


def build_adaptive_substitution(model, cal_loader, grid, n_extra, device, k_floor=10):
    """Same taxonomy and full-coverage substitution as reproduce.py, computed on
    a calibration batch drawn from `cal_loader` (COCO images in this script)."""
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.num_heads
    cal_x, _ = next(iter(cal_loader))
    cal_x = cal_x.to(device)
    attns = capture_attention(model, cal_x, n_extra)
    kappa_bar, diag, cat = {}, {}, {}
    for li in range(n_layers):
        a = attns[li].numpy()
        for hi in range(n_heads):
            ker, d = head_diagnostics(a[:, hi], *grid, K=5)
            kappa_bar[(li, hi)] = ker
            diag[(li, hi)] = d
            cat[(li, hi)] = classify(d)
    shift_cats = {"delta_shift", "soft_shift", "self_only"}
    shift_heads = [(l, h) for (l, h), c in cat.items() if c in shift_cats]
    kernel_map = {(l, h): kernel_from_mean(kappa_bar[(l, h)]) for (l, h) in shift_heads}
    head_to_k = {}
    for (l, h), c in cat.items():
        if (l, h) in shift_heads:
            continue
        head_to_k[(l, h)] = max(k_floor, int(math.ceil(diag[(l, h)]["eff_rank_90"])))
    layer_fwds = adaptive_layer_forwards(
        model, kernel_map=kernel_map, head_to_k_map=head_to_k,
        grid=grid, n_extra=n_extra, k_floor=k_floor,
    )
    return layer_fwds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch16_224.augreg2_in21k_ft_in1k")
    ap.add_argument("--coco-train-img", required=True)
    ap.add_argument("--coco-train-mask", required=True)
    ap.add_argument("--coco-val-img", required=True)
    ap.add_argument("--coco-val-mask", required=True)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-cal", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device}")
    model = timm.create_model(args.model, pretrained=True).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    grid, n_extra = detect_grid_and_extra(model)
    print(f"  grid {grid}, n_extra {n_extra}")

    train_ds = CocoStuffSeg(args.coco_train_img, args.coco_train_mask, n_max=args.n_train, seed=0)
    val_ds = CocoStuffSeg(args.coco_val_img, args.coco_val_mask, n_max=args.n_val, seed=1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    cal_loader = DataLoader(train_ds, batch_size=args.n_cal, shuffle=True, num_workers=2)

    print("Training linear segmentation probe on frozen baseline features...")
    probe = train_probe(model, train_loader, n_extra, grid, device, N_CLASSES, epochs=args.epochs)

    print("Evaluating baseline (unsubstituted) mIoU on val subset...")
    miou_base, n_present = eval_miou(model, probe, val_loader, n_extra, grid, device, N_CLASSES)
    print(f"  baseline mIoU: {miou_base:.4f} ({n_present} classes present)")

    print("Building adaptive (Cor. 3.1, 100% coverage) substitution from a COCO calibration batch...")
    layer_fwds = build_adaptive_substitution(model, cal_loader, grid, n_extra, device)
    saved = install(model, layer_fwds)
    print("Evaluating SOLA-substituted mIoU on the same val subset, same frozen probe...")
    miou_sub, n_present_sub = eval_miou(model, probe, val_loader, n_extra, grid, device, N_CLASSES)
    restore(model, saved)
    print(f"  substituted mIoU: {miou_sub:.4f} ({n_present_sub} classes present)")
    print(f"\nDelta mIoU (substituted - baseline): {miou_sub - miou_base:+.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {"model": args.model, "miou_baseline": miou_base,
                 "miou_substituted": miou_sub, "n_present": n_present,
                 "n_train": args.n_train, "n_val": args.n_val},
                f, indent=2,
            )
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
