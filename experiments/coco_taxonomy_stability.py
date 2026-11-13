"""Change of category per head between ImageNet and COCO calibration batches."""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

SOLA_CODE = Path(__file__).resolve().parent.parent / "sola"
sys.path.insert(0, str(SOLA_CODE))
from diagnostics import head_diagnostics, classify  # noqa: E402
from data import imagenet_val_loader, detect_grid_and_extra, capture_attention  # noqa: E402


class FlatImageFolder(Dataset):
    """Plain directory of images, no class subfolders (COCO val2017 layout)."""

    def __init__(self, root, image_size=224, n_max=None):
        self.root = Path(root)
        exts = (".jpg", ".jpeg", ".png")
        self.files = sorted(
            p for p in self.root.iterdir() if p.suffix.lower() in exts
        )
        if n_max is not None:
            self.files = self.files[:n_max]
        try:
            bicubic = T.InterpolationMode.BICUBIC
        except AttributeError:
            bicubic = 3
        self.tf = T.Compose(
            [
                T.Resize(256, interpolation=bicubic),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.tf(img), 0


def classify_batch(model, cal_x, grid, n_extra, n_layers, n_heads):
    attns = capture_attention(model, cal_x, n_extra)
    cat = {}
    for li in range(n_layers):
        a = attns[li].numpy()
        for hi in range(n_heads):
            _, d = head_diagnostics(a[:, hi], *grid, K=5)
            cat[(li, hi)] = classify(d)
    return cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="vit_base_patch16_224.augreg2_in21k_ft_in1k"
    )
    ap.add_argument("--imagenet-val-dir", required=True)
    ap.add_argument("--coco-dir", required=True, help="COCO val2017 image dir")
    ap.add_argument("--n-cal", type=int, default=32)
    ap.add_argument("--n-repeats", type=int, default=4,
                     help="independent calibration batches per dataset; flip rates are reported per batch pair")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device}")
    model = timm.create_model(args.model, pretrained=True).eval().to(device)
    grid, n_extra = detect_grid_and_extra(model)
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.num_heads
    print(f"  grid {grid}, {n_layers} layers, {n_heads} heads/layer")

    in_loader = imagenet_val_loader(args.imagenet_val_dir, args.n_cal, image_size=224)
    coco_ds = FlatImageFolder(args.coco_dir, image_size=224, n_max=args.n_cal * args.n_repeats)
    coco_loader = DataLoader(coco_ds, batch_size=args.n_cal, shuffle=False, num_workers=2)

    in_iter = iter(in_loader)
    coco_iter = iter(coco_loader)

    flip_counts = Counter()
    total_heads = n_layers * n_heads
    per_repeat_flip_rate = []
    imagenet_cats_by_repeat = []
    coco_cats_by_repeat = []

    for rep in range(args.n_repeats):
        in_x, _ = next(in_iter)
        coco_x, _ = next(coco_iter)
        in_x, coco_x = in_x.to(device), coco_x.to(device)

        cat_in = classify_batch(model, in_x, grid, n_extra, n_layers, n_heads)
        cat_coco = classify_batch(model, coco_x, grid, n_extra, n_layers, n_heads)
        imagenet_cats_by_repeat.append(cat_in)
        coco_cats_by_repeat.append(cat_coco)

        n_flip = sum(1 for k in cat_in if cat_in[k] != cat_coco[k])
        rate = n_flip / total_heads
        per_repeat_flip_rate.append(rate)
        for k in cat_in:
            if cat_in[k] != cat_coco[k]:
                flip_counts[(cat_in[k], cat_coco[k])] += 1
        print(f"  repeat {rep}: {n_flip}/{total_heads} heads flip category "
              f"({100 * rate:.1f}%)")

    print(f"\nMean flip rate over {args.n_repeats} calibration-batch pairs: "
          f"{100 * np.mean(per_repeat_flip_rate):.1f}% "
          f"(std {100 * np.std(per_repeat_flip_rate):.1f}%)")
    print("Most common flip directions (imagenet_cat -> coco_cat):")
    for (a, b), n in flip_counts.most_common(10):
        print(f"  {a:14s} -> {b:14s}: {n}")

    # majority-vote category per head across repeats, for a summary stability number
    def majority(cats_list):
        out = {}
        for k in cats_list[0]:
            votes = Counter(c[k] for c in cats_list)
            out[k] = votes.most_common(1)[0][0]
        return out

    maj_in = majority(imagenet_cats_by_repeat)
    maj_coco = majority(coco_cats_by_repeat)
    n_flip_maj = sum(1 for k in maj_in if maj_in[k] != maj_coco[k])
    print(f"\nMajority-vote (across {args.n_repeats} batches) flip rate: "
          f"{n_flip_maj}/{total_heads} ({100 * n_flip_maj / total_heads:.1f}%)")

    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "per_repeat_flip_rate": per_repeat_flip_rate,
                    "majority_flip_rate": n_flip_maj / total_heads,
                    "flip_directions": {f"{a}->{b}": n for (a, b), n in flip_counts.items()},
                },
                f, indent=2,
            )
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
