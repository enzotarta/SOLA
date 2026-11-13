"""Escape mass of every (layer, head): the attention mass outside a KxK window
around the query.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


# Attention recording (vectorized, runs on a timm-style ViT)

class AttentionRecorder:
    """
    Records softmax attention weights from every block of a timm ViT,
    without monkey-patching the attention forward (which is fragile across
    timm versions). Instead we register a forward-pre-hook on each attention
    module to capture its input, then re-run qkv ourselves to compute the
    attention weights. The forward pass of the model is untouched.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.records: dict[int, list[torch.Tensor]] = {}
        self._handles = []
        self._inputs: dict[int, list[torch.Tensor]] = {}
        self._install()

    def _install(self):
        for li, block in enumerate(self.model.blocks):
            self.records[li] = []
            self._inputs[li] = []
            attn = block.attn
            self._handles.append(
                attn.register_forward_pre_hook(self._make_pre_hook(li))
            )
            self._handles.append(
                attn.register_forward_hook(self._make_post_hook(li, attn))
            )

    def _make_pre_hook(self, li: int):
        def hook(module, args):
            # args[0] is the token tensor (B, N, C)
            self._inputs[li].append(args[0].detach())
        return hook

    def _make_post_hook(self, li: int, attn: nn.Module):
        recorder = self
        def hook(module, args, output):
            # Compute attention weights from the saved input.
            x = recorder._inputs[li].pop(0)
            B, N, C = x.shape
            with torch.no_grad():
                qkv = attn.qkv(x).reshape(
                    B, N, 3, attn.num_heads, C // attn.num_heads
                ).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)
                if hasattr(attn, "q_norm") and attn.q_norm is not None:
                    q = attn.q_norm(q)
                if hasattr(attn, "k_norm") and attn.k_norm is not None:
                    k = attn.k_norm(k)
                a = (q @ k.transpose(-2, -1)) * attn.scale
                a = a.softmax(dim=-1)
            recorder.records[li].append(a.detach().cpu())
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def clear(self):
        for li in self.records:
            self.records[li] = []
            self._inputs[li] = []

    def stacked(self) -> dict[int, torch.Tensor]:
        return {li: torch.cat(self.records[li], dim=0) for li in self.records}


# Escape-mass computation (vectorized)

def compute_escape_mass(
    attn: torch.Tensor,
    grid: tuple[int, int],
    kernel_sizes: list[int],
    has_cls: bool,
) -> dict[int, torch.Tensor]:
    H, W = grid
    expected = H * W
    if has_cls:
        attn = attn[:, :, 1:, 1:]
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    B, heads, N, _ = attn.shape
    assert N == expected, f"After CLS handling, expected {expected} tokens, got {N}"

    coords = torch.tensor(
        [(i, j) for i in range(H) for j in range(W)],
        dtype=torch.long, device=attn.device,
    )
    delta = coords.unsqueeze(0) - coords.unsqueeze(1)
    abs_dy = delta[..., 0].abs()
    abs_dx = delta[..., 1].abs()

    out: dict[int, torch.Tensor] = {}
    for K in kernel_sizes:
        half = K // 2
        mask = ((abs_dy <= half) & (abs_dx <= half)).to(attn.dtype)
        inside = (attn * mask[None, None, :, :]).sum(dim=-1)
        escape = 1.0 - inside
        out[K] = escape.mean(dim=(0, 2))
    return out


# End-to-end sweep

@torch.no_grad()
def run_sweep(
    model: nn.Module,
    dataloader: Iterable,
    grid: tuple[int, int],
    kernel_sizes: list[int],
    n_batches: int,
    has_cls: bool,
    device: str = "cuda",
) -> dict:
    rec = AttentionRecorder(model)
    model.eval().to(device)

    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.num_heads

    sums = {li: {K: torch.zeros(n_heads) for K in kernel_sizes}
            for li in range(n_layers)}
    counts = {li: 0 for li in range(n_layers)}

    for bi, batch in enumerate(dataloader):
        if bi >= n_batches:
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device, non_blocking=True)
        rec.clear()
        _ = model(x)
        layer_attns = rec.stacked()
        for li, A in layer_attns.items():
            em = compute_escape_mass(A, grid, kernel_sizes, has_cls=has_cls)
            for K in kernel_sizes:
                sums[li][K] += em[K] * A.shape[0]
            counts[li] += A.shape[0]
        print(f"  batch {bi+1}/{n_batches} done")

    escape = {}
    for li in range(n_layers):
        escape[li] = {K: (sums[li][K] / counts[li]).tolist()
                      for K in kernel_sizes}

    return {
        "kernel_sizes": kernel_sizes,
        "escape": escape,
        "n_layers": n_layers,
        "n_heads_per_layer": [n_heads] * n_layers,
    }


# Pretty-printing

def print_summary(result: dict, top_k: int = 10):
    K_min = min(result["kernel_sizes"])
    flat = []
    for li, by_K in result["escape"].items():
        for hi, val in enumerate(by_K[K_min]):
            flat.append((val, int(li), hi))
    flat.sort()

    print(f"\nTop {top_k} most 'convolutional' heads (lowest escape at K={K_min}):")
    print(f"{'rank':>4}  {'layer':>5}  {'head':>4}  " +
          "  ".join(f"K={K:>2}" for K in result["kernel_sizes"]))
    for rank, (val, li, hi) in enumerate(flat[:top_k]):
        row = "  ".join(f"{result['escape'][li][K][hi]:.3f}"
                        for K in result["kernel_sizes"])
        print(f"{rank+1:>4}  {li:>5}  {hi:>4}  {row}")

    print(f"\nBottom {top_k} most 'global' heads (highest escape at K={K_min}):")
    for rank, (val, li, hi) in enumerate(flat[-top_k:][::-1]):
        row = "  ".join(f"{result['escape'][li][K][hi]:.3f}"
                        for K in result["kernel_sizes"])
        print(f"{rank+1:>4}  {li:>5}  {hi:>4}  {row}")

    print("\nLayer-mean escape mass (averaged over heads):")
    print(f"{'layer':>5}  " + "  ".join(f"K={K:>2}" for K in result["kernel_sizes"]))
    for li in result["escape"]:
        row = "  ".join(
            f"{sum(result['escape'][li][K]) / len(result['escape'][li][K]):.3f}"
            for K in result["kernel_sizes"])
        print(f"{li:>5}  {row}")


# Data loaders: all return a stream of normalized images

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def loader_tiny_imagenet(batch_size: int, image_size: int):
    """Auto-download tiny-imagenet (200 classes, ~250MB) via HuggingFace."""
    from datasets import load_dataset
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    print("Loading zh-plus/tiny-imagenet from HuggingFace (~250MB on first run)...")
    ds = load_dataset("zh-plus/tiny-imagenet", split="valid")

    tf = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    def collate(items):
        imgs = []
        for it in items:
            img = it["image"].convert("RGB")
            imgs.append(tf(img))
        return torch.stack(imgs)

    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, collate_fn=collate)


def loader_folder(data_dir: str, batch_size: int, image_size: int):
    """Read every image file in a directory tree (no class structure required)."""
    import torchvision.transforms as T
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG", "*.PNG", "*.webp"):
        paths.extend(Path(data_dir).rglob(ext))
    if not paths:
        raise FileNotFoundError(f"No images found under {data_dir}")
    print(f"Found {len(paths)} images under {data_dir}")

    tf = T.Compose([
        T.Resize(int(image_size * 256 / 224)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    class _DS(Dataset):
        def __len__(self): return len(paths)
        def __getitem__(self, i):
            return tf(Image.open(paths[i]).convert("RGB"))

    return DataLoader(_DS(), batch_size=batch_size, shuffle=False,
                      num_workers=2, pin_memory=True)


def loader_cifar(batch_size: int, image_size: int):
    """Auto-download CIFAR-10, upscale to image_size."""
    import torchvision.transforms as T
    from torchvision.datasets import CIFAR10
    from torch.utils.data import DataLoader

    tf = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    print("Loading CIFAR-10 (auto-downloads to ./data, ~170MB)...")
    ds = CIFAR10(root="./data", train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=2, pin_memory=True)


def loader_noise(batch_size: int, image_size: int, n_total: int = 1024):
    """Pure normal-noise images, to check the pipeline: content-dependent
    heads produce uninformative attention patterns on them."""
    print("WARNING: using random noise. Content-dependent attention behavior")
    print("         won't be observable. For pipeline sanity-check only.")

    class _NoiseLoader:
        def __iter__(self):
            for _ in range(n_total // batch_size):
                yield torch.randn(batch_size, 3, image_size, image_size)

    return _NoiseLoader()


def build_loader(source: str, batch_size: int, image_size: int,
                  data_dir: str | None = None):
    if source == "tiny-imagenet":
        return loader_tiny_imagenet(batch_size, image_size)
    if source == "folder":
        if not data_dir:
            raise ValueError("--data-dir required for --source folder")
        return loader_folder(data_dir, batch_size, image_size)
    if source == "cifar":
        return loader_cifar(batch_size, image_size)
    if source == "noise":
        return loader_noise(batch_size, image_size)
    raise ValueError(f"Unknown source: {source}")


# Per-model auto-detection of grid, patch size, CLS, register tokens

def detect_model_config(model: nn.Module, image_size: int) -> dict:
    """
    Inspect a timm model and figure out:
      - patch_size (from the patch embedder)
      - grid (image_size // patch_size, image_size // patch_size)
      - has_cls (does the model prepend a CLS token?)
      - num_extra_tokens (CLS + register tokens that come BEFORE the spatial grid)

    Returns dict suitable for passing to compute_escape_mass and run_sweep.
    """
    # timm models expose the patch size via patch_embed.proj (a Conv2d).
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"):
        ps = model.patch_embed.proj.kernel_size[0]
    elif hasattr(model, "patch_embed") and hasattr(model.patch_embed, "patch_size"):
        ps = model.patch_embed.patch_size
        if isinstance(ps, (tuple, list)):
            ps = ps[0]
    else:
        ps = 16  # fall back to ViT default

    H = W = image_size // ps

    # CLS token: timm exposes either model.cls_token or model.has_class_token
    has_cls = bool(getattr(model, "cls_token", None) is not None)

    # Register tokens (DINOv2 with registers, some modern ViTs have these)
    n_reg = 0
    if hasattr(model, "reg_token") and model.reg_token is not None:
        n_reg = model.reg_token.shape[1]
    elif hasattr(model, "num_reg_tokens"):
        n_reg = int(model.num_reg_tokens)

    n_extra = (1 if has_cls else 0) + n_reg
    return {
        "patch_size": ps, "grid": (H, W),
        "has_cls": has_cls, "n_reg": n_reg,
        "n_extra_tokens": n_extra,
    }


# Updated escape-mass computation with arbitrary leading-token count

def compute_escape_mass_v2(
    attn: torch.Tensor,
    grid: tuple[int, int],
    kernel_sizes: list[int],
    n_extra_tokens: int,
) -> dict[int, torch.Tensor]:
    """
    Like compute_escape_mass but supports any number of non-spatial leading
    tokens (CLS + register tokens) which we drop entirely before measuring
    locality on the spatial grid.
    """
    H, W = grid
    expected = H * W
    if n_extra_tokens > 0:
        attn = attn[:, :, n_extra_tokens:, n_extra_tokens:]
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    B, heads, N, _ = attn.shape
    if N != expected:
        raise ValueError(
            f"After stripping {n_extra_tokens} leading tokens, expected "
            f"{expected} spatial tokens but got {N}. Grid={grid}, model "
            f"may have a different patch size."
        )

    coords = torch.tensor(
        [(i, j) for i in range(H) for j in range(W)],
        dtype=torch.long, device=attn.device,
    )
    delta = coords.unsqueeze(0) - coords.unsqueeze(1)
    abs_dy = delta[..., 0].abs()
    abs_dx = delta[..., 1].abs()

    out: dict[int, torch.Tensor] = {}
    for K in kernel_sizes:
        half = K // 2
        mask = ((abs_dy <= half) & (abs_dx <= half)).to(attn.dtype)
        inside = (attn * mask[None, None, :, :]).sum(dim=-1)
        escape = 1.0 - inside
        out[K] = escape.mean(dim=(0, 2))
    return out


# v2 sweep that uses the auto-detected config

@torch.no_grad()
def run_sweep_v2(
    model: nn.Module,
    dataloader: Iterable,
    config: dict,
    kernel_sizes: list[int],
    n_batches: int,
    device: str = "cuda",
) -> dict:
    rec = AttentionRecorder(model)
    model.eval().to(device)

    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.num_heads

    sums = {li: {K: torch.zeros(n_heads) for K in kernel_sizes}
            for li in range(n_layers)}
    counts = {li: 0 for li in range(n_layers)}

    for bi, batch in enumerate(dataloader):
        if bi >= n_batches:
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device, non_blocking=True)
        rec.clear()
        _ = model(x)
        layer_attns = rec.stacked()
        for li, A in layer_attns.items():
            em = compute_escape_mass_v2(
                A, config["grid"], kernel_sizes,
                n_extra_tokens=config["n_extra_tokens"])
            for K in kernel_sizes:
                sums[li][K] += em[K] * A.shape[0]
            counts[li] += A.shape[0]
        print(f"  batch {bi+1}/{n_batches} done")

    rec.remove()  # don't leave hooks on the model when we're done

    escape = {}
    for li in range(n_layers):
        escape[li] = {K: (sums[li][K] / counts[li]).tolist()
                      for K in kernel_sizes}

    return {
        "kernel_sizes": kernel_sizes,
        "escape": escape,
        "n_layers": n_layers,
        "n_heads_per_layer": [n_heads] * n_layers,
        "config": {**config, "grid": list(config["grid"])},  # JSON-friendly
    }


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Run escape-mass diagnostic on one or more pretrained ViTs.")
    parser.add_argument("--models", nargs="+",
                        default=["deit_tiny_patch16_224"],
                        help="One or more timm model names; will be run "
                             "in sequence on the same data loader.")
    parser.add_argument("--source", default="tiny-imagenet",
                        choices=["tiny-imagenet", "folder", "cifar", "noise"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--kernel-sizes", type=int, nargs="+",
                        default=[3, 5, 7, 9, 11])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--out-dir", default=".",
                        help="Directory to write per-model JSONs and a "
                             "combined comparison JSON.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--summary-top-k", type=int, default=10)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import timm

    # One loader, so that every model sees the same images.
    loader = build_loader(args.source, args.batch_size, args.image_size,
                          data_dir=args.data_dir)

    # Pre-fetch enough batches into memory so the loader can be replayed
    # for each model. (Safer than re-iterating a stateful loader.)
    print(f"Caching {args.batches} batches of {args.batch_size} images "
          f"for replay across {len(args.models)} model(s)...")
    cached = []
    for bi, batch in enumerate(loader):
        if bi >= args.batches:
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        cached.append(x)
    print(f"  Cached {len(cached)} batches "
          f"({sum(b.shape[0] for b in cached)} images total)\n")

    all_results = {}
    for mi, model_name in enumerate(args.models):
        print(f"Model {mi+1}/{len(args.models)}: {model_name}")
        try:
            model = timm.create_model(model_name, pretrained=True)
        except Exception as e:
            print(f"  FAILED to load model: {e}")
            continue

        config = detect_model_config(model, args.image_size)
        print(f"  patch_size = {config['patch_size']}")
        print(f"  grid       = {config['grid']}")
        print(f"  has_cls    = {config['has_cls']}")
        print(f"  n_reg      = {config['n_reg']}")
        print(f"  n_extra    = {config['n_extra_tokens']}")
        print(f"  layers     = {len(model.blocks)}")
        print(f"  heads      = {model.blocks[0].attn.num_heads}")

        # Replay cached images.
        def replay():
            for x in cached:
                yield x

        try:
            result = run_sweep_v2(
                model, replay(), config=config,
                kernel_sizes=args.kernel_sizes,
                n_batches=len(cached), device=device)
        except Exception as e:
            print(f"  FAILED during sweep: {e}")
            del model
            continue

        # Free GPU memory before loading next model.
        model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        out_file = out_dir / f"escape_{model_name}.json"
        out_file.write_text(json.dumps(result, indent=2))
        print(f"  Wrote {out_file}")

        all_results[model_name] = result

        print()
        print_summary(result, top_k=args.summary_top_k)
        print()

    # Combined comparison file
    if all_results:
        combined = {
            "models": list(all_results.keys()),
            "kernel_sizes": args.kernel_sizes,
            "results": all_results,
        }
        combined_path = out_dir / "escape_comparison.json"
        combined_path.write_text(json.dumps(combined, indent=2))
        print(f"\nWrote combined comparison to {combined_path}")

        # Side-by-side layer-mean summary
        print("Side-by-side layer-mean escape mass (averaged over heads)")
        K_focus = args.kernel_sizes[len(args.kernel_sizes) // 2]  # middle K
        print(f"\nAt K={K_focus}:\n")
        max_layers = max(r["n_layers"] for r in all_results.values())
        print(f"{'layer':>5}  " +
              "  ".join(f"{m:>26.26}" for m in all_results))
        for li in range(max_layers):
            cells = []
            for m, r in all_results.items():
                if li < r["n_layers"]:
                    vals = r["escape"][li][K_focus] if K_focus in r["escape"][li] else r["escape"][str(li)][str(K_focus)]
                    cells.append(f"{sum(vals)/len(vals):>26.3f}")
                else:
                    cells.append(f"{'-':>26}")
            print(f"{li:>5}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
