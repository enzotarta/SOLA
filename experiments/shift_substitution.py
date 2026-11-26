"""Replace candidate heads with a literal shift of V, and the shared data loaders.

Also defines the ImageNet val and tiny-ImageNet loaders used by the other
scripts. Both are unshuffled, so a run on n images uses the first n in class
order.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sola"))
from data import eval_transform  # noqa: E402


# Model config detection (same as escape_mass_sweep)

def image_embedding(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Image-level representation used by the CLS cos-sim metric.

    By default the first output token of forward_features, as in the stored
    results. That is the class token for every model with one; SigLIP has no
    class token, so for SigLIP it is the first patch token. With
    SOLA_EMBEDDING=pooled, models without a class token use the output of
    their pooling head instead (the MAP head for SigLIP).
    """
    feats = model.forward_features(x)
    if feats.ndim == 2:
        return feats
    if (os.environ.get("SOLA_EMBEDDING", "first_token") == "pooled"
            and getattr(model, "cls_token", None) is None):
        return model.forward_head(feats, pre_logits=True)
    return feats[:, 0]


def detect_model_config(model: nn.Module, image_size: int) -> dict:
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"):
        ps = model.patch_embed.proj.kernel_size[0]
    else:
        ps = 16
    H = W = image_size // ps
    has_cls = bool(getattr(model, "cls_token", None) is not None)
    n_reg = 0
    if hasattr(model, "reg_token") and model.reg_token is not None:
        n_reg = model.reg_token.shape[1]
    elif hasattr(model, "num_reg_tokens"):
        n_reg = int(model.num_reg_tokens)
    n_extra = (1 if has_cls else 0) + n_reg
    return {"patch_size": ps, "grid": (H, W),
            "has_cls": has_cls, "n_reg": n_reg, "n_extra_tokens": n_extra}


# extract dominant offset and consistency for a head

class _AttnGrabber:
    """Captures one specific (layer, head)'s attention weights for offset extraction."""
    def __init__(self, model, layer, head):
        self.layer = layer
        self.head = head
        self.attn_records = []
        self.input_records = []
        attn = model.blocks[layer].attn
        self._h1 = attn.register_forward_pre_hook(self._pre)
        self._h2 = attn.register_forward_hook(self._post(attn))

    def _pre(self, module, args):
        self.input_records.append(args[0].detach())

    def _post(self, attn):
        grabber = self
        def hook(module, args, output):
            x = grabber.input_records.pop(0)
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
            grabber.attn_records.append(a[:, grabber.head].detach().cpu())
        return hook

    def remove(self):
        self._h1.remove(); self._h2.remove()


@torch.no_grad()
def extract_offset(model, layer, head, dataloader, n_batches, config,
                   device, search_radius=5):
    """
    Determine the dominant integer offset (dy, dx) for a head.

    Returns: dict with offset, consistency_score, fingerprint_max_mass,
             per_query_argmax_match.
    """
    H, W = config["grid"]
    n_extra = config["n_extra_tokens"]

    grabber = _AttnGrabber(model, layer, head)

    # Collect attention from a few batches.
    for bi, batch in enumerate(dataloader):
        if bi >= n_batches:
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        _ = model(x.to(device, non_blocking=True))

    grabber.remove()

    # Each record: (B, N_full, N_full). Strip extras and renormalize.
    A = torch.cat(grabber.attn_records, dim=0)        # (Btot, N_full, N_full)
    if n_extra > 0:
        A = A[:, n_extra:, n_extra:]
        A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    A = A.numpy()                                     # (Btot, H*W, H*W)

    Btot = A.shape[0]

    # Recenter every (image, query) attention onto a (2R+1)x(2R+1) window.
    R = search_radius
    fp = np.zeros((2 * R + 1, 2 * R + 1))
    n = 0
    coords = [(i, j) for i in range(H) for j in range(W)]
    for b in range(Btot):
        for q_idx, (qy, qx) in enumerate(coords):
            for dy in range(-R, R + 1):
                for dx in range(-R, R + 1):
                    ky, kx = qy + dy, qx + dx
                    if 0 <= ky < H and 0 <= kx < W:
                        fp[dy + R, dx + R] += A[b, q_idx, ky * W + kx]
            n += 1
    fp /= n

    # Pick the dominant offset.
    fp_argmax_flat = int(fp.argmax())
    dy = fp_argmax_flat // (2 * R + 1) - R
    dx = fp_argmax_flat % (2 * R + 1) - R
    fp_norm = fp / (fp.sum() + 1e-12)
    dominant_mass = float(fp_norm.max())

    # Consistency: fraction of (image, query) pairs whose per-instance argmax
    # is the dominant offset
    target_idx_per_q = np.zeros(H * W, dtype=int)
    for q_idx, (qy, qx) in enumerate(coords):
        ty, tx = qy + dy, qx + dx
        if 0 <= ty < H and 0 <= tx < W:
            target_idx_per_q[q_idx] = ty * W + tx
        else:
            target_idx_per_q[q_idx] = -1   # boundary, mark as out-of-image

    matches = 0
    countable = 0
    for b in range(Btot):
        for q_idx in range(H * W):
            if target_idx_per_q[q_idx] >= 0:
                am = int(A[b, q_idx].argmax())
                if am == target_idx_per_q[q_idx]:
                    matches += 1
                countable += 1
    consistency = matches / max(countable, 1)

    return {
        "offset": (int(dy), int(dx)),
        "fingerprint_dominant_mass": dominant_mass,
        "argmax_consistency": float(consistency),
    }


# substitute one or more heads with torch.roll

def make_substituted_attention_forward(attn, substitutions, grid, n_extra_tokens,
                                        mode="head_output"):
    """
    Returns a new forward function for `attn` that swaps the listed heads.

    substitutions: dict of {head_idx: (dy, dx)}
    mode: 'head_output' or 'full_qkv'
    """
    original_forward = attn.forward
    H, W = grid

    def fast_shift_with_self_at_boundary(v_head, dy, dx):
        """
        Shift v_head's spatial tokens by (dy, dx). Boundary tokens that would
        fall off the image keep their own value.
        v_head: (B, n_extra + H*W, D_h)
        """
        B, N, D = v_head.shape
        v_extra = v_head[:, :n_extra_tokens, :] if n_extra_tokens > 0 else None
        v_spatial = v_head[:, n_extra_tokens:, :].reshape(B, H, W, D)

        qy_grid = torch.arange(H, device=v_head.device).view(H, 1).expand(H, W)
        qx_grid = torch.arange(W, device=v_head.device).view(1, W).expand(H, W)
        ty = qy_grid + dy
        tx = qx_grid + dx
        in_bounds = (ty >= 0) & (ty < H) & (tx >= 0) & (tx < W)
        ty = torch.where(in_bounds, ty, qy_grid)
        tx = torch.where(in_bounds, tx, qx_grid)

        out = v_spatial[:, ty, tx, :]                  # (B, H, W, D)
        out = out.reshape(B, H * W, D)
        if v_extra is not None:
            out = torch.cat([v_extra, out], dim=1)
        return out

    def new_forward(x, *args, **kwargs):
        B, N, C = x.shape
        Dh = C // attn.num_heads

        if mode == "full_qkv":
            # We still need V (and Q, K only for non-substituted heads).
            qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, Dh).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            non_sub_heads = [h for h in range(attn.num_heads) if h not in substitutions]
            attn_out = torch.zeros(B, attn.num_heads, N, Dh,
                                    device=x.device, dtype=x.dtype)
            if non_sub_heads:
                if hasattr(attn, "q_norm") and attn.q_norm is not None:
                    q = attn.q_norm(q); k = attn.k_norm(k)
                a = (q @ k.transpose(-2, -1)) * attn.scale
                a = a.softmax(dim=-1)
                # Only fill non-substituted heads
                for h in non_sub_heads:
                    attn_out[:, h] = a[:, h] @ v[:, h]
            for h, (dy, dx) in substitutions.items():
                v_h = v[:, h]
                attn_out[:, h] = fast_shift_with_self_at_boundary(v_h, dy, dx)
            out = attn_out.transpose(1, 2).reshape(B, N, C)
            out = attn.proj(out)
            if hasattr(attn, "proj_drop") and attn.proj_drop is not None:
                out = attn.proj_drop(out)
            return out
        else:
            # head_output mode: compute attention as usual, then overwrite
            # the listed heads' outputs with the shifted values.
            qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, Dh).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            if hasattr(attn, "q_norm") and attn.q_norm is not None:
                q = attn.q_norm(q); k = attn.k_norm(k)
            a = (q @ k.transpose(-2, -1)) * attn.scale
            a = a.softmax(dim=-1)
            attn_out = a @ v                                    # (B, heads, N, Dh)
            for h, (dy, dx) in substitutions.items():
                v_h = v[:, h]
                attn_out[:, h] = fast_shift_with_self_at_boundary(v_h, dy, dx)
            out = attn_out.transpose(1, 2).reshape(B, N, C)
            out = attn.proj(out)
            if hasattr(attn, "proj_drop") and attn.proj_drop is not None:
                out = attn.proj_drop(out)
            return out

    return new_forward


def install_substitutions(model, head_offset_map, grid, n_extra_tokens, mode):
    """
    head_offset_map: dict of (layer, head) -> (dy, dx).
    Mutates model in place (replaces attn.forward on each affected layer).
    Returns a dict of original forwards so you can restore.
    """
    by_layer = {}
    for (li, hi), off in head_offset_map.items():
        by_layer.setdefault(li, {})[hi] = off

    saved = {}
    for li, sub in by_layer.items():
        attn = model.blocks[li].attn
        saved[li] = attn.forward
        attn.forward = make_substituted_attention_forward(
            attn, sub, grid, n_extra_tokens, mode=mode)
    return saved


def restore_forwards(model, saved):
    for li, fwd in saved.items():
        model.blocks[li].attn.forward = fwd


# evaluate top-1 accuracy on tiny-imagenet

@torch.no_grad()
def evaluate(model, dataloader, n_batches, device):
    """Top-1 / top-5 accuracy. Examples with label=-1 are skipped."""
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0
    for bi, batch in enumerate(dataloader):
        if bi >= n_batches:
            break
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        top5 = logits.topk(5, dim=-1).indices
        valid = (y >= 0)
        correct1 += ((top5[:, 0] == y) & valid).sum().item()
        correct5 += ((top5 == y.unsqueeze(-1)).any(dim=-1) & valid).sum().item()
        total += valid.sum().item()
    return correct1 / max(total, 1), correct5 / max(total, 1), total


# Data loader: tiny-imagenet with ImageNet-1k label mapping

# tiny-imagenet's 200 classes are a subset of ImageNet-1k. To get meaningful
# absolute top-1 numbers from an ImageNet-1k-pretrained model, we need to
# map tiny-imagenet's label index -> ImageNet-1k label index.
#
# timm's pretrained model has a `pretrained_cfg`
# that includes the 1000 ImageNet wnids in canonical order. tiny-imagenet's
# label names ARE wnids (n01443537, etc.). So map name -> position in timm's
# ordering. If anything fails, we fall back to "delta-only" evaluation:
# absolute accuracies aren't meaningful but their CHANGE under substitution is.


def _try_get_imagenet_wnids_from_timm():
    """Pull ImageNet-1k wnids in canonical order from a known online source."""
    try:
        import urllib.request
        # This is the standard, deterministic mapping used by all timm models.
        url = ("https://raw.githubusercontent.com/raghakot/keras-vis/master/"
               "resources/imagenet_class_index.json")
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        # data is {"0": ["n01440764", "tench"], "1": [...], ...}
        wnids = [data[str(i)][0] for i in range(1000)]
        return wnids
    except Exception as e:
        print(f"  (couldn't fetch ImageNet wnids: {e})")
        return None


def build_imagenet_val_eval_loader(batch_size: int, image_size: int,
                                    val_dir: str = "data/imagenet/val",
                                    num_workers: int = 4):
    """
    Standard ImageNet-1k val loader from a directory laid out as
        val_dir/n01440764/ILSVRC2012_val_xxxxx.JPEG
        val_dir/n01443537/...
    The alphabetical wnid sort that ImageFolder uses to assign class
    indices coincides with the ImageNet-1k canonical label order, so
    no remapping is needed.

    Returns (dataloader, n_classes).
    """
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    ds = ImageFolder(val_dir, transform=eval_transform(image_size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    return loader, len(ds.classes)


def build_tiny_imagenet_eval_loader(batch_size: int, image_size: int):
    """
    Returns (dataloader, n_eval_classes).
    Each batch: (image_tensor, imagenet1k_class_index_or_minus_1).
    """
    from datasets import load_dataset
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    print("Loading zh-plus/tiny-imagenet...")
    ds = load_dataset("zh-plus/tiny-imagenet", split="valid")

    # Get tiny-imagenet's label-index -> wnid mapping.
    label_names = None
    if "label" in ds.features and hasattr(ds.features["label"], "names"):
        label_names = ds.features["label"].names

    # Get ImageNet-1k canonical wnid -> index.
    imagenet_wnids = _try_get_imagenet_wnids_from_timm()
    wnid_to_imagenet_idx = ({w: i for i, w in enumerate(imagenet_wnids)}
                            if imagenet_wnids else None)

    ok_for_absolute = (label_names is not None
                       and wnid_to_imagenet_idx is not None
                       and all(n.startswith("n") and len(n) >= 8
                               for n in label_names[:5]))
    if ok_for_absolute:
        # Pre-build an array: tiny_idx -> imagenet_idx (or -1 if missing)
        tiny_to_imagenet = [
            wnid_to_imagenet_idx.get(label_names[i], -1)
            for i in range(len(label_names))
        ]
        n_unmapped = sum(1 for x in tiny_to_imagenet if x < 0)
        if n_unmapped > 0:
            print(f"  (note: {n_unmapped}/200 tiny-imagenet classes "
                  f"didn't match any ImageNet-1k wnid)")
    else:
        tiny_to_imagenet = None
        print("  (could not build wnid -> ImageNet-1k map. Will report")
        print("   accuracies in delta-only mode: baseline numbers may look")
        print("   low, but substitution deltas remain meaningful.)")

    if os.environ.get("SOLA_PREPROCESS", "imagenet") == "imagenet":
        tf = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        tf = eval_transform(image_size)

    def collate(items):
        imgs, labels = [], []
        for it in items:
            img = it["image"].convert("RGB")
            imgs.append(tf(img))
            tiny_label = it["label"]
            if tiny_to_imagenet is not None:
                labels.append(tiny_to_imagenet[tiny_label])
            else:
                labels.append(tiny_label)
        return torch.stack(imgs), torch.tensor(labels)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate)
    return loader, (1000 if tiny_to_imagenet is not None else 200)


# Driver

def parse_heads(s_list):
    return [tuple(int(x) for x in s.split(",")) for s in s_list]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deit_tiny_patch16_224")
    ap.add_argument("--heads", nargs="+", required=True,
                    help="layer,head pairs to substitute.")
    ap.add_argument("--mode", default="head_output",
                    choices=["head_output", "full_qkv"])
    ap.add_argument("--n-extract-batches", type=int, default=4,
                    help="batches used to find each head's offset")
    ap.add_argument("--batches", type=int, default=32,
                    help="batches used in evaluation")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="shift_substitution_result.json")
    ap.add_argument("--cumulative", action="store_true",
                    help="Substitute heads one at a time, evaluating after each.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    heads = parse_heads(args.heads)

    import timm
    model = timm.create_model(args.model, pretrained=True).eval().to(device)
    cfg = detect_model_config(model, args.image_size)
    H, W = cfg["grid"]
    print(f"Model: {args.model}")
    print(f"Grid:  {H}x{W}, n_extra={cfg['n_extra_tokens']}")
    print(f"Heads to substitute: {heads}\n")

    eval_loader, n_eval = build_tiny_imagenet_eval_loader(
        args.batch_size, args.image_size)

    # extract offsets for each head (using a small subset of data)
    print("Extracting dominant offsets...")
    offsets = {}
    for (li, hi) in heads:
        info = extract_offset(model, li, hi, eval_loader,
                              n_batches=args.n_extract_batches,
                              config=cfg, device=device)
        offsets[(li, hi)] = info
        print(f"  L{li}H{hi}: offset=({info['offset'][0]:+d},{info['offset'][1]:+d})  "
              f"dom_mass={info['fingerprint_dominant_mass']:.3f}  "
              f"argmax_consistency={info['argmax_consistency']:.3f}")

    # Drop heads whose argmax consistency is too low to call them shift heads.
    print()
    bad = [(lh, info) for lh, info in offsets.items()
           if info["argmax_consistency"] < 0.3]
    if bad:
        print(f"WARNING: {len(bad)} head(s) have low argmax consistency (<0.3):")
        for lh, info in bad:
            print(f"  L{lh[0]}H{lh[1]} consistency={info['argmax_consistency']:.3f} - substitution will likely be lossy.")

    # baseline accuracy
    print("\nMeasuring baseline accuracy...")
    base_top1, base_top5, n_total = evaluate(
        model, eval_loader, args.batches, device)
    print(f"  baseline: top-1 = {base_top1*100:.2f}%, top-5 = {base_top5*100:.2f}%  "
          f"(over {n_total} images)")

    # substitute (either all-at-once or cumulatively)
    results = []
    if args.cumulative:
        applied = {}
        sorted_heads = sorted(offsets.keys())
        for (li, hi) in sorted_heads:
            applied[(li, hi)] = offsets[(li, hi)]["offset"]
            saved = install_substitutions(
                model, applied, cfg["grid"], cfg["n_extra_tokens"], args.mode)
            top1, top5, _ = evaluate(model, eval_loader, args.batches, device)
            restore_forwards(model, saved)
            d_top1 = (top1 - base_top1) * 100
            d_top5 = (top5 - base_top5) * 100
            print(f"  +L{li}H{hi}({offsets[(li,hi)]['offset']}): "
                  f"top-1={top1*100:.2f}% (delta {d_top1:+.2f})  "
                  f"top-5={top5*100:.2f}% (delta {d_top5:+.2f})")
            results.append({
                "heads_applied": list(applied.keys()),
                "top1": top1, "top5": top5,
                "delta_top1": d_top1, "delta_top5": d_top5,
            })
    else:
        all_subs = {lh: offsets[lh]["offset"] for lh in offsets}
        saved = install_substitutions(
            model, all_subs, cfg["grid"], cfg["n_extra_tokens"], args.mode)
        top1, top5, _ = evaluate(model, eval_loader, args.batches, device)
        restore_forwards(model, saved)
        d_top1 = (top1 - base_top1) * 100
        d_top5 = (top5 - base_top5) * 100
        print(f"\nAll heads substituted simultaneously:")
        print(f"  top-1 = {top1*100:.2f}% (delta {d_top1:+.2f})")
        print(f"  top-5 = {top5*100:.2f}% (delta {d_top5:+.2f})")
        results.append({
            "heads_applied": list(all_subs.keys()),
            "top1": top1, "top5": top5,
            "delta_top1": d_top1, "delta_top5": d_top5,
        })

    # Save
    out_data = {
        "model": args.model,
        "mode": args.mode,
        "n_eval_images": n_total,
        "baseline_top1": base_top1,
        "baseline_top5": base_top5,
        "offsets": {f"L{lh[0]}H{lh[1]}": offsets[lh] for lh in offsets},
        "substitutions": results,
    }
    Path(args.out).write_text(json.dumps(out_data, indent=2, default=str))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
