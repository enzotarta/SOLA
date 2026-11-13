"""Attention maps and recentered kernels for chosen heads."""

from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


# Attention recorder (same hooks-based approach as escape_mass_sweep)

class AttentionRecorder:
    def __init__(self, model: nn.Module, layers: list[int]):
        self.model = model
        self.layers = layers
        self.records: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
        self._handles = []
        self._inputs: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
        for li in layers:
            attn = model.blocks[li].attn
            self._handles.append(attn.register_forward_pre_hook(
                self._make_pre_hook(li)))
            self._handles.append(attn.register_forward_hook(
                self._make_post_hook(li, attn)))

    def _make_pre_hook(self, li):
        def hook(module, args):
            self._inputs[li].append(args[0].detach())
        return hook

    def _make_post_hook(self, li, attn):
        recorder = self
        def hook(module, args, output):
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


# Detection of model config (matches escape_mass_sweep)

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


# Structural-fingerprint computation

def recenter_map(alpha: np.ndarray, q_pos: tuple[int, int],
                  H: int, W: int, K: int) -> np.ndarray:
    """
    Take an attention distribution alpha over the HxW grid (with query at q),
    extract the (2K+1) x (2K+1) window centered on q. Pixels outside the
    image are zero. The result is in OFFSET coordinates: center entry is
    delta = (0, 0), i.e., the query pixel.
    """
    half = K
    qy, qx = q_pos
    a2 = alpha.reshape(H, W)
    out = np.zeros((2 * K + 1, 2 * K + 1))
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            ky, kx = qy + dy, qx + dx
            if 0 <= ky < H and 0 <= kx < W:
                out[dy + half, dx + half] = a2[ky, kx]
    return out


def fingerprint(attn_per_query: np.ndarray, H: int, W: int,
                K: int = 5) -> tuple[np.ndarray, dict]:
    """
    attn_per_query: (n_images, n_queries=H*W, n_keys=H*W), the attention of a
                    head over a batch of images, all query positions.
    Returns:
        fp:   (2K+1, 2K+1), average recentered attention (the kernel)
        diag: dict of diagnostics
    """
    n_img, n_q, n_k = attn_per_query.shape
    assert n_q == H * W and n_k == H * W

    recentered = np.zeros((2 * K + 1, 2 * K + 1))
    n = 0
    for i in range(n_img):
        for q in range(n_q):
            qy, qx = q // W, q % W
            recentered += recenter_map(attn_per_query[i, q], (qy, qx), H, W, K)
            n += 1
    recentered /= n

    # Diagnostics
    diag = {}

    # Self-attention mass: how much do queries attend to themselves?
    self_mass = np.array([attn_per_query[i, q, q]
                          for i in range(n_img) for q in range(n_q)])
    diag["self_mass_mean"] = float(self_mass.mean())
    diag["self_mass_max"]  = float(self_mass.max())

    # Fixed-target test: does the head attend to the same absolute pixel
    #    regardless of query? Compute argmax of attention per (image, query),
    #    then check how concentrated those argmaxes are. If a single key is
    #    the argmax for >50% of queries, flag it.
    argmaxes = attn_per_query.argmax(axis=-1)            # (n_img, n_q)
    diag["argmax_concentration"] = []
    for i in range(min(n_img, 8)):
        per_image_argmaxes = argmaxes[i]
        bincount = np.bincount(per_image_argmaxes, minlength=n_k)
        top_frac = bincount.max() / n_q
        diag["argmax_concentration"].append(float(top_frac))
    diag["argmax_concentration_mean"] = float(
        np.mean(diag["argmax_concentration"]))

    # Query-independence test: pearson correlation between attention
    #    pattern at q and at q'. If high for every (q, q'), the head is
    #    ignoring its query.
    sample_qs = np.random.RandomState(0).choice(n_q, size=min(20, n_q),
                                                 replace=False)
    corrs = []
    for i in range(min(n_img, 4)):
        for ii in range(len(sample_qs)):
            for jj in range(ii + 1, len(sample_qs)):
                a = attn_per_query[i, sample_qs[ii]]
                b = attn_per_query[i, sample_qs[jj]]
                if a.std() > 0 and b.std() > 0:
                    c = np.corrcoef(a, b)[0, 1]
                    if not np.isnan(c):
                        corrs.append(c)
    diag["query_pair_corr_mean"] = float(np.mean(corrs)) if corrs else 0.0

    # How concentrated is the recentered fingerprint? Entropy.
    fp_norm = recentered / (recentered.sum() + 1e-12)
    H_fp = float(-(fp_norm * np.log(fp_norm + 1e-12)).sum())
    diag["fingerprint_entropy"] = H_fp
    diag["fingerprint_max_offset_mass"] = float(fp_norm.max())
    diag["fingerprint_center_mass"] = float(fp_norm[K, K])

    # Generate flags
    flags = []
    if diag["self_mass_mean"] > 0.7:
        flags.append("SELF-ONLY (delta kernel)")
    if diag["argmax_concentration_mean"] > 0.4:
        flags.append("FIXED-TARGET (always attends same pixel)")
    if diag["query_pair_corr_mean"] > 0.7:
        flags.append("QUERY-INDEPENDENT (ignores q)")
    if diag["fingerprint_max_offset_mass"] > 0.5 and diag["fingerprint_center_mass"] < 0.25:
        flags.append("OFF-CENTER (kernel peaks away from q)")
    diag["flags"] = flags

    return recentered, diag


# Main pipeline

def parse_heads(s_list: list[str]) -> list[tuple[int, int]]:
    out = []
    for s in s_list:
        l, h = s.split(",")
        out.append((int(l), int(h)))
    return out


def get_images(source: str, n: int, image_size: int, data_dir: str = None):
    import torchvision.transforms as T
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    tf = T.Compose([T.Resize(int(image_size * 256 / 224)),
                    T.CenterCrop(image_size),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std)])

    if source == "tiny-imagenet":
        from datasets import load_dataset
        ds = load_dataset("zh-plus/tiny-imagenet", split="valid")
        out = []
        for i in range(n):
            img = ds[i]["image"].convert("RGB")
            out.append(tf(img))
        return torch.stack(out)
    elif source == "folder":
        from PIL import Image
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG"):
            paths.extend(Path(data_dir).rglob(ext))
        paths = sorted(paths)[:n]
        if not paths:
            raise FileNotFoundError(f"No images in {data_dir}")
        return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
    raise ValueError(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deit_tiny_patch16_224")
    parser.add_argument("--heads", nargs="+", required=True,
                        help="List of layer,head pairs e.g. 1,2 2,0 2,1 3,2")
    parser.add_argument("--n-images", type=int, default=4,
                        help="Images to use for raw attention plot")
    parser.add_argument("--n-fingerprint-images", type=int, default=32,
                        help="More images, used for the avg fingerprint+stats")
    parser.add_argument("--n-queries", type=int, default=4,
                        help="How many query positions to plot per image")
    parser.add_argument("--source", default="tiny-imagenet",
                        choices=["tiny-imagenet", "folder"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--out", default="heads_diagnostic.png")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    import timm
    model = timm.create_model(args.model, pretrained=True).eval().to(device)
    cfg = detect_model_config(model, args.image_size)
    H, W = cfg["grid"]
    print(f"Model:  {args.model}")
    print(f"Grid:   {H}x{W}, n_extra_tokens={cfg['n_extra_tokens']}")

    heads_to_show = parse_heads(args.heads)
    layers_needed = sorted(set(l for (l, _) in heads_to_show))
    print(f"Heads:  {heads_to_show}")

    # Pass 1: collect attention for the FIRST set of images for raw plot
    n_total_imgs = max(args.n_images, args.n_fingerprint_images)
    images = get_images(args.source, n_total_imgs, args.image_size,
                         data_dir=args.data_dir).to(device)
    print(f"Loaded {len(images)} images.")

    rec = AttentionRecorder(model, layers_needed)
    with torch.no_grad():
        # Process in chunks to avoid OOM
        chunk = 32
        for s in range(0, len(images), chunk):
            _ = model(images[s:s + chunk])
    rec.remove()

    # rec.records[li] is a list of (B_i, heads, N, N) tensors. Concatenate.
    raw_attn = {}
    for li in layers_needed:
        full = torch.cat(rec.records[li], dim=0)            # (N_imgs, heads, N, N)
        # Strip extra tokens (CLS + registers) and renormalize.
        if cfg["n_extra_tokens"] > 0:
            full = full[:, :, cfg["n_extra_tokens"]:, cfg["n_extra_tokens"]:]
            full = full / full.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        raw_attn[li] = full                                  # (N_imgs, heads, H*W, H*W)

    # Build the figure (one row per (head, image) pair)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.gridspec import GridSpec

    n_rows = len(heads_to_show) * args.n_images
    n_cols = args.n_queries + 2  # source image + queries + fingerprint
    fig = plt.figure(figsize=(2.0 * n_cols, 2.0 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.1)

    # Pick fixed query positions (interior) for visual consistency
    rng = np.random.RandomState(42)
    query_positions = []
    while len(query_positions) < args.n_queries:
        qy = int(rng.randint(2, H - 2))
        qx = int(rng.randint(2, W - 2))
        if (qy, qx) not in query_positions:
            query_positions.append((qy, qx))

    mean_v = np.array([0.485, 0.456, 0.406])
    std_v  = np.array([0.229, 0.224, 0.225])

    for ri, (li, hi) in enumerate(heads_to_show):
        # Fingerprint over many images
        head_attn = raw_attn[li][:args.n_fingerprint_images, hi].cpu().numpy()
        fp, diag = fingerprint(head_attn, H, W, K=5)

        print(f"\nL{li}H{hi}")
        print(f"  self_mass:            {diag['self_mass_mean']:.3f}")
        print(f"  argmax_concentration: {diag['argmax_concentration_mean']:.3f}")
        print(f"  query_pair_corr:      {diag['query_pair_corr_mean']:.3f}")
        print(f"  fingerprint_entropy:  {diag['fingerprint_entropy']:.3f}")
        print(f"  flags:                {diag['flags']}")

        for ii in range(args.n_images):
            row = ri * args.n_images + ii
            img_np = images[ii].cpu().numpy()
            attn_for_plot = raw_attn[li][ii, hi].cpu().numpy()

            # col 0: source image with query markers (only on first row of head)
            ax = fig.add_subplot(gs[row, 0])
            disp = img_np.transpose(1, 2, 0) * std_v + mean_v
            disp = np.clip(disp, 0, 1)
            ax.imshow(disp)
            ps = disp.shape[0] // H
            for (qy, qx) in query_positions:
                ax.add_patch(Rectangle((qx * ps, qy * ps), ps, ps,
                                       fill=False, edgecolor='red',
                                       linewidth=1.0))
            if ii == 0:
                title = f"L{li}H{hi}"
                if diag["flags"]:
                    title += "\n" + " | ".join(diag["flags"])
                ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

            # cols 1..n_queries: attention at each query
            for ci, (qy, qx) in enumerate(query_positions):
                ax = fig.add_subplot(gs[row, 1 + ci])
                q_idx = qy * W + qx
                amap = attn_for_plot[q_idx].reshape(H, W)
                ax.imshow(amap, cmap='magma', vmin=0,
                          vmax=max(amap.max(), 0.05))
                ax.add_patch(Rectangle((qx - 0.5, qy - 0.5), 1, 1,
                                       fill=False, edgecolor='cyan',
                                       linewidth=0.8))
                if ii == 0:
                    ax.set_title(f"q=({qy},{qx})", fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])

            # last col: recentered fingerprint (same for every row of this head)
            ax = fig.add_subplot(gs[row, -1])
            ax.imshow(fp, cmap='magma', vmin=0, vmax=fp.max())
            K = fp.shape[0] // 2
            ax.add_patch(Rectangle((K - 0.5, K - 0.5), 1, 1,
                                   fill=False, edgecolor='cyan',
                                   linewidth=0.8))
            if ii == 0:
                ax.set_title(
                    f"avg recentered\n"
                    f"self={diag['self_mass_mean']:.2f}  "
                    f"qcorr={diag['query_pair_corr_mean']:.2f}",
                    fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])

    fig.savefig(args.out, dpi=120, bbox_inches='tight')
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
