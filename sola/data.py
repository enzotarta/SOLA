"""ImageNet val loader, grid detection and attention capture for timm ViTs."""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder


def eval_transform(image_size=224):
    """Preprocessing of the evaluation and calibration loaders.

    By default Resize(256) + CenterCrop with ImageNet statistics for every model,
    as in the stored results. SOLA_PREPROCESS selects another one: "siglip" resizes
    to the input size with mean = std = 0.5, and a timm model name uses the data
    config of that model.
    """
    mode = os.environ.get("SOLA_PREPROCESS", "imagenet")
    bicubic = T.InterpolationMode.BICUBIC
    if mode == "imagenet":
        return T.Compose([
            T.Resize(256, interpolation=bicubic),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    if mode == "siglip":
        return T.Compose([
            T.Resize((image_size, image_size), interpolation=bicubic),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    import timm
    from timm.data import create_transform, resolve_data_config
    cfg = resolve_data_config(pretrained_cfg=timm.get_pretrained_cfg(mode).to_dict())
    return create_transform(**cfg, is_training=False)


def imagenet_val_loader(val_dir, batch_size=32, image_size=224, num_workers=2,
                        n=None, order="class", seed=0):
    """ImageNet val in ImageFolder layout.

    order="class" keeps the class order of the folder, so that the first 9,984
    images, the subset used in the experiments, cover classes 0 to 199 and the
    first 32 images all belong to class 0. order="random" draws n images
    uniformly with a fixed seed.
    """
    ds = ImageFolder(val_dir, transform=eval_transform(image_size))
    if n is not None and n < len(ds):
        if order == "random":
            idx = np.random.RandomState(seed).permutation(len(ds))[:n]
            ds = Subset(ds, sorted(idx.tolist()))
        else:
            ds = Subset(ds, list(range(n)))
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def detect_grid_and_extra(model, image_size=224):
    p = model.patch_embed
    if hasattr(p, "grid_size"):
        H, W = p.grid_size
    else:
        ps = p.patch_size if hasattr(p, "patch_size") else 16
        ps = ps[0] if isinstance(ps, (tuple, list)) else ps
        H = W = image_size // ps
    n_extra = 0
    if getattr(model, "cls_token", None) is not None:
        n_extra += 1
    if getattr(model, "reg_token", None) is not None:
        n_extra += int(model.reg_token.shape[1])
    return (H, W), n_extra


@torch.no_grad()
def capture_attention(model, x, n_extra, renormalize=False):
    """Attention of every layer on batch x, spatial tokens only.

    renormalize=True rescales each row to sum to one after the non-spatial tokens
    are removed, as in experiments/operator_taxonomy.py.
    """
    blocks = list(model.blocks)
    captured = [None] * len(blocks)
    handles = []

    def make_hook(li):
        def hk(blk, inp, out):
            attn = blk.attn
            x_in = blk.norm1(inp[0])
            B, N, C = x_in.shape
            Hh = attn.num_heads
            Dh = C // Hh
            qkv = attn.qkv(x_in).reshape(B, N, 3, Hh, Dh).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            if hasattr(attn, "q_norm") and attn.q_norm is not None:
                q = attn.q_norm(q)
                k = attn.k_norm(k)
            a = (q @ k.transpose(-2, -1)) * attn.scale
            a = a.softmax(-1)
            a = a[:, :, n_extra:, n_extra:]
            if renormalize:
                a = a / a.sum(-1, keepdim=True).clamp(min=1e-12)
            captured[li] = a.cpu()
        return hk

    for li, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(make_hook(li)))
    try:
        model(x)
    finally:
        for h in handles:
            h.remove()
    return captured
