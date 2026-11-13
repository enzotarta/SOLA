"""Closed-form substitutes for softmax(QK^T), installed on a timm ViT.

Each forward replaces the attention of the chosen heads and leaves the value
projection unchanged. Rank-k uses torch.svd_lowrank(q=k+4, niter=2) by default,
as in the experiments. This is a randomized approximation of the best rank-k
matrix; pass exact_svd=True for the exact truncation.
"""
import math
import numpy as np
import torch


def kernel_apply(v, kernel, grid, n_extra):
    B, N, D = v.shape
    H, W = grid
    assert N == n_extra + H * W
    if n_extra > 0:
        v_extra, v_sp = v[:, :n_extra], v[:, n_extra:]
    else:
        v_extra, v_sp = None, v
    v_sp = v_sp.reshape(B, H, W, D)

    K = (kernel.shape[0] - 1) // 2
    out = torch.zeros_like(v_sp)
    norm = torch.zeros(H, W, device=v.device, dtype=v.dtype)

    qy = torch.arange(H, device=v.device)
    qx = torch.arange(W, device=v.device)
    for dy in range(-K, K + 1):
        for dx in range(-K, K + 1):
            w = kernel[dy + K, dx + K]
            if w.abs().item() < 1e-9:
                continue
            sy = qy + dy
            sx = qx + dx
            mask = ((sy >= 0) & (sy < H)).unsqueeze(1) & \
                   ((sx >= 0) & (sx < W)).unsqueeze(0)
            mask_f = mask.to(v.dtype)
            shifted = torch.roll(v_sp, (-dy, -dx), (1, 2))
            out = out + shifted * (mask_f * w).unsqueeze(0).unsqueeze(-1)
            norm = norm + mask_f * w

    out = out / norm.unsqueeze(0).unsqueeze(-1).clamp(min=1e-8)
    zero = (norm.abs() < 1e-9).unsqueeze(0).unsqueeze(-1)
    if zero.any():
        out = torch.where(zero.expand_as(out), v_sp, out)

    out = out.reshape(B, H * W, D)
    if v_extra is not None:
        out = torch.cat([v_extra, out], dim=1)
    return out


def kernel_from_mean(mean_kernel):
    Z = float(np.sum(mean_kernel))
    if Z <= 0:
        raise ValueError("kernel has nonpositive mass")
    return (mean_kernel / Z).astype(np.float32)


def _qkv(attn, x):
    B, N, C = x.shape
    H = attn.num_heads
    Dh = C // H
    qkv = attn.qkv(x).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    if hasattr(attn, "q_norm") and attn.q_norm is not None:
        q = attn.q_norm(q)
        k = attn.k_norm(k)
    return q, k, v


def _wrap_out(attn, attn_out, B, N, C):
    out = attn_out.transpose(1, 2).reshape(B, N, C)
    out = attn.proj(out)
    if hasattr(attn, "proj_drop") and attn.proj_drop is not None:
        out = attn.proj_drop(out)
    return out


def make_kernel_forward(attn, kernels, grid, n_extra):
    def fwd(x, *a, **kw):
        B, N, C = x.shape
        q, k, v = _qkv(attn, x)
        a_mat = (q @ k.transpose(-2, -1)) * attn.scale
        a_mat = a_mat.softmax(-1)
        out = a_mat @ v
        for h, ker in kernels.items():
            out[:, h] = kernel_apply(v[:, h], ker, grid, n_extra)
        return _wrap_out(attn, out, B, N, C)
    return fwd


def make_broadcast_forward(attn, broadcast_heads):
    bset = list(broadcast_heads)
    def fwd(x, *a, **kw):
        B, N, C = x.shape
        q, k, v = _qkv(attn, x)
        a_mat = (q @ k.transpose(-2, -1)) * attn.scale
        a_mat = a_mat.softmax(-1)
        for h in bset:
            mean_row = a_mat[:, h].mean(dim=1, keepdim=True)  # (B, 1, N)
            a_mat[:, h] = mean_row.expand(-1, N, -1)
        out = a_mat @ v
        return _wrap_out(attn, out, B, N, C)
    return fwd


def rank_k(a_flat, k, exact_svd=False):
    """Rank-k approximation of a batch of square matrices."""
    if exact_svd:
        U, S, Vh = torch.linalg.svd(a_flat, full_matrices=False)
        return U[..., :k] @ torch.diag_embed(S[..., :k]) @ Vh[..., :k, :]
    U, S, V = torch.svd_lowrank(a_flat, q=min(k + 4, a_flat.shape[-1]), niter=2)
    return U[..., :k] @ torch.diag_embed(S[..., :k]) @ V[..., :k].transpose(-2, -1)


def make_adaptive_forward(attn, kernels, head_to_k, grid, n_extra, exact_svd=False):
    by_k = {}
    for h, k in head_to_k.items():
        by_k.setdefault(int(k), []).append(int(h))
    by_k_sorted = sorted(by_k.items())

    def fwd(x, *a, **kw):
        B, N, C = x.shape
        q, k_, v = _qkv(attn, x)
        a_mat = (q @ k_.transpose(-2, -1)) * attn.scale
        a_mat = a_mat.softmax(-1)

        for k_val, heads in by_k_sorted:
            if k_val >= N:
                continue
            idx = torch.tensor(heads, device=x.device, dtype=torch.long)
            sub = a_mat.index_select(1, idx)             # (B, ns, N, N)
            B0, ns, N0, _ = sub.shape
            flat = sub.reshape(B0 * ns, N0, N0)
            recon = rank_k(flat, k_val, exact_svd).reshape(B0, ns, N0, N0)
            for j, h in enumerate(heads):
                a_mat[:, h] = recon[:, j]

        out = a_mat @ v
        for h, ker in kernels.items():
            out[:, h] = kernel_apply(v[:, h], ker, grid, n_extra)
        return _wrap_out(attn, out, B, N, C)

    return fwd


def install(model, layer_to_fwd):
    saved = {}
    for li, fwd in layer_to_fwd.items():
        attn = model.blocks[li].attn
        saved[li] = attn.forward
        attn.forward = fwd
    return saved


def restore(model, saved):
    for li, fwd in saved.items():
        model.blocks[li].attn.forward = fwd


def adaptive_layer_forwards(model, kernel_map, head_to_k_map, grid, n_extra,
                             k_floor=10, exact_svd=False):
    by_layer = {}
    for (li, hi), ker in kernel_map.items():
        d = by_layer.setdefault(li, {"kernels": {}, "ranks": {}})
        d["kernels"][hi] = torch.as_tensor(ker, dtype=torch.float32)
    for (li, hi), kh in head_to_k_map.items():
        d = by_layer.setdefault(li, {"kernels": {}, "ranks": {}})
        d["ranks"][hi] = max(k_floor, int(math.ceil(kh)))

    out = {}
    for li, d in by_layer.items():
        attn = model.blocks[li].attn
        device = next(attn.parameters()).device
        kers = {h: t.to(device) for h, t in d["kernels"].items()}
        out[li] = make_adaptive_forward(attn, kers, d["ranks"], grid, n_extra, exact_svd)
    return out
