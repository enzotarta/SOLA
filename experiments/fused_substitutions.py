"""PyTorch fused forward: shift heads skip QK^T and softmax entirely."""
from __future__ import annotations
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_layer_shift_state(
    head_indices: List[int],
    kernels_per_head: Dict[int, torch.Tensor],
    grid: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Build the per-layer state needed to apply a depthwise conv
    over a stacked (n_shift, ...) V tensor:

        weight   : (n_shift, 1, 2K+1, 2K+1)
        norm     : (n_shift, H, W)         in-grid weight sum per query
        zero_msk : (n_shift, H, W)  bool   true where norm == 0

    head_indices : ordered list of shift-head indices for this layer.
    kernels_per_head : {head_idx: (2K+1, 2K+1) tensor}
    """
    H, W = grid
    K = (next(iter(kernels_per_head.values())).shape[0] - 1) // 2
    n_shift = len(head_indices)

    weight = torch.stack([kernels_per_head[h] for h in head_indices], dim=0)
    weight = weight.to(device=device, dtype=dtype).unsqueeze(1)
    # (n_shift, 1, 2K+1, 2K+1)

    # F.conv2d uses cross-correlation, not convolution, so apply the
    # kernel as-is to match the original kernel_substitution semantics
    # where the recentered mean kernel is applied directly.

    # Per-(head, y, x) sum of in-grid weights, computed by convolving
    # the kernel against an all-ones boundary indicator.
    qy = torch.arange(H, device=device)
    qx = torch.arange(W, device=device)
    norm = torch.zeros(n_shift, H, W, device=device, dtype=dtype)
    kernels_cpu = weight.detach().cpu().squeeze(1).numpy()  # (n_shift, 2K+1, 2K+1)
    for j in range(n_shift):
        for dy in range(-K, K+1):
            for dx in range(-K, K+1):
                w = float(kernels_cpu[j, dy+K, dx+K])
                if abs(w) < 1e-9:
                    continue
                sy = qy + dy
                sx = qx + dx
                valid = ((sy >= 0) & (sy < H)).unsqueeze(1) \
                      & ((sx >= 0) & (sx < W)).unsqueeze(0)
                norm[j] = norm[j] + valid.to(dtype) * w

    zero_msk = (norm.abs() < 1e-9)
    norm_safe = norm.clamp(min=1e-8)
    return {
        "weight":    weight,
        "K":         K,
        "norm_safe": norm_safe,                    # (n_shift, H, W)
        "zero_msk":  zero_msk,                     # (n_shift, H, W)
        "head_idx":  torch.tensor(head_indices, device=device, dtype=torch.long),
    }


def _apply_layer_shift_conv(
    v_shift: torch.Tensor,                          # (B, n_shift, N, Dh)
    state: dict,
    grid: Tuple[int, int],
    n_extra_tokens: int,
) -> torch.Tensor:
    """
    Apply the per-head conv kernels to V-shift in one F.conv2d call.

    Returns (B, n_shift, N, Dh).
    """
    B, n_shift, N, Dh = v_shift.shape
    H, W = grid
    K = state["K"]
    weight = state["weight"]                        # (n_shift, 1, 2K+1, 2K+1)
    norm_safe = state["norm_safe"]                  # (n_shift, H, W)
    zero_msk = state["zero_msk"]                    # (n_shift, H, W)

    # Split off non-spatial tokens (passed through unchanged).
    if n_extra_tokens > 0:
        v_extra = v_shift[:, :, :n_extra_tokens, :]
        v_sp = v_shift[:, :, n_extra_tokens:, :]
    else:
        v_extra = None
        v_sp = v_shift
    # v_sp : (B, n_shift, H*W, Dh) -> reshape to (B, n_shift, H, W, Dh)
    v_sp = v_sp.reshape(B, n_shift, H, W, Dh)

    # F.conv2d expects (B, C, H, W). We want a per-channel conv where
    # each channel pair (head, d) is independent. Pack as
    # (B*Dh, n_shift, H, W) and use groups=n_shift.
    v_sp_perm = v_sp.permute(0, 4, 1, 2, 3).reshape(B * Dh, n_shift, H, W)

    out_conv = F.conv2d(v_sp_perm, weight, padding=K, groups=n_shift)
    # (B*Dh, n_shift, H, W)

    # Renormalize for boundary: divide by per-query in-grid weight sum.
    out_conv = out_conv / norm_safe.unsqueeze(0)     # broadcast over batch dim
    # Fallback for queries whose entire mass is off-grid.
    # (Practically never triggers for unit-cardinal delta kernels.)
    if zero_msk.any():
        # zero_msk: (n_shift, H, W); v_sp_perm: (B*Dh, n_shift, H, W)
        out_conv = torch.where(
            zero_msk.unsqueeze(0).expand_as(out_conv),
            v_sp_perm, out_conv)

    # Reshape back: (B*Dh, n_shift, H, W) -> (B, n_shift, H*W, Dh).
    out_sp = out_conv.reshape(B, Dh, n_shift, H, W).permute(0, 2, 3, 4, 1)
    out_flat = out_sp.reshape(B, n_shift, H*W, Dh)
    if v_extra is not None:
        out_flat = torch.cat([v_extra, out_flat], dim=2)
    return out_flat


def make_fused_shift_attention_forward(
    attn,
    layer_state: dict,
    non_shift_idx_t: torch.Tensor,
    shift_idx_t: torch.Tensor,
    grid: Tuple[int, int],
    n_extra_tokens: int,
):
    """
    layer_state      : output of `_build_layer_shift_state`.
    non_shift_idx_t  : long tensor, head indices that go through SDPA
    shift_idx_t      : long tensor, shift-head indices (matching layer_state)
    """
    num_heads = attn.num_heads

    def new_forward(x, *args, **kwargs):
        B, N, C = x.shape
        Dh = C // num_heads
        qkv = attn.qkv(x).reshape(B, N, 3, num_heads, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                             # (B, h, N, Dh)
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            q = attn.q_norm(q); k = attn.k_norm(k)

        attn_out = torch.empty_like(v)                       # (B, h, N, Dh)

        # Non-shift heads via SDPA on indexed subset
        if non_shift_idx_t.numel() > 0:
            q_ns = q.index_select(1, non_shift_idx_t)
            k_ns = k.index_select(1, non_shift_idx_t)
            v_ns = v.index_select(1, non_shift_idx_t)
            out_ns = F.scaled_dot_product_attention(
                q_ns, k_ns, v_ns, scale=attn.scale)
            attn_out.index_copy_(1, non_shift_idx_t, out_ns)

        # Shift heads via single depthwise F.conv2d
        if shift_idx_t.numel() > 0:
            v_sh = v.index_select(1, shift_idx_t)             # (B, n_sh, N, Dh)
            out_sh = _apply_layer_shift_conv(
                v_sh, layer_state, grid, n_extra_tokens)
            attn_out.index_copy_(1, shift_idx_t, out_sh)

        out = attn_out.transpose(1, 2).reshape(B, N, C)
        out = attn.proj(out)
        if hasattr(attn, "proj_drop") and attn.proj_drop is not None:
            out = attn.proj_drop(out)
        return out

    return new_forward


def install_fused_shift_substitutions(
    model: nn.Module,
    kernel_map: Mapping[Tuple[int, int], np.ndarray],
    grid: Tuple[int, int],
    n_extra_tokens: int,
):
    """
    kernel_map : {(layer, head): ndarray of shape (2K+1, 2K+1)}.

    Install a fused-forward at every layer that has at least one
    shift head: shift heads skip softmax(QK^T) and go through one
    depthwise F.conv2d per layer; other heads go through SDPA.
    Returns a `saved` dict for restore_forwards().
    """
    by_layer: Dict[int, Dict[int, torch.Tensor]] = {}
    for (li, hi), ker in kernel_map.items():
        ker_t = torch.as_tensor(ker, dtype=torch.float32)
        by_layer.setdefault(li, {})[hi] = ker_t

    saved = {}
    for li, sub in by_layer.items():
        attn = model.blocks[li].attn
        device = next(attn.parameters()).device
        dtype = next(attn.parameters()).dtype
        head_indices = sorted(sub.keys())
        n_heads = attn.num_heads
        non_shift_indices = [h for h in range(n_heads) if h not in sub]
        non_shift_idx_t = torch.tensor(non_shift_indices,
                                        device=device, dtype=torch.long)
        shift_idx_t = torch.tensor(head_indices,
                                    device=device, dtype=torch.long)
        layer_state = _build_layer_shift_state(
            head_indices, sub, grid, device, dtype)
        saved[li] = attn.forward
        attn.forward = make_fused_shift_attention_forward(
            attn, layer_state, non_shift_idx_t, shift_idx_t,
            grid, n_extra_tokens)
    return saved


def restore_forwards(model, saved):
    for li, fwd in saved.items():
        model.blocks[li].attn.forward = fwd
