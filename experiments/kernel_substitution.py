"""Replace a head's output with a 2D convolution of V against a fixed kernel.

Out-of-grid offsets are dropped and the kernel is renormalised at the boundary;
a query with no in-grid mass falls back to its own value.
"""
from __future__ import annotations
from typing import Dict, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn


# Per-head kernel application (boundary-renormalized 2D conv on V)

def apply_kernel_to_values(v_head: torch.Tensor, kernel: torch.Tensor,
                            grid: Tuple[int, int], n_extra_tokens: int
                            ) -> torch.Tensor:
    """
    Compute output[q] = sum_d kernel[d] * v_head[q + d] for d in the
    (2K+1) x (2K+1) window around q, with out-of-grid offsets dropped
    and per-query renormalization so the surviving weights sum to the
    same total as the in-grid kernel mass.

    v_head : (B, n_extra_tokens + H*W, D)
    kernel : (2K+1, 2K+1) tensor on the same device, real-valued. The
             kernel does NOT need to sum to 1; if it does (the typical
             case from a softmax attention head's mean), the
             renormalization is trivial for queries with full window
             coverage.
    grid   : (H, W) spatial grid size
    n_extra_tokens : number of leading non-spatial tokens (CLS, registers)

    Returns the same shape as v_head. Extra tokens pass through unchanged.
    """
    B, N, D = v_head.shape
    H, W = grid
    assert N == n_extra_tokens + H * W, (
        f"v_head shape mismatch: {N=} vs {n_extra_tokens=} + {H*W=}")

    # Split off non-spatial tokens
    v_extra = v_head[:, :n_extra_tokens, :] if n_extra_tokens > 0 else None
    v_spatial = v_head[:, n_extra_tokens:, :].reshape(B, H, W, D)

    K = (kernel.shape[0] - 1) // 2
    assert kernel.shape == (2 * K + 1, 2 * K + 1)

    out = torch.zeros_like(v_spatial)
    norm = torch.zeros(H, W, device=v_head.device, dtype=v_head.dtype)

    qy = torch.arange(H, device=v_head.device)
    qx = torch.arange(W, device=v_head.device)

    for dy in range(-K, K + 1):
        for dx in range(-K, K + 1):
            w = kernel[dy + K, dx + K]
            if w.abs().item() < 1e-9:
                continue
            # source y/x for each q, shape (H,) and (W,)
            sy = qy + dy
            sx = qx + dx
            valid_y = (sy >= 0) & (sy < H)
            valid_x = (sx >= 0) & (sx < W)
            mask = (valid_y.unsqueeze(1) & valid_x.unsqueeze(0))   # (H, W)
            mask_f = mask.to(v_head.dtype)

            # gather v_spatial[q + (dy, dx)] for each q. Use roll then mask
            # off-grid entries (which would alias from the other side).
            shifted = torch.roll(v_spatial, shifts=(-dy, -dx), dims=(1, 2))
            out = out + shifted * (mask_f * w).unsqueeze(0).unsqueeze(-1)
            norm = norm + mask_f * w

    # Per-query renormalization: divide by total in-grid weight
    norm_safe = norm.unsqueeze(0).unsqueeze(-1).clamp(min=1e-8)
    out = out / norm_safe

    # Self-fallback for queries whose entire kernel mass is off-grid.
    zero_mask = (norm.abs() < 1e-9).unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
    if zero_mask.any():
        out = torch.where(zero_mask.expand_as(out), v_spatial, out)

    out_flat = out.reshape(B, H * W, D)
    if v_extra is not None:
        out_flat = torch.cat([v_extra, out_flat], dim=1)
    return out_flat


# Substituted attention forward

def make_kernel_substituted_attention_forward(attn, head_kernel_map,
                                                grid, n_extra_tokens):
    """
    head_kernel_map: {head_idx: torch.Tensor of shape (2K+1, 2K+1)}
      Each kernel is applied (with boundary-renormalized conv on V) in
      place of that head's standard attention output.
    """
    H, W = grid

    def new_forward(x, *args, **kwargs):
        B, N, C = x.shape
        Dh = C // attn.num_heads

        qkv = attn.qkv(x).reshape(
            B, N, 3, attn.num_heads, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            q = attn.q_norm(q); k = attn.k_norm(k)
        a = (q @ k.transpose(-2, -1)) * attn.scale
        a = a.softmax(dim=-1)
        attn_out = a @ v                                # (B, heads, N, Dh)

        for h, kernel in head_kernel_map.items():
            v_h = v[:, h]                                # (B, N, Dh)
            attn_out[:, h] = apply_kernel_to_values(
                v_h, kernel, grid, n_extra_tokens)

        out = attn_out.transpose(1, 2).reshape(B, N, C)
        out = attn.proj(out)
        if hasattr(attn, "proj_drop") and attn.proj_drop is not None:
            out = attn.proj_drop(out)
        return out

    return new_forward


def install_kernel_substitutions(model: nn.Module,
                                  kernel_map: Mapping[Tuple[int, int], np.ndarray],
                                  grid: Tuple[int, int],
                                  n_extra_tokens: int):
    """
    kernel_map : {(layer, head): ndarray of shape (2K+1, 2K+1)}.

    For each (layer, head), at forward time, replace the head's
    attention output with a 2D convolution of V against the kernel.

    Returns a `saved` dict to pass to restore_forwards().
    """
    by_layer: Dict[int, Dict[int, torch.Tensor]] = {}
    for (li, hi), ker in kernel_map.items():
        ker_t = torch.as_tensor(ker, dtype=torch.float32)
        by_layer.setdefault(li, {})[hi] = ker_t

    saved = {}
    for li, sub in by_layer.items():
        attn = model.blocks[li].attn
        # Move kernels to the attention module's device once
        device = next(attn.parameters()).device
        sub = {h: ker.to(device) for h, ker in sub.items()}
        saved[li] = attn.forward
        attn.forward = make_kernel_substituted_attention_forward(
            attn, sub, grid, n_extra_tokens)
    return saved


def restore_forwards(model, saved):
    for li, fwd in saved.items():
        model.blocks[li].attn.forward = fwd


# Kernel constructors

def make_delta_kernel(K: int, dy: int, dx: int) -> np.ndarray:
    """One-hot kernel at offset (dy, dx) within a (2K+1) x (2K+1) window."""
    assert -K <= dy <= K and -K <= dx <= K
    out = np.zeros((2 * K + 1, 2 * K + 1), dtype=np.float32)
    out[dy + K, dx + K] = 1.0
    return out


def kernel_from_diagnostics_mean(mean_kernel: np.ndarray) -> np.ndarray:
    """
    Take the mean recentered kernel produced by operator_taxonomy and
    return it normalized to sum to 1 (so it's a proper convex-combination
    weighting).
    """
    Z = float(mean_kernel.sum())
    if Z <= 0:
        raise ValueError("kernel has nonpositive mass")
    return (mean_kernel / Z).astype(np.float32)
