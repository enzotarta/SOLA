"""Triton kernels for the shift and broadcast substitutions and for the remaining
attention heads, dispatched per head inside one layer-level forward.
"""
from __future__ import annotations
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def shift_conv_kernel(
    V_ptr, O_ptr,
    HEAD_IDX_ptr, TERM_DY_ptr, TERM_DX_ptr, TERM_W_ptr, NORM_ptr,
    N, Dh: tl.constexpr,
    H_GRID: tl.constexpr, W_GRID: tl.constexpr,
    N_EXTRA: tl.constexpr, MAX_TERMS: tl.constexpr,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_tjj, stride_tjt,        # TERM_* strides: (j, term)
    stride_nj, stride_ny, stride_nx,
    BLOCK_Q: tl.constexpr,
):
    """
    Sparse-kernel variant: each head's kernel is precomputed as up to
    MAX_TERMS non-zero (dy, dx, weight) tuples (zero-padded). The
    Triton inner loop iterates only over MAX_TERMS, which is typically
    1 (delta) or 3-5 (soft-shift), against the (2K+1)^2 cells of a dense
    kernel.

    TERM_DY, TERM_DX : int32 tensors of shape (n_shift, MAX_TERMS)
    TERM_W           : float tensor of shape (n_shift, MAX_TERMS)
                       (rows are zero-padded; padded rows have weight 0)
    NORM             : (n_shift, H_GRID, W_GRID)  precomputed in-grid
                       weight sums (matches the active terms; equals
                       sum of in-grid weights at each query)
    """
    pid_b = tl.program_id(0)
    pid_j = tl.program_id(1)
    pid_q = tl.program_id(2)

    h_idx = tl.load(HEAD_IDX_ptr + pid_j)

    q_start = pid_q * BLOCK_Q
    q_offs = q_start + tl.arange(0, BLOCK_Q)
    q_mask = q_offs < N

    spatial_q = q_offs - N_EXTRA
    is_spatial = spatial_q >= 0
    qy = tl.where(is_spatial, spatial_q // W_GRID, 0)
    qx = tl.where(is_spatial, spatial_q % W_GRID, 0)

    d_offs = tl.arange(0, Dh)

    out_acc = tl.zeros((BLOCK_Q, Dh), dtype=tl.float32)

    for t in tl.static_range(0, MAX_TERMS):
        dy = tl.load(TERM_DY_ptr + pid_j * stride_tjj + t * stride_tjt)
        dx = tl.load(TERM_DX_ptr + pid_j * stride_tjj + t * stride_tjt)
        w = tl.load(TERM_W_ptr  + pid_j * stride_tjj + t * stride_tjt)
        sy = qy + dy
        sx = qx + dx
        in_grid = ((sy >= 0) & (sy < H_GRID)
                   & (sx >= 0) & (sx < W_GRID)
                   & is_spatial)
        sy_safe = tl.where(in_grid, sy, 0)
        sx_safe = tl.where(in_grid, sx, 0)
        src = N_EXTRA + sy_safe * W_GRID + sx_safe

        v_ptrs = (V_ptr
                  + pid_b * stride_vb
                  + h_idx * stride_vh
                  + src[:, None] * stride_vn
                  + d_offs[None, :] * stride_vd)
        v_block = tl.load(v_ptrs, mask=q_mask[:, None], other=0.0)

        in_grid_f = tl.where(in_grid, 1.0, 0.0).to(tl.float32)
        scale = (w * in_grid_f).to(tl.float32)
        out_acc += v_block.to(tl.float32) * scale[:, None]

    # Renormalize by per-(j, qy, qx) in-grid weight sum.
    norm_safe_qy = tl.where(is_spatial, qy, 0)
    norm_safe_qx = tl.where(is_spatial, qx, 0)
    n_ptrs = (NORM_ptr
              + pid_j * stride_nj
              + norm_safe_qy * stride_ny
              + norm_safe_qx * stride_nx)
    norm = tl.load(n_ptrs, mask=q_mask, other=1.0).to(tl.float32)
    norm_clamp = tl.where(norm > 1e-8, norm, 1.0)
    out_acc = out_acc / norm_clamp[:, None]

    # Self-fallback for non-spatial queries (CLS) or zero-norm queries.
    fallback_to_self = (~is_spatial) | (norm <= 1e-8)
    v_q_ptrs = (V_ptr
                + pid_b * stride_vb
                + h_idx * stride_vh
                + q_offs[:, None] * stride_vn
                + d_offs[None, :] * stride_vd)
    v_q = tl.load(v_q_ptrs, mask=q_mask[:, None], other=0.0)
    out_acc = tl.where(fallback_to_self[:, None], v_q.to(tl.float32), out_acc)

    o_ptrs = (O_ptr
              + pid_b * stride_ob
              + h_idx * stride_oh
              + q_offs[:, None] * stride_on
              + d_offs[None, :] * stride_od)
    tl.store(o_ptrs, out_acc.to(v_q.dtype), mask=q_mask[:, None])


def _build_layer_state(
    head_indices: List[int],
    kernels_per_head: Dict[int, np.ndarray],
    grid: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    weight_eps: float = 1e-9,
):
    """
    Pack each head's kernel into a sparse list of non-zero (dy, dx, w)
    terms. Pad to MAX_TERMS = max non-zero count across heads in this
    layer. Also precompute the per-query in-grid weight sum used for
    boundary renormalization.
    """
    H, W = grid
    K = (next(iter(kernels_per_head.values())).shape[0] - 1) // 2
    n_shift = len(head_indices)

    weights_np = np.stack([kernels_per_head[h] for h in head_indices], axis=0)
    # (n_shift, 2K+1, 2K+1)

    # Extract non-zero terms per head
    per_head_terms = []
    max_terms = 1
    for j in range(n_shift):
        terms = []
        for dy in range(-K, K + 1):
            for dx in range(-K, K + 1):
                w = float(weights_np[j, dy + K, dx + K])
                if abs(w) >= weight_eps:
                    terms.append((dy, dx, w))
        if not terms:
            terms = [(0, 0, 1.0)]                  # degenerate identity
        per_head_terms.append(terms)
        max_terms = max(max_terms, len(terms))

    # Pad to max_terms with zero-weight rows
    term_dy = np.zeros((n_shift, max_terms), dtype=np.int32)
    term_dx = np.zeros((n_shift, max_terms), dtype=np.int32)
    term_w  = np.zeros((n_shift, max_terms), dtype=np.float32)
    for j, terms in enumerate(per_head_terms):
        for t, (dy, dx, w) in enumerate(terms):
            term_dy[j, t] = dy
            term_dx[j, t] = dx
            term_w[j, t]  = w

    # Per-(j, y, x) in-grid weight sum from the active terms
    norm = torch.zeros(n_shift, H, W, device=device, dtype=dtype)
    for j, terms in enumerate(per_head_terms):
        for (dy, dx, w) in terms:
            sy = torch.arange(H, device=device).unsqueeze(1) + dy
            sx = torch.arange(W, device=device).unsqueeze(0) + dx
            valid = (sy >= 0) & (sy < H) & (sx >= 0) & (sx < W)
            norm[j] += valid.to(dtype) * w

    head_idx_t = torch.tensor(head_indices, device=device, dtype=torch.int32)
    term_dy_t = torch.as_tensor(term_dy, device=device).contiguous()
    term_dx_t = torch.as_tensor(term_dx, device=device).contiguous()
    term_w_t  = torch.as_tensor(term_w,  device=device, dtype=dtype).contiguous()

    return {
        "K": K,
        "max_terms": max_terms,
        "term_dy": term_dy_t,
        "term_dx": term_dx_t,
        "term_w":  term_w_t,
        "norm":    norm.contiguous(),
        "head_idx": head_idx_t,
    }


def shift_conv_triton(V: torch.Tensor,
                       O: torch.Tensor,
                       state: dict,
                       grid: Tuple[int, int],
                       n_extra: int,
                       block_q: int = 64):
    """
    V, O : (B, H, N, Dh) on CUDA, contiguous, same dtype.
    state : output of `_build_layer_state` for this layer.
    """
    assert V.is_cuda and O.is_cuda
    assert V.shape == O.shape
    B, H_heads, N, Dh = V.shape
    H_grid, W_grid = grid
    n_shift = state["head_idx"].shape[0]
    n_q_blocks = (N + block_q - 1) // block_q

    norm = state["norm"]
    term_dy = state["term_dy"]
    term_dx = state["term_dx"]
    term_w  = state["term_w"]
    grid_launch = (B, n_shift, n_q_blocks)
    shift_conv_kernel[grid_launch](
        V, O,
        state["head_idx"], term_dy, term_dx, term_w, norm,
        N, Dh,
        H_grid, W_grid, n_extra, state["max_terms"],
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        term_dy.stride(0), term_dy.stride(1),
        norm.stride(0), norm.stride(1), norm.stride(2),
        BLOCK_Q=block_q,
    )


@triton.jit
def broadcast_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    HEAD_IDX_ptr,
    Q_IDX, SCALE,
    N, Dh: tl.constexpr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    BLOCK_K: tl.constexpr,
    BLOCK_Q_W: tl.constexpr,
):
    """
    Per-(B, j) program. For each broadcast head h_idx[j]:

      out_vec = softmax( Q[b, h, Q_IDX, :] @ K[b, h, :, :]^T * SCALE ) @ V[b, h, :, :]
              : (Dh,)

    Then write O[b, h, q, :] = out_vec for every q in [0, N).

    Cost: 1 row of QK^T computed (BLOCK_K loop), 1 row of softmax, 1
    Q-block-of-V product. It equals the full broadcast only on heads whose
    attention does not depend on the query.
    """
    pid_b = tl.program_id(0)
    pid_j = tl.program_id(1)

    h_idx = tl.load(HEAD_IDX_ptr + pid_j)

    d_offs = tl.arange(0, Dh)

    # Load Q[b, h, Q_IDX, :] (one Dh-vector). Q_IDX is a runtime int.
    q_ptr = (Q_ptr + pid_b * stride_qb + h_idx * stride_qh
             + Q_IDX * stride_qn + d_offs * stride_qd)
    q_vec = tl.load(q_ptr).to(tl.float32)                            # (Dh,)

    # Online-softmax accumulators
    m_i = -float("inf")
    l_i = 0.0
    out_acc = tl.zeros((Dh,), dtype=tl.float32)

    for k_start in range(0, N, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < N

        k_ptrs = (K_ptr + pid_b * stride_kb + h_idx * stride_kh
                  + k_offs[:, None] * stride_kn
                  + d_offs[None, :] * stride_kd)
        k_block = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)

        v_ptrs = (V_ptr + pid_b * stride_vb + h_idx * stride_vh
                  + k_offs[:, None] * stride_vn
                  + d_offs[None, :] * stride_vd)
        v_block = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)

        # scores: (BLOCK_K,) = sum_d q_vec[d] * k_block[k, d]
        scores = tl.sum(q_vec[None, :] * k_block.to(tl.float32), axis=1) * SCALE
        scores = tl.where(k_mask, scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)                                   # (BLOCK_K,)

        l_i = l_i * alpha + tl.sum(p, axis=0)
        # accumulate sum_k p[k] * v_block[k, :]
        out_acc = out_acc * alpha + tl.sum(
            p[:, None] * v_block.to(tl.float32), axis=0)
        m_i = m_new

    out_vec = out_acc / l_i                                          # (Dh,) fp32

    # Broadcast write: for q in 0..N step BLOCK_Q_W, write out_vec
    # to O[b, h_idx, q : q+BLOCK_Q_W, :].
    out_block = tl.broadcast_to(out_vec[None, :], (BLOCK_Q_W, Dh))
    for q_start in range(0, N, BLOCK_Q_W):
        q_offs = q_start + tl.arange(0, BLOCK_Q_W)
        q_mask = q_offs < N
        o_ptrs = (O_ptr + pid_b * stride_ob + h_idx * stride_oh
                  + q_offs[:, None] * stride_on
                  + d_offs[None, :] * stride_od)
        # Cast back to output dtype implicitly via tl.store
        tl.store(o_ptrs, out_block, mask=q_mask[:, None])


def broadcast_triton(Q: torch.Tensor, K: torch.Tensor,
                       V: torch.Tensor, O: torch.Tensor,
                       broadcast_head_idx: torch.Tensor,
                       q_idx: int,
                       scale: float,
                       block_k: int = 128,
                       block_q_w: int = 128):
    """
    Q, K, V, O : (B, H, N, Dh) on CUDA, contiguous.
    broadcast_head_idx : (n_broadcast,) int32 on device.
    """
    B, H_heads, N, Dh = V.shape
    n_b = broadcast_head_idx.shape[0]
    if n_b == 0:
        return
    grid = (B, n_b)
    broadcast_kernel[grid](
        Q, K, V, O,
        broadcast_head_idx,
        q_idx, scale,
        N, Dh,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        BLOCK_K=block_k,
        BLOCK_Q_W=block_q_w,
    )


@triton.jit
def attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    HEAD_IDX_ptr,
    SCALE,
    N, Dh: tl.constexpr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    FA-1 style attention kernel with online softmax. One program per
    (B, head_in_subset, q_block). Writes O[b, head_idx[j], q, :].

    Uses fp32 accumulators throughout. Loads Q/K/V at native dtype;
    `tl.dot` will use Tensor Cores when inputs are fp16/bf16 on
    Turing+, scalar ops on fp32.
    """
    pid_b = tl.program_id(0)
    pid_j = tl.program_id(1)
    pid_q = tl.program_id(2)

    h_idx = tl.load(HEAD_IDX_ptr + pid_j)

    q_start = pid_q * BLOCK_Q
    q_offs = q_start + tl.arange(0, BLOCK_Q)
    q_mask = q_offs < N
    d_offs = tl.arange(0, Dh)

    q_ptrs = (Q_ptr + pid_b * stride_qb + h_idx * stride_qh
              + q_offs[:, None] * stride_qn + d_offs[None, :] * stride_qd)
    q_block = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    m_i = tl.full([BLOCK_Q], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, Dh], dtype=tl.float32)

    for k_start in range(0, N, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < N

        k_ptrs = (K_ptr + pid_b * stride_kb + h_idx * stride_kh
                  + k_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd)
        k_block = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)

        v_ptrs = (V_ptr + pid_b * stride_vb + h_idx * stride_vh
                  + k_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd)
        v_block = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)

        # logits: (BQ, Dh) @ (Dh, BK) -> (BQ, BK)
        logits = tl.dot(q_block, tl.trans(k_block)).to(tl.float32) * SCALE
        logits = tl.where(k_mask[None, :], logits, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(logits, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(logits - m_new[:, None])                  # (BQ, BK) fp32

        l_i = l_i * alpha + tl.sum(p, axis=1)
        # Cast p to v dtype for the second matmul to use Tensor Cores
        # when v is fp16/bf16; has no effect on fp32.
        acc = acc * alpha[:, None] + tl.dot(p.to(v_block.dtype), v_block).to(tl.float32)
        m_i = m_new

    out = acc / l_i[:, None]

    o_ptrs = (O_ptr + pid_b * stride_ob + h_idx * stride_oh
              + q_offs[:, None] * stride_on + d_offs[None, :] * stride_od)
    tl.store(o_ptrs, out.to(q_block.dtype), mask=q_mask[:, None])


def attention_triton(Q: torch.Tensor, K: torch.Tensor,
                       V: torch.Tensor, O: torch.Tensor,
                       head_idx_t: torch.Tensor,
                       scale: float,
                       block_q: int = 64,
                       block_k: int = 64):
    """
    Q, K, V, O : (B, H, N, Dh) on CUDA, contiguous.
    head_idx_t : (n_subset,) int32. The kernel writes only those head
                 positions in O; other heads in O are left untouched.
    """
    B, H_heads, N, Dh = V.shape
    n_sub = head_idx_t.shape[0]
    if n_sub == 0:
        return
    n_q_blocks = (N + block_q - 1) // block_q
    grid = (B, n_sub, n_q_blocks)
    attention_kernel[grid](
        Q, K, V, O,
        head_idx_t,
        scale,
        N, Dh,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
    )


def make_triton_fused_attention_forward(
    attn,
    sdpa_idx_t: torch.Tensor,
    shift_state: dict,
    broadcast_idx_t: torch.Tensor,
    grid: Tuple[int, int],
    n_extra_tokens: int,
    broadcast_q_idx: int = 0,
    use_triton_for_sdpa: bool = False,
):
    """
    Three-way head routing:
      - shift heads      : Triton conv-on-V (no QK^T)
      - broadcast heads  : Triton single-row attention + broadcast (1/N QK^T)
      - other heads      : F.scaled_dot_product_attention OR Triton attention
                           (controlled by `use_triton_for_sdpa`). Triton is
                           preferred when all heads in a layer are routed
                           through it: writing to O at head-indexed positions
                           avoids index_select / index_copy_ overhead.
    """
    num_heads = attn.num_heads
    has_shift = shift_state is not None
    has_broadcast = broadcast_idx_t.numel() > 0

    def new_forward(x, *args, **kwargs):
        B, N, C = x.shape
        Dh = C // num_heads
        qkv = attn.qkv(x).reshape(B, N, 3, num_heads, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                              # (B, h, N, Dh)
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            q = attn.q_norm(q); k = attn.k_norm(k)

        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        attn_out = torch.empty_like(v)

        if has_shift:
            shift_conv_triton(v, attn_out, shift_state, grid, n_extra_tokens)

        if has_broadcast:
            broadcast_triton(q, k, v, attn_out, broadcast_idx_t,
                              q_idx=broadcast_q_idx, scale=attn.scale)

        if sdpa_idx_t.numel() > 0:
            if use_triton_for_sdpa:
                attention_triton(q, k, v, attn_out, sdpa_idx_t,
                                   scale=attn.scale)
            else:
                q_ns = q.index_select(1, sdpa_idx_t)
                k_ns = k.index_select(1, sdpa_idx_t)
                v_ns = v.index_select(1, sdpa_idx_t)
                out_ns = F.scaled_dot_product_attention(
                    q_ns, k_ns, v_ns, scale=attn.scale)
                attn_out.index_copy_(1, sdpa_idx_t, out_ns)

        out = attn_out.transpose(1, 2).reshape(B, N, C)
        out = attn.proj(out)
        if hasattr(attn, "proj_drop") and attn.proj_drop is not None:
            out = attn.proj_drop(out)
        return out

    return new_forward


def install_triton_fused_substitutions(
    model: nn.Module,
    kernel_map: Mapping[Tuple[int, int], np.ndarray] = None,
    broadcast_heads: List[Tuple[int, int]] = None,
    grid: Tuple[int, int] = None,
    n_extra_tokens: int = 0,
    broadcast_q_idx: int = 0,
    use_triton_for_sdpa: bool = False,
):
    """
    Install a combined Triton fused substitution at every layer that
    has either a shift head (kernel_map entry) or a broadcast head.

    kernel_map     : {(layer, head): kernel ndarray}  for shifts
    broadcast_heads: [(layer, head), ...]              for per-image broadcast
    grid           : (H, W) spatial grid
    n_extra_tokens : leading non-spatial tokens
    broadcast_q_idx: which Q row to use for broadcast (default 0 = CLS)

    Returns `saved` dict for restore_forwards().
    """
    kernel_map = kernel_map or {}
    broadcast_heads = broadcast_heads or []

    shifts_by_layer: Dict[int, Dict[int, np.ndarray]] = {}
    for (li, hi), ker in kernel_map.items():
        shifts_by_layer.setdefault(li, {})[hi] = np.asarray(ker)
    bcast_by_layer: Dict[int, List[int]] = {}
    for (li, hi) in broadcast_heads:
        bcast_by_layer.setdefault(li, []).append(hi)

    if use_triton_for_sdpa:
        # Install on all layers: even layers with no shift/broadcast
        # benefit from running through the Triton attention kernel
        # (which writes directly into the output buffer), avoiding
        # F.scaled_dot_product_attention dispatch overhead for them.
        affected_layers = list(range(len(model.blocks)))
    else:
        affected_layers = sorted(set(shifts_by_layer.keys())
                                  | set(bcast_by_layer.keys()))
    saved = {}
    for li in affected_layers:
        attn = model.blocks[li].attn
        device = next(attn.parameters()).device
        dtype = next(attn.parameters()).dtype
        n_heads = attn.num_heads

        sub = shifts_by_layer.get(li, {})
        head_indices = sorted(sub.keys())
        if head_indices:
            shift_state = _build_layer_state(head_indices, sub, grid,
                                                device, dtype)
        else:
            shift_state = None

        bcast_indices = sorted(bcast_by_layer.get(li, []))
        broadcast_idx_t = torch.tensor(bcast_indices, device=device,
                                         dtype=torch.int32)

        substituted = set(head_indices) | set(bcast_indices)
        sdpa_indices = [h for h in range(n_heads) if h not in substituted]
        sdpa_idx_t = torch.tensor(sdpa_indices, device=device,
                                    dtype=torch.long)

        saved[li] = attn.forward
        attn.forward = make_triton_fused_attention_forward(
            attn, sdpa_idx_t, shift_state, broadcast_idx_t,
            grid, n_extra_tokens, broadcast_q_idx,
            use_triton_for_sdpa=use_triton_for_sdpa)
    return saved


def restore_forwards(model, saved):
    for li, fwd in saved.items():
        model.blocks[li].attn.forward = fwd
