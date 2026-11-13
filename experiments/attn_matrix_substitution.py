"""Forward functions that replace softmax(QK^T) of chosen heads.

Includes the fixed-matrix, per-image broadcast and per-image rank-k variants.
The rank-k variant uses torch.svd_lowrank(q=k+4, niter=2), a randomized
approximation of the best rank-k matrix.
"""
from __future__ import annotations
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn


def make_matrix_substituted_attention_forward(attn, head_matrix_map):
    """
    head_matrix_map : {head_idx: torch.Tensor of shape (N, N)}
       For each head listed, the head's attention matrix is replaced
       with the given fixed matrix.
    """
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
        # Override matrices at the chosen heads
        for h, A in head_matrix_map.items():
            if A.shape[-1] != N:
                raise ValueError(
                    f"A_bar size {A.shape[-1]} != model N {N} at head {h}")
            a[:, h] = A.unsqueeze(0).expand(B, N, N)
        attn_out = a @ v
        out = attn_out.transpose(1, 2).reshape(B, N, C)
        out = attn.proj(out)
        if hasattr(attn, "proj_drop") and attn.proj_drop is not None:
            out = attn.proj_drop(out)
        return out

    return new_forward


def install_matrix_substitutions(model: nn.Module,
                                  matrix_map: Mapping[Tuple[int, int], np.ndarray]):
    """
    matrix_map : {(layer, head): ndarray of shape (N, N)}.

    Replaces the head's attention matrix with the given fixed one
    (image-independent). Returns a `saved` dict to pass to
    restore_forwards().
    """
    by_layer: Dict[int, Dict[int, torch.Tensor]] = {}
    for (li, hi), A in matrix_map.items():
        A_t = torch.as_tensor(A, dtype=torch.float32)
        by_layer.setdefault(li, {})[hi] = A_t

    saved = {}
    for li, sub in by_layer.items():
        attn = model.blocks[li].attn
        device = next(attn.parameters()).device
        dtype = next(attn.parameters()).dtype
        sub = {h: A.to(device=device, dtype=dtype) for h, A in sub.items()}
        saved[li] = attn.forward
        attn.forward = make_matrix_substituted_attention_forward(attn, sub)
    return saved


def restore_forwards(model, saved):
    for li, fwd in saved.items():
        model.blocks[li].attn.forward = fwd


# Broadcast substitution: per-image but query-independent attention

def make_broadcast_substituted_attention_forward(attn, head_set):
    """
    For each head h in head_set: replace its attention matrix A[b, h, q, k]
    with the per-image mean over q, broadcast back to all queries:
        A_br[b, h, q, k] = (1/N) sum_q' A[b, h, q', k]
    The head remains image-dependent (uses real Q, K) but becomes
    query-independent (the same row for every q).
    """
    head_set = set(head_set)

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
        for h in head_set:
            row_mean = a[:, h].mean(dim=-2, keepdim=True)        # (B, 1, N)
            a[:, h] = row_mean.expand(B, N, N)
        attn_out = a @ v
        out = attn_out.transpose(1, 2).reshape(B, N, C)
        out = attn.proj(out)
        if hasattr(attn, "proj_drop") and attn.proj_drop is not None:
            out = attn.proj_drop(out)
        return out

    return new_forward


def install_broadcast_substitutions(model: nn.Module,
                                     heads: List[Tuple[int, int]]):
    """
    For each head in `heads`, replace its attention matrix with the
    per-image mean-over-queries row, broadcast across all queries. The
    head remains image-dependent and QK^T is still computed in full, so this
    version tests the structure and saves no FLOPs.
    Returns a `saved` dict to pass to restore_forwards().
    """
    by_layer: Dict[int, List[int]] = {}
    for (li, hi) in heads:
        by_layer.setdefault(li, []).append(hi)
    saved = {}
    for li, hi_list in by_layer.items():
        attn = model.blocks[li].attn
        saved[li] = attn.forward
        attn.forward = make_broadcast_substituted_attention_forward(attn, hi_list)
    return saved


# Efficient broadcast: compute ONE row of QK^T per substituted head.
# This is the practical FLOPs-saving variant of broadcast: instead of
# computing the full softmax(QK^T) and collapsing, we only ever compute
# the attention from one query (q_idx, default 0 = CLS) and broadcast it.
# Validation: should match install_broadcast_substitutions on heads where
# attention is empirically query-independent.

def make_efficient_broadcast_forward(attn, head_set, q_idx=0):
    head_set_t = sorted(set(head_set))

    def new_forward(x, *args, **kwargs):
        B, N, C = x.shape
        num_heads = attn.num_heads
        Dh = C // num_heads
        qkv = attn.qkv(x).reshape(
            B, N, 3, num_heads, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)               # each (B, h, N, Dh)
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            q = attn.q_norm(q); k = attn.k_norm(k)

        attn_out = torch.empty_like(v)        # (B, h, N, Dh)

        # Non-substituted heads: standard attention path
        all_h = set(range(num_heads))
        ns_heads = sorted(all_h - set(head_set_t))
        if ns_heads:
            ns_idx = torch.tensor(ns_heads, device=q.device)
            q_ns = q.index_select(1, ns_idx)
            k_ns = k.index_select(1, ns_idx)
            v_ns = v.index_select(1, ns_idx)
            a_ns = (q_ns @ k_ns.transpose(-2, -1)) * attn.scale
            a_ns = a_ns.softmax(dim=-1)
            if hasattr(attn, "attn_drop"):
                a_ns = attn.attn_drop(a_ns)
            attn_out[:, ns_idx] = a_ns @ v_ns

        # Substituted heads: one row of QK^T, broadcast to all queries
        if head_set_t:
            sub_idx = torch.tensor(head_set_t, device=q.device)
            q_sub_one = q.index_select(1, sub_idx)[:, :, q_idx:q_idx+1]
            #   (B, |sub|, 1, Dh)
            k_sub = k.index_select(1, sub_idx)            # (B, |sub|, N, Dh)
            v_sub = v.index_select(1, sub_idx)            # (B, |sub|, N, Dh)
            a_row = (q_sub_one @ k_sub.transpose(-2, -1)) * attn.scale
            a_row = a_row.softmax(dim=-1)                 # (B, |sub|, 1, N)
            if hasattr(attn, "attn_drop"):
                a_row = attn.attn_drop(a_row)
            out_one = a_row @ v_sub                       # (B, |sub|, 1, Dh)
            attn_out[:, sub_idx] = out_one.expand(B, len(head_set_t), N, Dh)

        out = attn_out.transpose(1, 2).reshape(B, N, C)
        out = attn.proj(out)
        if hasattr(attn, "proj_drop") and attn.proj_drop is not None:
            out = attn.proj_drop(out)
        return out

    return new_forward


def install_efficient_broadcast_substitutions(
        model: nn.Module, heads: List[Tuple[int, int]], q_idx: int = 0):
    """
    Like install_broadcast_substitutions but uses only ONE row of Q
    (q_idx, default 0 = CLS) per substituted head, skipping the per-query
    QK^T matmul and softmax. This is the practical FLOPs-saving variant.
    """
    by_layer: Dict[int, List[int]] = {}
    for (li, hi) in heads:
        by_layer.setdefault(li, []).append(hi)
    saved = {}
    for li, hi_list in by_layer.items():
        attn = model.blocks[li].attn
        saved[li] = attn.forward
        attn.forward = make_efficient_broadcast_forward(attn, hi_list, q_idx)
    return saved


# Rank-k attention substitution (SVD truncation per image, structural test)

def make_rank_k_substituted_attention_forward(attn, head_set, k_rank=1):
    head_set_t = sorted(set(head_set))

    def new_forward(x, *args, **kwargs):
        B, N, C = x.shape
        num_heads = attn.num_heads
        Dh = C // num_heads
        qkv = attn.qkv(x).reshape(
            B, N, 3, num_heads, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            q = attn.q_norm(q); k = attn.k_norm(k)
        a = (q @ k.transpose(-2, -1)) * attn.scale
        a = a.softmax(dim=-1)                          # (B, h, N, N)
        if hasattr(attn, "attn_drop"):
            a = attn.attn_drop(a)
        # Batched per-image rank-k via svd_lowrank (iterative, fast for small k)
        if head_set_t:
            sub_idx = torch.tensor(head_set_t, device=a.device, dtype=torch.long)
            A_sub = a.index_select(1, sub_idx)         # (B, n_sub, N, N)
            B0, ns, N0, _ = A_sub.shape
            A_flat = A_sub.reshape(B0 * ns, N0, N0)
            U, S, V = torch.svd_lowrank(A_flat, q=k_rank + 4, niter=2)
            # Truncate to top-k
            U_k = U[..., :k_rank]                       # (B*ns, N, k)
            S_k = S[..., :k_rank]                       # (B*ns, k)
            V_k = V[..., :k_rank]                       # (B*ns, N, k)
            A_rk_flat = U_k @ torch.diag_embed(S_k) @ V_k.transpose(-2, -1)
            A_rk = A_rk_flat.reshape(B0, ns, N0, N0)
            for j, h in enumerate(head_set_t):
                a[:, h] = A_rk[:, j]
        attn_out = a @ v
        out = attn_out.transpose(1, 2).reshape(B, N, C)
        out = attn.proj(out)
        if hasattr(attn, "proj_drop") and attn.proj_drop is not None:
            out = attn.proj_drop(out)
        return out

    return new_forward


def install_rank_k_substitutions(model: nn.Module,
                                   heads: List[Tuple[int, int]],
                                   k_rank: int = 1):
    """
    For each head in `heads`, replace its attention matrix, per image,
    with a rank-k approximation from torch.svd_lowrank. This is a
    structural test and saves no FLOPs.
    """
    by_layer: Dict[int, List[int]] = {}
    for (li, hi) in heads:
        by_layer.setdefault(li, []).append(hi)
    saved = {}
    for li, hi_list in by_layer.items():
        attn = model.blocks[li].attn
        saved[li] = attn.forward
        attn.forward = make_rank_k_substituted_attention_forward(
            attn, hi_list, k_rank=k_rank)
    return saved


# Mean attention recorder (for building A_bar per head)

def record_mean_attention(
    model: nn.Module,
    eval_loader,
    device: str,
    n_batches: int,
    head_set: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Record A_bar[q, k] for each (layer, head) in head_set, averaged
    over n_batches of images.
    """
    from operator_taxonomy import make_attention_recorder_class
    AttentionRecorder = make_attention_recorder_class()
    rec = AttentionRecorder(model)
    with torch.no_grad():
        for bi, batch in enumerate(eval_loader):
            if bi >= n_batches:
                break
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device)
            _ = model(batch)
    attn_by_layer = rec.collect()
    rec.remove()

    out = {}
    layers_needed = sorted({li for (li, _) in head_set})
    for li in layers_needed:
        a = attn_by_layer[li].numpy()  # (B', n_heads, N, N)
        for (li2, hi) in head_set:
            if li2 != li:
                continue
            out[(li, hi)] = a[:, hi].mean(axis=0).astype(np.float32)
    return out


# Scrambling: row-shuffle (preserve per-row distribution, randomize q->k)

def scramble_matrix_row_perm(A: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Random row permutation: A_scr[q, :] = A[perm[q], :]. This preserves
    the per-query attention distribution but maps it to a different
    query, breaking the q -> k association.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(A.shape[0])
    return A[perm].copy()
