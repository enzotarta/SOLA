"""
Tests for operator_taxonomy.py.

Build synthetic attention tensors representing the canonical kernel
shapes (delta-shift, identity, multi-delta, Gaussian, fixed-target,
random) and verify that:

  1. compute_diagnostics produces sensible numbers
  2. classify_head returns the right category
  3. kernel_l1_residual_mean correctly distinguishes fixed-kernel
     heads (small residual) from content-dependent heads (large)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from operator_taxonomy import (
    compute_diagnostics, classify_head,
    _count_local_maxima, _gaussian_fit_residual,
    recenter_one_query, recentered_kernel_stack,
)


# Synthetic-attention factories

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def make_delta_shift_attn(H: int, W: int, n_img: int, dy: int, dx: int,
                           noise: float = 0.0,
                           rng: np.random.Generator | None = None):
    """
    Per (i, q) attention is a delta at (q + (dy, dx)) when in-bounds, else
    at q itself (boundary). Optional softmax noise: small uniform noise on
    log-attention before softmax.
    """
    rng = rng or np.random.default_rng(0)
    n = H * W
    a = np.zeros((n_img, n, n))
    for q in range(n):
        qy, qx = q // W, q % W
        ky, kx = qy + dy, qx + dx
        if 0 <= ky < H and 0 <= kx < W:
            target = ky * W + kx
        else:
            target = q
        for i in range(n_img):
            logits = noise * rng.standard_normal(n)
            logits[target] += 10.0
            a[i, q] = softmax(logits)
    return a


def make_identity_attn(H: int, W: int, n_img: int):
    """alpha[i, q, q] = 1, all else 0: self-only."""
    n = H * W
    a = np.zeros((n_img, n, n))
    for q in range(n):
        a[:, q, q] = 1.0
    return a


def make_gaussian_attn(H: int, W: int, n_img: int, sigma: float):
    """
    Translation-equivariant Gaussian: alpha[q, j] proportional to
    exp(-||j - q||^2 / 2 sigma^2). Same kernel for all q (after
    boundary renormalization).
    """
    n = H * W
    a = np.zeros((n_img, n, n))
    for q in range(n):
        qy, qx = q // W, q % W
        for j in range(n):
            jy, jx = j // W, j % W
            d2 = (jy - qy) ** 2 + (jx - qx) ** 2
            a[0, q, j] = np.exp(-d2 / (2 * sigma ** 2))
        a[0, q] /= a[0, q].sum()
    a[1:] = a[0]  # same across images (deterministic kernel)
    return a


def make_fixed_target_attn(H: int, W: int, n_img: int, ty: int, tx: int):
    """All queries always attend to a single fixed key (ty, tx)."""
    n = H * W
    a = np.zeros((n_img, n, n))
    target = ty * W + tx
    for q in range(n):
        a[:, q, target] = 1.0
    return a


def make_multi_delta_attn(H: int, W: int, n_img: int,
                           offsets: list[tuple[int, int]],
                           weights: list[float]):
    """
    Per (i, q), attention concentrates on q + each offset with the given
    weights; out-of-bounds offsets renormalize.
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    n = H * W
    a = np.zeros((n_img, n, n))
    for q in range(n):
        qy, qx = q // W, q % W
        ws = []
        ks = []
        for (dy, dx), w in zip(offsets, weights):
            ky, kx = qy + dy, qx + dx
            if 0 <= ky < H and 0 <= kx < W:
                ks.append(ky * W + kx)
                ws.append(w)
        ws = np.asarray(ws) / sum(ws)
        for k, w in zip(ks, ws):
            a[:, q, k] += w
    return a


def make_random_attn(H: int, W: int, n_img: int,
                      rng: np.random.Generator | None = None):
    """Per-(i, q) random softmax: content-dependent, no fixed kernel."""
    rng = rng or np.random.default_rng(42)
    n = H * W
    logits = rng.standard_normal((n_img, n, n))
    return softmax(logits, axis=-1)


# Tests

def test_recenter_one_query_basic():
    H, W = 4, 4
    K = 2
    a = np.zeros(H * W)
    a[0 * W + 1] = 1.0  # key at (0, 1)
    out = recenter_one_query(a, q_pos=(0, 0), H=H, W=W, K=K)
    # offset (0, 1) -> array index (K, K+1) = (2, 3)
    assert out[K, K + 1] == 1.0
    assert out.sum() == 1.0


def test_recentered_stack_shape():
    H, W, K = 4, 4, 2
    n_img, n = 3, H * W
    a = make_delta_shift_attn(H, W, n_img, dy=0, dx=1)
    stack = recentered_kernel_stack(a, H, W, K)
    assert stack.shape == (n_img * n, 2 * K + 1, 2 * K + 1)


def test_count_local_maxima():
    K = 2
    sz = 2 * K + 1
    flat = np.zeros((sz, sz))
    flat[2, 3] = 1.0
    assert _count_local_maxima(flat) == 1
    flat[1, 1] = 0.5  # secondary peak above threshold
    assert _count_local_maxima(flat) == 2
    # tiny secondary below threshold
    flat[0, 0] = 0.01
    assert _count_local_maxima(flat) == 2


def test_gaussian_fit_residual_perfect_gaussian():
    K = 5
    sz = 2 * K + 1
    yy, xx = np.meshgrid(np.arange(sz) - K, np.arange(sz) - K, indexing="ij")
    g = np.exp(-(yy ** 2 + xx ** 2) / 2.0)
    g /= g.sum()
    res = _gaussian_fit_residual(g)
    assert res < 0.05, f"Gaussian fit residual on a Gaussian: {res:.3f}"


def test_gaussian_fit_residual_delta_with_minvar():
    """
    A delta is mathematically a Gaussian with variance --> 0; without a
    minimum bandwidth the fit is perfect (residual 0). With min_var = 0.5,
    fitting a Gaussian to a delta should leave a substantial residual.
    """
    K = 5
    sz = 2 * K + 1
    delta = np.zeros((sz, sz))
    delta[K, K + 1] = 1.0
    res = _gaussian_fit_residual(delta, min_var=0.5)
    assert res > 0.5, f"Delta with min_var=0.5: residual={res:.3f}"


def test_delta_shift_diagnostics_and_classification():
    H, W = 14, 14  # match ViT-B grid
    n_img = 4
    a = make_delta_shift_attn(H, W, n_img, dy=0, dx=1)
    _, diag = compute_diagnostics(a, H, W, K=5)
    # Expectations:
    #  - escape mass at K=3 should be ~0 (single delta within central 3x3)
    #  - peak at offset (0, 1)
    #  - top1 mass ~1
    #  - kernel_l1_residual very small (perfect TE within boundary)
    #  - modality 1
    #  - argmax_concentration small (< 0.05) since each query has its own argmax
    assert diag.escape_mass_K3 < 0.05
    assert diag.peak_offset == (0, 1)
    assert diag.top1_mass > 0.85
    # Interior queries see a perfect delta, so modality_count == 1 even
    # though the boundary fallback adds some mass at the origin in the
    # full-grid mean kernel; the modality counter uses a 5% threshold.
    assert diag.modality_count <= 2
    # L1 residual on interior queries should be near zero for a deterministic
    # delta-shift head.
    assert diag.kernel_l1_residual_mean < 0.05
    assert diag.argmax_concentration < 0.05
    cat = classify_head(diag)
    assert cat == "delta_shift", f"got {cat}"


def test_identity_classified_as_self_only():
    H, W = 14, 14
    a = make_identity_attn(H, W, n_img=2)
    _, diag = compute_diagnostics(a, H, W, K=5)
    assert diag.peak_offset == (0, 0)
    assert diag.self_mass > 0.99
    cat = classify_head(diag)
    assert cat == "self_only", f"got {cat}"


def test_gaussian_classified_as_gaussian_local():
    H, W = 14, 14
    a = make_gaussian_attn(H, W, n_img=2, sigma=1.5)
    _, diag = compute_diagnostics(a, H, W, K=5)
    # Smooth single-mode local kernel
    assert diag.modality_count == 1
    assert diag.peak_offset == (0, 0)
    assert diag.gaussian_fit_residual < 0.10
    assert diag.escape_mass_K3 < 0.5
    cat = classify_head(diag)
    assert cat == "gaussian_local", f"got {cat}"


def test_fixed_target_classified_as_fixed_target():
    H, W = 14, 14
    a = make_fixed_target_attn(H, W, n_img=2, ty=5, tx=5)
    _, diag = compute_diagnostics(a, H, W, K=5)
    assert diag.argmax_concentration > 0.95
    cat = classify_head(diag)
    assert cat == "fixed_target", f"got {cat}"


def make_soft_shift_attn(H: int, W: int, n_img: int, dy: int, dx: int,
                          spread: float = 0.3,
                          rng: np.random.Generator | None = None):
    """
    Off-center peak that's translation-equivariant but with mass spread
    across the dominant cell and its neighbors. Same kernel for all
    interior queries (boundary truncated).
    """
    rng = rng or np.random.default_rng(1)
    n = H * W
    a = np.zeros((n_img, n, n))
    # Build a small kernel: 0.5 on (dy, dx), spread / 4 on each of 4 neighbors
    kernel_offsets = [
        ((dy, dx), 0.5),
        ((dy + 1, dx), spread / 4),
        ((dy - 1, dx), spread / 4),
        ((dy, dx + 1), spread / 4),
        ((dy, dx - 1), spread / 4),
        ((0, 0), 0.5 - spread),  # residual on origin (boundary-like)
    ]
    for q in range(n):
        qy, qx = q // W, q % W
        for (oy, ox), w in kernel_offsets:
            ky, kx = qy + oy, qx + ox
            if 0 <= ky < H and 0 <= kx < W:
                a[:, q, ky * W + kx] += w
        a[:, q] /= a[:, q].sum()
    return a


def test_soft_shift_classified_as_soft_shift():
    """A directionally consistent off-center peak with diffuse mass should
    be flagged as soft_shift (not delta_shift, not diffuse_local)."""
    H, W = 14, 14
    a = make_soft_shift_attn(H, W, n_img=2, dy=0, dx=1, spread=0.4)
    _, diag = compute_diagnostics(a, H, W, K=5)
    assert diag.peak_offset == (0, 1)
    assert diag.top1_mass < 0.5  # not concentrated enough for delta_shift
    assert diag.kernel_l1_residual_mean < 0.40  # but TE
    cat = classify_head(diag)
    assert cat == "soft_shift", f"got {cat}"


def test_multi_delta_classified_as_multi_delta():
    H, W = 14, 14
    a = make_multi_delta_attn(H, W, n_img=2,
                                offsets=[(0, 1), (0, -1)],
                                weights=[1.0, 1.0])
    _, diag = compute_diagnostics(a, H, W, K=5)
    assert diag.modality_count >= 2
    assert diag.top3_mass > 0.85
    assert diag.fingerprint_entropy < 2.0
    cat = classify_head(diag)
    assert cat == "multi_delta", f"got {cat}"


def test_random_classified_as_near_global_or_other():
    H, W = 14, 14
    a = make_random_attn(H, W, n_img=2)
    _, diag = compute_diagnostics(a, H, W, K=5)
    # Random softmax over 196 keys: most mass is far from q.
    assert diag.escape_mass_K3 > 0.6
    assert diag.kernel_l1_residual_mean > 0.2  # NOT a fixed kernel
    cat = classify_head(diag)
    assert cat in {"near_global", "other"}, f"got {cat}"


def test_l1_residual_distinguishes_fixed_vs_random():
    """L1 residual on interior queries should clearly separate a
    deterministic translation-equivariant kernel (Gaussian) from a
    content-dependent random one."""
    H, W = 14, 14
    fixed = make_gaussian_attn(H, W, n_img=2, sigma=1.5)
    random_ = make_random_attn(H, W, n_img=2)
    _, diag_fixed = compute_diagnostics(fixed, H, W, K=5)
    _, diag_rand = compute_diagnostics(random_, H, W, K=5)
    # Fixed kernel: per-query == mean kernel exactly (no variation),
    # so interior L1 residual is ~0.
    assert diag_fixed.kernel_l1_residual_mean < 0.05, (
        f"fixed l1 = {diag_fixed.kernel_l1_residual_mean:.3f}")
    # Content-dependent random: per-query kernels differ from the mean
    # substantially.
    assert diag_rand.kernel_l1_residual_mean > 0.5, (
        f"random l1 = {diag_rand.kernel_l1_residual_mean:.3f}")
    # And the gap should be at least an order of magnitude.
    assert diag_rand.kernel_l1_residual_mean > 10 * diag_fixed.kernel_l1_residual_mean


# Runner

if __name__ == "__main__":
    tests = [
        test_recenter_one_query_basic,
        test_recentered_stack_shape,
        test_count_local_maxima,
        test_gaussian_fit_residual_perfect_gaussian,
        test_gaussian_fit_residual_delta_with_minvar,
        test_delta_shift_diagnostics_and_classification,
        test_identity_classified_as_self_only,
        test_gaussian_classified_as_gaussian_local,
        test_fixed_target_classified_as_fixed_target,
        test_soft_shift_classified_as_soft_shift,
        test_multi_delta_classified_as_multi_delta,
        test_random_classified_as_near_global_or_other,
        test_l1_residual_distinguishes_fixed_vs_random,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed} / {len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
