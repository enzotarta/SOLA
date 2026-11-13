"""
Tests for kernel_substitution.py.

When the kernel is a one-hot delta at offset (dy, dx), the
boundary-renormalized 2D-conv substitution reduces to torch.roll with self at
the boundary, which is what shift_substitution implements. Checked on synthetic
value tensors.

Other tests:
  - identity kernel (delta at origin) reproduces V exactly
  - normalized Gaussian kernel preserves mean across queries
  - off-grid-only kernel triggers the self-fallback path
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from kernel_substitution import (
    apply_kernel_to_values, make_delta_kernel,
    kernel_from_diagnostics_mean,
)


def fast_shift_with_self_at_boundary(v_spatial: torch.Tensor, dy: int, dx: int,
                                      H: int, W: int) -> torch.Tensor:
    """Reference impl: same logic as shift_substitution.py."""
    B, _, D = v_spatial.shape
    v2d = v_spatial.reshape(B, H, W, D)
    qy = torch.arange(H).view(H, 1).expand(H, W)
    qx = torch.arange(W).view(1, W).expand(H, W)
    ty = qy + dy
    tx = qx + dx
    in_bounds = (ty >= 0) & (ty < H) & (tx >= 0) & (tx < W)
    ty = torch.where(in_bounds, ty, qy)
    tx = torch.where(in_bounds, tx, qx)
    out = v2d[:, ty, tx, :]
    return out.reshape(B, H * W, D)


def test_delta_kernel_matches_shift():
    """Delta kernel substitution reproduces shift_substitution exactly."""
    H, W = 14, 14
    B, D = 2, 8
    n_extra = 1
    K = 5

    rng = np.random.default_rng(0)
    v_spatial_np = rng.standard_normal((B, H * W, D)).astype(np.float32)
    v_extra_np = rng.standard_normal((B, n_extra, D)).astype(np.float32)
    v_full = torch.from_numpy(np.concatenate([v_extra_np, v_spatial_np], axis=1))

    for dy, dx in [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-2, 3)]:
        kernel = torch.from_numpy(make_delta_kernel(K, dy, dx))
        out = apply_kernel_to_values(v_full, kernel, (H, W), n_extra)

        # Reference
        ref_spatial = fast_shift_with_self_at_boundary(
            torch.from_numpy(v_spatial_np), dy, dx, H, W)
        ref = torch.cat([torch.from_numpy(v_extra_np), ref_spatial], dim=1)

        diff = (out - ref).abs().max().item()
        assert diff < 1e-5, f"Delta ({dy},{dx}) mismatch: max diff {diff}"


def test_identity_kernel_returns_input():
    H, W = 14, 14
    B, D = 1, 4
    n_extra = 1
    K = 5
    v = torch.randn(B, n_extra + H * W, D)
    kernel = torch.from_numpy(make_delta_kernel(K, 0, 0))
    out = apply_kernel_to_values(v, kernel, (H, W), n_extra)
    diff = (out - v).abs().max().item()
    assert diff < 1e-6, f"Identity mismatch: {diff}"


def test_normalized_kernel_preserves_uniform_v():
    """If V is constant across queries, any normalized kernel returns V."""
    H, W = 10, 10
    B, D = 1, 4
    n_extra = 1
    K = 3
    # All spatial tokens have the same value vector
    spatial = torch.ones(B, H * W, D) * 1.7
    extra = torch.ones(B, n_extra, D) * -0.3
    v = torch.cat([extra, spatial], dim=1)

    # Random normalized kernel
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((2 * K + 1, 2 * K + 1)) ** 2
    raw = raw / raw.sum()
    kernel = torch.from_numpy(raw.astype(np.float32))

    out = apply_kernel_to_values(v, kernel, (H, W), n_extra)
    # Spatial output should still be 1.7 everywhere
    spatial_out = out[:, n_extra:, :]
    assert torch.allclose(spatial_out, spatial, atol=1e-5)
    # Extra tokens unchanged
    assert torch.allclose(out[:, :n_extra, :], extra)


def test_self_fallback_when_kernel_off_grid():
    """If the kernel has all mass off-grid for a particular query, that
    query's output should fall back to V[q] (self-value)."""
    H, W = 4, 4
    B, D = 1, 3
    n_extra = 0
    K = 2
    v = torch.arange(B * H * W * D, dtype=torch.float32).reshape(B, H * W, D)

    # A delta at (-2, 0): for queries in row 0 or 1, q + (-2, 0) is out of grid
    kernel = torch.from_numpy(make_delta_kernel(K, -2, 0))
    out = apply_kernel_to_values(v, kernel, (H, W), n_extra)
    # For (qy, qx) with qy < 2: out should equal V[qy, qx] (self)
    out2d = out.reshape(B, H, W, D)
    v2d = v.reshape(B, H, W, D)
    for qy in [0, 1]:
        for qx in range(W):
            assert torch.allclose(out2d[0, qy, qx], v2d[0, qy, qx]), (
                f"Boundary fallback failed at q=({qy},{qx})")
    # For (qy, qx) with qy >= 2: out should equal V[qy - 2, qx]
    for qy in [2, 3]:
        for qx in range(W):
            assert torch.allclose(out2d[0, qy, qx], v2d[0, qy - 2, qx]), (
                f"Interior shift failed at q=({qy},{qx})")


def test_kernel_from_diagnostics_mean_normalizes():
    raw = np.array([[0.1, 0.2, 0.0],
                    [0.0, 0.5, 0.1],
                    [0.0, 0.0, 0.0]], dtype=np.float32)
    out = kernel_from_diagnostics_mean(raw)
    assert abs(out.sum() - 1.0) < 1e-6
    assert out.dtype == np.float32


# Runner

if __name__ == "__main__":
    tests = [
        test_delta_kernel_matches_shift,
        test_identity_kernel_returns_input,
        test_normalized_kernel_preserves_uniform_v,
        test_self_fallback_when_kernel_off_grid,
        test_kernel_from_diagnostics_mean_normalizes,
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
