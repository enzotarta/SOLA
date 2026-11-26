"""Compare the Triton kernels with the PyTorch reference on synthetic inputs."""
import torch
import numpy as np

from fused_substitutions_triton import (
    shift_conv_triton, broadcast_triton, _build_layer_state,
)
from kernel_substitution import apply_kernel_to_values


def make_kernel(K, mode="delta"):
    if mode == "delta":
        ker = np.zeros((2*K+1, 2*K+1), dtype=np.float32)
        ker[K + 1, K] = 1.0
        return ker
    elif mode == "soft":
        ker = np.zeros((2*K+1, 2*K+1), dtype=np.float32)
        ker[K, K] = 0.1
        ker[K + 1, K] = 0.6
        ker[K - 1, K] = 0.1
        ker[K, K + 1] = 0.1
        ker[K, K - 1] = 0.1
        return ker / ker.sum()
    elif mode == "random":
        rng = np.random.default_rng(42)
        ker = rng.uniform(0, 1, (2*K+1, 2*K+1)).astype(np.float32)
        return ker / ker.sum()
    raise ValueError(mode)


def test_shift():
    torch.manual_seed(0)
    cases = [
        (2, 6,  32, 14, 14, 1, 3, "delta",  torch.float32),
        (2, 6,  32, 14, 14, 1, 3, "soft",   torch.float32),
        (4, 12, 64, 14, 14, 1, 3, "soft",   torch.float32),
        (4, 12, 64, 37, 37, 1, 3, "soft",   torch.float32),
        (2, 12, 64, 14, 14, 5, 3, "soft",   torch.float32),
        (2, 6,  32, 14, 14, 1, 3, "soft",   torch.float16),
        (2, 6,  32, 14, 14, 1, 3, "random", torch.float32),
    ]
    for B, H_heads, Dh, H_grid, W_grid, n_extra, K, mode, dtype in cases:
        N = n_extra + H_grid * W_grid
        V = torch.randn(B, H_heads, N, Dh, device='cuda', dtype=dtype)
        O = V.clone()
        head_indices = [1, 3, 4]
        kernel = make_kernel(K, mode=mode)
        kernels = {h: kernel for h in head_indices}
        state = _build_layer_state(head_indices, kernels, (H_grid, W_grid),
                                     device=V.device, dtype=dtype)
        shift_conv_triton(V.contiguous(), O, state,
                            (H_grid, W_grid), n_extra)
        ref = V.clone()
        kernel_t = torch.as_tensor(kernel, device=V.device, dtype=dtype)
        for h in head_indices:
            ref[:, h] = apply_kernel_to_values(
                V[:, h], kernel_t, (H_grid, W_grid), n_extra)
        max_diff = max(
            (O[:, h] - ref[:, h]).abs().max().item() for h in head_indices)
        for h in [0, 2, 5]:
            if h >= H_heads: continue
            assert (O[:, h] - V[:, h]).abs().max().item() == 0.0
        tol = 5e-3 if dtype == torch.float16 else 1e-5
        ok = max_diff <= tol
        print(f"[shift] B={B} H={H_heads} Dh={Dh} grid={H_grid}x{W_grid} "
              f"extra={n_extra} mode={mode:>6s} dtype={str(dtype):>15s}: "
              f"max_diff={max_diff:.3e}  {'OK' if ok else 'FAIL'}")
        assert ok


def broadcast_reference(Q, K, V, broadcast_heads, q_idx, scale):
    """One-row attention + broadcast, in PyTorch."""
    B, H, N, Dh = V.shape
    O = V.clone()
    for h in broadcast_heads:
        # (B, 1, Dh) @ (B, Dh, N) -> (B, 1, N)
        scores = (Q[:, h, q_idx:q_idx+1] @ K[:, h].transpose(-2, -1)) * scale
        weights = scores.softmax(dim=-1)                          # (B, 1, N)
        out_vec = weights @ V[:, h]                               # (B, 1, Dh)
        O[:, h] = out_vec.expand(B, N, Dh)
    return O


def test_broadcast():
    torch.manual_seed(0)
    cases = [
        # (B, H_heads, Dh, N, q_idx, dtype)
        (2, 6,  32, 197,  0, torch.float32),
        (4, 12, 64, 197,  0, torch.float32),
        (4, 12, 64, 577,  0, torch.float32),
        (2, 12, 64, 1374, 0, torch.float32),
        (2, 6,  32, 197,  3, torch.float32),
        (2, 6,  32, 197,  0, torch.float16),
    ]
    for B, H_heads, Dh, N, q_idx, dtype in cases:
        Q = torch.randn(B, H_heads, N, Dh, device='cuda', dtype=dtype)
        K = torch.randn(B, H_heads, N, Dh, device='cuda', dtype=dtype)
        V = torch.randn(B, H_heads, N, Dh, device='cuda', dtype=dtype)
        O = V.clone()
        broadcast_heads = [1, 3, 4]
        head_idx_t = torch.tensor(broadcast_heads, device='cuda',
                                    dtype=torch.int32)
        scale = Dh ** -0.5

        broadcast_triton(Q.contiguous(), K.contiguous(), V.contiguous(),
                          O, head_idx_t, q_idx, scale)
        ref = broadcast_reference(Q, K, V, broadcast_heads, q_idx, scale)

        max_diff = max(
            (O[:, h] - ref[:, h]).abs().max().item() for h in broadcast_heads)
        for h in [0, 2, 5]:
            if h >= H_heads: continue
            assert (O[:, h] - V[:, h]).abs().max().item() == 0.0
        tol = 1e-2 if dtype == torch.float16 else 1e-4
        ok = max_diff <= tol
        print(f"[bcast] B={B} H={H_heads} Dh={Dh} N={N:>5d} q_idx={q_idx} "
              f"dtype={str(dtype):>15s}: max_diff={max_diff:.3e}  "
              f"{'OK' if ok else 'FAIL'}")
        assert ok


def test_attention():
    """Triton FA-1 attention vs F.scaled_dot_product_attention."""
    import torch.nn.functional as F
    from fused_substitutions_triton import attention_triton
    torch.manual_seed(0)
    cases = [
        # (B, H_heads, Dh, N, dtype)
        (2, 6,  32, 197,  torch.float32),
        (4, 12, 64, 197,  torch.float32),
        (4, 12, 64, 577,  torch.float32),
        (2, 12, 64, 1374, torch.float32),
        (2, 6,  64, 197,  torch.float16),
        (4, 12, 64, 577,  torch.float16),
    ]
    for B, H_heads, Dh, N, dtype in cases:
        Q = torch.randn(B, H_heads, N, Dh, device='cuda', dtype=dtype)
        K = torch.randn(B, H_heads, N, Dh, device='cuda', dtype=dtype)
        V = torch.randn(B, H_heads, N, Dh, device='cuda', dtype=dtype)
        scale = Dh ** -0.5

        # Triton: write all heads
        O = torch.empty_like(V)
        head_idx_t = torch.arange(H_heads, device='cuda', dtype=torch.int32)
        attention_triton(Q.contiguous(), K.contiguous(), V.contiguous(),
                          O, head_idx_t, scale)

        # Reference
        ref = F.scaled_dot_product_attention(Q, K, V, scale=scale)

        max_diff = (O - ref).abs().max().item()
        # On compute capability 8.0 and newer, tl.dot runs fp32 inputs in TF32.
        tf32 = torch.cuda.get_device_capability()[0] >= 8
        tol = 1e-2 if dtype == torch.float16 else (5e-3 if tf32 else 1e-4)
        ok = max_diff <= tol
        print(f"[attn]  B={B} H={H_heads} Dh={Dh} N={N:>5d} "
              f"dtype={str(dtype):>15s}: max_diff={max_diff:.3e}  "
              f"{'OK' if ok else 'FAIL'}")
        assert ok


if __name__ == "__main__":
    test_shift()
    test_broadcast()
    test_attention()
    print("All cases passed.")
