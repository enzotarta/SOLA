"""Per-head diagnostics and the Stage-1 category cascade.

Writes one JSON per backbone with the category, the diagnostics and the mean
kernel of every (layer, head). The calibration batch is the first images of
tiny-ImageNet val.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np


# Recentered-kernel utilities

def recenter_one_query(alpha_q: np.ndarray, q_pos: tuple[int, int],
                        H: int, W: int, K: int) -> np.ndarray:
    """
    Reshape alpha_q (length H*W) to the (2K+1)x(2K+1) window centered on q.

    Off-grid offsets are 0; in particular the head's residual mass on
    keys outside the window is dropped (handled separately via escape_mass).
    """
    out = np.zeros((2 * K + 1, 2 * K + 1))
    qy, qx = q_pos
    a2 = alpha_q.reshape(H, W)
    for dy in range(-K, K + 1):
        for dx in range(-K, K + 1):
            ky, kx = qy + dy, qx + dx
            if 0 <= ky < H and 0 <= kx < W:
                out[dy + K, dx + K] = a2[ky, kx]
    return out


def recentered_kernel_stack(attn_per_query: np.ndarray,
                            H: int, W: int, K: int) -> np.ndarray:
    """
    attn_per_query: (n_img, n_q=H*W, n_k=H*W)

    Returns recentered: (n_img * n_q, 2K+1, 2K+1) array of recentered
    kernels, one per (image, query) pair.
    """
    n_img, n_q, _ = attn_per_query.shape
    assert n_q == H * W
    out = np.zeros((n_img * n_q, 2 * K + 1, 2 * K + 1))
    idx = 0
    for i in range(n_img):
        for q in range(n_q):
            qy, qx = q // W, q % W
            out[idx] = recenter_one_query(attn_per_query[i, q],
                                            (qy, qx), H, W, K)
            idx += 1
    return out


# Per-head diagnostics

@dataclass
class HeadDiagnostics:
    # escape mass at K=3 (locality)
    escape_mass_K3: float
    # self-attention mass
    self_mass: float
    # argmax-concentration (max fraction of queries with same absolute argmax)
    argmax_concentration: float
    # cross-query Pearson on attention vectors (high = query-independent)
    query_pair_corr: float
    # entropy of the recentered mean kernel (nats)
    fingerprint_entropy: float
    # per-(image,query) L1 residual against the mean kernel, the translation
    # equivariance test. Small => fixed kernel.
    kernel_l1_residual_mean: float
    # number of local maxima in the mean kernel above 5% of peak
    modality_count: int
    # relative L2 residual of a least-squares 2D-Gaussian fit to the mean
    # kernel. Small => Gaussian-like.
    gaussian_fit_residual: float
    # mass at the top cell and on the top-3 cells
    top1_mass: float
    top3_mass: float
    # location of the mean-kernel argmax (relative offset)
    peak_offset: tuple[int, int]
    # spread of the kernel (2nd-moment radius)
    kernel_radius: float

    def asdict(self):
        d = asdict(self)
        d["peak_offset"] = list(d["peak_offset"])
        return d


def compute_diagnostics(attn_per_query: np.ndarray, H: int, W: int,
                         K: int = 5) -> tuple[np.ndarray, HeadDiagnostics]:
    """
    attn_per_query : (n_img, n_q, n_k) softmax outputs for ONE head.
    Returns (mean_kernel, HeadDiagnostics).
    """
    n_img, n_q, n_k = attn_per_query.shape
    assert n_q == H * W and n_k == H * W

    # Recentered stack and mean kernel
    stack = recentered_kernel_stack(attn_per_query, H, W, K)  # (N, 2K+1, 2K+1)
    mean_kernel = stack.mean(axis=0)
    Z = mean_kernel.sum()
    mean_kernel_n = mean_kernel / max(Z, 1e-12)

    # escape mass at K=3 (mass outside the central 3x3)
    half3 = 1
    central3 = mean_kernel[K - half3 : K + half3 + 1,
                           K - half3 : K + half3 + 1].sum()
    escape_mass_K3 = float(1.0 - central3)

    # self-mass: fraction of attention on the query itself
    self_mass = float(np.array(
        [attn_per_query[i, q, q] for i in range(n_img) for q in range(n_q)]
    ).mean())

    # argmax concentration: per-image, the max fraction of queries whose
    #    (image-level) attention argmax lands on the same absolute key.
    argmaxes = attn_per_query.argmax(axis=-1)
    am_conc = []
    for i in range(min(n_img, 8)):
        bc = np.bincount(argmaxes[i], minlength=n_k)
        am_conc.append(bc.max() / n_q)
    argmax_concentration = float(np.mean(am_conc))

    # query-pair correlation
    rs = np.random.RandomState(0)
    sample_qs = rs.choice(n_q, size=min(20, n_q), replace=False)
    corrs = []
    for i in range(min(n_img, 4)):
        for ii in range(len(sample_qs)):
            for jj in range(ii + 1, len(sample_qs)):
                a = attn_per_query[i, sample_qs[ii]]
                b = attn_per_query[i, sample_qs[jj]]
                if a.std() > 0 and b.std() > 0:
                    corrs.append(float(np.corrcoef(a, b)[0, 1]))
    query_pair_corr = float(np.mean(corrs)) if corrs else 0.0

    # fingerprint entropy of the mean kernel
    p = mean_kernel_n.flatten()
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum())

    # L1 residual: mean over INTERIOR (image, query) of L1 distance
    #    between the per-query recentered kernel and the global mean
    #    kernel, both restricted to in-grid offsets and renormalized.
    #
    #    Restricting to interior queries (those whose entire (2K+1)x(2K+1)
    #    window is in-grid) is necessary because boundary queries trivially
    #    differ from the mean: out-of-grid mass shows up at different cells
    #    depending on q. For a true translation-equivariant head, the
    #    interior L1 residual is small even when the boundary one is large.
    interior_idx = []
    for i in range(n_img):
        for qi in range(n_q):
            qy, qx = qi // W, qi % W
            if K <= qy < H - K and K <= qx < W - K:
                interior_idx.append(i * n_q + qi)
    if interior_idx:
        interior_stack = stack[interior_idx]
        # Renormalize each recentered kernel to sum to 1 (interior queries
        # have no out-of-grid mass to drop, so this is a unit op).
        per_q_sum = interior_stack.sum(axis=(1, 2), keepdims=True).clip(min=1e-12)
        interior_stack_n = interior_stack / per_q_sum
        mean_interior = interior_stack_n.mean(axis=0)
        l1 = float(np.abs(interior_stack_n - mean_interior[None]).sum(axis=(1, 2)).mean())
    else:
        l1 = float("nan")

    # modality count (local maxima above 5% of peak in the mean kernel)
    modality = _count_local_maxima(mean_kernel_n, threshold=0.05)

    # Gaussian-fit residual
    gauss_res = _gaussian_fit_residual(mean_kernel_n)

    # top1 / top3 mass and peak offset
    flat = mean_kernel_n.flatten()
    top_idx = np.argsort(flat)[::-1]
    top1_mass = float(flat[top_idx[0]])
    top3_mass = float(flat[top_idx[:3]].sum())
    peak_flat = int(top_idx[0])
    peak_y = peak_flat // (2 * K + 1) - K
    peak_x = peak_flat % (2 * K + 1) - K
    peak_offset = (int(peak_y), int(peak_x))

    # kernel radius (2nd moment from origin of the mean kernel)
    yy, xx = np.meshgrid(np.arange(-K, K + 1), np.arange(-K, K + 1),
                          indexing="ij")
    rr = (yy ** 2 + xx ** 2)
    kernel_radius = float(np.sqrt((mean_kernel_n * rr).sum()))

    diag = HeadDiagnostics(
        escape_mass_K3=escape_mass_K3,
        self_mass=self_mass,
        argmax_concentration=argmax_concentration,
        query_pair_corr=query_pair_corr,
        fingerprint_entropy=entropy,
        kernel_l1_residual_mean=l1,
        modality_count=modality,
        gaussian_fit_residual=gauss_res,
        top1_mass=top1_mass,
        top3_mass=top3_mass,
        peak_offset=peak_offset,
        kernel_radius=kernel_radius,
    )
    return mean_kernel_n, diag


def _count_local_maxima(kernel: np.ndarray, threshold: float = 0.05) -> int:
    """
    Count cells that are >= threshold * peak AND are >= each of their 8
    neighbors. Edge cells use only the available neighbors.
    """
    H, W = kernel.shape
    peak = kernel.max()
    cutoff = threshold * peak
    count = 0
    for y in range(H):
        for x in range(W):
            v = kernel[y, x]
            if v < cutoff:
                continue
            is_max = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if kernel[ny, nx] > v:
                            is_max = False
                            break
                if not is_max:
                    break
            if is_max:
                count += 1
    return count


def _gaussian_fit_residual(kernel: np.ndarray,
                            min_var: float = 0.5) -> float:
    """
    Fit an isotropic 2D Gaussian to the kernel by moment matching with a
    minimum bandwidth, then return ||kernel - fit|| / ||kernel|| (relative
    L2 residual).

    Without the minimum bandwidth a delta kernel would be fit perfectly by a
    Gaussian with var --> 0. With var >= min_var a head counts as
    Gaussian-like only if its kernel has spread (sigma >= sqrt(min_var)).
    """
    H, W = kernel.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.meshgrid(np.arange(H) - cy, np.arange(W) - cx, indexing="ij")
    Z = kernel.sum()
    if Z <= 0:
        return float("inf")
    p = kernel / Z
    mu_y = (p * yy).sum()
    mu_x = (p * xx).sum()
    var_emp = (p * ((yy - mu_y) ** 2 + (xx - mu_x) ** 2)).sum() / 2.0
    var = max(var_emp, min_var)
    fit_unnorm = np.exp(-((yy - mu_y) ** 2 + (xx - mu_x) ** 2) / (2 * var))
    fit = fit_unnorm / fit_unnorm.sum() * Z
    num = np.sqrt(((kernel - fit) ** 2).sum())
    den = np.sqrt((kernel ** 2).sum())
    return float(num / max(den, 1e-12))


# Categorization

CATEGORIES = (
    "delta_shift",
    "soft_shift",
    "self_only",
    "multi_delta",
    "gaussian_local",
    "diffuse_local",
    "fixed_target",
    "near_global",
    "other",
)


def classify_head(diag: HeadDiagnostics) -> str:
    """
    Classify a head based on its structural diagnostics.

    The most restrictive detector is applied first, so that a head satisfying
    several criteria gets the tighter label.
    """
    # Fixed-target attractor: same absolute key wins for >50% of queries.
    #    Tested before near_global because such heads also have high escape
    #    mass; "fixed absolute key" is the more informative tag. Despite the
    #    name, this is not necessarily the CLS token; many vision heads in
    #    late layers fixate on a specific spatial position.
    if diag.argmax_concentration > 0.5:
        return "fixed_target"

    # Near-global: no local structure
    if diag.escape_mass_K3 > 0.7:
        return "near_global"

    # Self-only: single-cell kernel at the query
    if diag.self_mass > 0.7 and diag.peak_offset == (0, 0):
        return "self_only"

    # Delta shift: single-cell kernel off-center
    if (diag.top1_mass > 0.5
            and diag.peak_offset != (0, 0)
            and diag.modality_count <= 2):
        return "delta_shift"

    # Multi-delta: a few discrete cells dominate, low entropy. Tested
    #    before soft_shift so multi-modal kernels get the more specific
    #    label.
    if (diag.top3_mass > 0.7
            and diag.modality_count >= 2
            and diag.fingerprint_entropy < 2.0):
        return "multi_delta"

    # Soft shift: off-center peak that's directionally consistent (low
    #    interior L1 residual = translation-equivariant) but with mass
    #    spread across a few cells rather than concentrated on one. These
    #    are heads that substitute within the CI with torch.roll despite not
    #    looking like sharp deltas.
    if (diag.peak_offset != (0, 0)
            and diag.kernel_l1_residual_mean < 0.40
            and diag.modality_count <= 3
            and diag.fingerprint_entropy < 2.5):
        return "soft_shift"

    # Gaussian-local: smooth, single mode, fits a 2D Gaussian well
    if (diag.gaussian_fit_residual < 0.15
            and diag.modality_count == 1
            and diag.escape_mass_K3 < 0.5):
        return "gaussian_local"

    # Diffuse local: locally concentrated but no clear shape
    if diag.escape_mass_K3 < 0.5:
        return "diffuse_local"

    return "other"


# Generalized substitution kernel

def kernel_substitution_for_head(mean_kernel: np.ndarray) -> np.ndarray:
    """
    Given the recentered mean kernel for a head, return a kernel that can
    be used as a substitution: the (2K+1)x(2K+1) array, normalized to sum
    to 1.

    For boundary handling at substitution time, the per-query effective
    kernel restricts to in-grid offsets and renormalizes.
    """
    Z = mean_kernel.sum()
    return mean_kernel / max(Z, 1e-12)


# Public sweep API (PyTorch-side; lazy import so the diagnostic functions
# above can be tested without torch present)

def make_attention_recorder_class():
    """Attention recorder class, built here so that the diagnostics above import
    without torch."""
    import torch
    import torch.nn as nn

    class AttentionRecorder:
        def __init__(self, model: nn.Module, layers=None):
            self.model = model
            self.records: dict[int, list] = {}
            self._handles = []
            self._inputs: dict[int, list] = {}
            for li, block in enumerate(model.blocks):
                if layers is not None and li not in layers:
                    continue
                attn = block.attn
                self.records[li] = []
                self._inputs[li] = []
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

        def collect(self):
            import torch
            return {li: torch.cat(self.records[li], dim=0) for li in self.records}

        def remove(self):
            for h in self._handles:
                h.remove()

    return AttentionRecorder


def sweep_model_heads(model, eval_loader, device, n_batches: int = 4,
                       K: int = 5, layer_chunk: int = 0):
    """
    For every (layer, head) in `model`, compute diagnostics on
    `n_batches` worth of images, then categorize.

    layer_chunk > 0 records that many layers per pass over the images, which
    bounds host memory for the L-size models; the result is the same.

    Returns a list of dicts with model-side info plus diagnostics.
    """
    import torch
    from escape_mass_sweep import detect_model_config
    AttentionRecorder = make_attention_recorder_class()

    cfg = detect_model_config(model, image_size=224)
    H, W = cfg["grid"]
    n_extra = cfg["n_extra_tokens"]

    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.num_heads
    step = layer_chunk if layer_chunk > 0 else n_layers
    out = []
    for first in range(0, n_layers, step):
        layers = list(range(first, min(first + step, n_layers)))
        rec = AttentionRecorder(model, layers=set(layers))
        with torch.no_grad():
            for bi, batch in enumerate(eval_loader):
                if bi >= n_batches:
                    break
                # eval_loader yields either Tensor or (Tensor, label) depending
                # on the loader; handle both
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                _ = model(batch.to(device))
        attn_by_layer = rec.collect()
        rec.remove()

        for li in layers:
            a = attn_by_layer.pop(li).numpy()  # (B*n_batches, n_heads, N, N)
            # strip extra tokens
            a = a[:, :, n_extra:, n_extra:]  # (B', H, n_q, n_k)
            # renormalize after stripping
            a = a / a.sum(axis=-1, keepdims=True).clip(min=1e-12)
            for hi in range(n_heads):
                attn_h = a[:, hi]  # (B', n_q, n_k)
                mean_kernel, diag = compute_diagnostics(attn_h, H, W, K=K)
                out.append({
                    "layer": li,
                    "head": hi,
                    "category": classify_head(diag),
                    "diag": diag.asdict(),
                    "mean_kernel": mean_kernel.tolist(),
                })
    return out


# CLI

def main():
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Operator taxonomy sweep")
    ap.add_argument("--model", default="deit_tiny_patch16_224")
    ap.add_argument("--batches", type=int, default=4,
                    help="number of calibration batches (each is 32 images)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--K", type=int, default=5,
                    help="recentered-kernel half-window")
    ap.add_argument("--source", default="tiny-imagenet",
                    help="image source (currently only tiny-imagenet)")
    ap.add_argument("--out", default="taxonomy.json")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--device", default=None)
    ap.add_argument("--layer-chunk", type=int, default=0,
                    help="record this many layers per pass to bound host memory "
                         "(0: all layers in one pass)")
    args = ap.parse_args()

    import torch
    import timm
    from shift_substitution import build_tiny_imagenet_eval_loader

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device} ...")
    model = timm.create_model(args.model, pretrained=True,
                                 img_size=args.img_size).eval().to(device)
    eval_loader, _ = build_tiny_imagenet_eval_loader(args.batch_size, args.img_size)

    print(f"Sweeping all heads (K={args.K}, {args.batches} batches) ...")
    results = sweep_model_heads(model, eval_loader, device,
                                  n_batches=args.batches, K=args.K,
                                  layer_chunk=args.layer_chunk)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": args.model,
        "K": args.K,
        "n_batches": args.batches,
        "results": results,
    }, indent=2))
    print(f"Wrote {out_path}")

    # Tally
    tally: dict[str, int] = {}
    for r in results:
        tally[r["category"]] = tally.get(r["category"], 0) + 1
    print("\nTally by category:")
    for cat in CATEGORIES:
        if cat in tally:
            print(f"  {cat:18s}  {tally[cat]:3d}")

    print("\nNon-trivial heads (excluding near_global, other):")
    for r in results:
        if r["category"] in {"near_global", "other"}:
            continue
        d = r["diag"]
        print(f"  L{r['layer']}H{r['head']:<2}  "
              f"{r['category']:14s}  "
              f"escape={d['escape_mass_K3']:.2f}  "
              f"l1_resid={d['kernel_l1_residual_mean']:.2f}  "
              f"peak={tuple(d['peak_offset'])}  "
              f"top1={d['top1_mass']:.2f}  "
              f"entropy={d['fingerprint_entropy']:.2f}")


if __name__ == "__main__":
    main()
