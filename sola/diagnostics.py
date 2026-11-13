"""Per-head diagnostics and the Stage-1 category cascade.

Self-contained version of the diagnostics in experiments/operator_taxonomy.py.
Input is the attention of one head with the non-spatial tokens removed, shaped
(n_images, n_queries, n_keys).
"""
import numpy as np


def within_image_residual(attn):
    """Distance of each row from the mean row of the same image.

    This is the residual in the bound for the broadcast substitution.
    """
    mean_row = attn.mean(axis=1, keepdims=True)
    return float((0.5 * np.abs(attn - mean_row).sum(-1).mean(-1)).mean())


def across_image_residual(attn):
    """Distance of the attention matrix from its mean over images.

    Low values mean the head attends in the same way on every image. The
    experiments select the heads for the broadcast and rank-k tests with this
    quantity (r_abs_mean in experiments/fixed_target_diagnostic.py).
    """
    mean_matrix = attn.mean(axis=0, keepdims=True)
    return float((0.5 * np.abs(attn - mean_matrix).sum(-1).mean(-1)).mean())


def recenter(attn, H, W, K=5):
    n_img, n_q, n_k = attn.shape
    assert n_q == H * W and n_k == H * W
    out = np.zeros((n_img * n_q, 2 * K + 1, 2 * K + 1), attn.dtype)
    idx = 0
    for i in range(n_img):
        a2 = attn[i].reshape(n_q, H, W)
        for q in range(n_q):
            qy, qx = q // W, q % W
            for dy in range(-K, K + 1):
                for dx in range(-K, K + 1):
                    ky, kx = qy + dy, qx + dx
                    if 0 <= ky < H and 0 <= kx < W:
                        out[idx, dy + K, dx + K] = a2[q, ky, kx]
            idx += 1
    return out


def head_diagnostics(attn_per_query, H, W, K=5):
    n_img, n_q, n_k = attn_per_query.shape
    stack = recenter(attn_per_query, H, W, K)
    mean_kernel = stack.mean(axis=0)
    Z = mean_kernel.sum()
    kbar = mean_kernel / max(Z, 1e-12)

    central3 = mean_kernel[K - 1:K + 2, K - 1:K + 2].sum()
    escape3 = float(1.0 - central3)

    self_mass = float(np.mean(
        [attn_per_query[i, q, q] for i in range(n_img) for q in range(n_q)]
    ))
    am_concs = []
    for i in range(min(n_img, 8)):
        argm = attn_per_query[i].argmax(-1)
        am_concs.append(np.bincount(argm, minlength=n_k).max() / n_q)
    arg_conc = float(np.mean(am_concs))
    p = kbar.flatten()
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum())
    int_idx = []
    for i in range(n_img):
        for qi in range(n_q):
            qy, qx = qi // W, qi % W
            if K <= qy < H - K and K <= qx < W - K:
                int_idx.append(i * n_q + qi)
    if int_idx:
        s = stack[int_idx]
        sn = s / s.sum(axis=(1, 2), keepdims=True).clip(min=1e-12)
        r = float(np.abs(sn - sn.mean(axis=0)[None]).sum(axis=(1, 2)).mean())
    else:
        r = float("nan")
    r_row = within_image_residual(attn_per_query)
    r_img = across_image_residual(attn_per_query)
    er = {}
    for tau in (0.9, 0.99):
        ks = []
        for i in range(n_img):
            s = np.linalg.svd(attn_per_query[i], compute_uv=False)
            cum = np.cumsum(s ** 2)
            ks.append(int(np.searchsorted(cum, tau * cum[-1]) + 1))
        er[tau] = float(np.mean(ks))
    flat = kbar.flatten()
    order = np.argsort(flat)[::-1]
    top1 = float(flat[order[0]])
    top3 = float(flat[order[:3]].sum())
    py, px = int(order[0] // (2 * K + 1) - K), int(order[0] % (2 * K + 1) - K)
    mod = _local_maxima(kbar, 0.05)
    gres = _gaussian_fit_residual(kbar)

    return kbar, dict(
        escape3=escape3, self_mass=self_mass, arg_conc=arg_conc,
        entropy=entropy, r=r, r_row=r_row, r_img=r_img,
        eff_rank_90=er[0.9], eff_rank_99=er[0.99],
        top1_mass=top1, top3_mass=top3, peak=(py, px),
        modality=mod, gauss_residual=gres,
    )


def _local_maxima(k, thr):
    H, W = k.shape
    peak = k.max()
    if peak <= 0:
        return 0
    n = 0
    for y in range(H):
        for x in range(W):
            v = k[y, x]
            if v < thr * peak:
                continue
            ok = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and k[yy, xx] > v:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                n += 1
    return n


def _gaussian_fit_residual(k):
    H, W = k.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    Z = k.sum()
    if Z <= 0:
        return float("nan")
    mu_y = (k * yy).sum() / Z
    mu_x = (k * xx).sum() / Z
    var = (k * ((yy - mu_y) ** 2 + (xx - mu_x) ** 2)).sum() / Z / 2
    sig = max(np.sqrt(var), 0.5)
    g = np.exp(-((yy - mu_y) ** 2 + (xx - mu_x) ** 2) / (2 * sig ** 2))
    g /= g.sum()
    return float(np.linalg.norm(k / max(Z, 1e-12) - g) / max(np.linalg.norm(k / Z), 1e-12))


def classify(d):
    if d["arg_conc"] > 0.5:
        return "fixed_target"
    if d["escape3"] > 0.7:
        return "near_global"
    if d["self_mass"] > 0.7 and d["peak"] == (0, 0):
        return "self_only"
    if d["top1_mass"] > 0.5 and d["peak"] != (0, 0) and d["modality"] <= 2:
        return "delta_shift"
    if d["top3_mass"] > 0.7 and d["modality"] >= 2 and d["entropy"] < 2.0:
        return "multi_delta"
    if (d["peak"] != (0, 0) and d["r"] < 0.40
            and d["modality"] <= 3 and d["entropy"] < 2.5):
        return "soft_shift"
    if d["gauss_residual"] < 0.15 and d["modality"] == 1 and d["escape3"] < 0.5:
        return "gaussian_local"
    if d["escape3"] < 0.5:
        return "diffuse_local"
    return "other"


def stage2(category, eff_rank_90, cut=40):
    """Reassign the residual `other` class by effective rank."""
    if category != "other":
        return category
    return "near_global" if eff_rank_90 <= cut else "diffuse_local"
