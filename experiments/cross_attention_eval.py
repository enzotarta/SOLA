"""Broadcast and rank-k substitution of the decoder cross-attention in BLIP captioning."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# caption metrics

def jaccard(a: str, b: str) -> float:
    A = set(a.lower().split()); B = set(b.lower().split())
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)


def shared_prefix(a: List[int], b: List[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x == y: n += 1
        else: break
    return n


# substitutions

def per_image_broadcast(attn: torch.Tensor) -> torch.Tensor:
    """attn: (B, H, Tq, Tk) -> same shape, each row replaced by per-image row-mean."""
    row_mean = attn.mean(dim=-2, keepdim=True)         # (B, H, 1, Tk)
    return row_mean.expand_as(attn)


def per_image_rank_k(attn: torch.Tensor, k: int) -> torch.Tensor:
    """attn: (B, H, Tq, Tk) -> best rank-k SVD per (image, head)."""
    B, H, Tq, Tk = attn.shape
    a = attn.reshape(B * H, Tq, Tk).float()
    out = torch.zeros_like(a)
    for i in range(B * H):
        try:
            U, S, Vh = torch.linalg.svd(a[i], full_matrices=False)
            kk = min(k, S.shape[0])
            out[i] = U[:, :kk] @ torch.diag(S[:kk]) @ Vh[:kk, :]
        except Exception:
            out[i] = a[i]
    return out.reshape(B, H, Tq, Tk).to(attn.dtype)


# BLIP cross-attn instrumentation
# In transformers BlipTextModel, each decoder layer is a BlipTextLayer with
#   self.crossattention.self  -> BlipTextSelfAttention (used in cross mode)
# We capture attention probs in this module.

CAPTURED: Dict[Tuple[int, str], torch.Tensor] = {}
SUBSTITUTE_FN = None       # callable(attn, layer_idx, head_idx) -> attn'
ACTIVE_HEADS: Dict[int, List[int]] = {}    # layer_idx -> list of head indices to substitute


def patch_cross_attention(model):
    """Rewrite forward of BlipTextSelfAttention (in cross-attention mode)
    in every decoder layer. We mimic the current HF implementation but
    capture attn probs and optionally substitute selected heads."""
    import math
    text_decoder = model.text_decoder.bert
    for li, layer in enumerate(text_decoder.encoder.layer):
        ca = layer.crossattention.self  # BlipTextSelfAttention (cross mode)
        ca._layer_idx = li
        Hh = ca.num_attention_heads
        Dh = ca.attention_head_size

        def make_forward(self_mod, layer_idx):
            def fwd(hidden_states, attention_mask=None,
                    encoder_hidden_states=None, encoder_attention_mask=None,
                    past_key_values=None, **kwargs):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, Dh)
                q = self_mod.query(hidden_states).view(hidden_shape).transpose(1, 2)
                # cross-attn: K/V from encoder_hidden_states
                kv_shape = (*encoder_hidden_states.shape[:-1], -1, Dh)
                k = self_mod.key(encoder_hidden_states).view(kv_shape).transpose(1, 2)
                v = self_mod.value(encoder_hidden_states).view(kv_shape).transpose(1, 2)
                scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Dh)
                if encoder_attention_mask is not None:
                    scores = scores + encoder_attention_mask.to(scores.device)
                attn = F.softmax(scores, dim=-1)
                # capture
                CAPTURED[(layer_idx, "cross")] = attn.detach().to("cpu", dtype=torch.float32)
                # optional substitute
                if SUBSTITUTE_FN is not None:
                    heads = ACTIVE_HEADS.get(layer_idx, [])
                    if heads:
                        new_attn = SUBSTITUTE_FN(attn, layer_idx)
                        mask = torch.zeros(attn.shape[1], dtype=torch.bool, device=attn.device)
                        mask[heads] = True
                        attn = torch.where(mask.view(1, -1, 1, 1), new_attn, attn)
                # dropout
                if hasattr(self_mod, "dropout"):
                    attn_d = self_mod.dropout(attn)
                else:
                    attn_d = attn
                ctx = torch.matmul(attn_d, v)            # (B, H, Tq, Dh)
                ctx = ctx.transpose(1, 2).contiguous()    # (B, Tq, H, Dh)
                ctx = ctx.view(*input_shape, Hh * Dh)
                return ctx, attn
            return fwd
        ca.forward = make_forward(ca, li)


# main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-dir", required=True,
                    help="Directory with COCO val2017 *.jpg images.")
    ap.add_argument("--n-images", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rabs-thresh", type=float, default=0.4,
                    help="r_abs threshold for broadcast subset.")
    ap.add_argument("--rank-k", type=int, default=10)
    args = ap.parse_args()

    from transformers import BlipProcessor, BlipForConditionalGeneration

    print("Loading BLIP-base ...", flush=True)
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        use_safetensors=True).to(args.device).eval()
    patch_cross_attention(model)
    n_layers = len(model.text_decoder.bert.encoder.layer)
    n_heads = model.text_decoder.bert.encoder.layer[0].crossattention.self.num_attention_heads
    print(f"BLIP-base: {n_layers} decoder layers x {n_heads} heads of cross-attention",
          flush=True)

    images = sorted(Path(args.coco_dir).glob("*.jpg"))[:args.n_images]
    print(f"Using {len(images)} images.", flush=True)

    # Pass 1: baseline captions + capture cross-attn for diagnostics
    global SUBSTITUTE_FN, ACTIVE_HEADS
    SUBSTITUTE_FN = None
    ACTIVE_HEADS = {}

    baseline_captions = []          # str per image
    baseline_token_ids = []
    # accumulate per-layer per-head, per-image diagnostics
    sum_attn_per_head: Dict[Tuple[int, int], torch.Tensor] = {}  # not used, we compute r_abs per image
    r_abs_per_head: Dict[Tuple[int, int], List[float]] = {(li, h): [] for li in range(n_layers) for h in range(n_heads)}
    eff_rank_per_head: Dict[Tuple[int, int], List[float]] = {(li, h): [] for li in range(n_layers) for h in range(n_heads)}

    def eff_rank_90(s):
        s2 = s ** 2
        cum = torch.cumsum(s2, 0) / s2.sum()
        return int((cum < 0.9).sum().item()) + 1

    def caption_one_image(img):
        inputs = proc(img, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, num_beams=1, do_sample=False)
        text = proc.decode(out[0], skip_special_tokens=True)
        return text, out[0].tolist()

    print("Pass 1: baseline captions + diagnostic capture", flush=True)
    t0 = time.time()
    for ii, p in enumerate(images):
        img = Image.open(p).convert("RGB")
        # Generate baseline caption first (will populate CAPTURED at last decoding step but not for full
        # baseline-text; we'll do a teacher-forced forward instead for diagnostics)
        text, ids = caption_one_image(img)
        baseline_captions.append(text)
        baseline_token_ids.append(ids)

        # Teacher-forced forward on baseline caption to get full cross-attn
        # for diagnostics (Tq = caption length).
        CAPTURED.clear()
        inputs = proc(img, text=text, return_tensors="pt", padding=True).to(args.device)
        with torch.no_grad():
            _ = model(**inputs, return_dict=True)
        # CAPTURED now has (li, "cross") -> (1, H, Tq, Tk)
        for li in range(n_layers):
            attn = CAPTURED.get((li, "cross"))
            if attn is None: continue
            attn = attn[0]  # (H, Tq, Tk)
            row_mean = attn.mean(dim=1, keepdim=True)  # (H, 1, Tk)
            r_abs = (attn - row_mean).abs().sum(dim=-1).mean(dim=-1) / 2.0
            for h in range(n_heads):
                r_abs_per_head[(li, h)].append(float(r_abs[h]))
                # SVD for eff_rank
                try:
                    s = torch.linalg.svdvals(attn[h].float())
                    eff_rank_per_head[(li, h)].append(eff_rank_90(s))
                except Exception:
                    pass
        if (ii + 1) % 10 == 0:
            print(f"  baseline {ii+1}/{len(images)}  elapsed={time.time()-t0:.0f}s", flush=True)

    # Median diagnostics per head
    median_rabs = {(li, h): float(np.median(r_abs_per_head[(li, h)])) if r_abs_per_head[(li, h)] else float("nan")
                   for li in range(n_layers) for h in range(n_heads)}
    median_er90 = {(li, h): float(np.median(eff_rank_per_head[(li, h)])) if eff_rank_per_head[(li, h)] else float("nan")
                   for li in range(n_layers) for h in range(n_heads)}

    # Identify low-r_abs heads (broadcast candidates)
    low_rabs_heads: Dict[int, List[int]] = {li: [] for li in range(n_layers)}
    n_total = 0; n_low = 0
    for li in range(n_layers):
        for h in range(n_heads):
            n_total += 1
            if median_rabs[(li, h)] < args.rabs_thresh:
                low_rabs_heads[li].append(h)
                n_low += 1
    print(f"\nLow-r_abs heads (r_abs < {args.rabs_thresh}): {n_low}/{n_total}", flush=True)
    for li in range(n_layers):
        if low_rabs_heads[li]:
            print(f"  layer {li}: heads {low_rabs_heads[li]}", flush=True)

    # Pass 2: broadcast substitution at low-r_abs heads
    print("\nPass 2: broadcast substitution at low-r_abs cross-attn heads", flush=True)
    SUBSTITUTE_FN = lambda attn, li: per_image_broadcast(attn)
    ACTIVE_HEADS = low_rabs_heads
    bc_captions = []
    for ii, p in enumerate(images):
        img = Image.open(p).convert("RGB")
        text, _ = caption_one_image(img)
        bc_captions.append(text)
    SUBSTITUTE_FN = None; ACTIVE_HEADS = {}

    # Pass 3: rank-k substitution at ALL cross-attn heads
    print(f"\nPass 3: rank-{args.rank_k} substitution at all cross-attn heads", flush=True)
    SUBSTITUTE_FN = lambda attn, li: per_image_rank_k(attn, args.rank_k)
    ACTIVE_HEADS = {li: list(range(n_heads)) for li in range(n_layers)}
    rk_captions = []
    for ii, p in enumerate(images):
        img = Image.open(p).convert("RGB")
        text, _ = caption_one_image(img)
        rk_captions.append(text)
    SUBSTITUTE_FN = None; ACTIVE_HEADS = {}

    # Pass 4: SANITY uniform attention substitution at all heads
    # If hooks fire, captions should be wrecked.
    print("\nPass 4 (sanity): uniform attention at all cross-attn heads", flush=True)
    def _uniform(attn, li):
        return torch.full_like(attn, 1.0 / attn.shape[-1])
    SUBSTITUTE_FN = _uniform
    ACTIVE_HEADS = {li: list(range(n_heads)) for li in range(n_layers)}
    sanity_captions = []
    for ii, p in enumerate(images):
        img = Image.open(p).convert("RGB")
        text, _ = caption_one_image(img)
        sanity_captions.append(text)
    SUBSTITUTE_FN = None; ACTIVE_HEADS = {}

    # Metrics
    def metrics(refs, hyps):
        jacs = [jaccard(r, h) for r, h in zip(refs, hyps)]
        exact = sum(int(r.strip() == h.strip()) for r, h in zip(refs, hyps))
        return {"n": len(refs),
                "exact_match": exact,
                "exact_rate": exact / max(len(refs), 1),
                "median_jaccard": float(np.median(jacs)),
                "mean_jaccard": float(np.mean(jacs)),
                "frac_jaccard_ge_0.5": float(np.mean([j >= 0.5 for j in jacs]))}

    res = {
        "model": "Salesforce/blip-image-captioning-base",
        "n_images": len(images),
        "n_layers": n_layers,
        "n_heads_per_layer": n_heads,
        "rabs_thresh": args.rabs_thresh,
        "rank_k": args.rank_k,
        "n_low_rabs_heads": n_low,
        "median_rabs_layer_mean": [
            float(np.mean([median_rabs[(li, h)] for h in range(n_heads)]))
            for li in range(n_layers)],
        "median_er90_layer_mean": [
            float(np.mean([median_er90[(li, h)] for h in range(n_heads)
                          if not np.isnan(median_er90[(li, h)])]))
            for li in range(n_layers)],
        "broadcast_metrics": metrics(baseline_captions, bc_captions),
        "rank_k_metrics": metrics(baseline_captions, rk_captions),
        "sanity_uniform_metrics": metrics(baseline_captions, sanity_captions),
        "examples": [
            {"baseline": b, "broadcast": bc, f"rank_{args.rank_k}": rk, "uniform": un}
            for b, bc, rk, un in zip(baseline_captions[:8], bc_captions[:8],
                                      rk_captions[:8], sanity_captions[:8])],
        "per_head_diagnostics": [
            {"layer": li, "head": h,
             "median_r_abs": median_rabs[(li, h)],
             "median_eff_rank_90": median_er90[(li, h)]}
            for li in range(n_layers) for h in range(n_heads)],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nWrote {args.out}", flush=True)
    print(f"\nBROADCAST: {res['broadcast_metrics']}", flush=True)
    print(f"\nRANK-{args.rank_k}: {res['rank_k_metrics']}", flush=True)
    print(f"\nSANITY (uniform): {res['sanity_uniform_metrics']}", flush=True)


if __name__ == "__main__":
    main()
