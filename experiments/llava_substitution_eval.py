"""Shift substitution in the CLIP-L/14 vision tower of LLaVA-1.5, compared on
generated captions.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def apply_kernel_to_values_2d(v_head, kernel, grid, n_extra_tokens):
    """v_head: (B, N, Dh).  Same as kernel_substitution.apply_kernel_to_values
    but inlined and adapted to live in this file (avoids transitive deps)."""
    B, N, Dh = v_head.shape
    H, W = grid
    assert N == n_extra_tokens + H * W
    v_extra = v_head[:, :n_extra_tokens] if n_extra_tokens > 0 else None
    v_spatial = v_head[:, n_extra_tokens:].reshape(B, H, W, Dh)
    K = (kernel.shape[0] - 1) // 2
    out = torch.zeros_like(v_spatial)
    norm = torch.zeros(H, W, device=v_head.device, dtype=v_head.dtype)
    qy = torch.arange(H, device=v_head.device)
    qx = torch.arange(W, device=v_head.device)
    for dy in range(-K, K + 1):
        for dx in range(-K, K + 1):
            w = kernel[dy + K, dx + K]
            if w.abs().item() < 1e-9: continue
            sy = qy + dy; sx = qx + dx
            valid_y = (sy >= 0) & (sy < H); valid_x = (sx >= 0) & (sx < W)
            mask = valid_y.unsqueeze(1) & valid_x.unsqueeze(0)
            mask_f = mask.to(v_head.dtype)
            shifted = torch.roll(v_spatial, shifts=(-dy, -dx), dims=(1, 2))
            out = out + shifted * (mask_f * w).unsqueeze(0).unsqueeze(-1)
            norm = norm + mask_f * w
    norm_safe = norm.unsqueeze(0).unsqueeze(-1).clamp(min=1e-8)
    out = out / norm_safe
    zero_mask = (norm.abs() < 1e-9).unsqueeze(0).unsqueeze(-1)
    if zero_mask.any():
        out = torch.where(zero_mask.expand_as(out), v_spatial, out)
    out_flat = out.reshape(B, H * W, Dh)
    if v_extra is not None:
        out_flat = torch.cat([v_extra, out_flat], dim=1)
    return out_flat


def make_clip_attn_substituted_forward(self_attn, head_kernel_map, grid,
                                          n_extra_tokens):
    """Substituted forward for HF CLIPAttention.
    HF source: transformers.models.clip.modeling_clip.CLIPAttention.forward
    """

    def new_forward(hidden_states, attention_mask=None,
                     causal_attention_mask=None, output_attentions=False, **kwargs):
        B, N, C = hidden_states.shape
        nh = self_attn.num_heads
        Dh = self_attn.head_dim

        q = self_attn.q_proj(hidden_states).reshape(B, N, nh, Dh).transpose(1, 2)
        k = self_attn.k_proj(hidden_states).reshape(B, N, nh, Dh).transpose(1, 2)
        v = self_attn.v_proj(hidden_states).reshape(B, N, nh, Dh).transpose(1, 2)

        scale = Dh ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        # NOTE: CLIP vision encoder typically has no causal mask
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = attn.softmax(dim=-1)
        attn_out = attn @ v                           # (B, nh, N, Dh)

        for h, kernel in head_kernel_map.items():
            v_h = v[:, h]                              # (B, N, Dh)
            attn_out[:, h] = apply_kernel_to_values_2d(
                v_h, kernel, grid, n_extra_tokens)

        x = attn_out.transpose(1, 2).reshape(B, N, C)
        x = self_attn.out_proj(x)
        return (x, attn) if output_attentions else (x, None)

    return new_forward


def install_vlm_substitutions(vision_tower, kernel_map, grid, n_extra_tokens):
    """Patch self_attn.forward of each affected encoder layer.
    kernel_map: {(layer, head): np.ndarray (2K+1, 2K+1)}
    Returns saved dict for restore."""
    by_layer: Dict[int, Dict[int, torch.Tensor]] = {}
    for (li, hi), ker in kernel_map.items():
        ker_t = torch.as_tensor(ker, dtype=torch.float32)
        by_layer.setdefault(li, {})[hi] = ker_t
    saved = {}
    enc_layers = vision_tower.encoder.layers
    for li, sub in by_layer.items():
        if li >= len(enc_layers):
            print(f"  skip L{li}: only {len(enc_layers)} layers in this vision tower")
            continue
        attn = enc_layers[li].self_attn
        device = next(attn.parameters()).device
        dtype = next(attn.parameters()).dtype
        sub = {h: ker.to(device=device, dtype=dtype) for h, ker in sub.items()}
        saved[li] = attn.forward
        attn.forward = make_clip_attn_substituted_forward(
            attn, sub, grid, n_extra_tokens)
    return saved


def restore_vlm(vision_tower, saved):
    enc_layers = vision_tower.encoder.layers
    for li, fwd in saved.items():
        enc_layers[li].self_attn.forward = fwd


def kernel_from_diagnostics_mean(mean_kernel: np.ndarray) -> np.ndarray:
    Z = float(mean_kernel.sum())
    if Z <= 0:
        raise ValueError("kernel has nonpositive mass")
    return (mean_kernel / Z).astype(np.float32)


def shared_prefix_len(a: List[int], b: List[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llava-id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--taxonomy", required=True,
                    help="taxonomy_clip_L.json from a CLIP-L/14")
    ap.add_argument("--coco-val", default="data/coco/val2017",
                    help="COCO val2017 image directory")
    ap.add_argument("--n-images", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=30)
    ap.add_argument("--prompt", default="USER: <image>\nWhat is in this image? Answer in one sentence.\nASSISTANT:")
    ap.add_argument("--shift-cats", default="delta_shift,soft_shift")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float16",
                    choices=("float16", "bfloat16", "float32"))
    args = ap.parse_args()

    import os
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    print(f"Loading {args.llava_id} ({dtype}) on {device} ...", flush=True)
    proc = AutoProcessor.from_pretrained(args.llava_id)
    m = LlavaForConditionalGeneration.from_pretrained(
        args.llava_id, dtype=dtype).to(device).eval()

    vt = m.model.vision_tower
    n_layers = len(vt.encoder.layers)
    n_heads = vt.encoder.layers[0].self_attn.num_heads
    print(f"  vision_tower: {type(vt).__name__}, {n_layers} layers x "
          f"{n_heads} heads", flush=True)

    image_size = vt.config.image_size
    patch_size = vt.config.patch_size
    grid_side = image_size // patch_size
    print(f"  image_size={image_size}, patch_size={patch_size}, "
          f"grid={grid_side}x{grid_side}", flush=True)

    # Read taxonomy + extract shift heads
    tax = json.loads(Path(args.taxonomy).read_text())["results"]
    shift_cats = set(args.shift_cats.split(","))
    res_by_lh = {(r["layer"], r["head"]): r for r in tax}
    shift_heads = [(r["layer"], r["head"]) for r in tax
                    if r["category"] in shift_cats]
    print(f"  shift heads in taxonomy: {len(shift_heads)} -> {shift_heads}",
          flush=True)
    kernel_map = {(li, hi): kernel_from_diagnostics_mean(
                       np.asarray(res_by_lh[(li, hi)]["mean_kernel"]))
                   for (li, hi) in shift_heads}

    # Pick N coco val images deterministically
    img_dir = Path(args.coco_val)
    img_paths = sorted(img_dir.glob("*.jpg"))[:args.n_images]
    print(f"  evaluating on {len(img_paths)} COCO val images", flush=True)

    def generate_caption(img):
        inputs = proc(images=img, text=args.prompt, return_tensors="pt").to(device, dtype)
        # processor pixel_values are float; keep input_ids long
        if "input_ids" in inputs and inputs["input_ids"].dtype != torch.long:
            inputs["input_ids"] = inputs["input_ids"].long()
        if "attention_mask" in inputs and inputs["attention_mask"].dtype != torch.long:
            inputs["attention_mask"] = inputs["attention_mask"].long()
        out = m.generate(**inputs, max_new_tokens=args.max_new_tokens,
                          do_sample=False, num_beams=1)
        # trim prompt prefix
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out[0, prompt_len:].tolist()
        text = proc.decode(out[0, prompt_len:], skip_special_tokens=True)
        return gen_ids, text.strip()

    # Run baseline + substituted on each image
    results = []
    for i, p in enumerate(img_paths):
        img = Image.open(p).convert("RGB")
        b_ids, b_text = generate_caption(img)
        saved = install_vlm_substitutions(vt, kernel_map,
                                            grid=(grid_side, grid_side),
                                            n_extra_tokens=1)
        try:
            s_ids, s_text = generate_caption(img)
        finally:
            restore_vlm(vt, saved)
        spl = shared_prefix_len(b_ids, s_ids)
        match = (b_text == s_text)
        results.append({
            "img": p.name,
            "baseline": b_text,
            "substituted": s_text,
            "shared_prefix_tokens": spl,
            "exact_match": match,
            "n_baseline_tokens": len(b_ids),
            "n_substituted_tokens": len(s_ids),
        })
        print(f"[{i+1}/{len(img_paths)}] match={match}  spl={spl}/"
              f"{min(len(b_ids), len(s_ids))}", flush=True)
        print(f"   B: {b_text[:120]}", flush=True)
        print(f"   S: {s_text[:120]}", flush=True)

    # Summary
    n = len(results)
    n_match = sum(r["exact_match"] for r in results)
    spls = [r["shared_prefix_tokens"] for r in results]
    out = {
        "llava_id": args.llava_id,
        "taxonomy": args.taxonomy,
        "n_images": n,
        "n_shift_heads_substituted": len(shift_heads),
        "max_new_tokens": args.max_new_tokens,
        "exact_match_rate": n_match / max(n, 1),
        "mean_shared_prefix": float(np.mean(spls)) if spls else 0.0,
        "median_shared_prefix": float(np.median(spls)) if spls else 0.0,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    print(f"\nsummary")
    print(f"  n_images:         {n}")
    print(f"  exact match rate: {n_match}/{n} = {n_match/n*100:.1f}%")
    print(f"  median shared prefix tokens: {np.median(spls):.1f}/"
          f"{args.max_new_tokens}")
    print(f"  mean shared prefix tokens:   {np.mean(spls):.1f}/"
          f"{args.max_new_tokens}")


if __name__ == "__main__":
    main()
