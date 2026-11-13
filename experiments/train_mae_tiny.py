"""Train an MAE with a ViT-Tiny encoder on ImageNet-1k and save periodic checkpoints.

Mask ratio 0.75 and a 3-block decoder, with the same encoder and epoch budget as
train_deit_tiny.py.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Patch utilities

def patchify(imgs, patch_size=16):
    """imgs: (B, 3, H, W) -> (B, n_patches, 3*P*P)"""
    P = patch_size
    B, C, H, W = imgs.shape
    h = H // P; w = W // P
    x = imgs.reshape(B, C, h, P, w, P).permute(0, 2, 4, 3, 5, 1).contiguous()
    x = x.reshape(B, h * w, P * P * C)
    return x


def random_masking(x, mask_ratio):
    """x: (B, N, D). Returns (x_unmasked, mask, ids_restore).
    ids_restore: indices to scatter back to original order.
    """
    B, N, D = x.shape
    n_keep = int(N * (1 - mask_ratio))
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :n_keep]
    x_kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask = torch.ones(B, N, device=x.device)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return x_kept, mask, ids_restore


# Model

class MAETiny(nn.Module):
    def __init__(self, encoder, decoder_dim=96, decoder_depth=3,
                 decoder_heads=3, mask_ratio=0.75, patch_size=16,
                 img_size=224, in_chans=3):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.in_chans = in_chans

        # Drop the classifier head if present
        if hasattr(self.encoder, "head"):
            self.encoder.head = nn.Identity()

        n_patches = (img_size // patch_size) ** 2
        enc_dim = encoder.embed_dim

        self.dec_embed = nn.Linear(enc_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.dec_pos_embed = nn.Parameter(
            torch.zeros(1, n_patches + 1, decoder_dim))    # +1 cls

        from timm.models.vision_transformer import Block as ViTBlock
        self.dec_blocks = nn.ModuleList([
            ViTBlock(dim=decoder_dim, num_heads=decoder_heads,
                     mlp_ratio=4.0, qkv_bias=True)
            for _ in range(decoder_depth)
        ])
        self.dec_norm = nn.LayerNorm(decoder_dim)
        self.dec_pred = nn.Linear(decoder_dim, patch_size * patch_size * in_chans,
                                    bias=True)

        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.dec_pos_embed, std=0.02)

    def forward_encoder(self, x):
        # x: (B, 3, H, W)
        # We need a forward pass that:
        # patchifies + adds pos embed
        # masks 75% (random_masking)
        # runs through encoder transformer blocks on the kept tokens + cls
        # timm's vision_transformer does steps 1+3 in `forward_features`,
        # but inserts cls + pos embed first. Easiest: replicate the bits.

        enc = self.encoder
        x = enc.patch_embed(x)                                      # (B, N, D)
        # add positional embeddings (skip cls slot)
        pos = enc.pos_embed[:, 1:, :]                                # (1, N, D)
        x = x + pos

        # Random masking on patch tokens
        x_kept, mask, ids_restore = random_masking(x, self.mask_ratio)

        # Prepend cls token (with its own pos embed)
        cls_tok = enc.cls_token + enc.pos_embed[:, :1, :]
        cls_tok = cls_tok.expand(x_kept.shape[0], -1, -1)
        x_kept = torch.cat([cls_tok, x_kept], dim=1)
        x_kept = enc.norm_pre(x_kept) if hasattr(enc, "norm_pre") else x_kept
        x_kept = enc.pos_drop(x_kept) if hasattr(enc, "pos_drop") else x_kept

        for blk in enc.blocks:
            x_kept = blk(x_kept)
        x_kept = enc.norm(x_kept) if hasattr(enc, "norm") else x_kept
        return x_kept, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.dec_embed(x)
        # Append mask tokens for the masked positions, then unshuffle
        cls = x[:, :1, :]
        x_no_cls = x[:, 1:, :]
        mask_tokens = self.mask_token.expand(
            x.shape[0], ids_restore.shape[1] - x_no_cls.shape[1], -1)
        x_full = torch.cat([x_no_cls, mask_tokens], dim=1)
        x_full = torch.gather(
            x_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[-1]))
        x = torch.cat([cls, x_full], dim=1)
        x = x + self.dec_pos_embed
        for blk in self.dec_blocks:
            x = blk(x)
        x = self.dec_norm(x)
        x = self.dec_pred(x)
        return x[:, 1:, :]                                            # drop cls

    def forward(self, imgs):
        # imgs: (B, 3, H, W)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            x_kept, mask, ids_restore = self.forward_encoder(imgs)
            pred = self.forward_decoder(x_kept, ids_restore)         # (B, N, P*P*C)
        target = patchify(imgs, self.patch_size)
        # Per-patch normalization (as in MAE paper)
        target_mean = target.mean(dim=-1, keepdim=True)
        target_std  = target.std(dim=-1, keepdim=True).clamp(min=1e-6)
        target = (target - target_mean) / target_std
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                                      # (B, N)
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        return loss


# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--decoder-depth", type=int, default=3)
    ap.add_argument("--decoder-dim", type=int, default=96)
    ap.add_argument("--decoder-heads", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    device = "cuda"
    print(f"Device: {device}, GPU: {torch.cuda.get_device_name(0)}", flush=True)

    import timm
    from timm.data import create_dataset, create_loader
    from timm.optim import create_optimizer_v2
    from timm.scheduler import create_scheduler_v2

    # Encoder = same arch as the supervised run
    encoder = timm.create_model(
        "vit_tiny_patch16_224", pretrained=False,
        num_classes=0,                 # no head; we'll replace with Identity
        global_pool="",                # disable pooling, we want full token output
    )
    model = MAETiny(
        encoder, decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        mask_ratio=args.mask_ratio,
    ).to(device)
    n_params_enc = sum(p.numel() for p in encoder.parameters())
    n_params_tot = sum(p.numel() for p in model.parameters())
    print(f"Encoder: vit_tiny_patch16_224  params={n_params_enc/1e6:.2f}M",
          flush=True)
    print(f"Total (incl decoder)        params={n_params_tot/1e6:.2f}M",
          flush=True)

    # Data: we only use train split (MAE has no labels)
    train_ds = create_dataset(name="", root=args.data_dir, split="train",
                                is_training=True, batch_size=args.batch_size)
    train_loader = create_loader(
        train_ds, input_size=(3, 224, 224),
        batch_size=args.batch_size, is_training=True,
        no_aug=True, re_prob=0.0,                # MAE: minimal aug
        auto_augment=None,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        num_workers=args.workers, distributed=False,
        pin_memory=True, persistent_workers=True,
        crop_pct=1.0,
    )
    print(f"train: {len(train_ds)} samples, {len(train_loader)} batches/epoch",
          flush=True)

    optimizer = create_optimizer_v2(
        model, opt="adamw", lr=args.lr, weight_decay=args.weight_decay)
    scheduler, _ = create_scheduler_v2(
        optimizer, sched="cosine", num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs, warmup_lr=1e-6, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck.get("model_full", ck["model"]),
                              strict=("model_full" in ck))
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        model.train()
        t0 = time.time()
        loss_sum = 0.0; n_iter = 0
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model(x)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item(); n_iter += 1
            if i % 100 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  ep{epoch} it{i}/{len(train_loader)}  "
                      f"loss={loss.item():.4f} lr={lr_now:.2e}", flush=True)
        train_loss = loss_sum / max(n_iter, 1)
        train_time = time.time() - t0

        log_entry = {
            "epoch": epoch, "train_loss": train_loss,
            "train_time_s": train_time,
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(f"epoch {epoch}: train_loss={train_loss:.4f} "
              f"({train_time:.0f}s)", flush=True)
        with log_path.open("a") as f:
            f.write(json.dumps(log_entry) + "\n")

        save_now = ((epoch + 1) % args.ckpt_every == 0
                    or epoch == args.epochs - 1
                    or epoch == 0)
        if save_now:
            ck_path = out_dir / f"epoch_{epoch:03d}.pth"
            # Save only the encoder state_dict so it's loadable into a vanilla
            # timm vit_tiny_patch16_224 for the analysis pipeline.
            enc_sd = {k: v.detach().cpu()
                       for k, v in model.encoder.state_dict().items()}
            torch.save({
                "epoch": epoch,
                "model": enc_sd,
                "model_full": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "args": vars(args),
                "val_top1": None,                   # MAE has no top-1 by construction
            }, ck_path)
            print(f"  saved {ck_path}", flush=True)
        torch.save({
            "epoch": epoch,
            "model": {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()},
            "model_full": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "val_top1": None,
        }, out_dir / "last.pth")


if __name__ == "__main__":
    main()
