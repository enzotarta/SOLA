"""Train DeiT-Tiny from scratch on ImageNet-1k and save periodic checkpoints.

AdamW, cosine schedule, 5 warm-up epochs, label smoothing, RandAugment, Mixup,
CutMix and random erasing. No EMA and no repeated augmentation.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deit_tiny_patch16_224")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output", required=True,
                    help="output directory for checkpoints + log")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.8)
    ap.add_argument("--cutmix", type=float, default=1.0)
    ap.add_argument("--rand-aug", default="rand-m9-mstd0.5-inc1")
    ap.add_argument("--reprob", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=5,
                    help="save a checkpoint every N epochs")
    ap.add_argument("--resume", default=None,
                    help="path to checkpoint to resume from")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    device = "cuda"
    print(f"Device: {device}, GPU: {torch.cuda.get_device_name(0)}")

    import timm
    from timm.data import create_dataset, create_loader, Mixup, resolve_data_config
    from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
    from timm.optim import create_optimizer_v2
    from timm.scheduler import create_scheduler_v2

    # model
    model = timm.create_model(
        args.model, pretrained=False, num_classes=1000,
        drop_path_rate=args.drop_path,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}  params={n_params/1e6:.2f}M")

    # data
    dcfg = resolve_data_config({}, model=model, verbose=True)

    train_ds = create_dataset(
        name="", root=args.data_dir, split="train",
        is_training=True, batch_size=args.batch_size,
    )
    val_ds = create_dataset(
        name="", root=args.data_dir, split="val",
        is_training=False, batch_size=args.batch_size,
    )

    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
            label_smoothing=args.smoothing, num_classes=1000,
        )

    train_loader = create_loader(
        train_ds, input_size=dcfg["input_size"],
        batch_size=args.batch_size, is_training=True,
        no_aug=False, re_prob=args.reprob, re_mode="pixel",
        auto_augment=args.rand_aug,
        mean=dcfg["mean"], std=dcfg["std"],
        num_workers=args.workers, distributed=False,
        pin_memory=True, persistent_workers=True,
        crop_pct=dcfg.get("crop_pct", 0.875),
    )
    # Bump prefetch on the underlying DataLoader if accessible
    try:
        train_loader.loader.prefetch_factor = 4
    except Exception:
        pass
    val_loader = create_loader(
        val_ds, input_size=dcfg["input_size"],
        batch_size=args.batch_size, is_training=False,
        mean=dcfg["mean"], std=dcfg["std"],
        num_workers=args.workers, distributed=False,
        pin_memory=True, crop_pct=dcfg.get("crop_pct", 0.875),
    )
    print(f"train: {len(train_ds)} samples, {len(train_loader)} batches/epoch")
    print(f"val:   {len(val_ds)} samples, {len(val_loader)} batches")

    # optim + sched
    optimizer = create_optimizer_v2(
        model, opt="adamw", lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler, _ = create_scheduler_v2(
        optimizer, sched="cosine", num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs, warmup_lr=1e-6, min_lr=1e-5,
    )

    if mixup_fn is not None:
        train_loss_fn = SoftTargetCrossEntropy()
    else:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    val_loss_fn = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # train loop
    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n_iter = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if mixup_fn is not None:
                x, y = mixup_fn(x, y)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = train_loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
            n_iter += 1
            if i % 100 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  ep{epoch} it{i}/{len(train_loader)}  "
                      f"loss={loss.item():.4f} lr={lr_now:.2e}", flush=True)
        train_loss = loss_sum / max(n_iter, 1)
        train_time = time.time() - t0

        # eval
        model.eval()
        v_loss_sum = 0.0; v_correct = 0; v_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(x)
                v_loss_sum += val_loss_fn(logits, y).item() * x.size(0)
                v_correct += (logits.argmax(-1) == y).sum().item()
                v_total += x.size(0)
        val_loss = v_loss_sum / max(v_total, 1)
        val_top1 = v_correct / max(v_total, 1)

        log_entry = {
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_top1": val_top1,
            "train_time_s": train_time, "lr": optimizer.param_groups[0]["lr"],
        }
        print(f"epoch {epoch}: train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_top1={val_top1*100:.2f}% "
              f"({train_time:.0f}s)", flush=True)
        with log_path.open("a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # checkpoint
        save_now = ((epoch + 1) % args.ckpt_every == 0
                    or epoch == args.epochs - 1
                    or epoch == 0)
        if save_now:
            ck_path = out_dir / f"epoch_{epoch:03d}.pth"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "args": vars(args),
                "val_top1": val_top1,
            }, ck_path)
            print(f"  saved {ck_path}", flush=True)
        # always update last.pth for resume
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "val_top1": val_top1,
        }, out_dir / "last.pth")


if __name__ == "__main__":
    main()
