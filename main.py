# main.py

import os
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

from wsi_dataset import DynamicWSICortMedDataset


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# Augmentations
# ============================================================

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3
        ),

        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=10,
            val_shift_limit=10,
            p=0.2
        ),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2()
    ])


def get_val_transform():
    return A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


# ============================================================
# Model
# ============================================================

def build_model(args):
    if args.model == "deeplabv3":
        model = smp.DeepLabV3(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            in_channels=3,
            classes=args.num_classes
        )

    elif args.model == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            in_channels=3,
            classes=args.num_classes
        )

    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model


# ============================================================
# Loss
# ============================================================

class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, ce_weight=1.0, dice_weight=1.0):
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.ce = nn.CrossEntropyLoss(
            weight=class_weights
        )

        self.dice = smp.losses.DiceLoss(
            mode="multiclass",
            from_logits=True
        )

    def forward(self, logits, masks):
        ce_loss = self.ce(logits, masks)
        dice_loss = self.dice(logits, masks)

        loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        return loss, ce_loss.detach(), dice_loss.detach()


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def compute_metrics(logits, masks, num_classes):
    preds = torch.argmax(logits, dim=1)

    preds = preds.view(-1)
    masks = masks.view(-1)

    valid = (masks >= 0) & (masks < num_classes)
    preds = preds[valid]
    masks = masks[valid]

    hist = torch.bincount(
        num_classes * masks + preds,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes).float()

    intersection = torch.diag(hist)
    union = hist.sum(dim=1) + hist.sum(dim=0) - intersection

    iou = intersection / torch.clamp(union, min=1.0)
    miou = iou.mean().item()

    pixel_acc = intersection.sum() / torch.clamp(hist.sum(), min=1.0)

    return {
        "pixel_acc": pixel_acc.item(),
        "miou": miou,
        "iou_background": iou[0].item(),
        "iou_cortical": iou[1].item(),
        "iou_medullary": iou[2].item(),
    }


# ============================================================
# Train One Epoch / 100k Steps
# ============================================================

def train(args, model, train_loader, optimizer, scaler, criterion, device):
    model.train()

    running_loss = 0.0
    running_ce = 0.0
    running_dice = 0.0

    for step, batch in enumerate(train_loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].long().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(images)
            loss, ce_loss, dice_loss = criterion(logits, masks)

        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip
            )

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        running_ce += ce_loss.item()
        running_dice += dice_loss.item()

        if step % args.log_interval == 0:
            metrics = compute_metrics(
                logits.detach(),
                masks.detach(),
                num_classes=args.num_classes
            )

            avg_loss = running_loss / args.log_interval
            avg_ce = running_ce / args.log_interval
            avg_dice = running_dice / args.log_interval

            print(
                f"[Step {step:06d}/{args.num_steps}] "
                f"loss={avg_loss:.4f} "
                f"ce={avg_ce:.4f} "
                f"dice={avg_dice:.4f} "
                f"acc={metrics['pixel_acc']:.4f} "
                f"mIoU={metrics['miou']:.4f} "
                f"IoU_bg={metrics['iou_background']:.4f} "
                f"IoU_cort={metrics['iou_cortical']:.4f} "
                f"IoU_med={metrics['iou_medullary']:.4f}"
            )

            running_loss = 0.0
            running_ce = 0.0
            running_dice = 0.0

        if step % args.save_interval == 0:
            save_checkpoint(
                args=args,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                filename=f"checkpoint_step_{step}.pt"
            )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(args, model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_metrics = {
        "pixel_acc": 0.0,
        "miou": 0.0,
        "iou_background": 0.0,
        "iou_cortical": 0.0,
        "iou_medullary": 0.0,
    }

    n_batches = 0

    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].long().to(device, non_blocking=True)

        logits = model(images)
        loss, _, _ = criterion(logits, masks)

        metrics = compute_metrics(
            logits,
            masks,
            num_classes=args.num_classes
        )

        total_loss += loss.item()

        for k in total_metrics:
            total_metrics[k] += metrics[k]

        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)

    for k in total_metrics:
        total_metrics[k] /= max(n_batches, 1)

    print(
        f"[Validation] "
        f"loss={avg_loss:.4f} "
        f"acc={total_metrics['pixel_acc']:.4f} "
        f"mIoU={total_metrics['miou']:.4f} "
        f"IoU_bg={total_metrics['iou_background']:.4f} "
        f"IoU_cort={total_metrics['iou_cortical']:.4f} "
        f"IoU_med={total_metrics['iou_medullary']:.4f}"
    )

    return avg_loss, total_metrics


# ============================================================
# Checkpointing
# ============================================================

def save_checkpoint(args, model, optimizer, scaler, step, filename):
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    path = save_dir / filename

    checkpoint = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
    }

    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(model, optimizer, scaler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    start_step = checkpoint.get("step", 0)

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Resuming from step {start_step}")

    return start_step


# ============================================================
# Main
# ============================================================

def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_dataset = DynamicWSICortMedDataset(
        root_dir=args.train_root,
        tile_size=args.tile_size,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        transform=train_transform,
        min_class_pixels=args.min_class_pixels,
        max_sampling_attempts=args.max_sampling_attempts,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
    )

    val_loader = None

    if args.val_root is not None:
        val_dataset = DynamicWSICortMedDataset(
            root_dir=args.val_root,
            tile_size=args.tile_size,
            num_steps=args.val_steps,
            batch_size=args.batch_size,
            transform=val_transform,
            min_class_pixels=args.min_class_pixels,
            max_sampling_attempts=args.max_sampling_attempts,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
            worker_init_fn=seed_worker,
        )

    model = build_model(args)
    model = model.to(device)

    class_weights = torch.tensor(
        args.class_weights,
        dtype=torch.float32,
        device=device
    )

    criterion = CombinedLoss(
        class_weights=class_weights,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if args.resume is not None:
        load_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            checkpoint_path=args.resume,
            device=device
        )

    train(
        args=args,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scaler=scaler,
        criterion=criterion,
        device=device
    )

    save_checkpoint(
        args=args,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=args.num_steps,
        filename="final_model.pt"
    )

    if val_loader is not None:
        validate(
            args=args,
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeepLabV3 / DeepLabV3+ for cortical-medullary WSI segmentation"
    )

    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--val_root", type=str, default=None)

    parser.add_argument(
        "--model",
        type=str,
        default="deeplabv3plus",
        choices=["deeplabv3", "deeplabv3plus"]
    )

    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet50",
        help="Example: resnet34, resnet50, resnet101, efficientnet-b0"
    )

    parser.add_argument(
        "--encoder_weights",
        type=str,
        default="imagenet",
        help="Use 'imagenet' or None"
    )

    parser.add_argument("--num_classes", type=int, default=3)

    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--val_steps", type=int, default=500)

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument(
        "--class_weights",
        type=float,
        nargs=3,
        default=[0.25, 1.0, 1.5],
        help="Weights for background cortical medullary"
    )

    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)

    parser.add_argument("--min_class_pixels", type=int, default=100)
    parser.add_argument("--max_sampling_attempts", type=int, default=50)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=5000)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.encoder_weights == "None":
        args.encoder_weights = None

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)