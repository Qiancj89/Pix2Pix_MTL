from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import PairedManifestDataset, load_manifest, save_json, seed_everything
from models_comparison import PatchDiscriminator, build_generator, sobel_edges, ssim_index


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    totals = {"mae": 0.0, "mse": 0.0, "ssim": 0.0}
    count = 0
    for bmode, ceus, _, _ in loader:
        bmode, ceus = bmode.to(device), ceus.to(device)
        fake = model(bmode)
        batch = bmode.shape[0]
        totals["mae"] += F.l1_loss(fake, ceus).item() * batch
        totals["mse"] += F.mse_loss(fake, ceus).item() * batch
        totals["ssim"] += ssim_index((fake + 1) / 2, (ceus + 1) / 2).item() * batch
        count += batch
    return {key: value / count for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train only the newly added independent synthesis baseline.")
    parser.add_argument("--model", required=True, choices=["resnet18_unet_transfer", "swin_tiny_unet"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=10)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--lambda-edge", type=float, default=10.0)
    parser.add_argument("--lambda-ssim", type=float, default=5.0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--allow-patient-overlap", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = load_manifest(args.manifest, args.allow_patient_overlap)
    train = frame[frame.split.eq("train")]
    val = frame[frame.split.eq("val")]
    if train.empty or val.empty:
        raise RuntimeError("Both train and val rows are required.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        PairedManifestDataset(train, args.image_size, augment=True), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        PairedManifestDataset(val, args.image_size, augment=False), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda"
    )
    generator = build_generator(args.model, args.image_size, pretrained=args.pretrained).to(device)
    discriminator = PatchDiscriminator().to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    encoder_parameters = list(generator.encoder_parameters())
    best_ssim, stale, history = float("-inf"), 0, []

    for epoch in range(1, args.epochs + 1):
        encoder_trainable = epoch > args.freeze_encoder_epochs
        for parameter in encoder_parameters:
            parameter.requires_grad_(encoder_trainable)
        generator.train()
        discriminator.train()
        running_g = running_d = 0.0
        for bmode, ceus, _, _ in tqdm(train_loader, desc=f"{args.model} {epoch}/{args.epochs}"):
            bmode, ceus = bmode.to(device), ceus.to(device)
            optimizer_d.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                with torch.no_grad():
                    detached = generator(bmode)
                real_score = discriminator(bmode, ceus)
                fake_score = discriminator(bmode, detached)
                loss_d = 0.5 * (
                    F.mse_loss(real_score, torch.ones_like(real_score))
                    + F.mse_loss(fake_score, torch.zeros_like(fake_score))
                )
            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)

            optimizer_g.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                fake = generator(bmode)
                score = discriminator(bmode, fake)
                adversarial = F.mse_loss(score, torch.ones_like(score))
                l1 = F.l1_loss(fake, ceus)
                edge = F.l1_loss(sobel_edges(fake), sobel_edges(ceus))
                ssim_loss = 1 - ssim_index((fake + 1) / 2, (ceus + 1) / 2)
                loss_g = adversarial + args.lambda_l1 * l1 + args.lambda_edge * edge + args.lambda_ssim * ssim_loss
            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()
            running_g += loss_g.item()
            running_d += loss_d.item()

        metrics = validate(generator, val_loader, device)
        record = {
            "epoch": epoch, "encoder_frozen": not encoder_trainable,
            "generator_loss": running_g / len(train_loader),
            "discriminator_loss": running_d / len(train_loader),
            **{f"val_{key}": value for key, value in metrics.items()},
        }
        history.append(record)
        save_json(history, output / "history.json")
        print(record)
        if metrics["ssim"] > best_ssim:
            best_ssim, stale = metrics["ssim"], 0
            torch.save(
                {
                    "generator_state_dict": generator.state_dict(), "model": args.model,
                    "image_size": args.image_size, "epoch": epoch, "val_metrics": metrics,
                    "pretrained_initialization": bool(args.pretrained), "seed": args.seed,
                    "manifest": str(Path(args.manifest).resolve()),
                    "allow_patient_overlap": bool(args.allow_patient_overlap),
                    "overlapping_patient_count": len(frame.attrs["overlapping_patient_ids"]),
                },
                output / "best_generator.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    save_json(
        {"status": "complete", "epochs_finished": len(history), "best_val_ssim": best_ssim},
        output / "TRAINING_COMPLETE.json",
    )


if __name__ == "__main__":
    main()

