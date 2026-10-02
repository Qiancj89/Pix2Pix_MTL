from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import assert_patient_disjoint, json_dump, seed_everything
from data import PairedTranslationDataset
from models import PatchDiscriminator, Pix2PixGenerator, sobel_edges, ssim_index


def validate(generator, loader, device):
    generator.eval()
    values = {"mae": [], "ssim": [], "mse": []}
    with torch.no_grad():
        for bmode, ceus, _, _ in loader:
            fake = generator(bmode.to(device))
            target = ceus.to(device)
            values["mae"].append(F.l1_loss(fake, target).item())
            values["mse"].append(F.mse_loss(fake, target).item())
            values["ssim"].append(ssim_index(fake, target).item())
    return {k: sum(v) / len(v) for k, v in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train revised Pix2Pix only on the frozen patient-level train split.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--lambda-edge", type=float, default=10.0)
    parser.add_argument("--lambda-ssim", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--allow-patient-overlap", action="store_true",
                        help="Comparison-only: reproduce a documented image-level split with patient overlap.")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.manifest)
    overlap = assert_patient_disjoint(frame, allow_overlap=args.allow_patient_overlap)
    paired = frame[frame["bmode_path"].fillna("").ne("") & frame["real_ceus_path"].fillna("").ne("")]
    train = paired[paired.split.eq("train")]
    val = paired[paired.split.eq("val")]
    if train.empty or val.empty:
        raise RuntimeError("Paired B-mode/real-CEUS rows are required in both train and val splits.")
    train_loader = DataLoader(PairedTranslationDataset(train, args.image_size, True), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(PairedTranslationDataset(val, args.image_size, False), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    generator, discriminator = Pix2PixGenerator().to(device), PatchDiscriminator().to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    history, best_ssim, patience = [], float("-inf"), 0
    for epoch in range(1, args.epochs + 1):
        generator.train(); discriminator.train()
        running_g = running_d = 0.0
        for bmode, ceus, _, _ in tqdm(train_loader, desc=f"pix2pix {epoch}/{args.epochs}"):
            bmode, ceus = bmode.to(device), ceus.to(device)
            with torch.no_grad():
                fake_detached = generator(bmode)
            optimizer_d.zero_grad(set_to_none=True)
            d_real = discriminator(bmode, ceus)
            d_fake = discriminator(bmode, fake_detached)
            loss_d = 0.5 * (F.mse_loss(d_real, torch.ones_like(d_real)) + F.mse_loss(d_fake, torch.zeros_like(d_fake)))
            loss_d.backward(); optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            fake = generator(bmode)
            adversarial = F.mse_loss(discriminator(bmode, fake), torch.ones_like(d_real))
            l1 = F.l1_loss(fake, ceus)
            edge = F.l1_loss(sobel_edges(fake), sobel_edges(ceus))
            ssim_loss = 1 - ssim_index(fake, ceus)
            loss_g = adversarial + args.lambda_l1 * l1 + args.lambda_edge * edge + args.lambda_ssim * ssim_loss
            loss_g.backward(); optimizer_g.step()
            running_g += loss_g.item(); running_d += loss_d.item()

        metrics = validate(generator, val_loader, device)
        record = {"epoch": epoch, "generator_loss": running_g / len(train_loader),
                  "discriminator_loss": running_d / len(train_loader), **{f"val_{k}": v for k, v in metrics.items()}}
        history.append(record); json_dump(history, output / "history.json")
        print(record)
        if metrics["ssim"] > best_ssim:
            best_ssim, patience = metrics["ssim"], 0
            torch.save({"generator": generator.state_dict(), "epoch": epoch, "val_metrics": metrics,
                        "seed": args.seed, "manifest": str(Path(args.manifest).resolve()),
                        "allow_patient_overlap": bool(args.allow_patient_overlap),
                        "overlapping_patient_count": len(overlap)}, output / "best_generator.pt")
        else:
            patience += 1
            if patience >= args.patience:
                print("Early stopping")
                break


if __name__ == "__main__":
    main()
