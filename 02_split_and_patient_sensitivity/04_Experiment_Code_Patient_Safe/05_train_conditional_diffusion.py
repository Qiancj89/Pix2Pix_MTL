from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import assert_patient_disjoint, json_dump, seed_everything
from data import PairedTranslationDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Conditional DDPM baseline: noisy CEUS + B-mode -> CEUS noise.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--allow-patient-overlap", action="store_true",
                        help="Comparison-only: accept an explicitly audited image-level split.")
    args = parser.parse_args()
    try:
        from diffusers import DDPMScheduler, UNet2DModel
    except ImportError as exc:
        raise SystemExit("Install requirements.txt (diffusers and accelerate are required).") from exc

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(args.manifest)
    overlap = assert_patient_disjoint(frame, allow_overlap=args.allow_patient_overlap)
    paired = frame[frame["bmode_path"].fillna("").ne("") & frame["real_ceus_path"].fillna("").ne("")]
    train, val = paired[paired.split.eq("train")], paired[paired.split.eq("val")]
    if train.empty or val.empty:
        raise RuntimeError("Diffusion baseline needs paired train and val rows.")
    loaders = {
        "train": DataLoader(PairedTranslationDataset(train, args.image_size, True), batch_size=args.batch_size,
                            shuffle=True, num_workers=args.workers),
        "val": DataLoader(PairedTranslationDataset(val, args.image_size, False), batch_size=args.batch_size,
                          shuffle=False, num_workers=args.workers),
    }
    model = UNet2DModel(
        sample_size=args.image_size, in_channels=4, out_channels=3,
        layers_per_block=2, block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=args.timesteps, beta_schedule="squaredcos_cap_v2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_values = {}
        for phase in ("train", "val"):
            model.train(phase == "train")
            total = 0.0
            for bmode, ceus, _, _ in tqdm(loaders[phase], desc=f"ddpm {phase} {epoch}/{args.epochs}"):
                bmode, ceus = bmode.to(device), ceus.to(device)
                noise = torch.randn_like(ceus)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (ceus.shape[0],), device=device).long()
                noisy = scheduler.add_noise(ceus, noise, timesteps)
                with torch.set_grad_enabled(phase == "train"):
                    predicted = model(torch.cat([noisy, bmode], dim=1), timesteps).sample
                    loss = F.mse_loss(predicted, noise)
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True); loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                total += loss.item()
            epoch_values[f"{phase}_noise_mse"] = total / len(loaders[phase])
        history.append({"epoch": epoch, **epoch_values}); json_dump(history, output / "history.json")
        print(history[-1])
        if epoch_values["val_noise_mse"] < best:
            best = epoch_values["val_noise_mse"]
            model.save_pretrained(output / "best_unet")
            scheduler.save_pretrained(output / "scheduler")
            json_dump({"epoch": epoch, "val_noise_mse": best, "manifest": str(Path(args.manifest).resolve()),
                       "seed": args.seed, "image_size": args.image_size,
                       "allow_patient_overlap": bool(args.allow_patient_overlap),
                       "overlapping_patient_count": len(overlap)}, output / "best_metadata.json")


if __name__ == "__main__":
    main()
