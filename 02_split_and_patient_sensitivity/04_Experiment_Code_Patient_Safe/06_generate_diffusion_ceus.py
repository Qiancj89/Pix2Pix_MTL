from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

from common import assert_patient_disjoint, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the conditional-diffusion comparison images.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--scheduler-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--inference-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--allow-patient-overlap", action="store_true",
                        help="Comparison-only: accept an explicitly audited image-level split.")
    args = parser.parse_args()
    try:
        from diffusers import DDPMScheduler, UNet2DModel
    except ImportError as exc:
        raise SystemExit("Install diffusers first.") from exc
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2DModel.from_pretrained(args.model_dir).to(device).eval()
    scheduler = DDPMScheduler.from_pretrained(args.scheduler_dir)
    scheduler.set_timesteps(args.inference_steps, device=device)
    frame = pd.read_csv(args.manifest); assert_patient_disjoint(frame, allow_overlap=args.allow_patient_overlap)
    destination_root = Path(args.output_dir).resolve(); paths = []
    with torch.no_grad():
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="diffusion generate"):
            bmode = Image.open(row.bmode_path).convert("L").resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
            condition = (TF.to_tensor(bmode) * 2 - 1).unsqueeze(0).to(device)
            generator = torch.Generator(device=device).manual_seed(args.seed + int(str(row.row_uid)[:8], 16))
            sample = torch.randn((1, 3, args.image_size, args.image_size), generator=generator, device=device)
            for timestep in scheduler.timesteps:
                model_input = torch.cat([sample, condition], dim=1)
                predicted_noise = model(model_input, timestep).sample
                sample = scheduler.step(predicted_noise, timestep, sample).prev_sample
            destination = destination_root / str(row.split) / f"{row.row_uid}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            TF.to_pil_image(((sample.squeeze(0).cpu() + 1) / 2).clamp(0, 1)).save(destination)
            paths.append(str(destination))
    frame["diffusion_ceus_path"] = paths
    output_manifest = Path(args.output_manifest).resolve(); output_manifest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_manifest, index=False, encoding="utf-8-sig")
    print(output_manifest)


if __name__ == "__main__":
    main()
