from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

from common import assert_patient_disjoint
from models import Pix2PixGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CEUS-like images without changing the frozen split.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--allow-patient-overlap", action="store_true",
                        help="Comparison-only: accept an explicitly audited image-level split.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(args.manifest)
    assert_patient_disjoint(frame, allow_overlap=args.allow_patient_overlap)
    generator = Pix2PixGenerator().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    root = Path(args.output_dir).resolve()
    paths = []
    with torch.no_grad():
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="generate"):
            source = Image.open(row.bmode_path).convert("L").resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
            tensor = (TF.to_tensor(source) * 2 - 1).unsqueeze(0).to(device)
            fake = (generator(tensor).squeeze(0).cpu() + 1) / 2
            destination = root / str(row.split) / f"{row.row_uid}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            TF.to_pil_image(fake.clamp(0, 1)).save(destination)
            paths.append(str(destination))
    frame["synthetic_ceus_path"] = paths
    output_manifest = Path(args.output_manifest).resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_manifest, index=False, encoding="utf-8-sig")
    assert_patient_disjoint(frame, allow_overlap=args.allow_patient_overlap)
    print(f"Wrote generated images and immutable-split manifest: {output_manifest}")


if __name__ == "__main__":
    main()
