from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import PairedManifestDataset, load_manifest, tensor_to_pil
from models_comparison import build_generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CEUS outputs from a completed comparison-model checkpoint.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test_internal_temporal"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--allow-patient-overlap", action="store_true")
    args = parser.parse_args()

    frame = load_manifest(args.manifest, args.allow_patient_overlap)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_generator(checkpoint["model"], int(checkpoint["image_size"]), pretrained=False).to(device)
    model.load_state_dict(checkpoint["generator_state_dict"], strict=True)
    model.eval()
    output = Path(args.output_dir).resolve()
    generated_paths = {}

    with torch.no_grad():
        for split in args.splits:
            subset = frame[frame.split.eq(split)]
            if subset.empty:
                continue
            loader = DataLoader(
                PairedManifestDataset(subset, int(checkpoint["image_size"]), augment=False),
                batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                pin_memory=device.type == "cuda",
            )
            split_dir = output / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for bmode, _, row_uids, _ in tqdm(loader, desc=f"generate {checkpoint['model']} {split}"):
                fake = model(bmode.to(device))
                for image, row_uid in zip(fake, row_uids):
                    destination = split_dir / f"{row_uid}.png"
                    tensor_to_pil(image).save(destination)
                    generated_paths[str(row_uid)] = str(destination)

    column = f"generated_{checkpoint['model']}_path"
    result = pd.read_csv(args.manifest)
    result[column] = result.row_uid.astype(str).map(generated_paths).fillna("")
    result.to_csv(output / "manifest_with_generated.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {len(generated_paths)} generated images and manifest to {output}")


if __name__ == "__main__":
    main()

