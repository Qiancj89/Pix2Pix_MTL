from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_rgb(path: str, size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def clustered_ci(frame: pd.DataFrame, metric: str, iterations: int, seed: int):
    rng = np.random.default_rng(seed)
    patients = frame.patient_id.astype(str).unique()
    values = []
    grouped = {patient: group for patient, group in frame.groupby(frame.patient_id.astype(str))}
    for _ in range(iterations):
        selected = rng.choice(patients, size=len(patients), replace=True)
        sample = pd.concat([grouped[patient] for patient in selected], ignore_index=True)
        values.append(float(sample[metric].mean()))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one or more generated-image columns using identical metrics.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="model=generated_column pairs")
    parser.add_argument("--split-policy", required=True, choices=["patient_level", "original_image_level"])
    parser.add_argument("--eval-splits", nargs="+", default=["val", "test_internal_temporal"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    pairs = []
    for item in args.models:
        if "=" not in item:
            raise RuntimeError(f"Expected model=column, received: {item}")
        pairs.append(item.split("=", 1))
    records = []
    for model, column in pairs:
        if column not in frame.columns:
            raise RuntimeError(f"Missing generated-image column: {column}")
        for split in args.eval_splits:
            subset = frame[frame.split.eq(split)].copy()
            valid = subset.real_ceus_path.fillna("").ne("") & subset[column].fillna("").ne("")
            subset = subset[valid]
            for row in subset.itertuples(index=False):
                real = load_rgb(row.real_ceus_path, args.image_size)
                generated = load_rgb(getattr(row, column), args.image_size)
                records.append({
                    "model": model, "split_policy": args.split_policy, "eval_split": split,
                    "row_uid": row.row_uid, "patient_id": row.patient_id,
                    "mae": float(np.abs(real - generated).mean()),
                    "psnr": float(peak_signal_noise_ratio(real, generated, data_range=1.0)),
                    "ssim": float(structural_similarity(real, generated, channel_axis=2, data_range=1.0)),
                })
    per_image = pd.DataFrame(records)
    if per_image.empty:
        raise RuntimeError("No paired real/generated images were found.")
    summary = []
    for keys, group in per_image.groupby(["model", "split_policy", "eval_split"], sort=False):
        model, policy, split = keys
        for metric in ("mae", "psnr", "ssim"):
            low, high = clustered_ci(group, metric, args.bootstrap, args.seed)
            summary.append({
                "model": model, "split_policy": policy, "eval_split": split, "metric": metric,
                "mean": group[metric].mean(), "ci_low": low, "ci_high": high,
                "n_images": len(group), "n_patients": group.patient_id.nunique(),
            })
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(output / "synthesis_metrics_per_image.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(output / "synthesis_metrics_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()

