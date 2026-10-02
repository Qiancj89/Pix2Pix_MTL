from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from common import json_dump


def load_array(path, size):
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def metric_values(real, generated, lpips_model=None):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    values = {"mae": float(np.abs(real - generated).mean()),
              "psnr": float(peak_signal_noise_ratio(real, generated, data_range=1.0)),
              "ssim": float(structural_similarity(real, generated, channel_axis=2, data_range=1.0))}
    if lpips_model is not None:
        import torch
        real_tensor = torch.from_numpy(real.transpose(2, 0, 1)).unsqueeze(0) * 2 - 1
        generated_tensor = torch.from_numpy(generated.transpose(2, 0, 1)).unsqueeze(0) * 2 - 1
        if min(real_tensor.shape[-2:]) < 64:
            real_tensor = torch.nn.functional.interpolate(real_tensor, size=(64, 64), mode="bilinear", align_corners=False)
            generated_tensor = torch.nn.functional.interpolate(generated_tensor, size=(64, 64), mode="bilinear", align_corners=False)
        with torch.no_grad():
            values["lpips"] = float(lpips_model(real_tensor, generated_tensor).item())
    return values


def bootstrap_ci(frame, column, iterations, seed):
    rng = np.random.default_rng(seed); patients = frame.patient_id.astype(str).unique(); means = []
    for _ in range(iterations):
        chosen = rng.choice(patients, len(patients), replace=True)
        sample = pd.concat([frame[frame.patient_id.astype(str).eq(pid)] for pid in chosen], ignore_index=True)
        means.append(sample[column].mean())
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Whole-image and optional blinded lesion-ROI CEUS synthesis metrics.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--generated-column", default="synthetic_ceus_path")
    parser.add_argument("--split", default="test_internal_temporal")
    parser.add_argument("--roi-csv", default=None, help="Columns: row_uid,x,y,width,height,confirmed_by,blinded_to_generation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--lpips", action="store_true", help="Compute LPIPS (requires the lpips package and cached weights).")
    args = parser.parse_args()
    lpips_model = None
    if args.lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net="alex").eval()
        except ImportError as exc:
            raise SystemExit("Install lpips or omit --lpips.") from exc
    frame = pd.read_csv(args.manifest); frame = frame[frame.split.eq(args.split)].copy()
    valid = frame["real_ceus_path"].fillna("").ne("") & frame[args.generated_column].fillna("").ne("")
    frame = frame[valid].copy()
    if frame.empty:
        raise RuntimeError("No rows have both real and generated CEUS for the requested split.")
    roi = pd.read_csv(args.roi_csv) if args.roi_csv else None
    if roi is not None:
        required = {"row_uid", "x", "y", "width", "height", "confirmed_by", "blinded_to_generation"}
        if not required.issubset(roi.columns):
            raise RuntimeError(f"ROI CSV needs columns: {sorted(required)}")
        if not roi["blinded_to_generation"].astype(str).str.lower().isin(["1", "true", "yes"]).all():
            raise RuntimeError("All ROIs must be drawn/confirmed blinded to generated results.")
        frame = frame.merge(roi, on="row_uid", how="left", validate="one_to_one")
    records = []
    for row in frame.itertuples(index=False):
        real = load_array(row.real_ceus_path, args.image_size)
        generated = load_array(getattr(row, args.generated_column), args.image_size)
        whole = {"row_uid": row.row_uid, "patient_id": row.patient_id, "region": "whole",
                 **metric_values(real, generated, lpips_model)}
        records.append(whole)
        if roi is not None and not pd.isna(row.x):
            x, y, width, height = [int(getattr(row, key)) for key in ("x", "y", "width", "height")]
            x2, y2 = min(x + width, args.image_size), min(y + height, args.image_size)
            if x < 0 or y < 0 or x2 <= x or y2 <= y:
                raise RuntimeError(f"Invalid ROI for {row.row_uid}")
            records.append({"row_uid": row.row_uid, "patient_id": row.patient_id, "region": "lesion_roi",
                            **metric_values(real[y:y2, x:x2], generated[y:y2, x:x2], lpips_model)})
    result = pd.DataFrame(records); summary = []
    for region, group in result.groupby("region"):
        for metric in [x for x in ("mae", "psnr", "ssim", "lpips") if x in group.columns]:
            ci = bootstrap_ci(group, metric, args.bootstrap, args.seed)
            summary.append({"region": region, "metric": metric, "mean": group[metric].mean(),
                            "ci_low": ci[0], "ci_high": ci[1], "n_images": len(group),
                            "n_patients": group.patient_id.nunique()})
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "synthesis_metrics_per_image.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(output / "synthesis_metrics_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Saved synthesis metrics to {output}")


if __name__ == "__main__":
    main()
