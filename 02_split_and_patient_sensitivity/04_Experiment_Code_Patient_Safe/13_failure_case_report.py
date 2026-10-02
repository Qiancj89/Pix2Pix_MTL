from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Prespecify and export false-positive, false-negative, and synthesis-failure cases.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--synthesis-metrics", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-each", type=int, default=10)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest); predictions = pd.read_csv(args.predictions)
    data = predictions.merge(manifest[["row_uid", "bmode_path", "real_ceus_path", "synthetic_ceus_path"]],
                             on="row_uid", how="left", validate="one_to_one")
    data["error_type"] = "correct"
    data.loc[data.Malignant_true.eq(0) & data.Malignant_pred.eq(1), "error_type"] = "false_positive"
    data.loc[data.Malignant_true.eq(1) & data.Malignant_pred.eq(0), "error_type"] = "false_negative"
    false_positive = data[data.error_type.eq("false_positive")].sort_values("Malignant_prob", ascending=False).head(args.n_each)
    false_negative = data[data.error_type.eq("false_negative")].sort_values("Malignant_prob").head(args.n_each)
    selected = pd.concat([false_positive, false_negative], ignore_index=True)
    if args.synthesis_metrics:
        metrics = pd.read_csv(args.synthesis_metrics); metrics = metrics[metrics.region.eq("whole")]
        worst = metrics.sort_values(["ssim", "mae"], ascending=[True, False]).head(args.n_each)
        worst = worst.merge(data, on=["row_uid", "patient_id"], how="left"); worst["error_type"] = "worst_synthesis"
        selected = pd.concat([selected, worst], ignore_index=True, sort=False)
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "failure_case_index.csv", index=False, encoding="utf-8-sig")
    for _, row in selected.iterrows():
        case_dir = output / str(row.error_type) / str(row.row_uid); case_dir.mkdir(parents=True, exist_ok=True)
        for column in ("bmode_path", "real_ceus_path", "synthetic_ceus_path"):
            path = str(row.get(column, ""))
            if path and path != "nan" and Path(path).is_file():
                shutil.copy2(path, case_dir / f"{column}{Path(path).suffix.lower()}")
    print(f"Saved prespecified failure cases to {output}")


if __name__ == "__main__":
    main()

