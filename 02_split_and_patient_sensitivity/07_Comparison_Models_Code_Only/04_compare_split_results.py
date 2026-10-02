from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a patient-level versus original image-level metric table.")
    parser.add_argument("--summaries", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frames = [pd.read_csv(path) for path in args.summaries if Path(path).is_file()]
    if not frames:
        raise RuntimeError("No completed summary CSV files were supplied.")
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["model", "split_policy", "eval_split", "metric"], keep="last"
    )
    wide = combined.pivot_table(
        index=["model", "eval_split", "metric"], columns="split_policy", values="mean", aggfunc="first"
    ).reset_index()
    for column in ("patient_level", "original_image_level"):
        if column not in wide:
            wide[column] = pd.NA
    wide["original_minus_patient"] = wide["original_image_level"] - wide["patient_level"]
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(destination, index=False, encoding="utf-8-sig")
    print(f"Saved split comparison to {destination}")


if __name__ == "__main__":
    main()

