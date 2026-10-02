from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def exact_path_overlap(frame: pd.DataFrame, column: str) -> list[str]:
    subset = frame.loc[frame[column].fillna("").ne(""), [column, "split"]].copy()
    subset[column] = subset[column].map(lambda x: str(Path(x).resolve()).casefold())
    counts = subset.groupby(column)["split"].nunique()
    return counts[counts > 1].index.astype(str).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the recovered manuscript image-level split without hiding patient overlap.")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest, dtype={"patient_id": str, "lesion_id": str, "image_id": str})
    required = {"row_uid", "patient_id", "lesion_id", "image_id", "split", "bmode_path", "real_ceus_path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    expected = {"train": 179, "val": 78, "test_internal_temporal": 67}
    observed = frame["split"].value_counts().to_dict()
    if observed != expected:
        raise RuntimeError(f"Expected {expected}; found {observed}")
    if frame["row_uid"].duplicated().any():
        raise RuntimeError("Duplicate row_uid values found")
    missing_files = {}
    for column in ("bmode_path", "real_ceus_path"):
        bad = frame[column].fillna("").map(lambda x: not x or not Path(x).is_file())
        missing_files[column] = int(bad.sum())
        if bad.any():
            raise RuntimeError(f"Missing {column} files: {int(bad.sum())}")
        overlaps = exact_path_overlap(frame, column)
        if overlaps:
            raise RuntimeError(f"Exact path overlap in {column}: {overlaps[:10]}")

    patients = {split: set(group["patient_id"].astype(str)) for split, group in frame.groupby("split")}
    train_val = sorted(patients["train"] & patients["val"])
    train_test = sorted(patients["train"] & patients["test_internal_temporal"])
    val_test = sorted(patients["val"] & patients["test_internal_temporal"])
    if train_test or val_test:
        raise RuntimeError("The fixed temporal test cohort overlaps development patients")

    print(frame.groupby("split").agg(rows=("row_uid", "size"), patients=("patient_id", "nunique"),
                                      lesions=("lesion_id", "nunique")).to_string())
    print(f"EXPECTED COMPARISON CONDITION: train/val patient overlap = {len(train_val)} patients")
    print("PASS: exact 179/78/67 row counts, no exact-path overlap, and fixed test cohort isolation confirmed.")
    print("WARNING: this image-level split is intentionally not patient-disjoint and must not be presented as leakage-safe.")


if __name__ == "__main__":
    main()
