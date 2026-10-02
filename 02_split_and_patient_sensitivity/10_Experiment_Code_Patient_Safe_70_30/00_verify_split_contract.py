from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "row_uid", "patient_id", "lesion_id", "image_id", "cohort_role", "split",
    "bmode_path", "real_ceus_path", "Malignant",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_summary(frame: pd.DataFrame, split: str) -> dict:
    subset = frame.loc[frame["split"].eq(split)].copy()
    patient_labels = subset.groupby("patient_id")["Malignant"].max()
    return {
        "split": split,
        "patients": int(subset["patient_id"].nunique()),
        "benign_patients": int(patient_labels.eq(0).sum()),
        "malignant_patients": int(patient_labels.eq(1).sum()),
        "rows": int(len(subset)),
        "benign_rows": int(subset["Malignant"].eq(0).sum()),
        "malignant_rows": int(subset["Malignant"].eq(1).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen patient-level 70:30 split contract.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    fraction = float(config.get("validation_fraction", -1))
    if abs(fraction - 0.30) > 1e-12:
        raise RuntimeError(f"Expected validation_fraction=0.30, found {fraction!r}.")

    frame = pd.read_csv(manifest, dtype={"patient_id": str, "row_uid": str})
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise RuntimeError(f"Manifest is missing required columns: {missing}")
    if frame.empty:
        raise RuntimeError("Manifest is empty.")
    if frame["patient_id"].fillna("").str.strip().eq("").any():
        raise RuntimeError("Blank patient_id values are forbidden.")
    if frame["row_uid"].duplicated().any():
        raise RuntimeError("row_uid must be unique.")

    required_splits = {"train", "val", "test_internal_temporal"}
    actual_splits = set(frame["split"].astype(str))
    if actual_splits != required_splits:
        raise RuntimeError(f"Expected splits {sorted(required_splits)}, found {sorted(actual_splits)}")

    patient_sets = {
        split: set(frame.loc[frame["split"].eq(split), "patient_id"].astype(str))
        for split in sorted(required_splits)
    }
    overlaps = {
        "train_val": sorted(patient_sets["train"] & patient_sets["val"]),
        "train_test": sorted(patient_sets["train"] & patient_sets["test_internal_temporal"]),
        "val_test": sorted(patient_sets["val"] & patient_sets["test_internal_temporal"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Patient overlap detected: {overlaps}")

    development_patients = patient_sets["train"] | patient_sets["val"]
    expected_val = int(round(len(development_patients) * fraction))
    if len(patient_sets["val"]) != expected_val:
        raise RuntimeError(
            f"Validation contains {len(patient_sets['val'])} patients; "
            f"expected {expected_val} of {len(development_patients)} for a 70:30 patient split."
        )

    missing_files = []
    for column in ("bmode_path", "real_ceus_path"):
        for value in frame[column].fillna("").astype(str):
            if value and not Path(value).is_file():
                missing_files.append({"column": column, "path": value})
                if len(missing_files) >= 20:
                    break
    if missing_files:
        raise RuntimeError(f"Manifest contains missing paired-image files (first 20): {missing_files}")

    report = {
        "contract": "patient_level_stratified_70_30_v1",
        "validation_fraction": fraction,
        "seed": int(config["seed"]),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "patient_overlap": overlaps,
        "development_unique_patients": len(development_patients),
        "expected_validation_patients": expected_val,
        "all_unique_patients": len(set().union(*patient_sets.values())),
        "summary": [split_summary(frame, name) for name in ("train", "val", "test_internal_temporal")],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Patient-level 70:30 split contract verified.")
    for row in report["summary"]:
        print(
            f"{row['split']}: patients={row['patients']} "
            f"(benign={row['benign_patients']}, malignant={row['malignant_patients']}), "
            f"rows={row['rows']}"
        )
    print(f"Audit: {output}")


if __name__ == "__main__":
    main()
