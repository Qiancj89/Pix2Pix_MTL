from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REVISION_DIR = HERE.parent
PIX2PIX_ROOT = HERE.parents[2]
DEFAULT_MAPPING = REVISION_DIR / "04_Experiment_Code_Patient_Safe" / "work" / "patient_mapping_review.csv"
DEFAULT_TRAIN_SOURCE = PIX2PIX_ROOT / "augmented_data_Pix2Pix_enhanced_ceus" / "train_clinical_data.csv"
DEFAULT_VAL_SOURCE = PIX2PIX_ROOT / "augmented_data_Pix2Pix_enhanced_ceus" / "val_clinical_data.csv"


def normalized_stem(value: str) -> str:
    return Path(str(value)).stem.casefold()


def sha256_file(path: str, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def path_overlap(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame:
        return []
    subset = frame.loc[frame[column].fillna("").ne(""), [column, "split"]].copy()
    subset[column] = subset[column].map(lambda x: str(Path(x).resolve()).casefold())
    counts = subset.groupby(column)["split"].nunique()
    return counts[counts > 1].index.astype(str).tolist()


def content_hash_overlap(frame: pd.DataFrame, columns: list[str]) -> tuple[int, list[dict]]:
    records = []
    for column in columns:
        if column not in frame:
            continue
        for row in frame.loc[frame[column].fillna("").ne(""), ["row_uid", "split", column]].itertuples(index=False):
            path = getattr(row, column)
            records.append({
                "row_uid": str(row.row_uid), "split": str(row.split), "kind": column,
                "path": str(path), "sha256": sha256_file(path),
            })
    hashes = pd.DataFrame(records)
    if hashes.empty:
        return 0, []
    counts = hashes.groupby(["kind", "sha256"])["split"].nunique()
    bad_keys = set(counts[counts > 1].index)
    bad = hashes[hashes.apply(lambda r: (r["kind"], r["sha256"]) in bad_keys, axis=1)]
    return len(hashes), bad.to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover the exact 179/78 image-level development split described in the submitted manuscript."
    )
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    parser.add_argument("--original-train-csv", default=str(DEFAULT_TRAIN_SOURCE))
    parser.add_argument("--original-val-csv", default=str(DEFAULT_VAL_SOURCE))
    parser.add_argument("--output", default=str(HERE / "work" / "original_split_manifest.csv"))
    parser.add_argument("--hash-images", action="store_true")
    args = parser.parse_args()

    mapping_path = Path(args.mapping).resolve()
    train_source_path = Path(args.original_train_csv).resolve()
    val_source_path = Path(args.original_val_csv).resolve()
    output = Path(args.output).resolve()

    mapping = pd.read_csv(mapping_path, dtype={"patient_id": str, "lesion_id": str, "image_id": str})
    required = {
        "row_uid", "cohort_role", "patient_id", "lesion_id", "image_id", "confirmed",
        "include_in_analysis", "bmode_path", "real_ceus_path", "Malignant",
    }
    missing = sorted(required - set(mapping.columns))
    if missing:
        raise ValueError(f"Mapping is missing columns: {missing}")
    confirmed = pd.to_numeric(mapping["confirmed"], errors="coerce").fillna(0).eq(1)
    included = pd.to_numeric(mapping["include_in_analysis"], errors="coerce").fillna(0).eq(1)
    frame = mapping[confirmed & included].copy()
    if len(frame) != 324:
        raise RuntimeError(f"Expected 324 confirmed included rows, found {len(frame)}")

    source_train = pd.read_csv(train_source_path)
    source_val = pd.read_csv(val_source_path)
    if "English_Name" not in source_train or "English_Name" not in source_val:
        raise ValueError("Original split CSV files must contain English_Name")
    original_rows = source_train[source_train["English_Name"].astype(str).str.endswith("_original")].copy()
    train_members = set(original_rows["English_Name"].astype(str).str.replace(r"_original$", "", regex=True).map(normalized_stem))
    val_members = set(source_val["English_Name"].astype(str).map(normalized_stem))
    if len(train_members) != 179 or len(val_members) != 78:
        raise RuntimeError(f"Recovered membership must be 179/78, found {len(train_members)}/{len(val_members)}")
    if train_members & val_members:
        raise RuntimeError("The persisted original train and validation memberships overlap by image name")

    frame["source_stem"] = frame["bmode_path"].map(normalized_stem)
    development = frame["cohort_role"].eq("development")
    test = frame["cohort_role"].eq("test")
    if int(development.sum()) != 257 or int(test.sum()) != 67:
        raise RuntimeError(f"Expected 257 development and 67 test rows, found {development.sum()} and {test.sum()}")
    dev_stems = set(frame.loc[development, "source_stem"])
    if train_members | val_members != dev_stems:
        missing_from_mapping = sorted((train_members | val_members) - dev_stems)
        unassigned_mapping = sorted(dev_stems - train_members - val_members)
        raise RuntimeError(
            f"Original membership does not match the verified mapping. Missing={missing_from_mapping[:10]}, "
            f"unassigned={unassigned_mapping[:10]}"
        )

    frame["split"] = ""
    frame.loc[development & frame["source_stem"].isin(train_members), "split"] = "train"
    frame.loc[development & frame["source_stem"].isin(val_members), "split"] = "val"
    frame.loc[test, "split"] = "test_internal_temporal"
    if frame["split"].eq("").any():
        raise RuntimeError("Some rows were not assigned to a split")
    frame["split_version"] = "submitted_manuscript_exact_image_membership_v1"
    frame["split_seed"] = "not_reported_membership_recovered_from_persisted_csv"
    frame["split_policy"] = "image_level_179_78_plus_fixed_67_test"
    frame["comparison_only_allow_patient_overlap"] = 1

    counts = frame["split"].value_counts().to_dict()
    expected_counts = {"train": 179, "val": 78, "test_internal_temporal": 67}
    if counts != expected_counts:
        raise RuntimeError(f"Unexpected split counts: {counts}")

    train_patients = set(frame.loc[frame["split"].eq("train"), "patient_id"].astype(str))
    val_patients = set(frame.loc[frame["split"].eq("val"), "patient_id"].astype(str))
    test_patients = set(frame.loc[frame["split"].eq("test_internal_temporal"), "patient_id"].astype(str))
    train_val_overlap = sorted(train_patients & val_patients)
    development_test_overlap = sorted((train_patients | val_patients) & test_patients)
    if development_test_overlap:
        raise RuntimeError(f"Development/test patient overlap is not allowed: {development_test_overlap[:20]}")

    path_overlaps = {
        column: path_overlap(frame, column)
        for column in ("bmode_path", "real_ceus_path")
    }
    if any(path_overlaps.values()):
        raise RuntimeError(f"Exact file paths occur in multiple splits: {path_overlaps}")

    files_hashed, hash_overlaps = (0, [])
    if args.hash_images:
        files_hashed, hash_overlaps = content_hash_overlap(frame, ["bmode_path", "real_ceus_path"])
        if hash_overlaps:
            raise RuntimeError("Byte-identical image content occurs across splits; inspect the audit report")

    order = pd.Categorical(frame["split"], ["train", "val", "test_internal_temporal"], ordered=True)
    frame = frame.assign(_split_order=order).sort_values(["_split_order", "source_stem"]).drop(columns="_split_order")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")

    summary = frame.groupby("split", sort=False).agg(
        rows=("row_uid", "size"), patients=("patient_id", "nunique"),
        lesions=("lesion_id", "nunique"), malignant_rows=("Malignant", lambda x: pd.to_numeric(x).sum()),
    ).reset_index()
    summary.to_csv(output.with_name("original_split_summary.csv"), index=False, encoding="utf-8-sig")
    audit = {
        "manifest": str(output),
        "mapping": str(mapping_path),
        "original_train_membership_source": str(train_source_path),
        "original_val_membership_source": str(val_source_path),
        "membership_recovered_not_resampled": True,
        "reported_random_seed_available": False,
        "expected_row_counts": expected_counts,
        "summary": summary.to_dict("records"),
        "train_val_patient_overlap_count": len(train_val_overlap),
        "train_val_patient_overlap_ids": train_val_overlap,
        "development_test_patient_overlap_count": 0,
        "exact_path_overlap": path_overlaps,
        "content_hash_audit": {"files_hashed": files_hashed, "cross_split_duplicate_hashes": hash_overlaps},
        "interpretation": (
            "This manifest reproduces the submitted manuscript's persisted 179/78 image-level membership. "
            "Train/validation patient overlap is intentional for comparison and is not leakage-safe."
        ),
    }
    output.with_name("original_split_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Train/validation patient overlap: {len(train_val_overlap)} patients")
    print(f"Manifest: {output}")


if __name__ == "__main__":
    main()
