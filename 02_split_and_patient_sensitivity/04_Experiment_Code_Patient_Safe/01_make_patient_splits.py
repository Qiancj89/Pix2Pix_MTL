from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import pandas as pd

from common import (
    LABEL_COLUMNS, assert_no_duplicate_paths_across_splits, assert_patient_disjoint,
    json_dump, load_config, require_columns, sha256_file,
)


def blank_string_mask(series: pd.Series) -> pd.Series:
    """Return a blank-value mask even when pandas inferred a numeric/empty dtype."""
    return series.astype("string").fillna("").str.strip().eq("")


def allocate_validation_patients(patient_table: pd.DataFrame, fraction: float, seed: int) -> set[str]:
    """Stratify at patient level by malignancy; never split rows from one patient."""
    rng = random.Random(seed)
    selected: set[str] = set()
    for _, stratum in patient_table.groupby("Malignant", dropna=False):
        ids = sorted(stratum["patient_id"].astype(str).tolist())
        rng.shuffle(ids)
        if len(ids) < 2:
            n_val = 0
        else:
            n_val = max(1, int(round(len(ids) * fraction)))
            n_val = min(n_val, len(ids) - 1)
        selected.update(ids[:n_val])

    target = int(round(len(patient_table) * fraction))
    target = min(max(target, 1), max(len(patient_table) - 1, 1))
    all_ids = set(patient_table["patient_id"].astype(str))
    remaining = sorted(all_ids - selected)
    rng.shuffle(remaining)
    while len(selected) < target and remaining:
        selected.add(remaining.pop())
    while len(selected) > target:
        selected.remove(sorted(selected)[-1])
    return selected


def content_hash_audit(frame: pd.DataFrame, columns: list[str]) -> dict:
    records = []
    for column in columns:
        if column not in frame.columns:
            continue
        for row in frame.loc[frame[column].fillna("").ne(""), ["row_uid", "split", column]].itertuples(index=False):
            path = getattr(row, column)
            records.append({"row_uid": row.row_uid, "split": row.split, "kind": column,
                            "path": path, "sha256": sha256_file(path)})
    hashes = pd.DataFrame(records)
    if hashes.empty:
        return {"files_hashed": 0, "cross_split_duplicate_hashes": []}
    cross = hashes.groupby(["kind", "sha256"])["split"].nunique()
    bad_keys = set(cross[cross > 1].index)
    bad = hashes[hashes.apply(lambda r: (r["kind"], r["sha256"]) in bad_keys, axis=1)]
    return {"files_hashed": len(hashes), "cross_split_duplicate_hashes": bad.to_dict("records")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze patient-level train/validation/internal-test splits.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--mapping", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--allow-unconfirmed", action="store_true", help="Debug only; forbidden for final results.")
    parser.add_argument("--hash-images", action="store_true", help="Also detect byte-identical images across splits.")
    args = parser.parse_args()

    config = load_config(args.config)
    work_dir = Path(config["_work_dir"])
    mapping_path = Path(args.mapping).resolve() if args.mapping else work_dir / "patient_mapping_review.csv"
    output = Path(args.output).resolve() if args.output else work_dir / "frozen_manifest.csv"
    frame = pd.read_csv(mapping_path, dtype={"patient_id": str, "lesion_id": str, "image_id": str})
    require_columns(frame, ["row_uid", "cohort_role", "patient_id", "lesion_id", "image_id", "confirmed",
                            "include_in_analysis", "exclusion_reason",
                            "bmode_path", "real_ceus_path", *LABEL_COLUMNS], str(mapping_path))

    include = pd.to_numeric(frame["include_in_analysis"], errors="coerce").fillna(-1)
    if not include.isin([0, 1]).all():
        raise RuntimeError("include_in_analysis must be 0 or 1 for every row.")
    all_unconfirmed = pd.to_numeric(frame["confirmed"], errors="coerce").fillna(0).ne(1)
    if all_unconfirmed.any() and not args.allow_unconfirmed:
        raise RuntimeError(f"{int(all_unconfirmed.sum())} included or excluded rows are not clinically confirmed. Refusing to split.")
    excluded = frame[include.eq(0)].copy()
    missing_reason = blank_string_mask(excluded["exclusion_reason"])
    if missing_reason.any():
        raise RuntimeError("Every excluded row needs a non-empty exclusion_reason.")
    frame = frame[include.eq(1)].copy()
    if frame.empty:
        raise RuntimeError("No rows remain after exclusions.")
    missing_included_bmode = frame["bmode_path"].fillna("").map(lambda x: not x or not Path(x).is_file())
    if missing_included_bmode.any():
        raise RuntimeError(
            f"{int(missing_included_bmode.sum())} included rows lack B-mode files. Resolve them or confirm exclusion first."
        )
    empty_id = blank_string_mask(frame["patient_id"])
    empty_lesion = blank_string_mask(frame["lesion_id"])
    if (empty_id | empty_lesion).any():
        raise RuntimeError("patient_id and lesion_id must be populated for every row before splitting.")

    development = frame[frame["cohort_role"].eq("development")].copy()
    test = frame[frame["cohort_role"].eq("test")].copy()
    if development.empty or test.empty:
        raise RuntimeError("Both development and fixed temporal internal-test cohorts are required.")
    overlap = sorted(set(development["patient_id"]) & set(test["patient_id"]))
    if overlap:
        raise RuntimeError(f"Development/test patient overlap detected: {overlap[:20]}")

    patient_table = development.groupby("patient_id", as_index=False).agg(
        Malignant=("Malignant", "max"), n_lesions=("lesion_id", "nunique"), n_images=("image_id", "nunique")
    )
    val_ids = allocate_validation_patients(
        patient_table, float(config.get("validation_fraction", 0.2)), int(config.get("seed", 20260828))
    )
    frame["split"] = ""
    frame.loc[frame["cohort_role"].eq("development"), "split"] = frame.loc[
        frame["cohort_role"].eq("development"), "patient_id"
    ].map(lambda x: "val" if x in val_ids else "train")
    frame.loc[frame["cohort_role"].eq("test"), "split"] = "test_internal_temporal"
    frame["split_seed"] = int(config.get("seed", 20260828))
    frame["split_version"] = "patient_safe_v1"

    assert_patient_disjoint(frame)
    assert_no_duplicate_paths_across_splits(frame, ["bmode_path", "real_ceus_path"])
    hash_report = content_hash_audit(frame, ["bmode_path", "real_ceus_path"]) if args.hash_images else None
    if hash_report and hash_report["cross_split_duplicate_hashes"]:
        raise RuntimeError("Byte-identical images exist across splits; see the audit report after resolving them.")

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["split", "patient_id", "lesion_id", "image_id"]).to_csv(output, index=False, encoding="utf-8-sig")
    excluded.to_csv(output.with_name("excluded_records.csv"), index=False, encoding="utf-8-sig")
    summary = frame.groupby("split").agg(
        patients=("patient_id", "nunique"), lesions=("lesion_id", "nunique"), images=("image_id", "nunique"),
        malignant_rows=("Malignant", "sum"), rows=("row_uid", "size"),
    ).reset_index()
    summary.to_csv(output.with_name("split_summary.csv"), index=False, encoding="utf-8-sig")
    audit = {
        "mapping": str(mapping_path), "manifest": str(output), "seed": int(config.get("seed", 20260828)),
        "validation_fraction": float(config.get("validation_fraction", 0.2)),
        "patient_overlap": [], "path_overlap": [], "all_rows_confirmed": bool((~all_unconfirmed).all()),
        "excluded_records": int(len(excluded)),
        "provisional_debug_run": bool(args.allow_unconfirmed), "summary": summary.to_dict("records"),
        "content_hash_audit": hash_report,
    }
    json_dump(audit, output.with_name("zero_overlap_audit.json"))
    print(summary.to_string(index=False))
    print(f"Frozen manifest: {output}")
    print("Zero patient overlap confirmed. Do not edit this manifest after training starts.")


if __name__ == "__main__":
    main()
