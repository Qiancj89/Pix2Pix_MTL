from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import LABEL_COLUMNS, assert_no_duplicate_paths_across_splits, assert_patient_disjoint, require_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-fast audit run before every experiment.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--require-real-ceus", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest, dtype={"patient_id": str, "lesion_id": str, "image_id": str})
    require_columns(frame, ["row_uid", "patient_id", "lesion_id", "image_id", "split", "bmode_path", *LABEL_COLUMNS], args.manifest)
    assert_patient_disjoint(frame)
    assert_no_duplicate_paths_across_splits(frame, ["bmode_path", "real_ceus_path", "synthetic_ceus_path"])
    if frame[["patient_id", "lesion_id", "image_id", "split"]].isna().any().any():
        raise RuntimeError("Null identifiers or split values found.")
    missing_bmode = frame["bmode_path"].fillna("").map(lambda x: not x or not Path(x).is_file())
    if missing_bmode.any():
        raise RuntimeError(f"Missing B-mode files: {int(missing_bmode.sum())}")
    if args.require_real_ceus:
        missing = frame["real_ceus_path"].fillna("").map(lambda x: not x or not Path(x).is_file())
        if missing.any():
            raise RuntimeError(f"Missing real CEUS files: {int(missing.sum())}")
    print(frame.groupby("split").agg(patients=("patient_id", "nunique"), lesions=("lesion_id", "nunique"),
                                     images=("image_id", "nunique")).to_string())
    print("PASS: patient IDs, exact paths, identifiers, labels, and required files passed the audit.")


if __name__ == "__main__":
    main()

