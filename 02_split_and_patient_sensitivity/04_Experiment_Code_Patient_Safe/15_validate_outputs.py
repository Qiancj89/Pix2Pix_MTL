from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import assert_no_duplicate_paths_across_splits, assert_patient_disjoint, json_dump


def main() -> None:
    parser = argparse.ArgumentParser(description="Final consistency checks before copying numbers into the manuscript.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", nargs="*", default=[])
    parser.add_argument("--classification-metrics", default=None)
    parser.add_argument("--synthesis-metrics", default=None)
    parser.add_argument("--reader-results", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-patient-overlap", action="store_true",
                        help="Comparison-only: validate an explicitly audited image-level split.")
    args = parser.parse_args()
    report = {"checks": [], "warnings": []}
    frame = pd.read_csv(args.manifest)
    overlapping = assert_patient_disjoint(frame, allow_overlap=args.allow_patient_overlap)
    assert_no_duplicate_paths_across_splits(frame, ["bmode_path", "real_ceus_path", "synthetic_ceus_path", "diffusion_ceus_path"])
    if overlapping:
        report["warnings"].append(f"intentional_train_val_patient_overlap:{len(overlapping)}")
        report["checks"].append("exact_path_disjointness_passed_patient_overlap_documented")
    else:
        report["checks"].append("patient_and_exact_path_disjointness_passed")
    for path in args.predictions:
        predictions = pd.read_csv(path)
        if predictions.row_uid.duplicated().any():
            raise RuntimeError(f"Duplicate prediction row_uid in {path}")
        unknown = set(predictions.row_uid.astype(str)) - set(frame.row_uid.astype(str))
        if unknown:
            raise RuntimeError(f"Predictions contain unknown row_uid values in {path}")
        report["checks"].append(f"prediction_integrity:{Path(path).name}")
    for label, path in (("classification_metrics", args.classification_metrics),
                        ("synthesis_metrics", args.synthesis_metrics), ("reader_results", args.reader_results)):
        if path:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
            report["checks"].append(f"present:{label}")
        else:
            report["warnings"].append(f"not_supplied:{label}")
    test = frame[frame.split.eq("test_internal_temporal")]
    if "real_ceus_path" not in test or test.real_ceus_path.fillna("").eq("").any():
        report["warnings"].append("real_ceus_missing_for_some_or_all_internal_test_rows")
    report["status"] = "PASS_WITH_WARNINGS" if report["warnings"] else "PASS"
    json_dump(report, args.output); print(report)


if __name__ == "__main__":
    main()
