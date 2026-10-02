from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    LABEL_COLUMNS, build_image_index, find_image, load_config, proposed_lesion_id,
    proposed_patient_id, repo_path, require_columns, stable_uid,
)


def filter_named_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Remove spreadsheet separators/statistics that are not image records."""
    names = frame["English_Name"].astype("string")
    valid = names.notna() & names.str.strip().ne("")
    skipped = frame.loc[~valid]
    payload_columns = [column for column in frame.columns if column != "English_Name"]
    skipped_with_values = int(skipped[payload_columns].notna().any(axis=1).sum())

    cleaned = frame.loc[valid].copy()
    cleaned["English_Name"] = names.loc[valid].str.strip()
    return cleaned, int((~valid).sum()), skipped_with_values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a clinical-review patient/lesion mapping template. No split is made here."
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--accept-proposals", action="store_true",
        help="Debug only: copy name-based proposals into patient_id. Final analysis must use clinically confirmed IDs.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = Path(config["_repo_root"])
    output = Path(args.output) if args.output else Path(config["_work_dir"]) / "patient_mapping_review.csv"
    output = output.resolve()
    rows: list[dict] = []
    cohort_reports: list[dict] = []

    for cohort in config["cohorts"]:
        csv_path = repo_path(config, cohort["csv"])
        frame = pd.read_csv(csv_path)
        require_columns(frame, ["English_Name", *LABEL_COLUMNS], str(csv_path))
        frame, skipped_rows, skipped_with_values = filter_named_rows(frame)
        bmode_index = build_image_index([repo_path(config, x) for x in cohort.get("bmode_dirs", [])])
        ceus_index = build_image_index([repo_path(config, x) for x in cohort.get("real_ceus_dirs", [])])
        mask_dirs = cohort.get("mask_dirs", [])
        mask_index = build_image_index([repo_path(config, x) for x in mask_dirs])

        cohort_start = len(rows)

        for row_index, source in frame.iterrows():
            name = source["English_Name"]
            proposal = proposed_patient_id(name)
            bmode_path, bmode_issue = find_image(name, bmode_index)
            real_ceus_path, ceus_issue = find_image(name, ceus_index)
            mask_path, mask_issue = find_image(name, mask_index) if mask_dirs else ("", "NOT_CONFIGURED")
            item = {
                "row_uid": stable_uid(cohort["name"], row_index, name),
                "cohort": cohort["name"],
                "cohort_role": cohort["role"],
                "source_row": int(row_index),
                "original_name": name,
                "proposed_patient_id": proposal,
                "patient_id": proposal if args.accept_proposals else "",
                "lesion_id": proposed_lesion_id(name, proposal) if args.accept_proposals else "",
                "image_id": stable_uid(cohort["name"], name, length=20),
                "confirmed": 0,
                "bmode_path": bmode_path,
                "real_ceus_path": real_ceus_path,
                "mask_path": mask_path,
                "bmode_issue": bmode_issue,
                "real_ceus_issue": ceus_issue,
                "mask_issue": mask_issue,
                "include_in_analysis": 1 if bmode_path else 0,
                "exclusion_reason": "" if bmode_path else "UNRESOLVED_MISSING_BMODE_VERIFY_OR_CONFIRM_EXCLUSION",
                "clinical_notes": "",
            }
            for label in LABEL_COLUMNS:
                item[label] = source[label]
            rows.append(item)

        cohort_rows = rows[cohort_start:]
        cohort_reports.append({
            "name": cohort["name"],
            "rows": len(cohort_rows),
            "skipped_rows": skipped_rows,
            "skipped_with_values": skipped_with_values,
            "missing_bmode": sum(not row["bmode_path"] for row in cohort_rows),
            "missing_real_ceus": sum(not row["real_ceus_path"] for row in cohort_rows),
            "missing_mask": sum(not row["mask_path"] for row in cohort_rows) if mask_dirs else None,
        })

    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(result)} image rows to {output}")
    for report in cohort_reports:
        mask_text = "not configured" if report["missing_mask"] is None else str(report["missing_mask"])
        print(
            f"[{report['name']}] rows={report['rows']}, "
            f"skipped_blank_name_rows={report['skipped_rows']} "
            f"(with_summary_values={report['skipped_with_values']}), "
            f"missing_bmode={report['missing_bmode']}, "
            f"missing_real_ceus={report['missing_real_ceus']}, missing_mask={mask_text}"
        )
    print("NEXT: a clinical/data custodian must fill patient_id and lesion_id, set confirmed=1, and resolve issues.")
    print("Do not use --accept-proposals for manuscript results.")


if __name__ == "__main__":
    main()
