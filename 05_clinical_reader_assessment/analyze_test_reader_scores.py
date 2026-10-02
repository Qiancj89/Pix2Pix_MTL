from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook


KEY_FILE = Path(os.environ["BLINDING_KEY_CSV"]).expanduser().resolve()
MANIFEST_FILE = Path(os.environ["FROZEN_MANIFEST_CSV"]).expanduser().resolve()
EVIDENCE = Path(os.environ.get("READER_ANALYSIS_OUTPUT_DIR", "reader_analysis_outputs")).expanduser().resolve()
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260921
READERS = {
    "Reader 1": Path(os.environ["READER1_XLSX"]).expanduser().resolve(),
    "Reader 2": Path(os.environ["READER2_XLSX"]).expanduser().resolve(),
}


def read_scores(path: Path) -> pd.DataFrame:
    workbook = load_workbook(path, data_only=True, read_only=True)
    sheet = workbook.active
    rows = list(sheet.iter_rows(values_only=True))
    headers = rows[0]
    data = pd.DataFrame(rows[1:], columns=headers)
    data = data.loc[data["病例文件名前缀"].notna()].copy()
    rename = {
        "配对序号": "pair_index_sheet",
        "病例文件名前缀": "case_prefix",
        "图像1文件名": "file_1_sheet",
        "图像2文件名": "file_2_sheet",
        "判断哪张为生成图（填写1或2）": "predicted_generated_position",
        "判断信心（1–5分）": "confidence",
        "图像1边界清晰度（1–5分）": "boundary_1",
        "图像2边界清晰度（1–5分）": "boundary_2",
        "图像1灌注真实性（1–5分）": "perfusion_1",
        "图像2灌注真实性（1–5分）": "perfusion_2",
        "有助于良恶性判别（1或0）": "helpful",
        "备注": "comment",
    }
    data = data.rename(columns=rename)
    required = list(rename.values())
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")
    return data[required]


def cluster_bootstrap_interval(
    data: pd.DataFrame, value_fn, rng: np.random.Generator
) -> tuple[float, float]:
    patients = data["patient_id"].drop_duplicates().to_numpy()
    values = np.empty(BOOTSTRAP_REPS, dtype=float)
    grouped = {patient: group for patient, group in data.groupby("patient_id")}
    for i in range(BOOTSTRAP_REPS):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        resampled = pd.concat([grouped[patient] for patient in sampled], ignore_index=True)
        values[i] = float(value_fn(resampled))
    return tuple(np.quantile(values, [0.025, 0.975]))


def summarize_reader(data: pd.DataFrame, reader: str, seed_offset: int) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)

    def estimate(column: str):
        point = float(data[column].mean())
        low, high = cluster_bootstrap_interval(data, lambda frame: frame[column].mean(), rng)
        return point, low, high

    correct_rate, correct_low, correct_high = estimate("correct_identification")
    confidence, confidence_low, confidence_high = estimate("confidence")
    helpful_rate, helpful_low, helpful_high = estimate("helpful")

    boundary_real = float(data["boundary_real"].mean())
    boundary_synthetic = float(data["boundary_synthetic"].mean())
    boundary_difference = float(data["boundary_real_minus_synthetic"].mean())
    boundary_low, boundary_high = cluster_bootstrap_interval(
        data, lambda frame: frame["boundary_real_minus_synthetic"].mean(), rng
    )

    perfusion_real = float(data["perfusion_real"].mean())
    perfusion_synthetic = float(data["perfusion_synthetic"].mean())
    perfusion_difference = float(data["perfusion_real_minus_synthetic"].mean())
    perfusion_low, perfusion_high = cluster_bootstrap_interval(
        data, lambda frame: frame["perfusion_real_minus_synthetic"].mean(), rng
    )

    return {
        "reader": reader,
        "image_pairs": int(len(data)),
        "patients": int(data["patient_id"].nunique()),
        "correct_identification_n": int(data["correct_identification"].sum()),
        "correct_identification_rate": correct_rate,
        "correct_identification_ci_low": correct_low,
        "correct_identification_ci_high": correct_high,
        "confidence_mean": confidence,
        "confidence_ci_low": confidence_low,
        "confidence_ci_high": confidence_high,
        "boundary_real_mean": boundary_real,
        "boundary_synthetic_mean": boundary_synthetic,
        "boundary_real_minus_synthetic": boundary_difference,
        "boundary_difference_ci_low": boundary_low,
        "boundary_difference_ci_high": boundary_high,
        "perfusion_real_mean": perfusion_real,
        "perfusion_synthetic_mean": perfusion_synthetic,
        "perfusion_real_minus_synthetic": perfusion_difference,
        "perfusion_difference_ci_low": perfusion_low,
        "perfusion_difference_ci_high": perfusion_high,
        "helpful_n": int(data["helpful"].sum()),
        "helpful_rate": helpful_rate,
        "helpful_ci_low": helpful_low,
        "helpful_ci_high": helpful_high,
        "commented_pairs": int(data["comment"].fillna("").astype(str).str.strip().ne("").sum()),
    }


def validate_scores(data: pd.DataFrame, reader: str):
    checks = {
        "predicted_generated_position": {1, 2},
        "confidence": {1, 2, 3, 4, 5},
        "boundary_1": {1, 2, 3, 4, 5},
        "boundary_2": {1, 2, 3, 4, 5},
        "perfusion_1": {1, 2, 3, 4, 5},
        "perfusion_2": {1, 2, 3, 4, 5},
        "helpful": {0, 1},
    }
    for column, valid in checks.items():
        if data[column].isna().any():
            raise ValueError(f"{reader}: missing values in {column}")
        invalid = set(data[column].astype(int).unique()) - valid
        if invalid:
            raise ValueError(f"{reader}: invalid values in {column}: {sorted(invalid)}")
    if data["case_prefix"].duplicated().any():
        raise ValueError(f"{reader}: duplicate case prefixes")


def binary_kappa(left: pd.Series, right: pd.Series) -> float:
    left = left.astype(int).to_numpy()
    right = right.astype(int).to_numpy()
    observed = float(np.mean(left == right))
    expected = 0.0
    for value in (1, 2):
        expected += float(np.mean(left == value) * np.mean(right == value))
    if expected == 1.0:
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    key = pd.read_csv(KEY_FILE)
    key["case_prefix"] = key["case_prefix"].astype(str)

    manifest = pd.read_csv(MANIFEST_FILE)
    test = manifest.loc[manifest["split"].eq("test_internal_temporal")].copy()
    test["case_prefix"] = test["original_name"].astype(str)
    patient_map = test[["case_prefix", "patient_id"]].drop_duplicates()
    if patient_map["case_prefix"].duplicated().any():
        raise ValueError("Manifest has non-unique test case prefixes")

    joined_by_reader = {}
    summary_rows = []
    audit = {
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "key_rows": int(len(key)),
        "manifest_test_rows": int(len(test)),
        "manifest_test_patients": int(test["patient_id"].nunique()),
        "reader_files": {},
    }

    for offset, (reader, path) in enumerate(READERS.items()):
        scores = read_scores(path)
        validate_scores(scores, reader)
        joined = scores.merge(
            key[["case_prefix", "pair_index", "file_1", "file_2", "generated_position"]],
            on="case_prefix",
            how="left",
            validate="one_to_one",
        ).merge(patient_map, on="case_prefix", how="left", validate="many_to_one")

        if joined[["generated_position", "patient_id"]].isna().any().any():
            raise ValueError(f"{reader}: unmatched key or patient mapping")
        file_1_mismatches = int((joined["file_1_sheet"] != joined["file_1"]).sum())
        file_2_mismatches = int((joined["file_2_sheet"] != joined["file_2"]).sum())

        joined["correct_identification"] = (
            joined["predicted_generated_position"].astype(int)
            == joined["generated_position"].astype(int)
        ).astype(int)
        generated_is_1 = joined["generated_position"].astype(int).eq(1)
        joined["boundary_synthetic"] = np.where(
            generated_is_1, joined["boundary_1"], joined["boundary_2"]
        ).astype(float)
        joined["boundary_real"] = np.where(
            generated_is_1, joined["boundary_2"], joined["boundary_1"]
        ).astype(float)
        joined["perfusion_synthetic"] = np.where(
            generated_is_1, joined["perfusion_1"], joined["perfusion_2"]
        ).astype(float)
        joined["perfusion_real"] = np.where(
            generated_is_1, joined["perfusion_2"], joined["perfusion_1"]
        ).astype(float)
        joined["boundary_real_minus_synthetic"] = (
            joined["boundary_real"] - joined["boundary_synthetic"]
        )
        joined["perfusion_real_minus_synthetic"] = (
            joined["perfusion_real"] - joined["perfusion_synthetic"]
        )
        joined["confidence"] = joined["confidence"].astype(float)
        joined["helpful"] = joined["helpful"].astype(float)

        joined_by_reader[reader] = joined
        summary_rows.append(summarize_reader(joined, reader, offset * 100))

        comments = joined["comment"].fillna("").astype(str).str.strip()
        audit["reader_files"][reader] = {
            "source_file": path.name,
            "rows": int(len(joined)),
            "patients": int(joined["patient_id"].nunique()),
            "all_key_rows_matched": True,
            "file_1_label_mismatches": file_1_mismatches,
            "file_2_label_mismatches": file_2_mismatches,
            "filename_mismatch_note": (
                "One Reader 1 row contains a trailing '1' typo in the displayed file-1 name; "
                "case prefix, file-2 name, pair identity, and score-position columns remain unambiguous."
                if file_1_mismatches == 1 and file_2_mismatches == 0
                else ""
            ),
            "commented_pairs": int(comments.ne("").sum()),
            "comment_keyword_counts": {
                "blur_distortion": int(comments.str.contains("模糊|变形|失真", regex=True).sum()),
                "difficult_lesion_recognition": int(comments.str.contains("难以识别", regex=False).sum()),
                "artifact": int(comments.str.contains("伪影", regex=False).sum()),
                "composition_misrepresentation": int(comments.str.contains("实性|囊实性", regex=True).sum()),
                "perfusion_misrepresentation": int(comments.str.contains("灌注", regex=False).sum()),
                "malignancy_judgment_impacted": int(comments.str.contains("良恶性判断", regex=False).sum()),
            },
        }

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(EVIDENCE / "Table_R12_Test_Set_Reader_Assessment.csv", index=False)

    left = joined_by_reader["Reader 1"].set_index("case_prefix")
    right = joined_by_reader["Reader 2"].set_index("case_prefix")
    common = left.index.intersection(right.index)
    reader_agreement = float(
        (left.loc[common, "predicted_generated_position"].astype(int)
         == right.loc[common, "predicted_generated_position"].astype(int)).mean()
    )
    kappa = binary_kappa(
        left.loc[common, "predicted_generated_position"],
        right.loc[common, "predicted_generated_position"],
    )
    audit["between_reader_generated_position"] = {
        "pairs": int(len(common)),
        "raw_agreement": reader_agreement,
        "cohen_kappa": kappa,
    }
    audit["patient_cluster_structure"] = {
        "unique_patients": int(test["patient_id"].nunique()),
        "image_pairs": int(len(test)),
        "patients_with_multiple_rows": int((test.groupby("patient_id").size() > 1).sum()),
        "maximum_rows_per_patient": int(test.groupby("patient_id").size().max()),
    }

    with (EVIDENCE / "Test_Set_Reader_Assessment_Audit.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)

    print(summary.to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
