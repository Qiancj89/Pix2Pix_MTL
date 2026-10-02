from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc", "pr_auc", "brier"]


def split_stats(frame: pd.DataFrame, design: str) -> pd.DataFrame:
    return frame.groupby("split", sort=False).agg(
        rows=("row_uid", "size"), patients=("patient_id", "nunique"),
        lesions=("lesion_id", "nunique"), malignant_rows=("Malignant", lambda x: pd.to_numeric(x).sum()),
    ).reset_index().assign(design=design)[["design", "split", "rows", "patients", "lesions", "malignant_rows"]]


def patient_metrics(path: str, design: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame[
        frame["unit"].eq("patient_max_risk")
        & frame["task"].eq("Malignant")
        & frame["analysis_role"].eq("primary")
        & frame["threshold_strategy"].eq("fixed_0.5")
    ].copy()
    columns = ["model", "n", "reference_positive_count", "predicted_positive_count", *METRICS]
    selected = selected[columns]
    selected.insert(0, "design", design)
    return selected


def synthesis_metrics(path: str, design: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["region"].eq("whole")].copy()
    frame.insert(0, "design", design)
    return frame


def paired_auc(path: str, design: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["task"].eq("Malignant")].copy()
    frame.insert(0, "design", design)
    return frame


def fmt(value, digits=3):
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare patient-level and submitted-manuscript image-level split results.")
    parser.add_argument("--patient-manifest", required=True)
    parser.add_argument("--original-manifest", required=True)
    parser.add_argument("--patient-classification", required=True)
    parser.add_argument("--original-classification", required=True)
    parser.add_argument("--patient-synthesis", required=True)
    parser.add_argument("--original-synthesis", required=True)
    parser.add_argument("--patient-paired-auc", required=True)
    parser.add_argument("--original-paired-auc", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    patient_manifest = pd.read_csv(args.patient_manifest, dtype={"patient_id": str})
    original_manifest = pd.read_csv(args.original_manifest, dtype={"patient_id": str})
    split_table = pd.concat([
        split_stats(patient_manifest, "patient_level"),
        split_stats(original_manifest, "submitted_image_level"),
    ], ignore_index=True)
    split_table.to_csv(output / "split_comparison.csv", index=False, encoding="utf-8-sig")

    patient_overlap = len(
        set(patient_manifest.loc[patient_manifest.split.eq("train"), "patient_id"])
        & set(patient_manifest.loc[patient_manifest.split.eq("val"), "patient_id"])
    )
    original_overlap = len(
        set(original_manifest.loc[original_manifest.split.eq("train"), "patient_id"])
        & set(original_manifest.loc[original_manifest.split.eq("val"), "patient_id"])
    )
    overlap_table = pd.DataFrame([
        {"design": "patient_level", "train_val_overlapping_patients": patient_overlap},
        {"design": "submitted_image_level", "train_val_overlapping_patients": original_overlap},
    ])
    overlap_table.to_csv(output / "patient_overlap_comparison.csv", index=False, encoding="utf-8-sig")

    classification_long = pd.concat([
        patient_metrics(args.patient_classification, "patient_level"),
        patient_metrics(args.original_classification, "submitted_image_level"),
    ], ignore_index=True)
    classification_long.to_csv(output / "malignancy_patient_level_metrics_long.csv", index=False, encoding="utf-8-sig")
    patient_rows = classification_long[classification_long.design.eq("patient_level")].set_index("model")
    original_rows = classification_long[classification_long.design.eq("submitted_image_level")].set_index("model")
    common_models = sorted(set(patient_rows.index) & set(original_rows.index))
    comparison_records = []
    for model in common_models:
        record = {"model": model}
        for metric in METRICS:
            patient_value = patient_rows.loc[model, metric]
            original_value = original_rows.loc[model, metric]
            record[f"patient_level_{metric}"] = patient_value
            record[f"submitted_image_level_{metric}"] = original_value
            record[f"delta_original_minus_patient_{metric}"] = original_value - patient_value
        record["patient_level_predicted_positive_count"] = patient_rows.loc[model, "predicted_positive_count"]
        record["submitted_image_level_predicted_positive_count"] = original_rows.loc[model, "predicted_positive_count"]
        comparison_records.append(record)
    comparison = pd.DataFrame(comparison_records)
    comparison.to_csv(output / "malignancy_patient_level_comparison.csv", index=False, encoding="utf-8-sig")

    synthesis_long = pd.concat([
        synthesis_metrics(args.patient_synthesis, "patient_level"),
        synthesis_metrics(args.original_synthesis, "submitted_image_level"),
    ], ignore_index=True)
    synthesis_long.to_csv(output / "synthesis_validation_comparison.csv", index=False, encoding="utf-8-sig")
    paired_long = pd.concat([
        paired_auc(args.patient_paired_auc, "patient_level"),
        paired_auc(args.original_paired_auc, "submitted_image_level"),
    ], ignore_index=True)
    paired_long.to_csv(output / "paired_auc_comparison.csv", index=False, encoding="utf-8-sig")

    plot_models = [m for m in ["bmode", "synthetic", "bmode_synthetic", "real_ceus", "bmode_real_ceus"] if m in common_models]
    if plot_models:
        x = np.arange(len(plot_models))
        width = 0.36
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        for axis, metric, label in zip(axes, ["roc_auc", "pr_auc"], ["Patient-level ROC-AUC", "Patient-level PR-AUC"]):
            pvals = [float(patient_rows.loc[m, metric]) for m in plot_models]
            ovals = [float(original_rows.loc[m, metric]) for m in plot_models]
            axis.bar(x - width / 2, pvals, width, label="Patient-level split", color="#1f4e79")
            axis.bar(x + width / 2, ovals, width, label="Submitted image-level split", color="#c55a11")
            axis.set_xticks(x, [m.replace("_", "\n") for m in plot_models])
            axis.set_ylim(0, 1)
            axis.set_ylabel(label)
            axis.grid(axis="y", alpha=0.25)
            axis.legend(fontsize=8)
        fig.suptitle("Same temporal test cohort; models retrained under two development-split policies")
        fig.savefig(output / "patient_vs_original_split_auc_comparison.png", dpi=240)
        plt.close(fig)

    lines = [
        "# Patient-level versus submitted image-level split comparison",
        "",
        "## Interpretation boundary",
        "",
        f"The submitted-manuscript membership contains **{original_overlap} patients shared between training and validation**. "
        "The patient-level membership contains no such overlap. Both model families are evaluated on the same fixed 62-patient/67-row temporal internal test cohort. "
        "The image-level results are a sensitivity analysis and must not be described as leakage-safe evidence.",
        "",
        "The two development designs also use different train/validation sizes (submitted 179/78 rows versus patient-level 207/50 rows). "
        "Therefore, observed differences reflect the combined effect of allocation unit and development-set size, not patient overlap alone.",
        "",
        "## Primary patient-level malignancy comparison (fixed threshold 0.5)",
        "",
        "| Model | Patient ROC-AUC | Original ROC-AUC | Delta | Patient PR-AUC | Original PR-AUC | Delta | Patient sensitivity | Original sensitivity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in plot_models:
        p = patient_rows.loc[model]
        o = original_rows.loc[model]
        lines.append(
            f"| {model} | {fmt(p.roc_auc)} | {fmt(o.roc_auc)} | {fmt(o.roc_auc-p.roc_auc)} | "
            f"{fmt(p.pr_auc)} | {fmt(o.pr_auc)} | {fmt(o.pr_auc-p.pr_auc)} | "
            f"{fmt(p.sensitivity)} | {fmt(o.sensitivity)} |"
        )
    lines.extend([
        "",
        "## Files",
        "",
        "- `split_comparison.csv`: split sizes and composition.",
        "- `patient_overlap_comparison.csv`: explicit train/validation patient-overlap count.",
        "- `malignancy_patient_level_comparison.csv`: side-by-side primary test metrics and deltas.",
        "- `synthesis_validation_comparison.csv`: whole-image validation synthesis metrics; validation cohorts differ and are not paired.",
        "- `paired_auc_comparison.csv`: paired test-row AUC comparisons from each retrained model family.",
        "- `patient_vs_original_split_auc_comparison.png`: visual comparison of ROC-AUC and PR-AUC.",
    ])
    (output / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote comparison outputs to {output}")
    print(f"Train/validation patient overlap: patient-level={patient_overlap}, submitted image-level={original_overlap}")


if __name__ == "__main__":
    main()
