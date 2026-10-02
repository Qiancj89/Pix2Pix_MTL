from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def wilson(successes, total, z=1.959963984540054):
    if total == 0: return (float("nan"), float("nan"))
    p = successes / total; denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def fleiss_kappa_binary(ratings: pd.DataFrame):
    counts = ratings.dropna(subset=["guess_binary"]).groupby("image_code")["guess_binary"].value_counts().unstack(fill_value=0)
    for label in (0, 1):
        if label not in counts.columns: counts[label] = 0
    counts = counts[[0, 1]]; n = counts.sum(axis=1)
    if counts.empty:
        return float("nan")
    counts = counts[n.eq(n.mode().iloc[0])]; n_raters = int(counts.sum(axis=1).iloc[0])
    if n_raters < 2 or len(counts) < 2: return float("nan")
    agreement = ((counts.pow(2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))).mean()
    proportions = counts.sum(axis=0) / counts.to_numpy().sum(); expected = float((proportions ** 2).sum())
    return float((agreement - expected) / max(1 - expected, 1e-12))


def prepare(args):
    frame = pd.read_csv(args.manifest); frame = frame[frame.split.eq(args.split)].copy()
    frame = frame[frame.real_ceus_path.fillna("").ne("") & frame[args.generated_column].fillna("").ne("")]
    if frame.empty: raise RuntimeError("No real/generated pairs available for reader study.")
    rng = random.Random(args.seed)
    patient_rows = [group.sample(1, random_state=args.seed).iloc[0] for _, group in frame.groupby("patient_id")]
    rng.shuffle(patient_rows); patient_rows = patient_rows[:args.pairs]
    records = []
    for row in patient_rows:
        records.extend([{"row_uid": row.row_uid, "patient_id": row.patient_id, "truth": "real", "source": row.real_ceus_path},
                        {"row_uid": row.row_uid, "patient_id": row.patient_id, "truth": "synthetic", "source": row[args.generated_column]}])
    rng.shuffle(records); output = Path(args.output_dir).resolve(); images = output / "blinded_images"; images.mkdir(parents=True, exist_ok=True)
    key_rows = []
    for index, item in enumerate(records, 1):
        code = f"IMG_{index:04d}"; suffix = Path(item["source"]).suffix.lower() or ".png"
        destination = images / f"{code}{suffix}"; shutil.copy2(item["source"], destination)
        key_rows.append({"image_code": code, **item, "blinded_path": str(destination)})
    pd.DataFrame(key_rows).to_csv(output / "BLINDING_KEY_KEEP_SEPARATE.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=["image_code", "reader_id", "reader_experience_years", "real_or_synthetic_guess",
                          "boundary_clarity_1to5", "perfusion_realism_1to5", "false_enhancement_0or1",
                          "missed_enhancement_0or1", "confidence_1to5", "notes"]).to_csv(
        output / "ratings_template.csv", index=False, encoding="utf-8-sig")
    print(f"Prepared {len(records)} blinded images from {len(patient_rows)} distinct patients in {output}")


def analyze(args):
    ratings = pd.read_csv(args.ratings); key = pd.read_csv(args.key)
    data = ratings.merge(key[["image_code", "row_uid", "patient_id", "truth"]], on="image_code", validate="many_to_one")
    guess = data.real_or_synthetic_guess.astype(str).str.strip().str.lower().map(
        {"real": 1, "synthetic": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
    )
    if guess.isna().any():
        raise RuntimeError("real_or_synthetic_guess must be real/synthetic or 1/0 for every rating.")
    truth = data.truth.map({"real": 1, "synthetic": 0}); data["guess_binary"] = guess; data["truth_binary"] = truth
    data["correct"] = data.guess_binary.eq(data.truth_binary).astype(int)
    rows = []
    for reader, group in data.groupby("reader_id"):
        low, high = wilson(int(group.correct.sum()), len(group))
        rows.append({"reader_id": reader, "n": len(group), "accuracy": group.correct.mean(), "ci_low": low, "ci_high": high,
                     "real_sensitivity": group.loc[group.truth_binary.eq(1), "correct"].mean(),
                     "synthetic_specificity": group.loc[group.truth_binary.eq(0), "correct"].mean()})
    pooled_low, pooled_high = wilson(int(data.correct.sum()), len(data))
    rows.append({"reader_id": "POOLED_DESCRIPTIVE", "n": len(data), "accuracy": data.correct.mean(),
                 "ci_low": pooled_low, "ci_high": pooled_high})
    score_columns = ["boundary_clarity_1to5", "perfusion_realism_1to5", "false_enhancement_0or1",
                     "missed_enhancement_0or1", "confidence_1to5"]
    scores = data.groupby("truth")[score_columns].agg(["mean", "std", "count"])
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "reader_accuracy.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(output / "reader_scores_by_truth.csv", encoding="utf-8-sig")
    pd.DataFrame([{"fleiss_kappa_real_vs_synthetic_guess": fleiss_kappa_binary(data),
                   "n_readers": data.reader_id.nunique(), "n_images": data.image_code.nunique()}]).to_csv(
        output / "inter_reader_agreement.csv", index=False, encoding="utf-8-sig")
    print(f"Saved reader-study analysis to {output}")


def main():
    parser = argparse.ArgumentParser(description="Prepare or analyze a blinded CEUS visual Turing study.")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--manifest", required=True); prep.add_argument("--output-dir", required=True)
    prep.add_argument("--generated-column", default="synthetic_ceus_path"); prep.add_argument("--split", default="test_internal_temporal")
    prep.add_argument("--pairs", type=int, default=50); prep.add_argument("--seed", type=int, default=20260828)
    analysis = sub.add_parser("analyze"); analysis.add_argument("--ratings", required=True); analysis.add_argument("--key", required=True)
    analysis.add_argument("--output-dir", required=True)
    args = parser.parse_args(); prepare(args) if args.command == "prepare" else analyze(args)


if __name__ == "__main__":
    main()
