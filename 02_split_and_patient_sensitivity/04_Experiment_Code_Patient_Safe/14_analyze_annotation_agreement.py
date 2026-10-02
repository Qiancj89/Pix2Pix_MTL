from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from common import BINARY_TASKS, LABEL_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise inter-observer agreement for a blinded re-annotation subset.")
    parser.add_argument("--ratings", required=True, help="Columns: row_uid,reader_id and all eight task labels.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.ratings)
    required = {"row_uid", "reader_id", *LABEL_COLUMNS}
    if not required.issubset(data.columns):
        raise RuntimeError(f"Ratings file is missing: {sorted(required - set(data.columns))}")
    if data.duplicated(["row_uid", "reader_id"]).any():
        raise RuntimeError("Each reader may rate each row_uid only once.")
    reader_sets = data.groupby("row_uid").reader_id.nunique()
    if (reader_sets < 2).any():
        raise RuntimeError("Every re-annotated case needs at least two independent readers.")
    rows = []
    for left, right in combinations(sorted(data.reader_id.astype(str).unique()), 2):
        left_frame = data[data.reader_id.astype(str).eq(left)].set_index("row_uid")
        right_frame = data[data.reader_id.astype(str).eq(right)].set_index("row_uid")
        shared = left_frame.index.intersection(right_frame.index)
        if len(shared) < 2:
            continue
        for task in LABEL_COLUMNS:
            first = left_frame.loc[shared, task]; second = right_frame.loc[shared, task]
            weights = "quadratic" if task == "Internal_Echo" else None
            kappa = cohen_kappa_score(first, second, weights=weights)
            rows.append({"reader_1": left, "reader_2": right, "task": task, "n_shared": len(shared),
                         "percent_agreement": float((first.to_numpy() == second.to_numpy()).mean()),
                         "kappa": float(kappa), "kappa_type": "quadratic_weighted" if weights else "unweighted"})
    pairwise = pd.DataFrame(rows)
    summary = pairwise.groupby("task", as_index=False).agg(
        reader_pairs=("kappa", "size"), mean_kappa=("kappa", "mean"), min_kappa=("kappa", "min"),
        max_kappa=("kappa", "max"), mean_percent_agreement=("percent_agreement", "mean")
    )
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    pairwise.to_csv(output / "annotation_pairwise_agreement.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "annotation_agreement_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Saved annotation agreement to {output}")


if __name__ == "__main__":
    main()

