from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import BINARY_TASKS, json_dump


def safe_metric(function, *args, **kwargs):
    try:
        return float(function(*args, **kwargs))
    except ValueError:
        return float("nan")


def calibration_parameters(y, p):
    from scipy.optimize import minimize
    y = np.asarray(y, dtype=float); p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if np.unique(y).size < 2:
        return float("nan"), float("nan")
    x = np.log(p / (1 - p))
    def objective(parameters):
        linear = parameters[0] + parameters[1] * x
        return float(np.sum(np.logaddexp(0, linear) - y * linear))
    result = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    return float(result.x[0]), float(result.x[1])


def binary_metrics(y, p, threshold=0.5, include_calibration=True):
    from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                                 cohen_kappa_score, confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float); pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    has_both_reference_classes = np.unique(y).size == 2
    kappa_defined = not (np.unique(y).size == 1 and np.unique(pred).size == 1 and y[0] == pred[0])
    values = {
        "n": int(len(y)), "threshold": float(threshold),
        "reference_positive_count": int(y.sum()), "predicted_positive_count": int(pred.sum()),
        "true_positive_count": int(tp), "false_positive_count": int(fp),
        "true_negative_count": int(tn), "false_negative_count": int(fn),
        "prevalence": float(y.mean()), "accuracy": safe_metric(accuracy_score, y, pred),
        "sensitivity": safe_metric(recall_score, y, pred, zero_division=0),
        "specificity": float(tn / max(tn + fp, 1)),
        "precision": safe_metric(precision_score, y, pred, zero_division=0),
        "f1": safe_metric(f1_score, y, pred, zero_division=0),
        "kappa": safe_metric(cohen_kappa_score, y, pred) if kappa_defined else float("nan"),
        "roc_auc": safe_metric(roc_auc_score, y, p) if has_both_reference_classes else float("nan"),
        "pr_auc": safe_metric(average_precision_score, y, p) if y.sum() > 0 else float("nan"),
        "brier": safe_metric(brier_score_loss, y, p),
    }
    if include_calibration:
        values["calibration_intercept"], values["calibration_slope"] = calibration_parameters(y, p)
    return values


def cluster_bootstrap_indices(frame, iterations, seed):
    """Precompute patient-cluster resamples once and reuse them for every model/task."""
    patient_values = frame.patient_id.astype(str).to_numpy()
    patient_ids = pd.unique(patient_values)
    groups = [np.flatnonzero(patient_values == patient_id) for patient_id in patient_ids]
    rng = np.random.default_rng(seed)
    chosen_groups = rng.integers(0, len(groups), size=(iterations, len(groups)))
    return [np.concatenate([groups[index] for index in chosen]) for chosen in chosen_groups]


def bootstrap_row_weights(bootstrap_indices, row_count):
    weights = np.zeros((len(bootstrap_indices), row_count), dtype=float)
    for bootstrap_row, indices in enumerate(bootstrap_indices):
        np.add.at(weights[bootstrap_row], indices, 1.0)
    return weights


def _divide(numerator, denominator, default=np.nan):
    result = np.full(np.broadcast_shapes(np.shape(numerator), np.shape(denominator)), default, dtype=float)
    return np.divide(numerator, denominator, out=result, where=np.asarray(denominator) != 0)


def bootstrap_binary_metric_arrays(y, p, threshold, row_weights):
    """Vectorized weighted metrics; duplicated cluster rows are represented by integer weights."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    weights = np.asarray(row_weights, dtype=float)
    n = weights.sum(axis=1)
    true_positive = weights @ (y * pred)
    false_positive = weights @ ((1 - y) * pred)
    true_negative = weights @ ((1 - y) * (1 - pred))
    false_negative = weights @ (y * (1 - pred))
    actual_positive = true_positive + false_negative
    actual_negative = true_negative + false_positive
    predicted_positive = true_positive + false_positive
    predicted_negative = true_negative + false_negative

    accuracy = _divide(true_positive + true_negative, n)
    expected_accuracy = _divide(
        actual_positive * predicted_positive + actual_negative * predicted_negative, n * n
    )
    kappa = _divide(accuracy - expected_accuracy, 1 - expected_accuracy)

    positive = y == 1
    negative = ~positive
    pair_credit = ((p[:, None] > p[None, :]).astype(float) +
                   0.5 * (p[:, None] == p[None, :]).astype(float))
    pair_credit *= positive[:, None] & negative[None, :]
    auc_numerator = np.einsum("bi,ij,bj->b", weights, pair_credit, weights, optimize=True)
    roc_auc = _divide(auc_numerator, actual_positive * actual_negative)

    order = np.argsort(-p, kind="mergesort")
    sorted_p = p[order]
    sorted_y = y[order]
    sorted_weights = weights[:, order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_p) != 0) + 1]
    group_ends = np.r_[group_starts[1:], len(sorted_p)]
    group_total = np.stack([sorted_weights[:, start:end].sum(axis=1)
                            for start, end in zip(group_starts, group_ends)], axis=1)
    group_positive = np.stack([(sorted_weights[:, start:end] * sorted_y[start:end]).sum(axis=1)
                               for start, end in zip(group_starts, group_ends)], axis=1)
    cumulative_total = np.cumsum(group_total, axis=1)
    cumulative_positive = np.cumsum(group_positive, axis=1)
    precision_at_group = _divide(cumulative_positive, cumulative_total, default=0.0)
    pr_auc = _divide((precision_at_group * group_positive).sum(axis=1), actual_positive)

    return {
        "prevalence": _divide(actual_positive, n),
        "accuracy": accuracy,
        "sensitivity": _divide(true_positive, actual_positive, default=0.0),
        "specificity": _divide(true_negative, actual_negative, default=0.0),
        "precision": _divide(true_positive, predicted_positive, default=0.0),
        "f1": _divide(2 * true_positive, 2 * true_positive + false_positive + false_negative,
                      default=0.0),
        "kappa": kappa,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": _divide(weights @ ((p - y) ** 2), n),
    }


def clustered_bootstrap(frame, task, row_weights, threshold=0.5):
    y = frame[f"{task}_true"].to_numpy(dtype=int)
    p = frame[f"{task}_prob"].to_numpy(dtype=float)
    samples = bootstrap_binary_metric_arrays(y, p, threshold, row_weights)
    cis = {}
    for metric, values in samples.items():
        values = values[np.isfinite(values)]
        cis[metric] = [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if len(values) else [None, None]
    return cis


def validation_youden_threshold(frame, task):
    """Select a sensitivity-analysis threshold using validation data only."""
    from sklearn.metrics import roc_curve
    if frame.empty:
        return 0.5, "fallback_fixed_0.5_no_validation_rows"
    y = frame[f"{task}_true"].to_numpy(dtype=int)
    p = frame[f"{task}_prob"].to_numpy(dtype=float)
    if np.unique(y).size < 2:
        return 0.5, "fallback_fixed_0.5_single_validation_class"
    false_positive, true_positive, thresholds = roc_curve(y, p)
    usable = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    if not usable.any():
        return 0.5, "fallback_fixed_0.5_no_finite_threshold"
    thresholds = thresholds[usable]
    youden = (true_positive - false_positive)[usable]
    best = np.flatnonzero(np.isclose(youden, youden.max()))
    selected = best[np.argmin(np.abs(thresholds[best] - 0.5))]
    return float(thresholds[selected]), "validation_youden_j"


def calibration_rows(frame, task, bins=10):
    data = frame[[f"{task}_true", f"{task}_prob"]].copy()
    data["bin"] = pd.cut(data[f"{task}_prob"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    result = data.groupby("bin", observed=True).agg(
        n=(f"{task}_true", "size"), mean_probability=(f"{task}_prob", "mean"), observed_fraction=(f"{task}_true", "mean")
    ).reset_index()
    result["bin"] = result["bin"].astype(str)
    return result


def patient_malignancy(frame):
    inconsistent = frame.groupby("patient_id")["Malignant_true"].nunique()
    if (inconsistent > 1).any():
        print("WARNING: patients with mixed lesion malignancy labels are aggregated using max truth and max risk.")
    return frame.groupby("patient_id", as_index=False).agg(
        Malignant_true=("Malignant_true", "max"), Malignant_prob=("Malignant_prob", "max")
    )


def echo_metrics(frame):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
    y = frame["Internal_Echo_true"].astype(int); pred = frame["Internal_Echo_pred"].astype(int)
    probability_columns = [f"Internal_Echo_prob_{value}" for value in (0, 1, 3, 4)]
    result = {"n": len(frame), "accuracy": safe_metric(accuracy_score, y, pred),
              "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
              "kappa": safe_metric(cohen_kappa_score, y, pred)}
    if set(probability_columns).issubset(frame.columns):
        mapping = {0: 0, 1: 1, 3: 2, 4: 3}; encoded = y.map(mapping).to_numpy()
        try:
            result["macro_ovr_roc_auc"] = float(roc_auc_score(encoded, frame[probability_columns], multi_class="ovr", average="macro"))
        except ValueError:
            result["macro_ovr_roc_auc"] = float("nan")
    return result


def echo_clustered_bootstrap(frame, bootstrap_indices):
    values = [echo_metrics(frame.iloc[index]) for index in bootstrap_indices]
    result = {}
    for metric in values[0]:
        if metric == "n": continue
        sample_values = np.asarray([x.get(metric, np.nan) for x in values]); sample_values = sample_values[np.isfinite(sample_values)]
        result[metric] = [float(np.quantile(sample_values, .025)), float(np.quantile(sample_values, .975))] if len(sample_values) else [None, None]
    return result


def multitask_exact_metrics(frame):
    true_columns = ["Internal_Echo_true", *[f"{task}_true" for task in BINARY_TASKS]]
    pred_columns = ["Internal_Echo_pred", *[f"{task}_pred" for task in BINARY_TASKS]]
    correct = frame[true_columns].to_numpy() == frame[pred_columns].to_numpy()
    return {"n": len(frame), "hamming_loss": float(1 - correct.mean()), "subset_accuracy": float(correct.all(axis=1).mean())}


def holm_adjust(values):
    values = np.asarray(values, dtype=float); adjusted = np.full(len(values), np.nan)
    valid = np.where(np.isfinite(values))[0]
    order = valid[np.argsort(values[valid])]; running = 0.0; m = len(order)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * values[index]); adjusted[index] = min(running, 1.0)
    return adjusted


def paired_auc_difference(left, right, task, left_bootstrap_auc, right_bootstrap_auc):
    from sklearn.metrics import roc_auc_score
    if not left["row_uid"].equals(right["row_uid"]):
        raise RuntimeError("Paired prediction frames are not aligned by row_uid.")
    y_all = left[f"{task}_true"].to_numpy(dtype=int)
    left_probability = left[f"{task}_prob"].to_numpy(dtype=float)
    right_probability = right[f"{task}_prob"].to_numpy(dtype=float)
    values = np.asarray(left_bootstrap_auc) - np.asarray(right_bootstrap_auc)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"auc_difference": None, "ci_low": None, "ci_high": None, "p_two_sided": None,
                "n_matched": len(left)}
    observed = safe_metric(roc_auc_score, y_all, left_probability) - safe_metric(
        roc_auc_score, y_all, right_probability)
    p_value = 2 * min(float((values <= 0).mean()), float((values >= 0).mean()))
    return {"auc_difference": observed, "ci_low": float(np.quantile(values, .025)),
            "ci_high": float(np.quantile(values, .975)), "p_two_sided": min(p_value, 1.0),
            "n_matched": len(left)}


def parse_named_paths(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Each --predictions value must be NAME=CSV_PATH")
        name, path = value.split("=", 1); result[name] = Path(path).resolve()
    return result


def save_curve_figures(frames, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    figure_dir = output / "curves"; figure_dir.mkdir(parents=True, exist_ok=True)
    for task in BINARY_TASKS:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for name, frame in frames.items():
            y = frame[f"{task}_true"].to_numpy(); p = frame[f"{task}_prob"].to_numpy()
            if np.unique(y).size < 2:
                continue
            false_positive, true_positive, _ = roc_curve(y, p)
            precision, recall, _ = precision_recall_curve(y, p)
            axes[0].plot(false_positive, true_positive, label=name)
            axes[1].plot(recall, precision, label=name)
            bins = calibration_rows(frame, task)
            axes[2].plot(bins.mean_probability, bins.observed_fraction, marker="o", label=name)
        axes[0].plot([0, 1], [0, 1], "k--", linewidth=.8); axes[0].set(title=f"{task} ROC", xlabel="1-specificity", ylabel="sensitivity")
        axes[1].set(title=f"{task} precision-recall", xlabel="recall", ylabel="precision")
        axes[2].plot([0, 1], [0, 1], "k--", linewidth=.8); axes[2].set(title=f"{task} calibration", xlabel="predicted", ylabel="observed")
        for axis in axes:
            axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figure_dir / f"{task}_roc_pr_calibration.png", dpi=200); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Patient-clustered metrics, CIs, calibration, and paired comparisons.")
    parser.add_argument("--predictions", nargs="+", required=True, metavar="NAME=CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test_internal_temporal")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be at least 1")
    named = parse_named_paths(args.predictions)
    frames = {}; validation_frames = {}; rows = []; calibration = []; full = {}
    for name, path in named.items():
        all_rows = pd.read_csv(path)
        frame = all_rows[all_rows.split.eq(args.split)].copy()
        if frame.empty:
            raise RuntimeError(f"No {args.split} rows in {path}")
        validation = all_rows[all_rows.split.eq("val")].copy()
        if validation.empty:
            print(f"WARNING: {name} has no validation rows; sensitivity thresholds fall back to 0.5.", flush=True)
        frames[name] = frame
        validation_frames[name] = validation

    names = list(named)
    reference = frames[names[0]].sort_values("row_uid").reset_index(drop=True)
    reference_ids = reference["row_uid"].tolist()
    truth_columns = [f"{task}_true" for task in BINARY_TASKS]
    for name, frame in list(frames.items()):
        if frame["row_uid"].duplicated().any():
            raise RuntimeError(f"Duplicate row_uid values in predictions for {name}.")
        if set(frame["row_uid"]) != set(reference_ids):
            raise RuntimeError(f"Prediction rows do not match the reference model for {name}.")
        aligned = frame.set_index("row_uid").loc[reference_ids].reset_index()
        if not aligned["patient_id"].astype(str).equals(reference["patient_id"].astype(str)):
            raise RuntimeError(f"patient_id mismatch after row_uid alignment for {name}.")
        if not aligned[truth_columns].reset_index(drop=True).equals(reference[truth_columns].reset_index(drop=True)):
            raise RuntimeError(f"Ground-truth labels differ across prediction files for {name}.")
        frames[name] = aligned

    bootstrap_indices = cluster_bootstrap_indices(reference, args.bootstrap, args.seed)
    bootstrap_weights = bootstrap_row_weights(bootstrap_indices, len(reference))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    for name in names:
        frame = frames[name]
        validation = validation_frames[name]
        print(f"[{name}] evaluating {len(frame)} test rows with {args.bootstrap} shared patient-cluster resamples...",
              flush=True)
        full[name] = {}
        for task in BINARY_TASKS:
            primary = binary_metrics(frame[f"{task}_true"], frame[f"{task}_prob"], threshold=0.5)
            primary_ci = clustered_bootstrap(frame, task, bootstrap_weights, threshold=0.5)
            validation_threshold, threshold_status = validation_youden_threshold(validation, task)
            sensitivity = binary_metrics(frame[f"{task}_true"], frame[f"{task}_prob"],
                                         threshold=validation_threshold)
            sensitivity_ci = clustered_bootstrap(frame, task, bootstrap_weights,
                                                 threshold=validation_threshold)
            full[name][task] = {
                "primary_fixed_0.5": {"point": primary, "patient_clustered_95_ci": primary_ci},
                "sensitivity_validation_threshold": {
                    "selection_method": threshold_status, "source_split": "val",
                    "point": sensitivity, "patient_clustered_95_ci": sensitivity_ci,
                },
            }
            rows.append({"model": name, "unit": "image_or_lesion_row", "task": task,
                         "analysis_role": "primary", "threshold_strategy": "fixed_0.5",
                         "threshold_source_split": "predefined", **primary,
                         **{f"{key}_ci_low": value[0] for key, value in primary_ci.items()},
                         **{f"{key}_ci_high": value[1] for key, value in primary_ci.items()}})
            rows.append({"model": name, "unit": "image_or_lesion_row", "task": task,
                         "analysis_role": "sensitivity", "threshold_strategy": threshold_status,
                         "threshold_source_split": "val", **sensitivity,
                         **{f"{key}_ci_low": value[0] for key, value in sensitivity_ci.items()},
                         **{f"{key}_ci_high": value[1] for key, value in sensitivity_ci.items()}})
            if primary["predicted_positive_count"] == 0:
                print(f"  NOTE: {task} fixed_0.5 predicted_positive_count=0; precision/f1 recorded as 0.",
                      flush=True)
            bins = calibration_rows(frame, task); bins["model"] = name; bins["task"] = task; calibration.append(bins)
        echo = echo_metrics(frame); echo_ci = echo_clustered_bootstrap(frame, bootstrap_indices)
        full[name]["Internal_Echo"] = {"point": echo, "patient_clustered_95_ci": echo_ci}
        rows.append({"model": name, "unit": "image_or_lesion_row", "task": "Internal_Echo",
                     "analysis_role": "primary", "threshold_strategy": "not_applicable_multiclass",
                     "threshold_source_split": "not_applicable", **echo,
                     **{f"{key}_ci_low": value[0] for key, value in echo_ci.items()},
                     **{f"{key}_ci_high": value[1] for key, value in echo_ci.items()}})
        rows.append({"model": name, "unit": "image_or_lesion_row", "task": "ALL_TASKS",
                     "analysis_role": "primary", "threshold_strategy": "fixed_0.5_binary_outputs",
                     "threshold_source_split": "predefined", **multitask_exact_metrics(frame)})
        patient = patient_malignancy(frame)
        validation_patient = patient_malignancy(validation) if not validation.empty else validation
        patient_bootstrap = cluster_bootstrap_indices(patient, args.bootstrap, args.seed)
        patient_bootstrap_weights = bootstrap_row_weights(patient_bootstrap, len(patient))
        patient_primary = binary_metrics(patient.Malignant_true, patient.Malignant_prob, threshold=0.5)
        patient_primary_ci = clustered_bootstrap(patient, "Malignant", patient_bootstrap_weights,
                                                 threshold=0.5)
        patient_threshold, patient_status = validation_youden_threshold(validation_patient, "Malignant")
        patient_sensitivity = binary_metrics(patient.Malignant_true, patient.Malignant_prob,
                                             threshold=patient_threshold)
        patient_sensitivity_ci = clustered_bootstrap(patient, "Malignant", patient_bootstrap_weights,
                                                     threshold=patient_threshold)
        full[name]["Malignant_patient_max_risk"] = {
            "primary_fixed_0.5": {"point": patient_primary, "patient_clustered_95_ci": patient_primary_ci},
            "sensitivity_validation_threshold": {
                "selection_method": patient_status, "source_split": "val",
                "point": patient_sensitivity, "patient_clustered_95_ci": patient_sensitivity_ci,
            },
        }
        rows.append({"model": name, "unit": "patient_max_risk", "task": "Malignant",
                     "analysis_role": "primary", "threshold_strategy": "fixed_0.5",
                     "threshold_source_split": "predefined", **patient_primary,
                     **{f"{key}_ci_low": value[0] for key, value in patient_primary_ci.items()},
                     **{f"{key}_ci_high": value[1] for key, value in patient_primary_ci.items()}})
        rows.append({"model": name, "unit": "patient_max_risk", "task": "Malignant",
                     "analysis_role": "sensitivity", "threshold_strategy": patient_status,
                     "threshold_source_split": "val", **patient_sensitivity,
                     **{f"{key}_ci_low": value[0] for key, value in patient_sensitivity_ci.items()},
                     **{f"{key}_ci_high": value[1] for key, value in patient_sensitivity_ci.items()}})

    comparisons = []
    print("Computing paired AUC comparisons with the same cluster-resample plan...", flush=True)
    bootstrap_auc = {
        (name, task): bootstrap_binary_metric_arrays(
            frames[name][f"{task}_true"].to_numpy(dtype=int),
            frames[name][f"{task}_prob"].to_numpy(dtype=float),
            threshold=0.5,
            row_weights=bootstrap_weights,
        )["roc_auc"]
        for name in names for task in BINARY_TASKS
    }
    for i, left_name in enumerate(names):
        for right_name in names[i + 1:]:
            for task in BINARY_TASKS:
                result = paired_auc_difference(
                    frames[left_name], frames[right_name], task,
                    bootstrap_auc[(left_name, task)], bootstrap_auc[(right_name, task)],
                )
                comparisons.append({"left": left_name, "right": right_name, "task": task, **result})
    comparison_frame = pd.DataFrame(comparisons)
    if not comparison_frame.empty:
        comparison_frame["p_holm_all_reported_comparisons"] = holm_adjust(comparison_frame["p_two_sided"])
    pd.DataFrame(rows).to_csv(output / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    comparison_frame.to_csv(output / "paired_auc_comparisons.csv", index=False, encoding="utf-8-sig")
    pd.concat(calibration, ignore_index=True).to_csv(output / "calibration_bins.csv", index=False, encoding="utf-8-sig")
    json_dump(full, output / "metrics_full.json")
    save_curve_figures(frames, output)
    print(f"Saved classification analysis to {output}")


if __name__ == "__main__":
    main()
