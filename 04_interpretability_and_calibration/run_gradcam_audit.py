from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image


def bootstrap_ci(frame: pd.DataFrame, column: str, seed: int, n_boot: int = 2000) -> tuple[float, float, float]:
    patient_values = frame.groupby("patient_id", sort=True)[column].mean()
    values = patient_values.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        estimates[index] = rng.choice(values, size=len(values), replace=True).mean()
    return float(values.mean()), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def load_mask(path: str, size: int) -> np.ndarray:
    mask = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    array = np.asarray(mask, dtype=np.uint8)
    values, counts = np.unique(array, return_counts=True)
    if len(values) < 2:
        return np.zeros_like(array, dtype=bool)
    background = values[int(np.argmax(counts))]
    # Some exported class masks use 1/2 rather than 0/255; the modal value is
    # the background, so treating every nonzero pixel as lesion would invert them.
    return array != background


def save_case_panel(
    output: Path,
    row_uid: str,
    bmode_path: str,
    synthetic_path: str,
    mask: np.ndarray,
    cam: np.ndarray,
    probability: float,
    label: int,
) -> None:
    bmode = Image.open(bmode_path).convert("RGB").resize((cam.shape[1], cam.shape[0]), Image.Resampling.BICUBIC)
    synthetic = Image.open(synthetic_path).convert("RGB").resize((cam.shape[1], cam.shape[0]), Image.Resampling.BICUBIC)
    figure, axes = plt.subplots(1, 4, figsize=(11.2, 2.8))
    axes[0].imshow(bmode)
    axes[0].set_title("B-mode")
    axes[1].imshow(synthetic)
    axes[1].set_title("Synthetic CEUS")
    axes[2].imshow(bmode)
    axes[2].contour(mask.astype(float), levels=[0.5], colors=["lime"], linewidths=1.5)
    axes[2].set_title("Lesion mask")
    axes[3].imshow(bmode)
    axes[3].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[3].contour(mask.astype(float), levels=[0.5], colors=["white"], linewidths=1.0)
    axes[3].set_title("Malignancy Grad-CAM")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"Case {row_uid} | reference={label} | predicted risk={probability:.3f}", fontsize=10)
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Whole-test-set malignancy Grad-CAM audit with lesion-mask overlap.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test_internal_temporal")
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    code_dir = str(Path(args.code_dir).resolve())
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from data import MultiConditionDataset  # noqa: PLC0415
    from models import MultiTaskClassifier  # noqa: PLC0415

    output_dir = Path(args.output_dir).resolve()
    case_dir = output_dir / "GradCAM_Per_Case"
    case_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MultiTaskClassifier(checkpoint["backbone"], False, checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    manifest = pd.read_csv(args.manifest)
    frame = manifest.loc[manifest["split"].eq(args.split)].copy().reset_index(drop=True)
    required = ["row_uid", "patient_id", "bmode_path", "synthetic_ceus_path", "mask_path", "Malignant"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Manifest is missing required columns: {missing}")
    if frame.empty:
        raise RuntimeError(f"No rows found for split {args.split}")
    if frame[required].isna().any().any():
        raise RuntimeError("Test rows contain missing Grad-CAM inputs or masks")

    state: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, activation):
        state["activation"] = activation
        activation.retain_grad()

    handle = model.gradcam_layer.register_forward_hook(hook)
    records: list[dict[str, object]] = []
    cams: dict[str, np.ndarray] = {}
    try:
        for _, row in frame.iterrows():
            dataset = MultiConditionDataset(pd.DataFrame([row]), checkpoint["condition"], checkpoint["image_size"], False)
            image, _, _ = dataset[0]
            image = image.unsqueeze(0).to(device)
            model.zero_grad(set_to_none=True)
            outputs = model(image)
            logit = outputs["Malignant"].view(-1)[0]
            probability = float(torch.sigmoid(logit).detach().cpu().item())
            logit.backward()

            activation = state["activation"]
            gradient = activation.grad
            weights = gradient.mean(dim=tuple(range(2, gradient.ndim)), keepdim=True)
            cam_tensor = torch.relu((weights * activation).sum(dim=1, keepdim=True))
            cam_tensor = F.interpolate(
                cam_tensor,
                size=(checkpoint["image_size"], checkpoint["image_size"]),
                mode="bilinear",
                align_corners=False,
            )
            cam = cam_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
            cam = (cam - cam.min()) / max(float(cam.max() - cam.min()), 1e-8)
            mask = load_mask(str(row.mask_path), checkpoint["image_size"])
            if not mask.any() or mask.all():
                raise RuntimeError(f"Invalid lesion mask for row_uid={row.row_uid}")

            mask_area = float(mask.mean())
            total_energy = float(cam.sum())
            energy_inside = float(cam[mask].sum() / max(total_energy, 1e-8))
            enrichment = float(energy_inside / mask_area)
            maximum = np.unravel_index(int(np.argmax(cam)), cam.shape)
            pointing_hit = int(mask[maximum])
            top_threshold = float(np.quantile(cam, 0.80))
            top_region = cam >= top_threshold
            intersection = int(np.logical_and(top_region, mask).sum())
            union = int(np.logical_or(top_region, mask).sum())
            top20_iou = float(intersection / max(union, 1))
            top20_recall = float(intersection / max(int(mask.sum()), 1))
            row_uid = str(row.row_uid)
            cams[row_uid] = cam
            records.append(
                {
                    "row_uid": row_uid,
                    "patient_id": str(row.patient_id),
                    "reference_malignant": int(row.Malignant),
                    "predicted_malignancy_probability": probability,
                    "mask_area_fraction": mask_area,
                    "cam_energy_inside_lesion": energy_inside,
                    "cam_enrichment_over_area": enrichment,
                    "pointing_game_hit": pointing_hit,
                    "top20_iou": top20_iou,
                    "top20_lesion_recall": top20_recall,
                }
            )
            save_case_panel(
                case_dir / f"{row_uid}_Malignant_GradCAM.png",
                row_uid,
                str(row.bmode_path),
                str(row.synthetic_ceus_path),
                mask,
                cam,
                probability,
                int(row.Malignant),
            )
    finally:
        handle.remove()

    metrics = pd.DataFrame.from_records(records).sort_values("row_uid")
    metrics.to_csv(output_dir / "Table_R6_GradCAM_Lesion_Overlap_Per_Case.csv", index=False)
    summary: dict[str, object] = {
        "analysis": "malignancy-head Grad-CAM on every image in the prespecified temporal internal test split",
        "condition": checkpoint["condition"],
        "split": args.split,
        "n_images": int(len(metrics)),
        "n_patients": int(metrics["patient_id"].nunique()),
        "n_benign_images": int((metrics["reference_malignant"] == 0).sum()),
        "n_malignant_images": int((metrics["reference_malignant"] == 1).sum()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selection_rule": "all test images; displayed examples selected by predicted-risk quartiles before viewing attribution maps",
        "interpretation_limit": "lesion-mask overlap supports localization only; it does not establish causal reasoning, perfusion fidelity, or clinical safety",
    }
    for offset, column in enumerate(
        ["cam_energy_inside_lesion", "cam_enrichment_over_area", "pointing_game_hit", "top20_iou", "top20_lesion_recall"]
    ):
        mean, low, high = bootstrap_ci(metrics, column, args.seed + offset)
        summary[column] = {"patient_mean": mean, "patient_clustered_95_ci": [low, high]}
    summary["proportion_images_enrichment_gt_1"] = float((metrics["cam_enrichment_over_area"] > 1.0).mean())
    (output_dir / "Table_R6_GradCAM_Lesion_Overlap_Summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sorted_metrics = metrics.sort_values("predicted_malignancy_probability").reset_index(drop=True)
    positions = [0.10, 0.40, 0.70, 0.95]
    selected_indices = sorted({int(round(position * (len(sorted_metrics) - 1))) for position in positions})
    selected = sorted_metrics.iloc[selected_indices]
    figure, axes = plt.subplots(len(selected), 4, figsize=(10.8, 2.25 * len(selected)))
    if len(selected) == 1:
        axes = np.asarray([axes])
    for row_index, (_, metric_row) in enumerate(selected.iterrows()):
        source_row = frame.loc[frame["row_uid"].astype(str).eq(str(metric_row.row_uid))].iloc[0]
        cam = cams[str(metric_row.row_uid)]
        mask = load_mask(str(source_row.mask_path), checkpoint["image_size"])
        bmode = Image.open(source_row.bmode_path).convert("RGB").resize((checkpoint["image_size"], checkpoint["image_size"]), Image.Resampling.BICUBIC)
        synthetic = Image.open(source_row.synthetic_ceus_path).convert("RGB").resize((checkpoint["image_size"], checkpoint["image_size"]), Image.Resampling.BICUBIC)
        axes[row_index, 0].imshow(bmode)
        axes[row_index, 1].imshow(synthetic)
        axes[row_index, 2].imshow(bmode)
        axes[row_index, 2].contour(mask.astype(float), levels=[0.5], colors=["lime"], linewidths=1.3)
        axes[row_index, 3].imshow(bmode)
        axes[row_index, 3].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axes[row_index, 3].contour(mask.astype(float), levels=[0.5], colors=["white"], linewidths=0.9)
        axes[row_index, 0].set_ylabel(
            f"Ref={int(metric_row.reference_malignant)}\nRisk={metric_row.predicted_malignancy_probability:.3f}\nEnrich={metric_row.cam_enrichment_over_area:.2f}",
            fontsize=8,
        )
    titles = ["B-mode", "Synthetic CEUS", "Lesion mask", "Malignancy Grad-CAM"]
    for column_index, title in enumerate(titles):
        axes[0, column_index].set_title(title, fontsize=10)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(output_dir / "Figure_R3_GradCAM_Malignancy_Examples.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    compact = selected.iloc[[0, -1]] if len(selected) > 1 else selected
    figure, axes = plt.subplots(len(compact), 4, figsize=(7.2, 1.65 * len(compact)))
    if len(compact) == 1:
        axes = np.asarray([axes])
    for row_index, (_, metric_row) in enumerate(compact.iterrows()):
        source_row = frame.loc[frame["row_uid"].astype(str).eq(str(metric_row.row_uid))].iloc[0]
        cam = cams[str(metric_row.row_uid)]
        mask = load_mask(str(source_row.mask_path), checkpoint["image_size"])
        bmode = Image.open(source_row.bmode_path).convert("RGB").resize((checkpoint["image_size"], checkpoint["image_size"]), Image.Resampling.BICUBIC)
        synthetic = Image.open(source_row.synthetic_ceus_path).convert("RGB").resize((checkpoint["image_size"], checkpoint["image_size"]), Image.Resampling.BICUBIC)
        axes[row_index, 0].imshow(bmode)
        axes[row_index, 1].imshow(synthetic)
        axes[row_index, 2].imshow(bmode)
        axes[row_index, 2].contour(mask.astype(float), levels=[0.5], colors=["lime"], linewidths=1.1)
        axes[row_index, 3].imshow(bmode)
        axes[row_index, 3].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axes[row_index, 3].contour(mask.astype(float), levels=[0.5], colors=["white"], linewidths=0.8)
        axes[row_index, 0].set_ylabel(
            f"Ref={int(metric_row.reference_malignant)}\nRisk={metric_row.predicted_malignancy_probability:.3f}", fontsize=7
        )
    for column_index, title in enumerate(titles):
        axes[0, column_index].set_title(title, fontsize=8)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout(pad=0.5)
    figure.savefig(output_dir / "Figure_M1_GradCAM_Compact.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
