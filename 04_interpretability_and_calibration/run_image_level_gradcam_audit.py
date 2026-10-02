from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision import transforms


class ImageLevelViTL16(nn.Module):
    """Exact ViT-L/16 multi-task architecture used by the original image-level code."""

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        base_model = models.vit_l_16(weights=None)
        old_conv = base_model.conv_proj
        new_conv = nn.Conv2d(
            6,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=True,
        )
        base_model.conv_proj = new_conv
        self.vit_model = base_model
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.internal_echo_head = nn.Linear(256, 4)
        self.morphology_head = nn.Linear(256, 1)
        self.boundary_head = nn.Linear(256, 1)
        self.solid_head = nn.Linear(256, 1)
        self.separation_head = nn.Linear(256, 1)
        self.nipple_head = nn.Linear(256, 1)
        self.blood_flow_head = nn.Linear(256, 1)
        self.malignant_head = nn.Linear(256, 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.vit_model._process_input(image)
        class_token = self.vit_model.class_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([class_token, tokens], dim=1)
        tokens = self.vit_model.encoder(tokens)
        shared = self.classifier(tokens[:, 0])
        return {
            "Internal_Echo": self.internal_echo_head(shared),
            "Morphology": self.morphology_head(shared),
            "Boundary": self.boundary_head(shared),
            "Solid": self.solid_head(shared),
            "Separation": self.separation_head(shared),
            "Nipple": self.nipple_head(shared),
            "Blood_Flow": self.blood_flow_head(shared),
            "Malignant": self.malignant_head(shared),
        }


def find_image(directory: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Image not found for {stem} under {directory}")


def load_mask(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    array = np.asarray(image, dtype=np.uint8)
    values, counts = np.unique(array, return_counts=True)
    if len(values) < 2:
        raise RuntimeError(f"Mask has fewer than two values: {path}")
    background = values[int(np.argmax(counts))]
    mask = array != background
    if not mask.any() or mask.all():
        raise RuntimeError(f"Invalid lesion mask: {path}")
    return mask


def bootstrap_ci(frame: pd.DataFrame, column: str, seed: int, n_boot: int = 2000) -> tuple[float, float, float]:
    values = frame.groupby("patient_id", sort=True)[column].mean().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        estimates[index] = rng.choice(values, size=len(values), replace=True).mean()
    return float(values.mean()), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def save_panel(
    output: Path,
    title: str,
    bmode: Image.Image,
    synthetic: Image.Image,
    mask: np.ndarray,
    cam: np.ndarray,
) -> None:
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
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def save_examples(output: Path, selected: pd.DataFrame, cache: dict[str, dict[str, object]], compact: bool) -> None:
    figure, axes = plt.subplots(len(selected), 4, figsize=(7.2 if compact else 10.8, (1.65 if compact else 2.25) * len(selected)))
    if len(selected) == 1:
        axes = np.asarray([axes])
    for row_index, (_, row) in enumerate(selected.iterrows()):
        item = cache[str(row.row_uid)]
        bmode = item["bmode"]
        synthetic = item["synthetic"]
        mask = item["mask"]
        cam = item["cam"]
        axes[row_index, 0].imshow(bmode)
        axes[row_index, 1].imshow(synthetic)
        axes[row_index, 2].imshow(bmode)
        axes[row_index, 2].contour(mask.astype(float), levels=[0.5], colors=["lime"], linewidths=1.2)
        axes[row_index, 3].imshow(bmode)
        axes[row_index, 3].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axes[row_index, 3].contour(mask.astype(float), levels=[0.5], colors=["white"], linewidths=0.9)
        axes[row_index, 0].set_ylabel(
            f"Ref={int(row.reference_malignant)}\nRisk={row.predicted_malignancy_probability:.3f}\nEnrich={row.cam_enrichment_over_area:.2f}",
            fontsize=7 if compact else 8,
        )
    for column, title in enumerate(("B-mode", "Synthetic CEUS", "Lesion mask", "Malignancy Grad-CAM")):
        axes[0, column].set_title(title, fontsize=9 if compact else 10)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM audit for the archived original image-level classifier.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bmode-dir", required=True)
    parser.add_argument("--synthetic-dir", required=True)
    parser.add_argument("--archived-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    bmode_dir = Path(args.bmode_dir).resolve()
    synthetic_dir = Path(args.synthetic_dir).resolve()
    archived_predictions_path = Path(args.archived_predictions).resolve()
    output_dir = Path(args.output_dir).resolve()
    case_dir = output_dir / "GradCAM_Per_Case"
    case_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(manifest_path)
    frame = frame.loc[frame["split"].eq("test_internal_temporal")].copy()
    frame["case_name"] = frame["original_name"].astype(str)
    if len(frame) != 67:
        raise RuntimeError(f"Expected 67 temporal-test images, found {len(frame)}")
    if frame["case_name"].duplicated().any():
        raise RuntimeError("Temporal-test case names are not unique")

    archived = pd.read_csv(archived_predictions_path)
    archived = archived[["English_Name", "Malignant_prob"]].rename(
        columns={"English_Name": "case_name", "Malignant_prob": "archived_malignancy_probability"}
    )
    frame = frame.merge(archived, on="case_name", how="left", validate="one_to_one")
    if frame["archived_malignancy_probability"].isna().any():
        missing = frame.loc[frame["archived_malignancy_probability"].isna(), "case_name"].tolist()
        raise RuntimeError(f"Missing archived predictions for: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ImageLevelViTL16(dropout=0.5).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    hook_state: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, activation):
        hook_state["activation"] = activation
        activation.retain_grad()

    handle = model.vit_model.encoder.layers[-1].ln_1.register_forward_hook(forward_hook)
    records: list[dict[str, object]] = []
    cache: dict[str, dict[str, object]] = {}
    try:
        for _, row in frame.sort_values("case_name").iterrows():
            bmode_path = find_image(bmode_dir, str(row.case_name))
            synthetic_path = find_image(synthetic_dir, str(row.case_name))
            mask_path = Path(str(row.mask_path)).resolve()
            bmode_source = Image.open(bmode_path).convert("RGB")
            synthetic_source = Image.open(synthetic_path).convert("RGB")
            combined = torch.cat([transform(bmode_source), transform(synthetic_source)], dim=0).unsqueeze(0).to(device)

            model.zero_grad(set_to_none=True)
            logit = model(combined)["Malignant"].view(-1)[0]
            probability = float(torch.sigmoid(logit).detach().cpu().item())
            logit.backward()
            activation = hook_state["activation"]
            gradient = activation.grad
            patch_activation = activation[:, 1:, :].transpose(1, 2)
            patch_gradient = gradient[:, 1:, :].transpose(1, 2)
            grid_size = int(round(patch_activation.shape[-1] ** 0.5))
            if grid_size * grid_size != patch_activation.shape[-1]:
                raise RuntimeError(f"ViT patch count is not square: {patch_activation.shape[-1]}")
            patch_activation = patch_activation.reshape(activation.shape[0], activation.shape[2], grid_size, grid_size)
            patch_gradient = patch_gradient.reshape(activation.shape[0], activation.shape[2], grid_size, grid_size)
            weights = patch_gradient.mean(dim=(2, 3), keepdim=True)
            cam_tensor = torch.relu((weights * patch_activation).sum(dim=1, keepdim=True))
            cam_tensor = F.interpolate(cam_tensor, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
            cam = cam_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
            cam = (cam - cam.min()) / max(float(cam.max() - cam.min()), 1e-8)
            mask = load_mask(mask_path, args.image_size)

            mask_area = float(mask.mean())
            energy_inside = float(cam[mask].sum() / max(float(cam.sum()), 1e-8))
            enrichment = float(energy_inside / mask_area)
            maximum = np.unravel_index(int(np.argmax(cam)), cam.shape)
            pointing_hit = int(mask[maximum])
            top_region = cam >= float(np.quantile(cam, 0.80))
            intersection = int(np.logical_and(top_region, mask).sum())
            union = int(np.logical_or(top_region, mask).sum())
            row_uid = str(row.row_uid)
            record = {
                "row_uid": row_uid,
                "case_name": str(row.case_name),
                "patient_id": str(row.patient_id),
                "reference_malignant": int(row.Malignant),
                "predicted_malignancy_probability": probability,
                "archived_malignancy_probability": float(row.archived_malignancy_probability),
                "absolute_probability_difference": abs(probability - float(row.archived_malignancy_probability)),
                "mask_area_fraction": mask_area,
                "cam_energy_inside_lesion": energy_inside,
                "cam_enrichment_over_area": enrichment,
                "pointing_game_hit": pointing_hit,
                "top20_iou": float(intersection / max(union, 1)),
                "top20_lesion_recall": float(intersection / max(int(mask.sum()), 1)),
            }
            records.append(record)
            bmode = bmode_source.resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
            synthetic = synthetic_source.resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
            cache[row_uid] = {"bmode": bmode, "synthetic": synthetic, "mask": mask, "cam": cam}
            save_panel(
                case_dir / f"{row_uid}_{row.case_name}_ImageLevel_GradCAM.png",
                f"{row.case_name} | reference={int(row.Malignant)} | predicted risk={probability:.3f}",
                bmode,
                synthetic,
                mask,
                cam,
            )
    finally:
        handle.remove()

    metrics = pd.DataFrame.from_records(records).sort_values("case_name")
    max_difference = float(metrics["absolute_probability_difference"].max())
    if max_difference > 1e-5:
        raise RuntimeError(f"Reconstructed model does not reproduce archived predictions (max abs difference={max_difference})")
    metrics.to_csv(output_dir / "Table_IL_GradCAM_Lesion_Overlap_Per_Case.csv", index=False)

    summary: dict[str, object] = {
        "analysis": "malignancy-head Grad-CAM for the archived original image-level classifier on all temporal internal test images",
        "development_split_policy": "original image/lesion-row-level 70:30 split",
        "evaluation_split": "test_internal_temporal",
        "model": "ViT-L/16 multi-task classifier",
        "input_condition": "B-mode plus Enhanced Pix2Pix synthetic CEUS",
        "checkpoint": str(checkpoint_path),
        "checkpoint_selection": "best validation Hamming loss",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "n_images": int(len(metrics)),
        "n_patients": int(metrics["patient_id"].nunique()),
        "n_benign_images": int((metrics["reference_malignant"] == 0).sum()),
        "n_malignant_images": int((metrics["reference_malignant"] == 1).sum()),
        "archived_prediction_max_absolute_difference": max_difference,
        "selection_rule": "all test images; displayed examples selected by predicted-risk quantiles before inspecting attribution maps",
        "interpretation_limit": "Grad-CAM is post-hoc and model-specific; lesion overlap does not establish causal reasoning, clinical validity, or direct comparability with a different backbone.",
    }
    columns = ["cam_energy_inside_lesion", "cam_enrichment_over_area", "pointing_game_hit", "top20_iou", "top20_lesion_recall"]
    for offset, column in enumerate(columns):
        mean, low, high = bootstrap_ci(metrics, column, args.seed + offset)
        summary[column] = {"patient_mean": mean, "patient_clustered_95_ci": [low, high]}
    summary["proportion_images_enrichment_gt_1"] = float((metrics["cam_enrichment_over_area"] > 1.0).mean())
    (output_dir / "Table_IL_GradCAM_Lesion_Overlap_Summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ordered = metrics.sort_values("predicted_malignancy_probability").reset_index(drop=True)
    indices = sorted({int(round(position * (len(ordered) - 1))) for position in (0.10, 0.40, 0.70, 0.95)})
    selected = ordered.iloc[indices]
    save_examples(output_dir / "Figure_IL_GradCAM_Malignancy_Examples.png", selected, cache, compact=False)
    compact = selected.iloc[[0, -1]] if len(selected) > 1 else selected
    save_examples(output_dir / "Figure_IL_GradCAM_Compact.png", compact, cache, compact=True)

    readme = f"""# 图像级方法 Grad-CAM 输出说明

- 模型：原始图像级拆分训练的 ViT-L/16 多任务分类器。
- 输入：B-mode + Enhanced Pix2Pix 生成的 CEUS。
- 检查点：`{checkpoint_path}`（按验证集 Hamming loss 选择）。
- 评价集：固定的时间内部测试集，共 {len(metrics)} 张图像、{metrics['patient_id'].nunique()} 位患者。
- 完整性校验：重新计算的恶性概率与原存档预测最大绝对差为 {max_difference:.3g}。
- `GradCAM_Per_Case/`：每张测试图像一张四联图。
- `Figure_IL_GradCAM_Malignancy_Examples.png`：按预测风险分位数预先选择的四个示例。
- `Figure_IL_GradCAM_Compact.png`：论文排版用紧凑示例图。
- `Table_IL_GradCAM_Lesion_Overlap_Per_Case.csv`：逐图病灶重叠指标。
- `Table_IL_GradCAM_Lesion_Overlap_Summary.json`：患者聚类 bootstrap 95% CI 汇总。

注意：该图像级模型与患者级补充实验使用的 ResNet18 不是相同骨干网络，因此只能分别报告，不能将 Grad-CAM 外观差异解释为拆分策略本身造成的因果差异。
"""
    (output_dir / "README_CN.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
