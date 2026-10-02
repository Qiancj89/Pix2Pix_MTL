from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage
from skimage.metrics import structural_similarity


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
METRICS = ("mae", "psnr", "ssim")


def image_map(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    result = {
        path.name: path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if not result:
        raise RuntimeError(f"No images found in {root}")
    return result


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def mask_from_label_image(
    path: Path, target_size: tuple[int, int], lesion_label: int, tolerance: float
) -> tuple[np.ndarray | None, dict[str, object]]:
    with Image.open(path) as image:
        gray_image = ImageOps.exif_transpose(image).convert("L")
        original_size = gray_image.size

    audit: dict[str, object] = {
        "mask_width": original_size[0],
        "mask_height": original_size[1],
        "mask_geometry_matches_real_ceus": original_size == target_size,
        "mask_geometry_transform": "none",
    }
    if original_size != target_size:
        width_difference = abs(original_size[0] - target_size[0])
        height_difference = abs(original_size[1] - target_size[1])
        if width_difference <= 2 and height_difference <= 2:
            gray_image = gray_image.resize(target_size, Image.Resampling.NEAREST)
            audit["mask_geometry_transform"] = "nearest_neighbor_to_real_ceus"
        else:
            audit["mask_status"] = "invalid_geometry_mismatch"
            return None, audit

    gray = np.asarray(gray_image, dtype=np.uint8)
    audit["mask_effective_width"] = gray_image.width
    audit["mask_effective_height"] = gray_image.height
    audit["mask_min"] = int(gray.min())
    audit["mask_max"] = int(gray.max())
    if int(gray.max()) == 0:
        audit["mask_status"] = "invalid_all_zero"
        return None, audit

    # The source masks are JPEG-compressed label maps: lesion=1 and background=2.
    # Compression can move lesion boundary pixels down to 0, so the threshold is
    # intentionally asymmetric. It includes 0/1 but excludes background label 2.
    mask = gray.astype(np.float32) <= float(lesion_label) + tolerance
    if not mask.any():
        audit["mask_status"] = "invalid_no_lesion_label"
        return None, audit

    components, component_count = ndimage.label(mask)
    sizes = np.bincount(components.ravel())[1:]
    largest_label = int(np.argmax(sizes) + 1)
    largest = components == largest_label
    retained_fraction = float(largest.sum() / mask.sum())
    audit.update(
        {
            "mask_status": "valid"
            if audit["mask_geometry_transform"] == "none"
            else "valid_resized_nearest",
            "raw_component_count": int(component_count),
            "largest_component_retained_fraction": retained_fraction,
            "roi_pixels": int(largest.sum()),
            "roi_fraction": float(largest.mean()),
        }
    )
    return largest, audit


def psnr_from_mse(mse: float) -> float:
    if mse == 0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def calculate_metrics(
    real: np.ndarray, generated: np.ndarray, roi: np.ndarray | None
) -> tuple[dict[str, float], dict[str, float] | None]:
    difference = real - generated
    whole_mae = float(np.abs(difference).mean())
    whole_mse = float(np.square(difference).mean())
    whole_ssim, ssim_map = structural_similarity(
        real,
        generated,
        channel_axis=2,
        data_range=1.0,
        full=True,
    )
    whole = {
        "mae": whole_mae,
        "psnr": psnr_from_mse(whole_mse),
        "ssim": float(whole_ssim),
    }
    if roi is None:
        return whole, None

    roi_difference = difference[roi]
    roi_mae = float(np.abs(roi_difference).mean())
    roi_mse = float(np.square(roi_difference).mean())
    roi_ssim = float(ssim_map[roi].mean())
    lesion_real_mean = float(real[roi].mean())
    lesion_generated_mean = float(generated[roi].mean())
    roi_metrics = {
        "mae": roi_mae,
        "psnr": psnr_from_mse(roi_mse),
        "ssim": roi_ssim,
        "real_mean_intensity": lesion_real_mean,
        "generated_mean_intensity": lesion_generated_mean,
        "mean_intensity_error": lesion_generated_mean - lesion_real_mean,
    }
    return whole, roi_metrics


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    x1, x2 = int(cols.min()), int(cols.max()) + 1
    y1, y2 = int(rows.min()), int(rows.max()) + 1
    return x1, y1, x2 - x1, y2 - y1


def save_overlay(
    real_image: Image.Image,
    generated_image: Image.Image,
    mask: np.ndarray,
    path: Path,
) -> None:
    boundary = mask & ~ndimage.binary_erosion(mask, iterations=1)
    panels: list[Image.Image] = []
    for source in (real_image, generated_image):
        panel = np.asarray(source, dtype=np.uint8).copy()
        panel[boundary] = np.array([255, 0, 0], dtype=np.uint8)
        panels.append(Image.fromarray(panel))
    canvas = Image.new("RGB", (real_image.width * 2, real_image.height), "black")
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (real_image.width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def cluster_bootstrap_ci(
    frame: pd.DataFrame,
    value_column: str,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    finite = frame[np.isfinite(frame[value_column].astype(float))].copy()
    if finite.empty:
        return float("nan"), float("nan")
    grouped = {
        patient: group[value_column].astype(float).to_numpy()
        for patient, group in finite.groupby("patient_id", sort=False)
    }
    patients = np.asarray(list(grouped), dtype=object)
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = rng.choice(patients, size=len(patients), replace=True)
        sampled = np.concatenate([grouped[patient] for patient in selected])
        means[index] = sampled.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def make_summary(
    per_image: pd.DataFrame, iterations: int, seed: int
) -> pd.DataFrame:
    subsets = {
        "train": per_image[per_image["split"].eq("train")],
        "validation": per_image[per_image["split"].eq("val")],
        "development_combined": per_image[per_image["split"].isin(["train", "val"])],
        "test_internal_temporal": per_image[
            per_image["split"].eq("test_internal_temporal")
        ],
    }
    rows: list[dict[str, object]] = []
    for subset_name, subset in subsets.items():
        if subset.empty:
            continue
        for region, prefix in (("whole", "whole"), ("lesion_roi", "roi")):
            region_frame = subset
            if region == "lesion_roi":
                region_frame = subset[subset["mask_status"].str.startswith("valid")]
            for metric_index, metric in enumerate(METRICS):
                column = f"{prefix}_{metric}"
                valid = region_frame[np.isfinite(region_frame[column].astype(float))]
                ci_low, ci_high = cluster_bootstrap_ci(
                    valid,
                    column,
                    iterations,
                    seed + metric_index + (100 if region == "lesion_roi" else 0),
                )
                rows.append(
                    {
                        "subset": subset_name,
                        "region": region,
                        "metric": metric,
                        "mean": float(valid[column].mean()),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n_images": int(len(valid)),
                        "n_patients": int(valid["patient_id"].nunique()),
                        "excluded_mask_images": int(
                            len(subset)
                            - subset["mask_status"].str.startswith("valid").sum()
                        )
                        if region == "lesion_roi"
                        else 0,
                        "bootstrap_unit": "patient",
                        "bootstrap_iterations": iterations,
                    }
                )
    return pd.DataFrame(rows)


def save_summary_figure(summary: pd.DataFrame, path: Path) -> None:
    subset_order = ["train", "validation", "development_combined", "test_internal_temporal"]
    labels = ["Train", "Validation", "Development", "Test"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = {"whole": "#4C78A8", "lesion_roi": "#E45756"}
    for axis, metric in zip(axes, METRICS):
        x = np.arange(len(subset_order), dtype=float)
        width = 0.36
        for offset, region in ((-width / 2, "whole"), (width / 2, "lesion_roi")):
            values, low, high = [], [], []
            for subset in subset_order:
                row = summary[
                    summary["subset"].eq(subset)
                    & summary["region"].eq(region)
                    & summary["metric"].eq(metric)
                ].iloc[0]
                values.append(row["mean"])
                low.append(row["mean"] - row["ci_low"])
                high.append(row["ci_high"] - row["mean"])
            axis.bar(
                x + offset,
                values,
                width,
                color=colors[region],
                label="Whole image" if region == "whole" else "Lesion ROI",
                yerr=np.vstack([low, high]),
                capsize=3,
                alpha=0.9,
            )
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_title(metric.upper())
        axis.grid(axis="y", alpha=0.25)
        if metric == "psnr":
            axis.set_ylabel("dB")
    axes[0].legend(frameon=False)
    fig.suptitle("Whole-image and geometrically aligned lesion-ROI agreement")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_qc_montage(per_image: pd.DataFrame, root: Path, path: Path) -> None:
    selected: list[pd.Series] = []
    for dataset in ("development", "test"):
        valid = per_image[
            per_image["dataset"].eq(dataset)
            & per_image["mask_status"].str.startswith("valid")
        ].sort_values("roi_fraction")
        for position in (0, len(valid) // 2, len(valid) - 1):
            selected.append(valid.iloc[position])

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for axis, row in zip(axes.ravel(), selected):
        overlay_path = root / row["overlay_path"]
        axis.imshow(Image.open(overlay_path))
        axis.set_title(
            f"{row['dataset']} | {row['image_name']} | ROI {row['roi_fraction']:.1%}\n"
            "Real CEUS (left) | Generated CEUS (right)"
        )
        axis.axis("off")
    fig.suptitle("Geometric alignment QC: red line is the lesion-mask boundary")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute whole-image and geometrically aligned lesion-ROI CEUS metrics."
    )
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parent
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = (config_path.parent / config["manifest"]).resolve()
    manifest = pd.read_csv(manifest_path)
    manifest["image_name"] = manifest["real_ceus_path"].map(lambda value: Path(value).name)
    if manifest["image_name"].duplicated().any():
        raise RuntimeError("Manifest image basenames are not unique")
    metadata = manifest.set_index("image_name", drop=False)

    resized_root = script_root / "resized_generated"
    binary_mask_root = script_root / "binary_lesion_masks"
    overlay_root = script_root / "roi_overlays"
    results_root = script_root / "results"
    figures_root = script_root / "figures"
    for directory in (resized_root, binary_mask_root, overlay_root, results_root, figures_root):
        directory.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []

    lesion_label = int(config["mask_lesion_label"])
    tolerance = float(config["mask_jpeg_tolerance"])

    for dataset_name, dataset_config in config["datasets"].items():
        roots = {
            key: Path(dataset_config[key]).resolve()
            for key in ("bmode_dir", "real_ceus_dir", "mask_dir", "generated_ceus_dir")
        }
        maps = {key: image_map(path) for key, path in roots.items()}
        reference_names = set(maps["real_ceus_dir"])
        expected = int(dataset_config["expected_images"])
        if len(reference_names) != expected:
            raise RuntimeError(
                f"{dataset_name}: expected {expected} real CEUS images, found {len(reference_names)}"
            )
        for source_name, source_map in maps.items():
            missing = sorted(reference_names - set(source_map))
            extra = sorted(set(source_map) - reference_names)
            mapping_rows.append(
                {
                    "dataset": dataset_name,
                    "source": source_name,
                    "directory": str(roots[source_name]),
                    "n_files": len(source_map),
                    "missing_vs_real_ceus": len(missing),
                    "extra_vs_real_ceus": len(extra),
                    "missing_examples": "|".join(missing[:10]),
                    "extra_examples": "|".join(extra[:10]),
                }
            )
            if missing or extra:
                raise RuntimeError(
                    f"{dataset_name}/{source_name} is not one-to-one with real CEUS: "
                    f"missing={missing[:10]}, extra={extra[:10]}"
                )

        for image_name in sorted(reference_names):
            if image_name not in metadata.index:
                raise RuntimeError(f"No manifest metadata for {image_name}")
            meta = metadata.loc[image_name]
            if meta["split"] not in dataset_config["allowed_splits"]:
                raise RuntimeError(
                    f"Unexpected split for {image_name}: {meta['split']} not in "
                    f"{dataset_config['allowed_splits']}"
                )

            bmode_path = maps["bmode_dir"][image_name]
            real_path = maps["real_ceus_dir"][image_name]
            mask_path = maps["mask_dir"][image_name]
            generated_path = maps["generated_ceus_dir"][image_name]

            bmode_image = load_rgb(bmode_path)
            real_image = load_rgb(real_path)
            generated_original = load_rgb(generated_path)
            generated_resized = generated_original.resize(
                real_image.size, Image.Resampling.LANCZOS
            )
            resized_path = resized_root / dataset_name / image_name
            resized_path.parent.mkdir(parents=True, exist_ok=True)
            generated_resized.save(
                resized_path,
                format="JPEG",
                quality=95,
                subsampling=0,
                optimize=True,
            )
            generated_for_metrics = load_rgb(resized_path)

            roi, mask_audit = mask_from_label_image(
                mask_path, real_image.size, lesion_label, tolerance
            )
            if roi is not None:
                binary_path = binary_mask_root / dataset_name / f"{Path(image_name).stem}.png"
                binary_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(roi.astype(np.uint8) * 255).save(binary_path)
                overlay_path = overlay_root / dataset_name / f"{Path(image_name).stem}.png"
                save_overlay(real_image, generated_for_metrics, roi, overlay_path)
                x, y, width, height = bbox_from_mask(roi)
            else:
                binary_path = None
                overlay_path = None
                x = y = width = height = None
                invalid_rows.append(
                    {
                        "dataset": dataset_name,
                        "split": meta["split"],
                        "row_uid": meta["row_uid"],
                        "patient_id": meta["patient_id"],
                        "image_name": image_name,
                        "mask_path": str(mask_path),
                        "mask_status": mask_audit["mask_status"],
                        "mask_width": mask_audit["mask_width"],
                        "mask_height": mask_audit["mask_height"],
                        "real_ceus_width": real_image.width,
                        "real_ceus_height": real_image.height,
                    }
                )

            real = image_array(real_image)
            generated = image_array(generated_for_metrics)
            whole, roi_metrics = calculate_metrics(real, generated, roi)
            record: dict[str, object] = {
                "dataset": dataset_name,
                "split": meta["split"],
                "row_uid": meta["row_uid"],
                "patient_id": meta["patient_id"],
                "image_name": image_name,
                "bmode_path": str(bmode_path),
                "real_ceus_path": str(real_path),
                "mask_path": str(mask_path),
                "generated_source_path": str(generated_path),
                "resized_generated_path": resized_path.relative_to(script_root).as_posix(),
                "binary_mask_path": binary_path.relative_to(script_root).as_posix()
                if binary_path
                else "",
                "overlay_path": overlay_path.relative_to(script_root).as_posix()
                if overlay_path
                else "",
                "bmode_width": bmode_image.width,
                "bmode_height": bmode_image.height,
                "real_ceus_width": real_image.width,
                "real_ceus_height": real_image.height,
                "generated_source_width": generated_original.width,
                "generated_source_height": generated_original.height,
                "generated_resized_width": generated_for_metrics.width,
                "generated_resized_height": generated_for_metrics.height,
                **mask_audit,
                "roi_bbox_x": x,
                "roi_bbox_y": y,
                "roi_bbox_width": width,
                "roi_bbox_height": height,
                "whole_mae": whole["mae"],
                "whole_psnr": whole["psnr"],
                "whole_ssim": whole["ssim"],
                "roi_mae": roi_metrics["mae"] if roi_metrics else np.nan,
                "roi_psnr": roi_metrics["psnr"] if roi_metrics else np.nan,
                "roi_ssim": roi_metrics["ssim"] if roi_metrics else np.nan,
                "roi_real_mean_intensity": roi_metrics["real_mean_intensity"]
                if roi_metrics
                else np.nan,
                "roi_generated_mean_intensity": roi_metrics["generated_mean_intensity"]
                if roi_metrics
                else np.nan,
                "roi_mean_intensity_error": roi_metrics["mean_intensity_error"]
                if roi_metrics
                else np.nan,
            }
            records.append(record)

    per_image = pd.DataFrame(records)
    per_image.to_csv(
        results_root / "roi_metrics_per_image.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(mapping_rows).to_csv(
        results_root / "dataset_mapping_audit.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(invalid_rows).to_csv(
        results_root / "excluded_or_invalid_masks.csv", index=False, encoding="utf-8-sig"
    )

    summary = make_summary(
        per_image,
        int(config["bootstrap_iterations"]),
        int(config["bootstrap_seed"]),
    )
    summary.to_csv(
        results_root / "roi_metrics_summary.csv", index=False, encoding="utf-8-sig"
    )
    save_summary_figure(summary, figures_root / "whole_vs_lesion_roi_metrics.png")
    save_qc_montage(
        per_image, script_root, figures_root / "geometric_alignment_qc_montage.png"
    )

    summary_json = {
        "input_images": int(len(per_image)),
        "valid_roi_images": int(
            per_image["mask_status"].str.startswith("valid").sum()
        ),
        "invalid_roi_images": int(
            (~per_image["mask_status"].str.startswith("valid")).sum()
        ),
        "development_images": int(per_image["dataset"].eq("development").sum()),
        "test_images": int(per_image["dataset"].eq("test").sum()),
        "mask_rule": f"pixel <= {lesion_label}+{tolerance}; retain largest component",
        "generated_resize": "Pillow LANCZOS to corresponding real CEUS width and height; JPEG quality 95, subsampling 0",
        "bootstrap": {
            "unit": "patient",
            "iterations": int(config["bootstrap_iterations"]),
            "seed": int(config["bootstrap_seed"]),
        },
        "lpips": "not computed: local PyTorch DLL blocked by Windows application-control policy",
    }
    (results_root / "analysis_metadata.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Processed images: {len(per_image)}")
    print(f"Valid ROI masks: {summary_json['valid_roi_images']}")
    print(f"Invalid ROI masks: {summary_json['invalid_roi_images']}")
    print(f"Summary: {results_root / 'roi_metrics_summary.csv'}")
    print(f"QC montage: {figures_root / 'geometric_alignment_qc_montage.png'}")


if __name__ == "__main__":
    main()
