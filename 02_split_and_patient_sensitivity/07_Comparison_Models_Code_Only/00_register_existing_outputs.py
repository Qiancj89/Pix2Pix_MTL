from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import IMAGE_SUFFIXES, save_json


def image_index(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        raise RuntimeError(f"Existing output directory not found: {directory}")
    result = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            key = path.stem.casefold()
            if key in result:
                raise RuntimeError(f"Duplicate image stem in {directory}: {path.stem}")
            result[key] = str(path.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register existing original-split main/basic/ablation outputs without retraining or inference."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--registry", default=str(Path(__file__).with_name("model_registry.json")))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-ablation", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    is_original = (
        "comparison_only_allow_patient_overlap" in frame.columns
        and pd.to_numeric(frame["comparison_only_allow_patient_overlap"], errors="coerce").fillna(0).eq(1).all()
    )
    if not is_original:
        raise RuntimeError(
            "Existing root checkpoints/outputs belong to the documented original image-level workflow. "
            "They must not be registered as patient-level results."
        )

    repo = Path(args.repo_root).resolve()
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    selected = {
        name: item for name, item in registry.items()
        if item["role"] != "ablation" or args.include_ablation
    }
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit = []

    for model_name, item in selected.items():
        checkpoint = repo / item["checkpoint"]
        source = repo / item["source"]
        if not checkpoint.is_file() or not source.is_file():
            raise RuntimeError(f"Missing source/checkpoint for {model_name}: {source}; {checkpoint}")
        val_index = image_index(repo / item["val_images"])
        test_index = image_index(repo / item["test_images"])
        column = f"generated_{model_name}_path"
        values, missing = [], []
        for row in frame.itertuples(index=False):
            split = str(row.split)
            stem = Path(str(row.bmode_path)).stem.casefold()
            path = val_index.get(stem, "") if split == "val" else test_index.get(stem, "") if split == "test_internal_temporal" else ""
            values.append(path)
            if split in {"val", "test_internal_temporal"} and not path:
                missing.append({"model": model_name, "split": split, "row_uid": str(row.row_uid), "stem": stem})
        frame[column] = values
        audit.append({
            "model": model_name, "role": item["role"], "source": str(source),
            "checkpoint": str(checkpoint), "registered": int(pd.Series(values).ne("").sum()),
            "missing": len(missing), "missing_rows": missing,
        })
        if missing:
            raise RuntimeError(f"{model_name} is missing {len(missing)} expected validation/test outputs; see source folders.")

    frame.to_csv(output / "original_manifest_with_existing_outputs.csv", index=False, encoding="utf-8-sig")
    save_json(audit, output / "existing_outputs_audit.json")
    print(f"Registered {len(selected)} existing models in {output}")


if __name__ == "__main__":
    main()
