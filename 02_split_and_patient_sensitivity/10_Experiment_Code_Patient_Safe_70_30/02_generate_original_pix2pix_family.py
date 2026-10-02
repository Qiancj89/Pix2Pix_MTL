from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_module(source: Path, variant: str):
    spec = importlib.util.spec_from_file_location(f"original_pix2pix_generate_{variant}", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import original source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def supported_kwargs(callable_object, values: dict) -> dict:
    parameters = inspect.signature(callable_object).parameters
    return {key: value for key, value in values.items() if key in parameters}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CEUS images with a patient-level checkpoint from the submitted root implementation."
    )
    parser.add_argument("--variant", required=True)
    parser.add_argument("--registry", default="model_registry.json")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test_internal_temporal"],
        choices=["train", "val", "test_internal_temporal"],
    )
    parser.add_argument("--no-sharpening", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The submitted root generation functions use CUDA autocast directly. "
            "A CUDA GPU is required."
        )

    repo_root = Path(args.repo_root).resolve()
    registry_path = Path(args.registry).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    if args.variant not in registry:
        raise KeyError(f"Unknown variant {args.variant!r}; choices: {sorted(registry)}")
    entry = registry[args.variant]
    source = (repo_root / entry["source"]).resolve()
    if source.parent != repo_root or not source.is_file():
        raise RuntimeError(f"Original root source is missing or outside the repository root: {source}")

    manifest_path = Path(args.manifest).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    frame = pd.read_csv(manifest_path, dtype={"patient_id": str, "row_uid": str})
    required = {"row_uid", "patient_id", "split", "bmode_path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Manifest missing columns: {missing}")
    generated_column = str(entry["generated_column"])
    output_manifest = Path(args.output_manifest).resolve()
    if output_manifest.exists() and not args.overwrite:
        raise FileExistsError(f"Output manifest already exists: {output_manifest}")

    module = load_source_module(source, args.variant)
    module.device = torch.device("cuda")
    generator_class = getattr(module, entry["generator_class"])
    generator = generator_class(**supported_kwargs(generator_class, {
        "in_channels": 1, "out_channels": 3, "features": 64, "use_attention": True,
    })).to(module.device)
    checkpoint = torch.load(checkpoint_path, map_location=module.device, weights_only=False)
    state = checkpoint.get("generator_state_dict", checkpoint)
    generator.load_state_dict(state, strict=True)
    generator.eval()
    generation_function = getattr(module, entry["generation_function"])

    output_dir = Path(args.output_dir).resolve()
    generated_paths: dict[str, str] = {}
    selected = frame.loc[frame["split"].isin(args.splits)].copy()
    if selected.empty:
        raise RuntimeError(f"No rows found for requested splits: {args.splits}")

    for row in tqdm(selected.itertuples(index=False), total=len(selected), desc=args.variant):
        source_bmode = Path(str(row.bmode_path))
        if not source_bmode.is_file():
            raise FileNotFoundError(source_bmode)
        split_dir = output_dir / str(row.split)
        split_dir.mkdir(parents=True, exist_ok=True)
        destination = split_dir / f"{row.row_uid}.png"
        if destination.exists() and not args.overwrite:
            generated_paths[str(row.row_uid)] = str(destination)
            continue
        call_values = {
            "generator": generator,
            "bmode_path": str(source_bmode),
            "output_path": str(destination),
            "apply_sharpening": not args.no_sharpening,
        }
        generation_function(**supported_kwargs(generation_function, call_values))
        if not destination.is_file():
            raise RuntimeError(f"Generation function did not write: {destination}")
        generated_paths[str(row.row_uid)] = str(destination)

    result = frame.copy()
    result[generated_column] = result["row_uid"].astype(str).map(generated_paths).fillna("")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_manifest, index=False, encoding="utf-8-sig")
    provenance = {
        "variant": args.variant,
        "role": entry["role"],
        "generated_column": generated_column,
        "original_source": str(source),
        "original_source_sha256": sha256(source),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256(manifest_path),
        "output_manifest": str(output_manifest),
        "requested_splits": args.splits,
        "generated_images": len(generated_paths),
        "sharpening": not args.no_sharpening,
    }
    (output_dir / "GENERATION_COMPLETE.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Generated {len(generated_paths)} images for {args.variant}.")
    print(f"Manifest: {output_manifest}")


if __name__ == "__main__":
    main()
