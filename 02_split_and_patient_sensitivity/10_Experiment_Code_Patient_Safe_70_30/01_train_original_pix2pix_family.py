from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_module(source: Path, variant: str):
    spec = importlib.util.spec_from_file_location(f"original_pix2pix_{variant}", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import original source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class ManifestPairDataset(Dataset):
    """Patient-safe paired dataset that preserves the submitted model preprocessing."""

    def __init__(self, frame: pd.DataFrame, augmentation_class, augment: bool, augment_factor: int):
        self.rows = frame.reset_index(drop=True)
        self.augment = bool(augment)
        self.augment_factor = int(augment_factor) if augment else 1
        if self.augment_factor < 1:
            raise ValueError("augment_factor must be at least 1")
        self.medical_augmentation = augmentation_class(prob=0.7) if augment else None
        self.bmode_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.ceus_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.rows) * self.augment_factor

    def __getitem__(self, index: int):
        row = self.rows.iloc[index // self.augment_factor]
        with Image.open(row["bmode_path"]) as source:
            bmode = source.convert("L")
        with Image.open(row["real_ceus_path"]) as source:
            ceus = source.convert("RGB")
        if self.medical_augmentation is not None:
            bmode, ceus = self.medical_augmentation(bmode, ceus)
        return self.bmode_transform(bmode), self.ceus_transform(ceus)


def only_supported(function, values: dict) -> dict:
    parameters = inspect.signature(function).parameters
    return {key: value for key, value in values.items() if key in parameters}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the submitted root Pix2Pix implementation on a frozen patient-level manifest."
    )
    parser.add_argument("--variant", required=True)
    parser.add_argument("--registry", default="model_registry.json")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--augment-factor", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The submitted root implementations use CUDA autocast/GradScaler directly. "
            "A CUDA GPU is required; do not silently substitute a CPU implementation."
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
    frame = pd.read_csv(manifest_path, dtype={"patient_id": str, "row_uid": str})
    required = {"split", "patient_id", "row_uid", "bmode_path", "real_ceus_path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Manifest missing columns: {missing}")
    train = frame.loc[frame["split"].eq("train")].copy()
    val = frame.loc[frame["split"].eq("val")].copy()
    if train.empty or val.empty:
        raise RuntimeError("Both train and val rows are required.")
    overlap = set(train["patient_id"]) & set(val["patient_id"])
    if overlap:
        raise RuntimeError(f"Patient leakage detected between train and val: {sorted(overlap)[:20]}")
    for name, subset in (("train", train), ("val", val)):
        bad = subset.loc[
            ~subset["bmode_path"].map(lambda value: Path(str(value)).is_file())
            | ~subset["real_ceus_path"].map(lambda value: Path(str(value)).is_file())
        ]
        if not bad.empty:
            raise RuntimeError(f"{name} contains {len(bad)} missing B-mode/CEUS pairs.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_generator.pth"
    if checkpoint.exists() and not args.overwrite:
        raise FileExistsError(f"Checkpoint already exists: {checkpoint}. Use --overwrite only intentionally.")

    seed_everything(args.seed)
    module = load_source_module(source, args.variant)
    module.device = torch.device("cuda")
    augmentation_name = str(entry["augmentation_class"])
    try:
        augmentation_class = getattr(module, augmentation_name)
    except AttributeError as error:
        raise RuntimeError(
            f"Registry mismatch for {args.variant}: {source.name} does not define "
            f"augmentation class {augmentation_name!r}."
        ) from error
    train_dataset = ManifestPairDataset(train, augmentation_class, augment=True, augment_factor=args.augment_factor)
    val_dataset = ManifestPairDataset(val, augmentation_class, augment=False, augment_factor=1)
    batch_size = int(args.batch_size or entry["batch_size"])
    data_generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "generator": data_generator,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_options)

    generator_class = getattr(module, entry["generator_class"])
    discriminator_class = getattr(module, entry["discriminator_class"])
    generator = generator_class(**only_supported(generator_class, {
        "in_channels": 1, "out_channels": 3, "features": 64, "use_attention": True,
    })).to(module.device)
    discriminator = discriminator_class(**only_supported(discriminator_class, {
        "in_channels": 4, "features": 64, "use_attention": True,
    })).to(module.device)

    train_function = getattr(module, entry["train_function"])
    train_arguments = {
        "generator": generator,
        "discriminator": discriminator,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "num_epochs": args.epochs,
        "lr_g": 2e-4,
        "lr_d": 2e-4,
        "save_path": str(checkpoint),
        "patience": args.patience,
        "min_delta": 0.001,
        **entry.get("loss_weights", {}),
    }
    provenance = {
        "variant": args.variant,
        "role": entry["role"],
        "original_source": str(source),
        "original_source_sha256": sha256(source),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "seed": args.seed,
        "train_patients": int(train["patient_id"].nunique()),
        "val_patients": int(val["patient_id"].nunique()),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "batch_size": batch_size,
        "augment_factor": args.augment_factor,
        "epochs": args.epochs,
        "patience": args.patience,
        "training_function": entry["train_function"],
        "training_arguments": {
            key: value for key, value in only_supported(train_function, train_arguments).items()
            if key not in {"generator", "discriminator", "train_loader", "val_loader"}
        },
    }
    (output_dir / "run_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Variant: {args.variant}")
    print(f"Original source: {source}")
    print(f"Train: {train['patient_id'].nunique()} patients / {len(train)} rows")
    print(f"Validation: {val['patient_id'].nunique()} patients / {len(val)} rows")
    train_function(**only_supported(train_function, train_arguments))
    if not checkpoint.is_file():
        raise RuntimeError(f"Training returned without writing the expected checkpoint: {checkpoint}")
    provenance["checkpoint"] = str(checkpoint)
    provenance["checkpoint_sha256"] = sha256(checkpoint)
    (output_dir / "TRAINING_COMPLETE.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Completed checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
