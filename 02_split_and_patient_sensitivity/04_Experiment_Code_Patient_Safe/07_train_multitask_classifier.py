from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import BINARY_TASKS, ECHO_VALUES, assert_patient_disjoint, json_dump, seed_everything
from data import MultiConditionDataset
from models import MultiTaskClassifier, MultiTaskCriterion


def weights_from_training_rows(frame: pd.DataFrame):
    echo_counts = frame["Internal_Echo"].value_counts()
    echo_weights = [len(frame) / max(4 * echo_counts.get(value, 0), 1) for value in ECHO_VALUES]
    pos_weights = {}
    for task in BINARY_TASKS:
        positives = float(frame[task].sum())
        negatives = float(len(frame) - positives)
        pos_weights[task] = negatives / max(positives, 1.0)
    return echo_weights, pos_weights


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training); total = 0.0; malignant_y = []; malignant_p = []
    for images, targets, _ in tqdm(loader, desc="train" if training else "val"):
        images = images.to(device); targets = {k: v.to(device) for k, v in targets.items()}
        with torch.set_grad_enabled(training):
            outputs = model(images); loss = criterion(outputs, targets)
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        total += loss.item()
        malignant_y.extend(targets["Malignant"].detach().cpu().numpy().tolist())
        malignant_p.extend(torch.sigmoid(outputs["Malignant"]).detach().cpu().view(-1).numpy().tolist())
    metrics = {"loss": total / len(loader)}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        metrics["malignant_roc_auc"] = float(roc_auc_score(malignant_y, malignant_p))
        metrics["malignant_pr_auc"] = float(average_precision_score(malignant_y, malignant_p))
    except (ImportError, ValueError):
        metrics["malignant_roc_auc"] = None; metrics["malignant_pr_auc"] = None
    return metrics


def train_one(frame, condition, args, device, overlapping_patient_count=0):
    required = "synthetic_ceus_path" if "synthetic" in condition else "real_ceus_path" if "real_ceus" in condition else None
    usable = frame.copy()
    if required:
        usable = usable[usable[required].fillna("").ne("") & usable[required].map(lambda x: Path(x).is_file())]
    train, val = usable[usable.split.eq("train")], usable[usable.split.eq("val")]
    if train.empty or val.empty:
        raise RuntimeError(f"No usable train/val rows for {condition}; missing column/files: {required}")
    loaders = {
        "train": DataLoader(MultiConditionDataset(train, condition, args.image_size, True), batch_size=args.batch_size,
                            shuffle=True, num_workers=args.workers),
        "val": DataLoader(MultiConditionDataset(val, condition, args.image_size, False), batch_size=args.batch_size,
                          shuffle=False, num_workers=args.workers),
    }
    echo_weights, pos_weights = weights_from_training_rows(train)
    model = MultiTaskClassifier(args.backbone, args.pretrained, args.dropout).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    criterion = MultiTaskCriterion(args.loss_mode, echo_weights, pos_weights, args.focal_gamma).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    output = Path(args.output_dir).resolve() / condition / args.loss_mode
    output.mkdir(parents=True, exist_ok=True)
    json_dump({"condition": condition, "loss_mode": args.loss_mode, "backbone": args.backbone,
               "parameter_count": parameter_count, "train_rows": len(train), "val_rows": len(val),
               "train_patients": train.patient_id.nunique(), "val_patients": val.patient_id.nunique(),
               "manifest": str(Path(args.manifest).resolve()), "seed": args.seed,
               "allow_patient_overlap": bool(args.allow_patient_overlap),
               "overlapping_patient_count": int(overlapping_patient_count),
               "six_channel_equal_capacity_design": True}, output / "run_config.json")
    history, best, stale = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_metrics = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step(val_metrics["loss"])
        record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                  **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record); json_dump(history, output / "history.json"); print(condition, record)
        if val_metrics["loss"] < best:
            best, stale = val_metrics["loss"], 0
            torch.save({"model": model.state_dict(), "backbone": args.backbone, "condition": condition,
                        "loss_mode": args.loss_mode, "pretrained": args.pretrained, "dropout": args.dropout,
                        "image_size": args.image_size, "manifest": str(Path(args.manifest).resolve()),
                        "seed": args.seed, "echo_weights": echo_weights, "pos_weights": pos_weights,
                        "parameter_count": parameter_count,
                        "epoch": epoch, "val_metrics": val_metrics}, output / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Train equal-capacity six-channel models for controlled input comparisons.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition", default="all", choices=["all", *sorted(MultiConditionDataset.CONDITIONS)])
    parser.add_argument("--loss-mode", default="unweighted", choices=["unweighted", "class_weighted", "focal"])
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50", "efficientnet_b0",
                                                                    "efficientnet_b7", "vit_b_16", "vit_l_32"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--allow-patient-overlap", action="store_true",
                        help="Comparison-only: accept an explicitly audited image-level split.")
    args = parser.parse_args()
    seed_everything(args.seed)
    frame = pd.read_csv(args.manifest)
    overlap = assert_patient_disjoint(frame, allow_overlap=args.allow_patient_overlap)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conditions = sorted(MultiConditionDataset.CONDITIONS) if args.condition == "all" else [args.condition]
    for condition in conditions:
        train_one(frame, condition, args, device, len(overlap))


if __name__ == "__main__":
    main()
