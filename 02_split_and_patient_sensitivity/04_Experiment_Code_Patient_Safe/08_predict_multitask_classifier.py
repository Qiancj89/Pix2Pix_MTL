from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import BINARY_TASKS, ECHO_VALUES, LABEL_COLUMNS, assert_patient_disjoint
from data import MultiConditionDataset
from models import MultiTaskClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict every frozen split without model selection on test data.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test_internal_temporal"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--allow-patient-overlap", action="store_true",
                        help="Comparison-only: accept an explicitly audited image-level split.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    condition = checkpoint["condition"]
    frame = pd.read_csv(args.manifest); assert_patient_disjoint(frame, allow_overlap=args.allow_patient_overlap)
    if args.split != "all":
        frame = frame[frame.split.eq(args.split)].copy()
    required = "synthetic_ceus_path" if "synthetic" in condition else "real_ceus_path" if "real_ceus" in condition else None
    if required:
        usable = frame[required].fillna("").map(lambda x: bool(x) and Path(x).is_file())
        if (~usable).any():
            print(f"WARNING: skipping {(~usable).sum()} rows missing {required}; report this explicitly.")
        frame = frame[usable].copy()
    dataset = MultiConditionDataset(frame, condition, int(checkpoint["image_size"]), False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = MultiTaskClassifier(checkpoint["backbone"], False, checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    records = []
    with torch.no_grad():
        for images, targets, metadata in tqdm(loader, desc="predict"):
            outputs = model(images.to(device))
            echo_prob = torch.softmax(outputs["Internal_Echo"], dim=1).cpu()
            binary_prob = {task: torch.sigmoid(outputs[task]).view(-1).cpu() for task in BINARY_TASKS}
            for i in range(images.shape[0]):
                record = {k: metadata[k][i] for k in metadata}
                source = frame.loc[frame.row_uid.astype(str).eq(str(record["row_uid"]))].iloc[0]
                record["split"] = source["split"]; record["condition"] = condition
                record["Internal_Echo_true"] = int(targets["Internal_Echo"][i])
                record["Internal_Echo_pred"] = ECHO_VALUES[int(torch.argmax(echo_prob[i]))]
                for j, value in enumerate(ECHO_VALUES):
                    record[f"Internal_Echo_prob_{value}"] = float(echo_prob[i, j])
                for task in BINARY_TASKS:
                    probability = float(binary_prob[task][i])
                    record[f"{task}_true"] = int(targets[task][i])
                    record[f"{task}_prob"] = probability
                    record[f"{task}_pred"] = int(probability >= 0.5)
                records.append(record)
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(records)} predictions to {output}")


if __name__ == "__main__":
    main()
