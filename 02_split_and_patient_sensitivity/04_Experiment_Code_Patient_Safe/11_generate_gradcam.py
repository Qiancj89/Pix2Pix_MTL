from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from data import MultiConditionDataset
from models import MultiTaskClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate auditable Grad-CAM maps for prespecified cases.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases-csv", required=True, help="Must contain row_uid; optional task and reason columns.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = MultiTaskClassifier(checkpoint["backbone"], False, checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    manifest = pd.read_csv(args.manifest); cases = pd.read_csv(args.cases_csv)
    if "row_uid" not in cases.columns:
        raise RuntimeError("cases CSV must contain row_uid")
    selected = cases.merge(manifest, on="row_uid", how="left", validate="one_to_one")
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    state = {}

    def hook(_module, _inputs, activation):
        state["activation"] = activation
        activation.retain_grad()

    handle = model.gradcam_layer.register_forward_hook(hook)
    try:
        for _, row in selected.iterrows():
            task = str(row.get("task", "Malignant"))
            dataset = MultiConditionDataset(pd.DataFrame([row]), checkpoint["condition"], checkpoint["image_size"], False)
            image, _, _ = dataset[0]; image = image.unsqueeze(0).to(device)
            model.zero_grad(set_to_none=True); outputs = model(image)
            if task == "Internal_Echo":
                class_index = int(torch.argmax(outputs[task], dim=1).item()); score = outputs[task][0, class_index]
            else:
                score = outputs[task].view(-1)[0]
            score.backward()
            activation = state["activation"]; gradient = activation.grad
            weights = gradient.mean(dim=tuple(range(2, gradient.ndim)), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=(checkpoint["image_size"], checkpoint["image_size"]), mode="bilinear", align_corners=False)
            cam = cam.squeeze().detach().cpu().numpy(); cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-8)
            base = Image.open(row.bmode_path).convert("RGB").resize((checkpoint["image_size"], checkpoint["image_size"]))
            fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
            axes[0].imshow(base); axes[0].set_title("B-mode")
            axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1); axes[1].set_title(f"Grad-CAM: {task}")
            axes[2].imshow(base); axes[2].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1); axes[2].set_title("Overlay")
            for axis in axes: axis.axis("off")
            fig.suptitle(f"{row.row_uid} | {row.get('reason', '')}"); fig.tight_layout()
            fig.savefig(output / f"{row.row_uid}_{task}.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    finally:
        handle.remove()
    print(f"Saved Grad-CAM figures to {output}")


if __name__ == "__main__":
    main()

