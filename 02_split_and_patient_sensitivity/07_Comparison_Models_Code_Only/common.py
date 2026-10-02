from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path: Union[str, Path], allow_patient_overlap: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"row_uid", "patient_id", "split", "bmode_path", "real_ceus_path"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Manifest is missing columns: {sorted(missing)}")
    if "include_in_analysis" in frame:
        included = pd.to_numeric(frame["include_in_analysis"], errors="coerce").fillna(0).eq(1)
        frame = frame[included].copy()
    train_patients = set(frame.loc[frame.split.eq("train"), "patient_id"].astype(str))
    val_patients = set(frame.loc[frame.split.eq("val"), "patient_id"].astype(str))
    overlap = sorted(train_patients & val_patients)
    if overlap and not allow_patient_overlap:
        raise RuntimeError(
            f"Detected {len(overlap)} train/validation patient overlaps. "
            "Use --allow-patient-overlap only for the documented original image-level comparison."
        )
    frame.attrs["overlapping_patient_ids"] = overlap
    return frame


class PairedManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int, augment: bool = False):
        valid = frame["bmode_path"].fillna("").ne("") & frame["real_ceus_path"].fillna("").ne("")
        self.frame = frame[valid].reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        bmode = Image.open(row.bmode_path).convert("L")
        ceus = Image.open(row.real_ceus_path).convert("RGB")
        bmode = TF.resize(bmode, [self.image_size, self.image_size], antialias=True)
        ceus = TF.resize(ceus, [self.image_size, self.image_size], antialias=True)
        if self.augment and random.random() < 0.5:
            bmode = TF.hflip(bmode)
            ceus = TF.hflip(ceus)
        bmode_tensor = TF.to_tensor(bmode).mul(2).sub(1)
        ceus_tensor = TF.to_tensor(ceus).mul(2).sub(1)
        return bmode_tensor, ceus_tensor, str(row.row_uid), str(row.patient_id)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(-1, 1).add(1).div(2)
    return TF.to_pil_image(tensor)
