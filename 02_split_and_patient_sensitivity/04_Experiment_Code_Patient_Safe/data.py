from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from common import BINARY_TASKS, LABEL_COLUMNS


def _load_rgb(path: str) -> Image.Image:
    if not path or not Path(path).is_file():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def _load_gray(path: str) -> Image.Image:
    if not path or not Path(path).is_file():
        raise FileNotFoundError(path)
    return Image.open(path).convert("L")


class PairedTranslationDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, size: int = 256, augment: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        bmode = _load_gray(str(row.bmode_path)).resize((self.size, self.size), Image.Resampling.BICUBIC)
        ceus = _load_rgb(str(row.real_ceus_path)).resize((self.size, self.size), Image.Resampling.BICUBIC)
        if self.augment:
            if random.random() < 0.5:
                bmode, ceus = TF.hflip(bmode), TF.hflip(ceus)
            angle = random.uniform(-10, 10)
            bmode = TF.rotate(bmode, angle, interpolation=transforms.InterpolationMode.BICUBIC)
            ceus = TF.rotate(ceus, angle, interpolation=transforms.InterpolationMode.BICUBIC)
        bmode_tensor = TF.to_tensor(bmode) * 2 - 1
        ceus_tensor = TF.to_tensor(ceus) * 2 - 1
        return bmode_tensor, ceus_tensor, str(row.row_uid), str(row.patient_id)


class MultiConditionDataset(Dataset):
    CONDITIONS = {"bmode", "synthetic", "bmode_synthetic", "real_ceus", "bmode_real_ceus"}

    def __init__(self, frame: pd.DataFrame, condition: str, size: int = 224, augment: bool = False):
        if condition not in self.CONDITIONS:
            raise ValueError(f"Unknown condition {condition}; choose from {sorted(self.CONDITIONS)}")
        self.frame = frame.reset_index(drop=True)
        self.condition = condition
        self.size = size
        self.augment = augment
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.frame)

    def _transform_pair(self, first: Image.Image | None, second: Image.Image | None):
        images = [x.resize((self.size, self.size), Image.Resampling.BICUBIC) if x is not None else None
                  for x in (first, second)]
        if self.augment:
            flip = random.random() < 0.5
            angle = random.uniform(-10, 10)
            images = [TF.rotate(TF.hflip(x) if flip else x, angle,
                                interpolation=transforms.InterpolationMode.BICUBIC) if x is not None else None
                      for x in images]
        tensors = [self.normalize(TF.to_tensor(x)) if x is not None else torch.zeros(3, self.size, self.size)
                   for x in images]
        return torch.cat(tensors, dim=0)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        bmode = _load_rgb(str(row.bmode_path))
        synthetic = _load_rgb(str(row.synthetic_ceus_path)) if self.condition in {"synthetic", "bmode_synthetic"} else None
        real = _load_rgb(str(row.real_ceus_path)) if self.condition in {"real_ceus", "bmode_real_ceus"} else None
        if self.condition == "bmode":
            image = self._transform_pair(bmode, None)
        elif self.condition == "synthetic":
            image = self._transform_pair(None, synthetic)
        elif self.condition == "bmode_synthetic":
            image = self._transform_pair(bmode, synthetic)
        elif self.condition == "real_ceus":
            image = self._transform_pair(None, real)
        else:
            image = self._transform_pair(bmode, real)
        labels = {name: torch.tensor(row[name], dtype=torch.long if name == "Internal_Echo" else torch.float32)
                  for name in LABEL_COLUMNS}
        metadata = {"row_uid": str(row.row_uid), "patient_id": str(row.patient_id),
                    "lesion_id": str(row.lesion_id), "image_id": str(row.image_id)}
        return image, labels, metadata

