from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Iterable, Sequence

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
LABEL_COLUMNS = [
    "Internal_Echo", "Morphology", "Boundary", "Solid", "Separation",
    "Nipple", "Blood_Flow", "Malignant",
]
BINARY_TASKS = [
    "Morphology", "Boundary", "Solid", "Separation", "Nipple",
    "Blood_Flow", "Malignant",
]
ECHO_VALUES = (0, 1, 3, 4)


def load_config(path: str | os.PathLike) -> dict:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    repo_root = (base / config.get("repo_root", "../../..")).resolve()
    config["_config_path"] = str(config_path)
    config["_repo_root"] = str(repo_root)
    for key in ("work_dir", "outputs_dir"):
        value = config.get(key)
        if value:
            config[f"_{key}"] = str((repo_root / value).resolve())
    return config


def repo_path(config: dict, value: str | os.PathLike) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(config["_repo_root"]) / path).resolve()


def stable_uid(*parts: object, length: int = 16) -> str:
    text = "|".join(str(p) for p in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def proposed_patient_id(name: str) -> str:
    """Create a review suggestion only; never treat it as clinically confirmed."""
    stem = Path(str(name)).stem.strip()
    stem = re.sub(r"^\d+[_\-\s]+", "", stem)
    stem = re.sub(
        r"(?:_original|_rot[+\-]?\d+|_flip(?:ped)?|_aug\d+|_copy\d*)$",
        "", stem, flags=re.IGNORECASE,
    )
    stem = re.sub(r"\d+$", "", stem)
    normalized = re.sub(r"[^a-z0-9]+", "", stem.lower())
    return normalized or stable_uid(name, length=12)


def proposed_lesion_id(name: str, patient_id: str) -> str:
    stem = Path(str(name)).stem
    base = re.sub(r"(?:_original|_rot[+\-]?\d+|_flip(?:ped)?|_aug\d+|_copy\d*)$", "", stem,
                  flags=re.IGNORECASE)
    suffix = re.search(r"(\d+)$", re.sub(r"^\d+[_\-\s]+", "", base))
    lesion_suffix = suffix.group(1) if suffix else "1"
    return f"{patient_id}_L{lesion_suffix}"


def build_image_index(directories: Sequence[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                index.setdefault(path.stem.lower(), []).append(path.resolve())
    return index


def find_image(name: str, index: dict[str, list[Path]]) -> tuple[str, str]:
    matches = index.get(Path(str(name)).stem.lower(), [])
    if len(matches) == 1:
        return str(matches[0]), ""
    if len(matches) > 1:
        return "", "MULTIPLE_MATCHES:" + "|".join(str(x) for x in matches)
    return "", "NOT_FOUND"


def require_columns(frame, columns: Iterable[str], label: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def assert_patient_disjoint(frame, split_col: str = "split", allow_overlap: bool = False) -> list[str]:
    """Enforce patient disjointness unless an explicitly labeled comparison allows overlap.

    The default remains fail-fast for the patient-safe pipeline.  The opt-out exists only
    so the historically used image-level split can be reproduced in a separate output
    tree and compared transparently with the patient-level analysis.
    """
    require_columns(frame, ["patient_id", split_col], "manifest")
    memberships = frame.groupby("patient_id")[split_col].nunique(dropna=False)
    bad = memberships[memberships > 1]
    overlapping = bad.index.astype(str).tolist()
    if overlapping and not allow_overlap:
        raise RuntimeError(f"Patient leakage detected for {len(bad)} patients: {bad.index.tolist()[:20]}")
    return overlapping


def assert_no_duplicate_paths_across_splits(frame, path_columns: Sequence[str]) -> None:
    for column in path_columns:
        if column not in frame.columns:
            continue
        subset = frame.loc[frame[column].fillna("").astype(str).ne(""), [column, "split"]].copy()
        if subset.empty:
            continue
        subset[column] = subset[column].map(lambda x: str(Path(x).resolve()).lower())
        counts = subset.groupby(column)["split"].nunique()
        bad = counts[counts > 1]
        if not bad.empty:
            raise RuntimeError(f"Exact file-path leakage in {column}: {bad.index.tolist()[:10]}")


def sha256_file(path: str | os.PathLike, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def json_dump(data: object, path: str | os.PathLike) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def relative_or_absolute(path: str, root: Path) -> str:
    if not path:
        return ""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)
