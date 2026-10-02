from __future__ import annotations

import csv
import hashlib
import os
import random
import secrets
import shutil
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(os.environ["BLINDED_OUTPUT_DIR"]).expanduser().resolve()
ORIGINAL = Path(os.environ["CEUS_REAL_DIR"]).expanduser().resolve()
GENERATED = Path(os.environ["CEUS_SYNTHETIC_DIR"]).expanduser().resolve()
IMAGES = ROOT / "Images_FOR_READERS"
PRIVATE = ROOT / "Private_Key_DO_NOT_SHARE_WITH_READERS"
RESIZED_GENERATED = PRIVATE / "Resized_Generated_Matched_Original_Size"
KEY = PRIVATE / "blinding_key.csv"
AUDIT = PRIVATE / "package_integrity_audit.csv"
SEED_FILE = PRIVATE / "randomization_seed_DO_NOT_SHARE.txt"
SCORING = ROOT / "reader_scoring_sheet.csv"
README = ROOT / "README_CN.txt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_map(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resize_generated(source: Path, reference: Path, destination: Path) -> tuple[int, int]:
    with Image.open(reference) as ref_image:
        target_size = ref_image.size
    with Image.open(source) as generated_image:
        generated_image = ImageOps.exif_transpose(generated_image)
        generated_image = generated_image.convert("RGB")
        resized = generated_image.resize(target_size, Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        resized.save(destination, format="JPEG", quality=95, subsampling=0, optimize=True)
    with Image.open(destination) as check_image:
        if check_image.size != target_size:
            raise RuntimeError(
                f"Resized image has the wrong size: {destination.name}: "
                f"{check_image.size} != {target_size}"
            )
    return target_size


def main() -> None:
    original = image_map(ORIGINAL)
    generated = image_map(GENERATED)
    if set(original) != set(generated):
        only_original = sorted(set(original) - set(generated))
        only_generated = sorted(set(generated) - set(original))
        raise RuntimeError(
            "Original/generated mapping is not one-to-one. "
            f"only_original={only_original[:10]}, only_generated={only_generated[:10]}"
        )
    if len(original) != 67:
        raise RuntimeError(f"Expected 67 test cases, found {len(original)}")
    if ROOT.exists() and any(path for path in ROOT.iterdir() if path.name not in {"_admin", "_spreadsheet_work"}):
        raise RuntimeError(
            "The output package already contains generated files. Refusing to overwrite or rerandomize it."
        )

    IMAGES.mkdir(parents=True, exist_ok=False)
    RESIZED_GENERATED.mkdir(parents=True, exist_ok=False)

    seed_hex = secrets.token_hex(32)
    rng = random.Random(int(seed_hex, 16))
    relative_names = sorted(original)
    rng.shuffle(relative_names)

    generated_positions = [1] * (len(relative_names) // 2) + [2] * (
        len(relative_names) - len(relative_names) // 2
    )
    rng.shuffle(generated_positions)

    key_rows: list[dict[str, object]] = []
    scoring_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for pair_index, (relative, generated_position) in enumerate(
        zip(relative_names, generated_positions), start=1
    ):
        original_path = original[relative]
        generated_path = generated[relative]
        stem = Path(relative).stem
        resized_generated_path = RESIZED_GENERATED / f"{stem}.jpg"
        width, height = resize_generated(generated_path, original_path, resized_generated_path)

        file_1 = f"{stem}_1.jpg"
        file_2 = f"{stem}_2.jpg"
        if generated_position == 1:
            source_1, role_1 = resized_generated_path, "generated"
            source_2, role_2 = original_path, "original"
        else:
            source_1, role_1 = original_path, "original"
            source_2, role_2 = resized_generated_path, "generated"

        destination_1 = IMAGES / file_1
        destination_2 = IMAGES / file_2
        shutil.copy2(source_1, destination_1)
        shutil.copy2(source_2, destination_2)

        for output_path in (destination_1, destination_2):
            with Image.open(output_path) as image:
                if image.size != (width, height):
                    raise RuntimeError(
                        f"Pair dimension mismatch after packaging: {output_path.name}: {image.size}"
                    )

        key_rows.append(
            {
                "pair_index": pair_index,
                "case_prefix": stem,
                "source_relative_name": relative,
                "file_1": file_1,
                "true_type_1": role_1,
                "file_2": file_2,
                "true_type_2": role_2,
                "generated_position": generated_position,
                "width": width,
                "height": height,
                "original_source_sha256": sha256(original_path),
                "generated_source_sha256": sha256(generated_path),
                "resized_generated_sha256": sha256(resized_generated_path),
            }
        )
        scoring_rows.append(
            {
                "pair_index": pair_index,
                "case_prefix": stem,
                "file_1": file_1,
                "file_2": file_2,
                "reader_id": "",
                "which_file_is_generated_1_or_2": "",
                "confidence_1_to_5": "",
                "boundary_clarity_file_1_1_to_5": "",
                "boundary_clarity_file_2_1_to_5": "",
                "perfusion_realism_file_1_1_to_5": "",
                "perfusion_realism_file_2_1_to_5": "",
                "comments": "",
            }
        )

        for output_path, source_path, role in (
            (destination_1, source_1, role_1),
            (destination_2, source_2, role_2),
        ):
            with Image.open(output_path) as image:
                output_size = image.size
                output_mode = image.mode
            audit_rows.append(
                {
                    "pair_index": pair_index,
                    "case_prefix": stem,
                    "output_file": output_path.name,
                    "true_type": role,
                    "width": output_size[0],
                    "height": output_size[1],
                    "mode": output_mode,
                    "source_sha256": sha256(source_path),
                    "output_sha256": sha256(output_path),
                    "byte_identical_copy_from_packaging_source": sha256(source_path)
                    == sha256(output_path),
                }
            )

    PRIVATE.mkdir(parents=True, exist_ok=True)
    SEED_FILE.write_text(seed_hex + "\n", encoding="ascii")
    write_csv(
        KEY,
        [
            "pair_index",
            "case_prefix",
            "source_relative_name",
            "file_1",
            "true_type_1",
            "file_2",
            "true_type_2",
            "generated_position",
            "width",
            "height",
            "original_source_sha256",
            "generated_source_sha256",
            "resized_generated_sha256",
        ],
        key_rows,
    )
    write_csv(
        SCORING,
        [
            "pair_index",
            "case_prefix",
            "file_1",
            "file_2",
            "reader_id",
            "which_file_is_generated_1_or_2",
            "confidence_1_to_5",
            "boundary_clarity_file_1_1_to_5",
            "boundary_clarity_file_2_1_to_5",
            "perfusion_realism_file_1_1_to_5",
            "perfusion_realism_file_2_1_to_5",
            "comments",
        ],
        scoring_rows,
    )
    write_csv(
        AUDIT,
        [
            "pair_index",
            "case_prefix",
            "output_file",
            "true_type",
            "width",
            "height",
            "mode",
            "source_sha256",
            "output_sha256",
            "byte_identical_copy_from_packaging_source",
        ],
        audit_rows,
    )

    generated_as_1 = sum(row["generated_position"] == 1 for row in key_rows)
    generated_as_2 = len(key_rows) - generated_as_1
    README.write_text(
        "测试集临床专家盲法真实CEUS/生成CEUS识别评估包\n"
        "================================================\n"
        f"病例对数：{len(key_rows)}\n"
        f"供专家查看的图片总数：{len(audit_rows)}\n"
        f"生成图位于 _1 的病例数：{generated_as_1}\n"
        f"生成图位于 _2 的病例数：{generated_as_2}\n"
        "\n"
        "请向每位临床专家提供：\n"
        "1. Images_FOR_READERS 文件夹（或 Images_FOR_READERS.zip）\n"
        "2. 临床专家评估评分表_中文版.xlsx 的独立副本\n"
        "\n"
        "请勿向临床专家提供：\n"
        "Private_Key_DO_NOT_SHARE_WITH_READERS 文件夹及其任何内容。\n"
        "\n"
        "处理规则：\n"
        "1. 每张生成图按对应测试集原图的宽度和高度进行 LANCZOS 缩放。\n"
        "2. 同一病例的两张图除末尾 _1/_2 外文件名完全一致。\n"
        "3. 病例在评分表中的顺序已随机打乱。\n"
        "4. 真实图与生成图在 _1/_2 之间随机且位置数量近似平衡。\n"
        "5. 原图在混合打包阶段采用字节级复制；生成图从尺寸匹配版本字节级复制。\n"
        "6. 所有专家完成并锁定评分表后，方可使用 blinding_key.csv 解盲。\n"
        "\n"
        "注意：本流程只统一图像尺寸，没有裁剪或遮挡源图中已有的文字、标尺或设备标记。\n"
        "如果这些标记可能提示图像来源，应在正式评估前采用对所有图像一致的预处理方案。\n",
        encoding="utf-8",
    )

    print(f"Pairs: {len(key_rows)}")
    print(f"Reader images: {len(audit_rows)}")
    print(f"Generated as _1: {generated_as_1}")
    print(f"Generated as _2: {generated_as_2}")
    print(f"Reader folder: {IMAGES}")
    print(f"Private key: {KEY}")
    print(f"Scoring CSV: {SCORING}")


if __name__ == "__main__":
    main()
