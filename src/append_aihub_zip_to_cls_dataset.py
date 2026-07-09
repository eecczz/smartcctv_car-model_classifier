from __future__ import annotations

import argparse
import csv
import io
import os
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from crop_aihub_zip_to_cls_dataset import (
    CropAnn,
    build_image_index,
    collect_annotations,
    crop_box,
    normalize_class,
    split_by_class,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append AIHub vehicle crops to an existing YOLO classification dataset while respecting "
            "train-exc/val-exc/test-exc and promoting classes once train count reaches a threshold."
        )
    )
    parser.add_argument("--image-zip", required=True, type=Path)
    parser.add_argument("--label-zip", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--class-field", default="model_id", choices=["model_id", "class_id", "brand_id"])
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-size", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--min-box-width", type=float, default=0)
    parser.add_argument("--min-box-height", type=float, default=0)
    parser.add_argument("--min-box-area", type=float, default=0)
    parser.add_argument("--promote-threshold", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--exclude-class",
        action="append",
        default=["unknown"],
        help="Normalized class name to skip. Repeatable. Default: unknown",
    )
    return parser.parse_args()


def image_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def existing_class_state(root: Path, class_name: str) -> str:
    if (root / "train" / class_name).exists():
        return "active"
    return "excluded"


def split_root_for_class(root: Path, split: str, class_name: str) -> Path:
    state = existing_class_state(root, class_name)
    if state == "active":
        return root / split / class_name
    return root / f"{split}-exc" / class_name


def unique_crop_path(dst_dir: Path, image_name: str, counter: int) -> Path:
    stem = Path(image_name).stem
    dst = dst_dir / f"{stem}_{counter:06d}.jpg"
    if not dst.exists():
        return dst
    suffix = 1
    while True:
        candidate = dst_dir / f"{stem}_{counter:06d}_{suffix}.jpg"
        if not candidate.exists():
            return candidate
        suffix += 1


def write_audit(root: Path, rows: list[dict[str, Any]]) -> None:
    audit_path = root / "append_audit.csv"
    append = audit_path.exists()
    with audit_path.open("a" if append else "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "source",
            "split",
            "class",
            "target_bucket",
            "image",
            "label",
            "bbox_xywh",
            "status",
            "crop_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def merge_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    os.chmod(src, 0o700)
    for file in src.iterdir():
        target = dst / file.name
        if target.exists():
            suffix = 1
            while True:
                candidate = dst / f"{file.stem}_{suffix}{file.suffix}"
                if not candidate.exists():
                    target = candidate
                    break
                suffix += 1
        shutil.move(str(file), str(target))
    try:
        src.rmdir()
    except PermissionError:
        os.chmod(src, 0o700)
        shutil.rmtree(src, ignore_errors=True)


def promote_ready_classes(root: Path, threshold: int) -> list[str]:
    promoted: list[str] = []
    train_exc = root / "train-exc"
    if not train_exc.exists():
        return promoted

    for cls_dir in sorted(p for p in train_exc.iterdir() if p.is_dir()):
        cls = cls_dir.name
        if image_count(cls_dir) < threshold:
            continue
        merge_dir(root / "train-exc" / cls, root / "train" / cls)
        merge_dir(root / "val-exc" / cls, root / "val" / cls)
        merge_dir(root / "test-exc" / cls, root / "test" / cls)
        (root / "train" / cls).mkdir(parents=True, exist_ok=True)
        (root / "val" / cls).mkdir(parents=True, exist_ok=True)
        (root / "test" / cls).mkdir(parents=True, exist_ok=True)
        promoted.append(cls)
    return promoted


def crop_and_append(
    image_zip: Path,
    root: Path,
    anns: list[CropAnn],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    min_size: int,
) -> Counter:
    stats: Counter = Counter()
    audit_rows: list[dict[str, Any]] = []
    image_index = build_image_index(image_zip)
    splits = split_by_class(anns, val_ratio, test_ratio, seed)
    tasks_by_entry: dict[str, list[tuple[str, CropAnn, zipfile.ZipInfo]]] = defaultdict(list)

    for split, split_anns in splits.items():
        for ann in split_anns:
            entry = image_index.get(ann.image_name.lower()) or image_index.get(Path(ann.image_name).stem.lower())
            if entry is None:
                stats["missing_image"] += 1
                audit_rows.append(
                    {
                        "source": image_zip.name,
                        "split": split,
                        "class": ann.class_name,
                        "target_bucket": "",
                        "image": ann.image_name,
                        "label": ann.label_name,
                        "bbox_xywh": list(ann.bbox_xywh),
                        "status": "missing_image",
                        "crop_path": "",
                    }
                )
                continue
            tasks_by_entry[entry.filename].append((split, ann, entry))

    with zipfile.ZipFile(image_zip) as zf:
        counter = 0
        for _, tasks in tasks_by_entry.items():
            entry = tasks[0][2]
            try:
                with zf.open(entry) as f:
                    image = Image.open(io.BytesIO(f.read())).convert("RGB")
            except OSError:
                for split, ann, _ in tasks:
                    stats["image_open_error"] += 1
                    audit_rows.append(
                        {
                            "source": image_zip.name,
                            "split": split,
                            "class": ann.class_name,
                            "target_bucket": "",
                            "image": ann.image_name,
                            "label": ann.label_name,
                            "bbox_xywh": list(ann.bbox_xywh),
                            "status": "image_open_error",
                            "crop_path": "",
                        }
                    )
                continue

            for split, ann, _ in tasks:
                target_dir = split_root_for_class(root, split, ann.class_name)
                target_bucket = target_dir.parent.name
                row = {
                    "source": image_zip.name,
                    "split": split,
                    "class": ann.class_name,
                    "target_bucket": target_bucket,
                    "image": ann.image_name,
                    "label": ann.label_name,
                    "bbox_xywh": list(ann.bbox_xywh),
                    "status": "",
                    "crop_path": "",
                }
                try:
                    crop = crop_box(image, ann.bbox_xywh, min_size)
                except OSError:
                    stats["image_open_error"] += 1
                    row["status"] = "image_open_error"
                    audit_rows.append(row)
                    continue

                if crop is None:
                    stats["small_or_invalid_bbox"] += 1
                    row["status"] = "small_or_invalid_bbox"
                    audit_rows.append(row)
                    continue

                target_dir.mkdir(parents=True, exist_ok=True)
                dst = unique_crop_path(target_dir, ann.image_name, counter)
                crop.save(dst, quality=95)
                counter += 1

                stats[f"saved_{target_bucket}"] += 1
                stats[f"class_{ann.class_name}"] += 1
                row["status"] = "saved"
                row["crop_path"] = str(dst)
                audit_rows.append(row)

    write_audit(root, audit_rows)
    return stats


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    strip_year = True
    exclude_classes = {cls for cls in (normalize_class(v, strip_year=strip_year) for v in args.exclude_class) if cls}
    anns = collect_annotations(
        args.label_zip,
        args.class_field,
        args.limit,
        exclude_classes,
        strip_year,
        max(1, args.frame_stride),
        args.min_box_width,
        args.min_box_height,
        args.min_box_area,
    )

    splits = split_by_class(anns, args.val_ratio, args.test_ratio, args.seed)
    planned: Counter = Counter()
    for split, split_anns in splits.items():
        for ann in split_anns:
            bucket = split if existing_class_state(args.out, ann.class_name) == "active" else f"{split}-exc"
            planned[bucket] += 1

    class_counts = Counter(ann.class_name for ann in anns)
    print(f"Image zip: {args.image_zip}")
    print(f"Label zip: {args.label_zip}")
    print(f"Parsed crop annotations: {len(anns)}")
    print("Planned target buckets:")
    for key, value in sorted(planned.items()):
        print(f"  {key}: {value}")
    print("Top parsed classes:")
    for cls, count in class_counts.most_common(30):
        print(f"  {cls}: {count}")

    if args.dry_run:
        return

    stats = crop_and_append(
        args.image_zip,
        args.out,
        anns,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        args.min_size,
    )
    promoted = promote_ready_classes(args.out, args.promote_threshold)

    print("Append stats:")
    for key, value in stats.most_common():
        if not key.startswith("class_"):
            print(f"  {key}: {value}")
    print(f"Promoted classes: {len(promoted)}")
    for cls in promoted[:80]:
        print(f"  {cls}")
    if len(promoted) > 80:
        print(f"  ... {len(promoted) - 80} more")
    print(f"Audit written: {args.out / 'append_audit.csv'}")


if __name__ == "__main__":
    main()
