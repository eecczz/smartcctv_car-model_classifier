from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class CropAnn:
    image_name: str
    bbox_xywh: tuple[float, float, float, float]
    class_name: str
    label_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop AIHub vehicle bbox annotations from source/label zip files into YOLO cls folders."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Dataset root containing TS_*.zip and TL_*.zip. Used when --image-zip/--label-zip are omitted.",
    )
    parser.add_argument("--image-zip", type=Path, default=None, help="원천데이터 TS_*.zip")
    parser.add_argument("--label-zip", type=Path, default=None, help="라벨링데이터 TL_*.zip")
    parser.add_argument("--out", type=Path, required=True, help="Output YOLO classification dataset root.")
    parser.add_argument(
        "--class-field",
        default="model_id",
        choices=["model_id", "class_id", "brand_id"],
        help="JSON annotation field to use as the classification folder name.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-size", type=int, default=16)
    parser.add_argument("--min-box-width", type=float, default=0)
    parser.add_argument("--min-box-height", type=float, default=0)
    parser.add_argument("--min-box-area", type=float, default=0)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Use only frames whose trailing A-number is divisible by this value.",
    )
    parser.add_argument(
        "--keep-year",
        action="store_true",
        help="Keep trailing parenthesized year/model text in class folder names.",
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=1,
        help="Drop classes with fewer parsed annotations after all filters.",
    )
    parser.add_argument(
        "--exclude-class",
        action="append",
        default=["unknown"],
        help="Normalized class name to skip. Repeatable. Default: unknown",
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug limit for number of JSON labels.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def find_zip(src: Path, prefix: str) -> Path:
    matches = sorted(src.rglob(f"{prefix}*.zip"))
    if not matches:
        raise FileNotFoundError(f"Could not find {prefix}*.zip under {src}")
    if len(matches) > 1:
        print(f"Found {len(matches)} {prefix} zip files. Using first: {matches[0]}")
    return matches[0]


def normalize_class(value: Any, strip_year: bool = True) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if strip_year:
        text = re.sub(r"\([^)]*\)\s*$", "", text).strip()
    text = text.replace("\u2162", "III")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or None


def parse_coord(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x, y, w, h = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    return x, y, w, h


def read_json_entry(zf: zipfile.ZipFile, entry: zipfile.ZipInfo) -> dict[str, Any] | None:
    try:
        with zf.open(entry) as f:
            return json.loads(f.read().decode("utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None


def frame_number(image_stem: str) -> int | None:
    match = re.search(r"_A(\d+)$", image_stem)
    return int(match.group(1)) if match else None


def bbox_passes_filters(
    bbox_xywh: tuple[float, float, float, float],
    min_box_width: float,
    min_box_height: float,
    min_box_area: float,
) -> bool:
    _, _, w, h = bbox_xywh
    return w >= min_box_width and h >= min_box_height and (w * h) >= min_box_area


def parse_label(
    data: dict[str, Any],
    label_name: str,
    class_field: str,
    strip_year: bool,
    frame_stride: int,
    min_box_width: float,
    min_box_height: float,
    min_box_area: float,
) -> list[CropAnn]:
    source_info = data.get("Source Data Info", {})
    learning_info = data.get("Learning Data Info", {})
    image_stem = source_info.get("source_data_id") or learning_info.get("json_data_id") or Path(label_name).stem
    image_ext = source_info.get("file_extension") or "jpg"
    image_name = f"{image_stem}.{str(image_ext).lstrip('.')}"
    frame_no = frame_number(str(image_stem))
    if frame_stride > 1 and frame_no is not None and frame_no % frame_stride != 0:
        return []

    anns = learning_info.get("annotations", [])
    if not isinstance(anns, list):
        return []

    parsed: list[CropAnn] = []
    for ann in anns:
        if not isinstance(ann, dict) or ann.get("type") != "bbox":
            continue
        bbox = parse_coord(ann.get("coord"))
        cls = normalize_class(ann.get(class_field), strip_year=strip_year)
        if bbox and cls and bbox_passes_filters(bbox, min_box_width, min_box_height, min_box_area):
            parsed.append(CropAnn(image_name=image_name, bbox_xywh=bbox, class_name=cls, label_name=label_name))
    return parsed


def collect_annotations(
    label_zip: Path,
    class_field: str,
    limit: int | None,
    exclude_classes: set[str],
    strip_year: bool,
    frame_stride: int,
    min_box_width: float,
    min_box_height: float,
    min_box_area: float,
) -> list[CropAnn]:
    anns: list[CropAnn] = []
    errors = 0
    with zipfile.ZipFile(label_zip) as zf:
        entries = [e for e in zf.infolist() if e.filename.lower().endswith(".json")]
        if limit:
            entries = entries[:limit]
        for entry in entries:
            data = read_json_entry(zf, entry)
            if data is None:
                errors += 1
                continue
            anns.extend(
                ann
                for ann in parse_label(
                    data,
                    entry.filename,
                    class_field,
                    strip_year,
                    frame_stride,
                    min_box_width,
                    min_box_height,
                    min_box_area,
                )
                if ann.class_name not in exclude_classes
            )
    if errors:
        print(f"Label parse errors: {errors}")
    return anns


def build_image_index(image_zip: Path) -> dict[str, zipfile.ZipInfo]:
    with zipfile.ZipFile(image_zip) as zf:
        index: dict[str, zipfile.ZipInfo] = {}
        for entry in zf.infolist():
            path = Path(entry.filename)
            if path.suffix.lower() in IMAGE_EXTS:
                index.setdefault(path.name.lower(), entry)
                index.setdefault(path.stem.lower(), entry)
        return index


def split_by_class(anns: list[CropAnn], val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[CropAnn]]:
    grouped: dict[str, list[CropAnn]] = defaultdict(list)
    for ann in anns:
        grouped[ann.class_name].append(ann)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for class_anns in grouped.values():
        shuffled = class_anns[:]
        rng.shuffle(shuffled)
        n_test = int(round(len(shuffled) * test_ratio))
        n_val = int(round(len(shuffled) * val_ratio))
        splits["test"].extend(shuffled[:n_test])
        splits["val"].extend(shuffled[n_test : n_test + n_val])
        splits["train"].extend(shuffled[n_test + n_val :])
    return splits


def filter_min_class_count(anns: list[CropAnn], min_class_count: int) -> list[CropAnn]:
    if min_class_count <= 1:
        return anns
    counts = Counter(ann.class_name for ann in anns)
    return [ann for ann in anns if counts[ann.class_name] >= min_class_count]


def crop_box(image: Image.Image, bbox_xywh: tuple[float, float, float, float], min_size: int) -> Image.Image | None:
    x, y, w, h = bbox_xywh
    x1 = max(0, min(image.width, int(round(x))))
    y1 = max(0, min(image.height, int(round(y))))
    x2 = max(0, min(image.width, int(round(x + w))))
    y2 = max(0, min(image.height, int(round(y + h))))
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return image.crop((x1, y1, x2, y2))


def write_audit(out: Path, rows: list[dict[str, Any]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    audit_path = out / "crop_audit.csv"
    append = audit_path.exists()
    with audit_path.open("a" if append else "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "class",
                "image",
                "label",
                "bbox_xywh",
                "status",
                "crop_path",
            ],
        )
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def unique_crop_path(out: Path, split: str, class_name: str, image_name: str, counter: int) -> Path:
    image_stem = Path(image_name).stem
    dst = out / split / class_name / f"{image_stem}_{counter:06d}.jpg"
    if not dst.exists():
        return dst

    suffix = 1
    while True:
        candidate = out / split / class_name / f"{image_stem}_{counter:06d}_{suffix}.jpg"
        if not candidate.exists():
            return candidate
        suffix += 1


def crop_dataset(
    image_zip: Path,
    out: Path,
    anns: list[CropAnn],
    image_index: dict[str, zipfile.ZipInfo],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    min_size: int,
) -> Counter:
    stats: Counter = Counter()
    audit_rows: list[dict[str, Any]] = []
    splits = split_by_class(anns, val_ratio, test_ratio, seed)
    tasks_by_entry: dict[str, list[tuple[str, CropAnn, zipfile.ZipInfo]]] = defaultdict(list)

    for split, split_anns in splits.items():
        for ann in split_anns:
            entry = image_index.get(ann.image_name.lower()) or image_index.get(Path(ann.image_name).stem.lower())
            if entry is None:
                stats["missing_image"] += 1
                audit_rows.append(
                    {
                        "split": split,
                        "class": ann.class_name,
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
                            "split": split,
                            "class": ann.class_name,
                            "image": ann.image_name,
                            "label": ann.label_name,
                            "bbox_xywh": list(ann.bbox_xywh),
                            "status": "image_open_error",
                            "crop_path": "",
                        }
                    )
                continue

            for split, ann, _ in tasks:
                audit_row = {
                    "split": split,
                    "class": ann.class_name,
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
                    audit_row["status"] = "image_open_error"
                    audit_rows.append(audit_row)
                    continue

                if crop is None:
                    stats["small_or_invalid_bbox"] += 1
                    audit_row["status"] = "small_or_invalid_bbox"
                    audit_rows.append(audit_row)
                    continue

                dst = unique_crop_path(out, split, ann.class_name, ann.image_name, counter)
                dst.parent.mkdir(parents=True, exist_ok=True)
                crop.save(dst, quality=95)
                counter += 1

                stats[f"saved_{split}"] += 1
                stats[f"class_{ann.class_name}"] += 1
                audit_row["status"] = "saved"
                audit_row["crop_path"] = str(dst)
                audit_rows.append(audit_row)

    write_audit(out, audit_rows)
    return stats


def main() -> None:
    args = parse_args()
    if args.src:
        image_zip = args.image_zip or find_zip(args.src, "TS_")
        label_zip = args.label_zip or find_zip(args.src, "TL_")
    elif args.image_zip and args.label_zip:
        image_zip = args.image_zip
        label_zip = args.label_zip
    else:
        raise SystemExit("Provide --src or both --image-zip and --label-zip.")

    strip_year = not args.keep_year
    exclude_classes = {cls for cls in (normalize_class(v, strip_year=strip_year) for v in args.exclude_class) if cls}
    anns = collect_annotations(
        label_zip,
        args.class_field,
        args.limit,
        exclude_classes,
        strip_year,
        max(1, args.frame_stride),
        args.min_box_width,
        args.min_box_height,
        args.min_box_area,
    )
    anns = filter_min_class_count(anns, args.min_class_count)
    class_counts = Counter(ann.class_name for ann in anns)

    print(f"Image zip: {image_zip}")
    print(f"Label zip: {label_zip}")
    print(f"Parsed crop annotations: {len(anns)}")
    if exclude_classes:
        print(f"Excluded classes: {', '.join(sorted(exclude_classes))}")
    print(f"Strip trailing parentheses/year: {strip_year}")
    print(f"Frame stride: {max(1, args.frame_stride)}")
    print(
        "BBox filters: "
        f"min_width={args.min_box_width}, min_height={args.min_box_height}, min_area={args.min_box_area}"
    )
    print(f"Min class count: {args.min_class_count}")
    print("Top classes:")
    for cls, count in class_counts.most_common(30):
        print(f"  {cls}: {count}")

    if args.dry_run:
        return

    image_index = build_image_index(image_zip)
    stats = crop_dataset(
        image_zip=image_zip,
        out=args.out,
        anns=anns,
        image_index=image_index,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_size=args.min_size,
    )
    print("Crop stats:")
    for key, value in stats.most_common():
        if key.startswith("class_"):
            continue
        print(f"  {key}: {value}")
    print(f"Dataset written: {args.out}")
    print(f"Audit written: {args.out / 'crop_audit.csv'}")


if __name__ == "__main__":
    main()
