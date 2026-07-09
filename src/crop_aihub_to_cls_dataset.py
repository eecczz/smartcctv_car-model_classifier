from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTS = {".json"}


CLASS_KEYS = (
    "차종",
    "차량종류",
    "차량_종류",
    "차량모델",
    "모델명",
    "model",
    "car_model",
    "vehicle_model",
    "vehicle_type",
    "type",
    "category_name",
    "category",
    "class_name",
    "class",
    "label",
    "name",
)
IMAGE_KEYS = (
    "file_name",
    "filename",
    "image_file",
    "image_filename",
    "imageName",
    "image_name",
    "imagePath",
    "path",
)


@dataclass
class CropItem:
    image_name: str
    bbox: tuple[float, float, float, float]
    class_name: str
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract AIHub-style image/label zips and crop bbox regions into an "
            "Ultralytics classification dataset."
        )
    )
    parser.add_argument(
        "--src",
        nargs="+",
        type=Path,
        required=True,
        help="Zip files and/or directories containing source images and JSON labels.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output classification dataset root.")
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("data/extracted/aihub_vehicle"),
        help="Where zip files are extracted.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-size", type=int, default=16, help="Skip crops smaller than this width/height.")
    parser.add_argument(
        "--class-key",
        default=None,
        help="Prefer a specific JSON key for the class name, for example model or vehicle_type.",
    )
    parser.add_argument(
        "--image-key",
        default=None,
        help="Prefer a specific JSON key for the image filename.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug limit for number of labels to parse.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report only; do not crop images.")
    return parser.parse_args()


def normalize_class(name: Any) -> str | None:
    if name is None:
        return None
    text = str(name).strip()
    if not text:
        return None
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9A-Za-z가-힣_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or None


def safe_extract_zip(zip_path: Path, extract_root: Path) -> Path:
    target_dir = extract_root / zip_path.stem
    marker = target_dir / ".extracted"
    if marker.exists():
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            member_path = (target_dir / member.filename).resolve()
            if not str(member_path).startswith(str(target_dir.resolve())):
                raise ValueError(f"Unsafe zip member path: {member.filename}")
            zf.extract(member, target_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return target_dir


def collect_roots(srcs: list[Path], extract_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for src in srcs:
        if src.is_file() and src.suffix.lower() == ".zip":
            roots.append(safe_extract_zip(src, extract_dir))
        elif src.is_dir():
            roots.append(src)
            for zip_path in sorted(src.rglob("*.zip")):
                roots.append(safe_extract_zip(zip_path, extract_dir))
        else:
            raise FileNotFoundError(src)
    return roots


def collect_files(roots: list[Path], suffixes: set[str]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        files.extend(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
    return sorted(set(files))


def image_indexes(images: list[Path]) -> tuple[dict[str, Path], dict[str, Path]]:
    by_name: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}
    for path in images:
        by_name.setdefault(path.name.lower(), path)
        by_stem.setdefault(path.stem.lower(), path)
    return by_name, by_stem


def walk_dicts(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        found.append(value)
        for child in value.values():
            found.extend(walk_dicts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(walk_dicts(child))
    return found


def first_value_by_keys(node: dict[str, Any], keys: tuple[str, ...], preferred_key: str | None = None) -> Any:
    if preferred_key and preferred_key in node:
        return node[preferred_key]
    lowered = {str(k).lower(): v for k, v in node.items()}
    for key in keys:
        if key in node:
            return node[key]
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def find_image_name(data: dict[str, Any], label_path: Path, image_key: str | None) -> str:
    direct = first_value_by_keys(data, IMAGE_KEYS, image_key)
    if isinstance(direct, str) and direct.strip():
        return Path(direct).name

    for node in walk_dicts(data):
        value = first_value_by_keys(node, IMAGE_KEYS, image_key)
        if isinstance(value, str) and value.strip() and Path(value).suffix.lower() in IMAGE_EXTS:
            return Path(value).name

    return label_path.with_suffix(".jpg").name


def parse_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        if all(k in value for k in ("x", "y", "w", "h")):
            x, y, w, h = float(value["x"]), float(value["y"]), float(value["w"]), float(value["h"])
            return x, y, x + w, y + h
        if all(k in value for k in ("left", "top", "width", "height")):
            x, y = float(value["left"]), float(value["top"])
            return x, y, x + float(value["width"]), y + float(value["height"])
        if all(k in value for k in ("xmin", "ymin", "xmax", "ymax")):
            return float(value["xmin"]), float(value["ymin"]), float(value["xmax"]), float(value["ymax"])
        if all(k in value for k in ("x1", "y1", "x2", "y2")):
            return float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"])

    if isinstance(value, list) and len(value) >= 4:
        nums = [float(v) for v in value[:4]]
        x1, y1, third, fourth = nums
        if third > x1 and fourth > y1:
            return x1, y1, third, fourth
        return x1, y1, x1 + third, y1 + fourth

    return None


def bbox_from_node(node: dict[str, Any]) -> tuple[float, float, float, float] | None:
    for key in ("bbox", "bounding_box", "box", "bndbox", "rect"):
        if key in node:
            bbox = parse_bbox(node[key])
            if bbox:
                return bbox
    return parse_bbox(node)


def class_from_node(node: dict[str, Any], class_key: str | None) -> str | None:
    value = first_value_by_keys(node, CLASS_KEYS, class_key)
    cls = normalize_class(value)
    if cls:
        return cls

    for nested_key in ("attributes", "attribute", "properties", "metadata"):
        nested = node.get(nested_key)
        if isinstance(nested, dict):
            cls = normalize_class(first_value_by_keys(nested, CLASS_KEYS, class_key))
            if cls:
                return cls
    return None


def parse_coco(data: dict[str, Any], label_path: Path) -> list[CropItem] | None:
    if not all(key in data for key in ("images", "annotations", "categories")):
        return None

    images = {item.get("id"): item.get("file_name") for item in data["images"] if isinstance(item, dict)}
    categories = {item.get("id"): item.get("name") for item in data["categories"] if isinstance(item, dict)}
    items: list[CropItem] = []

    for ann in data["annotations"]:
        if not isinstance(ann, dict):
            continue
        image_name = images.get(ann.get("image_id"))
        cls = normalize_class(categories.get(ann.get("category_id")))
        bbox = parse_bbox(ann.get("bbox"))
        if image_name and cls and bbox:
            items.append(CropItem(Path(image_name).name, bbox, cls, label_path))
    return items


def parse_label_file(label_path: Path, class_key: str | None, image_key: str | None) -> list[CropItem]:
    with label_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return []

    coco_items = parse_coco(data, label_path)
    if coco_items is not None:
        return coco_items

    image_name = find_image_name(data, label_path, image_key)
    items: list[CropItem] = []

    for node in walk_dicts(data):
        bbox = bbox_from_node(node)
        if not bbox:
            continue
        cls = class_from_node(node, class_key)
        if not cls:
            cls = class_from_node(data, class_key)
        if cls:
            items.append(CropItem(image_name, bbox, cls, label_path))

    # Some AIHub labels repeat bbox data at parent and child levels. Keep exact duplicates once.
    unique: dict[tuple[str, tuple[int, int, int, int], str], CropItem] = {}
    for item in items:
        rounded = tuple(int(round(v)) for v in item.bbox)
        unique[(item.image_name.lower(), rounded, item.class_name)] = item
    return list(unique.values())


def split_items(items: list[CropItem], val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[CropItem]]:
    by_class: dict[str, list[CropItem]] = defaultdict(list)
    for item in items:
        by_class[item.class_name].append(item)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for cls_items in by_class.values():
        shuffled = cls_items[:]
        rng.shuffle(shuffled)
        n_test = int(round(len(shuffled) * test_ratio))
        n_val = int(round(len(shuffled) * val_ratio))
        splits["test"].extend(shuffled[:n_test])
        splits["val"].extend(shuffled[n_test : n_test + n_val])
        splits["train"].extend(shuffled[n_test + n_val :])
    return splits


def clamp_bbox(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1i = max(0, min(width, int(round(x1))))
    y1i = max(0, min(height, int(round(y1))))
    x2i = max(0, min(width, int(round(x2))))
    y2i = max(0, min(height, int(round(y2))))
    return x1i, y1i, x2i, y2i


def unique_crop_path(out: Path, split: str, cls: str, image_path: Path, index: int) -> Path:
    dst = out / split / cls / f"{image_path.stem}_{index:05d}{image_path.suffix.lower()}"
    if not dst.exists():
        return dst
    return out / split / cls / f"{image_path.stem}_{index:05d}_{abs(hash(str(image_path))) % 100000}{image_path.suffix.lower()}"


def crop_items(
    items: list[CropItem],
    images_by_name: dict[str, Path],
    images_by_stem: dict[str, Path],
    out: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    min_size: int,
) -> Counter:
    stats: Counter = Counter()
    splits = split_items(items, val_ratio, test_ratio, seed)
    crop_index = 0

    for split, split_items_ in splits.items():
        for item in split_items_:
            image_path = images_by_name.get(item.image_name.lower())
            if image_path is None:
                image_path = images_by_stem.get(Path(item.image_name).stem.lower())
            if image_path is None:
                stats["missing_image"] += 1
                continue

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    x1, y1, x2, y2 = clamp_bbox(item.bbox, image.width, image.height)
                    if x2 - x1 < min_size or y2 - y1 < min_size:
                        stats["small_or_invalid_bbox"] += 1
                        continue
                    crop = image.crop((x1, y1, x2, y2))
                    dst = unique_crop_path(out, split, item.class_name, image_path, crop_index)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    crop.save(dst, quality=95)
                    crop_index += 1
                    stats[f"saved_{split}"] += 1
                    stats[f"class_{item.class_name}"] += 1
            except OSError:
                stats["image_open_error"] += 1

    return stats


def main() -> None:
    args = parse_args()
    roots = collect_roots(args.src, args.extract_dir)
    images = collect_files(roots, IMAGE_EXTS)
    labels = collect_files(roots, LABEL_EXTS)
    if args.limit:
        labels = labels[: args.limit]

    images_by_name, images_by_stem = image_indexes(images)
    all_items: list[CropItem] = []
    parse_errors = 0

    for label in labels:
        try:
            all_items.extend(parse_label_file(label, args.class_key, args.image_key))
        except (json.JSONDecodeError, OSError, ValueError):
            parse_errors += 1

    class_counts = Counter(item.class_name for item in all_items)

    print(f"Roots: {len(roots)}")
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")
    print(f"Parsed crop annotations: {len(all_items)}")
    print(f"Label parse errors: {parse_errors}")
    print("Top parsed classes:")
    for cls, count in class_counts.most_common(30):
        print(f"  {cls}: {count}")

    if args.dry_run:
        return

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    stats = crop_items(
        all_items,
        images_by_name,
        images_by_stem,
        args.out,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        args.min_size,
    )
    print("Crop stats:")
    for key, value in stats.most_common():
        if key.startswith("class_"):
            continue
        print(f"  {key}: {value}")
    print(f"Dataset written: {args.out}")


if __name__ == "__main__":
    main()
