from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an Ultralytics classification dataset from raw vehicle images."
    )
    parser.add_argument("--src", required=True, type=Path, help="Raw image root.")
    parser.add_argument("--out", required=True, type=Path, help="Output dataset root.")
    parser.add_argument(
        "--class-regex",
        default=None,
        help=(
            "Regex with one capture group for class name. "
            "Default uses the first filename token before _, -, space, or dot."
        ),
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--dry-run", action="store_true", help="Only write audit CSV.")
    return parser.parse_args()


def iter_images(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def class_from_path(path: Path, src: Path, class_regex: str | None) -> str | None:
    rel = path.relative_to(src)

    if len(rel.parts) > 1:
        parent = rel.parts[0].strip()
        if parent:
            return normalize_class(parent)

    stem = path.stem
    if class_regex:
        match = re.search(class_regex, stem)
        if not match:
            return None
        return normalize_class(match.group(1))

    token = re.split(r"[_\-\s.]+", stem, maxsplit=1)[0]
    return normalize_class(token) if token else None


def normalize_class(name: str) -> str:
    normalized = re.sub(r"\s+", "_", name.strip())
    normalized = re.sub(r"[^0-9A-Za-z가-힣_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def split_items(items: list[Path], val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))

    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def unique_destination(root: Path, split: str, cls: str, src: Path) -> Path:
    base = root / split / cls / src.name
    if not base.exists():
        return base

    digest = abs(hash(src.resolve())) % 10_000_000
    return root / split / cls / f"{src.stem}_{digest}{src.suffix.lower()}"


def write_audit(out: Path, rows: list[dict[str, str]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    audit_path = out / "class_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class", "status"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    images = iter_images(args.src)

    grouped: dict[str, list[Path]] = defaultdict(list)
    audit_rows: list[dict[str, str]] = []

    for image in images:
        cls = class_from_path(image, args.src, args.class_regex)
        status = "ok" if cls else "unparsed"
        audit_rows.append({"path": str(image), "class": cls or "", "status": status})
        if cls:
            grouped[cls].append(image)

    write_audit(args.out, audit_rows)

    counts = Counter({cls: len(paths) for cls, paths in grouped.items()})
    print(f"Images found: {len(images)}")
    print(f"Parsed images: {sum(counts.values())}")
    print(f"Unparsed images: {len(images) - sum(counts.values())}")
    print("Top classes:")
    for cls, count in counts.most_common(30):
        print(f"  {cls}: {count}")
    print(f"Audit written: {args.out / 'class_audit.csv'}")

    if args.dry_run:
        return

    for cls, paths in grouped.items():
        splits = split_items(paths, args.val_ratio, args.test_ratio, args.seed)
        for split, split_paths in splits.items():
            for src_path in split_paths:
                dst_path = unique_destination(args.out, split, cls, src_path)
                copy_or_link(src_path, dst_path, args.copy_mode)

    print(f"Classification dataset written: {args.out}")


if __name__ == "__main__":
    main()
