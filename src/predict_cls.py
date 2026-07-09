from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict vehicle model classes with a YOLO classification model.")
    parser.add_argument("--weights", required=True, type=Path, help="Classification model weights, usually best.pt.")
    parser.add_argument("--source", required=True, type=Path, help="Image file or image directory.")
    parser.add_argument("--out", default="runs/vehicle_cls_predict", type=Path)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", action="store_true", help="Ask Ultralytics to save prediction visualizations.")
    return parser.parse_args()


def iter_images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTS:
        return [source]
    return sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights), task="classify")
    rows: list[dict[str, str]] = []

    for image_path in iter_images(args.source):
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            device=args.device,
            save=args.save,
            project=str(args.out),
            name="visualized",
            exist_ok=True,
            verbose=False,
        )[0]

        top_indices = result.probs.top5[: args.topk]
        row = {"image": str(image_path)}
        for rank, class_idx in enumerate(top_indices, start=1):
            row[f"top{rank}_class"] = result.names[int(class_idx)]
            row[f"top{rank}_conf"] = f"{float(result.probs.data[int(class_idx)]):.6f}"
        rows.append(row)

    fieldnames = ["image"]
    for rank in range(1, args.topk + 1):
        fieldnames.extend([f"top{rank}_class", f"top{rank}_conf"])

    csv_path = args.out / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Images predicted: {len(rows)}")
    print(f"Predictions CSV: {csv_path}")


if __name__ == "__main__":
    main()
