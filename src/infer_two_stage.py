from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


COCO_VEHICLE_CLASS_IDS = [2, 7]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run car/truck detection and vehicle-type classification.")
    parser.add_argument("--det-weights", default="yolo26s.pt")
    parser.add_argument("--cls-weights", required=True)
    parser.add_argument("--source", required=True, type=Path, help="Image file or image directory.")
    parser.add_argument("--out", default="runs/two_stage", type=Path)
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--cls-imgsz", type=int, default=224)
    parser.add_argument("--classes", type=int, nargs="*", default=COCO_VEHICLE_CLASS_IDS)
    parser.add_argument(
        "--dedupe-iou",
        type=float,
        default=0.8,
        help="Remove lower-confidence boxes that overlap a kept box above this IoU. Set >1 to disable.",
    )
    return parser.parse_args()


def iter_images(source: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def draw_label(image, box, label: str) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), (32, 180, 80), 2)
    cv2.putText(
        image,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (32, 180, 80),
        2,
        cv2.LINE_AA,
    )


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedupe_boxes(boxes, iou_threshold: float):
    if iou_threshold > 1:
        return list(boxes)

    sorted_boxes = sorted(boxes, key=lambda box: float(box.conf[0].item()), reverse=True)
    kept = []
    kept_xyxy = []
    for box in sorted_boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        if any(box_iou(xyxy, kept_box) >= iou_threshold for kept_box in kept_xyxy):
            continue
        kept.append(box)
        kept_xyxy.append(xyxy)
    return kept


def main() -> None:
    args = parse_args()
    det_model = YOLO(args.det_weights, task="detect") #탐지 - 1단계
    cls_model = YOLO(args.cls_weights, task="classify") #분류 - 2단계

    crops_dir = args.out / "crops"
    annotated_dir = args.out / "annotated"
    crops_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for image_path in iter_images(args.source):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        det_results = det_model.predict(
            source=str(image_path),
            conf=args.det_conf,
            classes=args.classes,
            verbose=False,
        )[0]
        
        #중복 탐지 방지 필터 거친 뒤 탐지된 차량에 대해서 분류 실행함 (for문)
        boxes = dedupe_boxes(det_results.boxes, args.dedupe_iou)  # 중복 탐지 방지(car, truck)

        for idx, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            crop = image[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
            if crop.size == 0:
                continue

            crop_path = crops_dir / f"{image_path.stem}_{idx:03d}{image_path.suffix.lower()}"
            cv2.imwrite(str(crop_path), crop)

            cls_result = cls_model.predict(source=str(crop_path), imgsz=args.cls_imgsz, verbose=False)[0]
            top1 = int(cls_result.probs.top1)
            cls_name = cls_result.names[top1]
            cls_conf = float(cls_result.probs.top1conf)

            det_cls_id = int(box.cls[0].item())
            det_name = det_results.names[det_cls_id]
            det_conf = float(box.conf[0].item())
            label = f"{det_name} {det_conf:.2f} | {cls_name} {cls_conf:.2f}"

            draw_label(image, xyxy, label)
            rows.append(
                {
                    "image": str(image_path),
                    "crop": str(crop_path),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "det_class": det_name,
                    "det_conf": f"{det_conf:.4f}",
                    "cls_class": cls_name,
                    "cls_conf": f"{cls_conf:.4f}",
                }
            )

        cv2.imwrite(str(annotated_dir / image_path.name), image)

    csv_path = args.out / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "image",
            "crop",
            "x1",
            "y1",
            "x2",
            "y2",
            "det_class",
            "det_conf",
            "cls_class",
            "cls_conf",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Predictions written: {csv_path}")
    print(f"Annotated images: {annotated_dir}")
    print(f"Crops: {crops_dir}")


if __name__ == "__main__":
    main()
