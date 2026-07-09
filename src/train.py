from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Ultralytics YOLO model.")
    parser.add_argument("--task", choices=["detect", "classify"], required=True)
    parser.add_argument("--model", required=True, help="Example: yolo26s.pt or yolo26s-cls.pt")
    parser.add_argument("--data", required=True, help="Dataset YAML for detect, dataset root for classify.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model, task=args.task)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
