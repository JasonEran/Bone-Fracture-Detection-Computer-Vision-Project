"""
Run inference with a trained YOLOv8s-seg model on FracAtlas images and visualize predictions.

Example:
    python infer_yolo.py --weights runs/fracatlas/yolov8s-seg-baseline/weights/best.pt \\
        --source BoneFractureYolo8/test/images --save-dir predictions
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FracAtlas YOLOv8 inference utility.")
    parser.add_argument("--weights", required=True, type=Path, help="Path to trained YOLO weights.")
    parser.add_argument(
        "--source",
        required=True,
        type=str,
        help="Image, directory, glob, or video source accepted by Ultralytics.",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference resolution.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS.")
    parser.add_argument("--device", type=str, default="auto", help="Device string (e.g., '0' or 'cpu').")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("runs/fracatlas_infer"),
        help="Directory to store visualizations.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Limit number of samples processed (-1 for all).",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to dump detections in JSON format.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display predictions after saving (may block execution).",
    )
    return parser.parse_args()


def ndarray_to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim != 3:
        raise ValueError("Expected HxWxC image for visualization.")
    # Ultralytics returns BGR arrays from OpenCV.
    rgb = image[..., ::-1]
    return Image.fromarray(rgb)


def result_summary(result) -> Dict[str, int]:
    counts: Counter = Counter()
    if result.boxes is not None and result.boxes.cls is not None:
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for cls_id in classes:
            counts[result.names[int(cls_id)]] += 1
    return dict(counts)


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights))
    args.save_dir.mkdir(parents=True, exist_ok=True)
    aggregated = Counter()
    json_payload: List[Dict] = []

    predictions = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        stream=True,
        save=False,
        verbose=True,
    )

    for idx, result in enumerate(predictions, start=1):
        if 0 < args.max_samples < idx:
            break
        summary = result_summary(result)
        aggregated.update(summary)
        plotted = result.plot()
        output_path = args.save_dir / f"{Path(result.path).stem}_pred.jpg"
        ndarray_to_pil(plotted).save(output_path)
        if args.show:
            ndarray_to_pil(plotted).show()
        detections = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(xyxy, confs, classes):
                detections.append(
                    {
                        "class_id": int(cls_id),
                        "class_name": result.names[int(cls_id)],
                        "confidence": float(conf),
                        "bbox_xyxy": [float(v) for v in box.tolist()],
                    }
                )
        json_payload.append(
            {
                "image_path": str(result.path),
                "output_path": str(output_path),
                "detections": detections,
                "summary": summary,
            }
        )
        print(f"[{idx}] Saved visualization to {output_path} | detections={summary}")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
        print(f"Wrote detection metadata to {args.save_json}")

    print("Aggregated detections:")
    for class_name, count in aggregated.most_common():
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
