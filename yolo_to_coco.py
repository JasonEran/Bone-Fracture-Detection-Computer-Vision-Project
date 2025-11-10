"""
Convert the Roboflow YOLO export under ``BoneFractureYolo8`` into a single COCO json.

The resulting file matches the FracAtlasDataset expectation (6 canonical classes).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image

BASE = Path("BoneFractureYolo8").resolve()
SPLITS = ("train", "valid", "test")

# Map Roboflow class indices to the 6 canonical fracture categories.
YOLO_TO_CANONICAL = {
    0: "Elbow",
    1: "Fingers",
    2: "Forearm",
    3: "Humerus",
    4: "Humerus",  # Collapse duplicate humerus labels into one canonical class.
    5: "Shoulder",
    6: "Wrist",
}

CATEGORIES = [
    {"id": 1, "name": "Elbow"},
    {"id": 2, "name": "Fingers"},
    {"id": 3, "name": "Forearm"},
    {"id": 4, "name": "Humerus"},
    {"id": 5, "name": "Shoulder"},
    {"id": 6, "name": "Wrist"},
]

CATEGORY_NAME_TO_ID = {category["name"]: category["id"] for category in CATEGORIES}


def polygon_area(coords_xy: Sequence[float]) -> float:
    """Return polygon area using the shoelace formula."""
    if len(coords_xy) < 6:
        return 0.0
    xs = coords_xy[0::2] + coords_xy[0:1]
    ys = coords_xy[1::2] + coords_xy[1:2]
    area = 0.0
    for i in range(len(xs) - 1):
        area += xs[i] * ys[i + 1] - xs[i + 1] * ys[i]
    return abs(area) * 0.5


def convert_dataset() -> Dict[str, List[Dict]]:
    images: List[Dict] = []
    annotations: List[Dict] = []
    image_id = 1
    annotation_id = 1

    for split in SPLITS:
        image_dir = BASE / split / "images"
        label_dir = BASE / split / "labels"
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                with Image.open(image_path) as image_fp:
                    width, height = image_fp.size
            except OSError:
                continue
            rel_name = str(image_path.relative_to(BASE))
            images.append(
                {
                    "id": image_id,
                    "file_name": rel_name.replace("\\", "/"),
                    "width": width,
                    "height": height,
                }
            )
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                for line in label_path.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(float(parts[0]))
                    if class_id not in YOLO_TO_CANONICAL:
                        continue
                    coords = list(map(float, parts[1:]))
                    if len(coords) % 2 != 0:
                        continue
                    abs_coords: List[float] = []
                    for x, y in zip(coords[0::2], coords[1::2]):
                        abs_coords.extend([x * width, y * height])
                    if len(abs_coords) < 6:
                        continue
                    xs = abs_coords[0::2]
                    ys = abs_coords[1::2]
                    x_min = max(0.0, min(xs))
                    y_min = max(0.0, min(ys))
                    x_max = min(width, max(xs))
                    y_max = min(height, max(ys))
                    bbox = [
                        x_min,
                        y_min,
                        max(0.0, x_max - x_min),
                        max(0.0, y_max - y_min),
                    ]
                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": CATEGORY_NAME_TO_ID[YOLO_TO_CANONICAL[class_id]],
                            "segmentation": [abs_coords],
                            "bbox": bbox,
                            "area": polygon_area(abs_coords),
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1
            image_id += 1

    return {
        "info": {
            "description": "FracAtlas-style COCO JSON converted from Roboflow YOLO labels.",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }


def main() -> None:
    output_path = BASE / "fracatlas_coco.json"
    payload = convert_dataset()
    output_path.write_text(json.dumps(payload))
    print(
        f"Wrote {output_path} | images={len(payload['images'])} annotations={len(payload['annotations'])}"
    )


if __name__ == "__main__":
    main()
