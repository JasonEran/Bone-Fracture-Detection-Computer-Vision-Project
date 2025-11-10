"""
Comprehensive dataset analysis tool for fracture detection datasets stored in COCO format.

The script computes core statistics, validates annotations, measures image quality,
and generates visual reports plus a Markdown summary.

Example:
    python analyze_dataset.py \
        --images-root BoneFractureYolo8 \
        --annotations BoneFractureYolo8/fracatlas_coco.json \
        --splits train valid test \
        --sample-count 20
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError

try:
    import pycocotools.mask as mask_utils  # type: ignore

    HAS_PYCOCO = True
except Exception:  # pragma: no cover
    HAS_PYCOCO = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a COCO-format fracture dataset.")
    parser.add_argument("--images-root", type=Path, required=True, help="Root folder containing images.")
    parser.add_argument("--annotations", type=Path, required=True, help="Path to COCO annotations JSON.")
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "valid", "test"],
        help="Dataset splits to inspect under images-root (expects <split>/images).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis"),
        help="Directory to store plots, samples, and the Markdown report.",
    )
    parser.add_argument("--sample-count", type=int, default=20, help="Number of random samples to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    return parser.parse_args()


def load_coco(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_image_path(images_root: Path, file_name: str) -> Path:
    candidate = Path(file_name)
    if candidate.is_absolute():
        return candidate
    return (images_root / candidate).resolve()


def compute_split_stats(images_root: Path, splits: Sequence[str]) -> Dict[str, int]:
    split_counts: Dict[str, int] = {}
    for split in splits:
        image_dir = images_root / split / "images"
        if image_dir.exists():
            count = sum(1 for _ in image_dir.glob("*.*"))
            split_counts[split] = count
    return split_counts


def plot_class_distribution(class_counts: Dict[str, int], output_path: Path) -> None:
    labels = list(class_counts.keys())
    values = [class_counts[label] for label in labels]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color=plt.cm.tab20(np.linspace(0, 1, len(labels))))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Annotations")
    plt.title("Class Distribution")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def ensure_dirs(output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    return output_dir, samples_dir


def build_image_index(images: Iterable[Dict]) -> Dict[int, Dict]:
    return {image["id"]: image for image in images}


def build_annotation_index(annotations: Iterable[Dict]) -> Dict[int, List[Dict]]:
    mapping: Dict[int, List[Dict]] = defaultdict(list)
    for ann in annotations:
        mapping[ann["image_id"]].append(ann)
    return mapping


def polygon_to_array(polygon: Sequence[float]) -> np.ndarray:
    arr = np.array(polygon).reshape(-1, 2)
    return arr


def decode_segmentation(segmentation: Dict, size: Tuple[int, int]) -> Optional[np.ndarray]:
    if HAS_PYCOCO:
        try:
            mask = mask_utils.decode(segmentation)
            return mask
        except Exception:
            return None
    width, height = size
    return np.zeros((height, width), dtype=np.uint8)


def visualize_samples(
    sample_ids: List[int],
    images_root: Path,
    image_index: Dict[int, Dict],
    annotation_index: Dict[int, List[Dict]],
    categories: Dict[int, str],
    output_dir: Path,
) -> None:
    from matplotlib import patches

    cmap = plt.cm.get_cmap("tab20", len(categories))
    for image_id in sample_ids:
        image_info = image_index[image_id]
        annotations = annotation_index.get(image_id, [])
        image_path = resolve_image_path(images_root, image_info["file_name"])
        if not image_path.exists():
            continue
        try:
            with Image.open(image_path) as img:
                image = ImageOps.exif_transpose(img).convert("RGB")
        except (UnidentifiedImageError, OSError):
            continue
        width, height = image.size
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        ax.set_axis_off()
        for ann in annotations:
            color = cmap(ann["category_id"] % cmap.N)
            x, y, w, h = ann["bbox"]
            rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(
                x,
                y - 2,
                categories.get(ann["category_id"], str(ann["category_id"])),
                color="white",
                fontsize=8,
                bbox=dict(facecolor=color, alpha=0.6, edgecolor="none"),
            )
            segmentation = ann.get("segmentation")
            if not segmentation:
                continue
            if isinstance(segmentation, list):
                polygons = segmentation if isinstance(segmentation[0], list) else [segmentation]
                for polygon in polygons:
                    if len(polygon) >= 6:
                        poly = polygon_to_array(polygon)
                        poly_patch = patches.Polygon(poly, closed=True, facecolor=color, alpha=0.3, edgecolor="none")
                        ax.add_patch(poly_patch)
            elif isinstance(segmentation, dict):
                mask = decode_segmentation(segmentation, (width, height))
                if mask is not None:
                    ax.imshow(np.ma.masked_where(mask.squeeze() == 0, mask.squeeze()), cmap="spring", alpha=0.3)
        output_path = output_dir / f"sample_{image_id}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close(fig)


def compute_image_quality(
    images_root: Path,
    image_index: Dict[int, Dict],
) -> Tuple[List[Tuple[int, int]], List[float], List[float], List[int]]:
    resolutions: List[Tuple[int, int]] = []
    brightness_values: List[float] = []
    contrast_values: List[float] = []
    corrupted: List[int] = []
    for image_id, info in image_index.items():
        image_path = resolve_image_path(images_root, info["file_name"])
        try:
            with Image.open(image_path) as img:
                image = ImageOps.exif_transpose(img).convert("RGB")
        except (UnidentifiedImageError, OSError):
            corrupted.append(image_id)
            continue
        width, height = image.size
        resolutions.append((width, height))
        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        brightness_values.append(stat.mean[0])
        contrast_values.append(stat.stddev[0])
    return resolutions, brightness_values, contrast_values, corrupted


def plot_resolution(resolutions: List[Tuple[int, int]], output_path: Path) -> None:
    if not resolutions:
        return
    widths = [w for w, _ in resolutions]
    heights = [h for _, h in resolutions]
    plt.figure(figsize=(6, 6))
    plt.scatter(widths, heights, alpha=0.4, s=10)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Resolution Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_hist(values: List[float], title: str, xlabel: str, output_path: Path, bins: int = 30) -> None:
    if not values:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=bins, color="#4C72B0", alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def categorize_bbox_sizes(bbox_ratios: List[float]) -> Dict[str, int]:
    buckets = {"small": 0, "medium": 0, "large": 0}
    for ratio in bbox_ratios:
        if ratio < 0.01:
            buckets["small"] += 1
        elif ratio < 0.05:
            buckets["medium"] += 1
        else:
            buckets["large"] += 1
    return buckets


def plot_size_pie(buckets: Dict[str, int], output_path: Path) -> None:
    plt.figure(figsize=(5, 5))
    labels = list(buckets.keys())
    sizes = [buckets[label] for label in labels]
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=plt.cm.Pastel1(np.linspace(0, 1, len(labels))))
    plt.title("Fracture Size Distribution (bbox area ratio)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_report(
    output_path: Path,
    summary: Dict[str, object],
    figure_paths: Dict[str, Path],
) -> None:
    def rel(path: Optional[Path]) -> str:
        if path is None:
            return ""
        try:
            return str(path.relative_to(output_path.parent))
        except ValueError:
            return str(path)

    lines = [
        "# Dataset Analysis Report",
        "",
        "## Dataset Overview",
        f"- Total images: **{summary['total_images']}**",
        f"- Total annotations: **{summary['total_annotations']}**",
        f"- Average annotations per image: **{summary['avg_annotations_per_image']:.2f}**",
    ]
    if summary.get("split_counts"):
        lines.append("- Split distribution:")
        for split, count in summary["split_counts"].items():
            lines.append(f"  - {split}: {count} images")
    lines.extend(
        [
            "",
            "## Class Distribution",
            f"![Class Distribution]({rel(figure_paths.get('class_dist'))})" if figure_paths.get("class_dist") else "",
            "",
            "## Annotation Quality",
            f"- Images without annotations: **{summary['images_without_annotations']}**",
            f"- Sample visualizations saved under `{rel(summary['samples_dir'])}`",
            f"- Corrupted / unreadable images: **{len(summary['corrupted_images'])}**",
            "",
            "## Image Quality",
            f"- Resolution range: {summary['resolution_min']} to {summary['resolution_max']} pixels",
            f"- Mean brightness: {summary['brightness_mean']:.2f}",
            f"- Mean contrast: {summary['contrast_mean']:.2f}",
            f"![Resolution Scatter]({rel(figure_paths.get('resolution'))})" if figure_paths.get("resolution") else "",
            f"![Brightness Histogram]({rel(figure_paths.get('brightness'))})" if figure_paths.get("brightness") else "",
            f"![Contrast Histogram]({rel(figure_paths.get('contrast'))})" if figure_paths.get("contrast") else "",
            "",
            "## Fracture Difficulty Metrics",
            f"- Bounding box area ratio (mean): **{summary['bbox_area_mean']:.4f}**",
            f"- Median bbox area ratio: **{summary['bbox_area_median']:.4f}**",
            f"- Small/Medium/Large counts: {summary['size_buckets']}",
            f"- Images with multiple fractures: **{summary['multi_fracture_images']}**",
            f"- Images with >=2 classes: **{summary['multi_class_images']}**",
            f"![BBox Area Histogram]({rel(figure_paths.get('bbox_hist'))})" if figure_paths.get("bbox_hist") else "",
            f"![Size Pie]({rel(figure_paths.get('size_pie'))})" if figure_paths.get("size_pie") else "",
            "",
            "## Notes",
            "- Review the sampled overlays for potential labeling issues (missing masks, misaligned boxes).",
            "- Brightness and contrast outliers may indicate imaging inconsistencies that affect training.",
            "- Size distribution helps determine anchor settings and augmentation requirements.",
        ]
    )
    output_path.write_text("\n".join(line for line in lines if line is not None), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_dir, samples_dir = ensure_dirs(args.output_dir)
    coco = load_coco(args.annotations)
    image_index = build_image_index(coco.get("images", []))
    annotation_index = build_annotation_index(coco.get("annotations", []))
    category_lookup = {category["id"]: category["name"] for category in coco.get("categories", [])}

    total_images = len(image_index)
    total_annotations = len(coco.get("annotations", []))
    split_counts = compute_split_stats(args.images_root, args.splits)

    class_counts = Counter()
    for ann in coco.get("annotations", []):
        class_counts[category_lookup.get(ann["category_id"], str(ann["category_id"]))] += 1
    class_plot = output_dir / "class_distribution.png"
    plot_class_distribution(class_counts, class_plot)

    annotations_per_image = [len(annotation_index.get(image_id, [])) for image_id in image_index]
    avg_annotations = statistics.mean(annotations_per_image) if annotations_per_image else 0.0
    images_without_annotations = sum(1 for count in annotations_per_image if count == 0)

    sample_ids = random.sample(list(image_index.keys()), min(args.sample_count, total_images))
    visualize_samples(sample_ids, args.images_root, image_index, annotation_index, category_lookup, samples_dir)

    resolutions, brightness_values, contrast_values, corrupted_images = compute_image_quality(
        args.images_root, image_index
    )
    if resolutions:
        widths = [w for w, _ in resolutions]
        heights = [h for _, h in resolutions]
        resolution_min = f"{min(widths)}x{min(heights)}"
        resolution_max = f"{max(widths)}x{max(heights)}"
    else:
        resolution_min = resolution_max = "N/A"
    brightness_mean = statistics.mean(brightness_values) if brightness_values else 0.0
    contrast_mean = statistics.mean(contrast_values) if contrast_values else 0.0

    resolution_plot = output_dir / "resolution_scatter.png"
    plot_resolution(resolutions, resolution_plot)
    brightness_plot = output_dir / "brightness_hist.png"
    plot_hist(brightness_values, "Brightness Distribution", "Brightness (mean pixel)", brightness_plot)
    contrast_plot = output_dir / "contrast_hist.png"
    plot_hist(contrast_values, "Contrast Distribution", "Contrast (std dev)", contrast_plot)

    bbox_area_ratios: List[float] = []
    for image_id, ann_list in annotation_index.items():
        image_info = image_index.get(image_id)
        if not image_info:
            continue
        width = image_info.get("width") or 0
        height = image_info.get("height") or 0
        image_area = max(width * height, 1)
        for ann in ann_list:
            area = float(ann.get("area") or (ann["bbox"][2] * ann["bbox"][3]))
            bbox_area_ratios.append(area / image_area)
    bbox_hist_path = output_dir / "bbox_area_hist.png"
    plot_hist(bbox_area_ratios, "Bounding Box Area Ratio", "Area / Image Area", bbox_hist_path, bins=40)
    bbox_area_mean = statistics.mean(bbox_area_ratios) if bbox_area_ratios else 0.0
    bbox_area_median = statistics.median(bbox_area_ratios) if bbox_area_ratios else 0.0

    size_buckets = categorize_bbox_sizes(bbox_area_ratios)
    size_pie_path = output_dir / "size_distribution.png"
    plot_size_pie(size_buckets, size_pie_path)

    multi_fracture_images = sum(1 for count in annotations_per_image if count >= 2)
    multi_class_images = 0
    for image_id, anns in annotation_index.items():
        unique_classes = {ann["category_id"] for ann in anns}
        if len(unique_classes) >= 2:
            multi_class_images += 1

    figure_paths = {
        "class_dist": class_plot,
        "resolution": resolution_plot,
        "brightness": brightness_plot,
        "contrast": contrast_plot,
        "bbox_hist": bbox_hist_path,
        "size_pie": size_pie_path,
    }
    summary = {
        "total_images": total_images,
        "total_annotations": total_annotations,
        "avg_annotations_per_image": avg_annotations,
        "images_without_annotations": images_without_annotations,
        "split_counts": split_counts,
        "brightness_mean": brightness_mean,
        "contrast_mean": contrast_mean,
        "resolution_min": resolution_min,
        "resolution_max": resolution_max,
        "bbox_area_mean": bbox_area_mean,
        "bbox_area_median": bbox_area_median,
        "size_buckets": size_buckets,
        "multi_fracture_images": multi_fracture_images,
        "multi_class_images": multi_class_images,
        "corrupted_images": corrupted_images,
        "samples_dir": samples_dir,
    }
    report_path = output_dir / "dataset_report.md"
    build_report(report_path, summary, figure_paths)
    print(f"Analysis complete. Report saved to {report_path}")


if __name__ == "__main__":
    main()
