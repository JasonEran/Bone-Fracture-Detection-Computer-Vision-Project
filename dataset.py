"""
FracAtlas PyTorch dataset utilities.

This module provides a production-ready :class:`FracAtlasDataset` that can read the
FracAtlas bone fracture dataset stored in COCO format, apply light augmentations, and
deliver tensors that work for classification, detection, or segmentation tasks.

Example
-------
```python
from torch.utils.data import DataLoader
from dataset import FracAtlasDataset, fracatlas_collate_fn

dataset = FracAtlasDataset(
    root="path/to/FracAtlas/images",
    annotation_file="path/to/FracAtlas/annotations.json",
    split="train",
    image_size=1024,
)

loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=fracatlas_collate_fn)
batch = next(iter(loader))
print(batch["image"].shape)          # torch.Size([2, 3, 1024, 1024])
print(len(batch["bbox"]))            # 2 (per-sample bounding boxes)
print(batch["metadata"][0]["file_name"])
```
"""
from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

LOGGER = logging.getLogger("fracatlas.dataset")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

FRACTURE_CLASSES: Tuple[str, ...] = (
    "Elbow",
    "Fingers",
    "Forearm",
    "Humerus",
    "Shoulder",
    "Wrist",
)

TARGET_KEYS = {"class_label", "bbox", "mask"}


def _canonical_label(name: str) -> str:
    """Canonicalize a class name to ease matching between annotations and model labels."""
    return "".join(name.lower().replace("-", " ").split())


CLASS_NAME_TO_ID = {_canonical_label(name): idx for idx, name in enumerate(FRACTURE_CLASSES)}


@dataclass
class InstanceRecord:
    bbox_xyxy: Tuple[float, float, float, float]
    class_id: int
    segmentation: Optional[Any]
    annotation_id: int
    area: float


@dataclass
class ImageRecord:
    image_id: int
    file_name: str
    file_path: Path
    width: int
    height: int
    annotations: List[InstanceRecord]
    primary_label: int


class FracAtlasSplitBuilder:
    """Utility that creates deterministic stratified train/val/test splits."""

    def __init__(
        self,
        ratios: Optional[Dict[str, float]] = None,
        *,
        seed: int = 17,
        split_order: Sequence[str] = ("train", "val", "test"),
    ) -> None:
        ratios = ratios or {"train": 0.7, "val": 0.15, "test": 0.15}
        self.ratios = self._normalize_ratios(ratios, split_order)
        self.split_order = tuple(split_order)
        self.rng = random.Random(seed)

    @staticmethod
    def _normalize_ratios(
        ratios: Dict[str, float], split_order: Sequence[str]
    ) -> Dict[str, float]:
        missing = set(split_order) - set(ratios)
        if missing:
            raise ValueError(f"Missing split ratios for: {', '.join(sorted(missing))}")
        total = sum(ratios[split] for split in split_order)
        if total <= 0:
            raise ValueError("Split ratios must sum to a positive number.")
        return {split: ratios[split] / total for split in split_order}

    def build(self, labels: Sequence[int]) -> Dict[str, List[int]]:
        """Assign dataset indices to splits using per-class stratification."""
        split_indices: Dict[str, List[int]] = {split: [] for split in self.split_order}
        label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            label_to_indices[label].append(idx)

        for label, indices in label_to_indices.items():
            self.rng.shuffle(indices)
            bucket_sizes = self._bucket_sizes(len(indices))
            cursor = 0
            for split in self.split_order:
                span = bucket_sizes[split]
                if span == 0:
                    continue
                split_indices[split].extend(indices[cursor : cursor + span])
                cursor += span

        for split in self.split_order:
            self.rng.shuffle(split_indices[split])
        return split_indices

    def _bucket_sizes(self, n_items: int) -> Dict[str, int]:
        raw = {split: int(math.floor(self.ratios[split] * n_items)) for split in self.split_order}
        assigned = sum(raw.values())
        remainder = n_items - assigned
        if remainder > 0:
            ranked = sorted(self.split_order, key=lambda split: (-self.ratios[split], split))
            for i in range(remainder):
                raw[ranked[i % len(ranked)]] += 1
        return raw


def fracatlas_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function that stacks images and keeps variable-sized annotations in lists."""
    images = torch.stack([item["image"] for item in batch], dim=0)
    return {
        "image": images,
        "class_label": [item["class_label"] for item in batch],
        "bbox": [item["bbox"] for item in batch],
        "mask": [item["mask"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


class FracAtlasDataset(Dataset):
    """
    PyTorch dataset for the FracAtlas bone fracture benchmark.

    Parameters
    ----------
    root:
        Folder that contains the image files or sub-folders referenced by `file_name`.
    annotation_file:
        Path to the COCO-format annotation JSON.
    split:
        One of ``"train"``, ``"val"``, or ``"test"``. Splits are stratified per class.
    split_ratios:
        Optional custom ratios for ``train/val/test``. Defaults to ``0.7/0.15/0.15``.
    image_size:
        Either an ``int`` for square resize or ``(height, width)`` tuple. Defaults to 1024.
    output_targets:
        Iterable subset of ``{"class_label", "bbox", "mask"}`` to populate in the sample.
        All keys are still returned but unrequested ones contain empty tensors.
    image_root:
        If provided, overrides ``root`` for image lookups (useful when annotations and
        images live in different folders).
    split_cache_path:
        Optional path to persist or read cached split indices. Defaults to
        ``<annotation_file>.splits.json``.
    persist_splits:
        When ``True`` the computed splits are written to ``split_cache_path``.
    enable_random_flip:
        Overrides whether random horizontal flips are applied. ``None`` uses
        ``True`` for train split and ``False`` otherwise.
    random_flip_prob:
        Probability for the random flip augmentation. Only applied when flips enabled.
    normalize_stats:
        Mean and std sequences passed to ``torchvision`` normalization. Defaults to ImageNet.

    Returns
    -------
    Dict[str, Any]
        A dictionary with keys ``["image", "class_label", "bbox", "mask", "metadata"]``.
    """

    def __init__(
        self,
        *,
        root: Union[str, Path],
        annotation_file: Union[str, Path],
        split: Literal["train", "val", "test"] = "train",
        split_ratios: Optional[Dict[str, float]] = None,
        image_size: Union[int, Tuple[int, int]] = 1024,
        output_targets: Optional[Sequence[str]] = None,
        image_root: Optional[Union[str, Path]] = None,
        split_cache_path: Optional[Union[str, Path]] = None,
        persist_splits: bool = True,
        enable_random_flip: Optional[bool] = None,
        random_flip_prob: float = 0.5,
        normalize_stats: Tuple[Sequence[float], Sequence[float]] = (IMAGENET_MEAN, IMAGENET_STD),
        seed: int = 17,
        skip_missing_images: bool = True,
    ) -> None:
        self.root = Path(root)
        self.annotation_file = Path(annotation_file)
        self.split = split
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unsupported split '{split}'. Expected train/val/test.")

        self.image_dir = Path(image_root) if image_root else self.root
        self.image_size = self._parse_image_size(image_size)
        self.normalize_mean = tuple(float(m) for m in normalize_stats[0])
        self.normalize_std = tuple(float(s) for s in normalize_stats[1])
        self.target_components = self._parse_target_components(output_targets)
        self.skip_missing_images = skip_missing_images

        if not (0.0 <= random_flip_prob <= 1.0):
            raise ValueError("random_flip_prob must be within [0, 1].")
        if enable_random_flip is None:
            self.enable_random_flip = split == "train"
        else:
            self.enable_random_flip = bool(enable_random_flip)
        self.random_flip_prob = random_flip_prob

        self.records: List[ImageRecord] = []
        self._validation_issues: List[str] = []
        self._load_records()

        if not self.records:
            raise RuntimeError("No valid records found. Check dataset paths and annotations.")

        self.split_cache_path = (
            Path(split_cache_path)
            if split_cache_path
            else self.annotation_file.with_suffix(".splits.json")
        )
        self.persist_splits = persist_splits
        self._indices_by_split = self._resolve_splits(split_ratios, seed)
        if split not in self._indices_by_split:
            raise RuntimeError(f"Split '{split}' not available in computed splits.")
        self._active_indices = self._indices_by_split[split]

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._active_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[self._active_indices[idx]]
        image = self._load_image(record.file_path)
        target = self._build_target(record)
        image_tensor, target = self._apply_transforms(image, target)
        sample = {
            "image": image_tensor,
            "class_label": target["labels"] if "class_label" in self.target_components else torch.zeros(0, dtype=torch.long),
            "bbox": target["boxes"] if "bbox" in self.target_components else torch.zeros((0, 4), dtype=torch.float32),
            "mask": target["masks"] if "mask" in self.target_components else torch.zeros((0, self.image_size[0], self.image_size[1]), dtype=torch.float32),
            "metadata": target["metadata"],
        }
        return sample

    @property
    def class_names(self) -> Tuple[str, ...]:
        return FRACTURE_CLASSES

    @property
    def num_classes(self) -> int:
        return len(FRACTURE_CLASSES)

    @property
    def validation_issues(self) -> List[str]:
        return list(self._validation_issues)

    def get_class_distribution(self) -> Dict[str, int]:
        counter = Counter(record.primary_label for record in self.records if record.primary_label >= 0)
        return {self.class_names[class_id]: counter.get(class_id, 0) for class_id in range(len(self.class_names))}

    def describe(self) -> str:
        """Return a short human-readable summary of the dataset."""
        info = [
            f"FracAtlasDataset(split={self.split}, size={len(self)})",
            f"  image_dir: {self.image_dir}",
            f"  annotation_file: {self.annotation_file}",
            f"  image_size: {self.image_size}",
            f"  targets: {sorted(self.target_components)}",
            f"  random_flip: {self.enable_random_flip} (p={self.random_flip_prob})",
        ]
        return "\n".join(info)

    # --------------------------------------------------------------------- #
    # Construction helpers
    # --------------------------------------------------------------------- #
    def _parse_target_components(
        self, components: Optional[Sequence[str]]
    ) -> Tuple[str, ...]:
        if not components:
            return tuple(sorted(TARGET_KEYS))
        invalid = set(components) - TARGET_KEYS
        if invalid:
            raise ValueError(f"Unknown target components: {', '.join(sorted(invalid))}")
        return tuple(sorted(set(components)))

    @staticmethod
    def _parse_image_size(image_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        if isinstance(image_size, int):
            if image_size <= 0:
                raise ValueError("image_size must be positive.")
            return (int(image_size), int(image_size))
        if (
            isinstance(image_size, (tuple, list))
            and len(image_size) == 2
            and all(isinstance(dim, int) for dim in image_size)
        ):
            h, w = image_size
            if h <= 0 or w <= 0:
                raise ValueError("image_size dimensions must be positive.")
            return (int(h), int(w))
        raise TypeError("image_size must be an int or a tuple/list of two ints (h, w).")

    def _load_records(self) -> None:
        coco = self._read_json(self.annotation_file)
        category_lookup = self._build_category_lookup(coco.get("categories", []))
        annotations_by_image = defaultdict(list)
        for annotation in coco.get("annotations", []):
            annotations_by_image[annotation.get("image_id")].append(annotation)

        for image in coco.get("images", []):
            image_id = image.get("id")
            file_name = image.get("file_name")
            if file_name is None:
                self._log_issue(f"Image entry without file_name skipped (image_id={image_id}).")
                continue
            file_path = self._resolve_image_path(file_name)
            width = int(image.get("width") or 0)
            height = int(image.get("height") or 0)
            if width <= 0 or height <= 0:
                inferred = self._probe_image_size(file_path)
                if inferred:
                    width, height = inferred
                else:
                    self._log_issue(f"Missing size metadata for {file_name}; record skipped.")
                    continue

            instance_records = self._build_instance_records(
                annotations_by_image.get(image_id, []), category_lookup, width, height
            )
            primary_label = self._infer_primary_label(instance_records)
            record = ImageRecord(
                image_id=image_id,
                file_name=file_name,
                file_path=file_path,
                width=width,
                height=height,
                annotations=instance_records,
                primary_label=primary_label,
            )
            if not self._validate_record(record):
                continue
            self.records.append(record)

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _build_category_lookup(
        self, categories: Iterable[Dict[str, Any]]
    ) -> Dict[int, int]:
        lookup = {}
        for category in categories:
            cat_id = category.get("id")
            name = category.get("name", "")
            canonical = _canonical_label(name)
            if canonical in CLASS_NAME_TO_ID:
                lookup[cat_id] = CLASS_NAME_TO_ID[canonical]
            else:
                LOGGER.debug("Skipping unknown category '%s' (id=%s).", name, cat_id)
        return lookup

    def _build_instance_records(
        self,
        annotations: Iterable[Dict[str, Any]],
        category_lookup: Dict[int, int],
        width: int,
        height: int,
    ) -> List[InstanceRecord]:
        instances: List[InstanceRecord] = []
        for annotation in annotations:
            cat_id = annotation.get("category_id")
            if cat_id not in category_lookup:
                continue
            bbox = annotation.get("bbox")
            clean_bbox = self._sanitize_bbox(bbox, width, height)
            if clean_bbox is None:
                self._log_issue(
                    f"Invalid bbox skipped (annotation_id={annotation.get('id')})."
                )
                continue
            segmentation = annotation.get("segmentation")
            instances.append(
                InstanceRecord(
                    bbox_xyxy=clean_bbox,
                    class_id=category_lookup[cat_id],
                    segmentation=segmentation,
                    annotation_id=annotation.get("id", -1),
                    area=float(annotation.get("area", 0.0)),
                )
            )
        return instances

    @staticmethod
    def _sanitize_bbox(
        bbox: Optional[Sequence[float]], width: int, height: int
    ) -> Optional[Tuple[float, float, float, float]]:
        if not bbox or len(bbox) != 4:
            return None
        x_min, y_min, bw, bh = bbox
        x_min = float(x_min)
        y_min = float(y_min)
        bw = max(float(bw), 0.0)
        bh = max(float(bh), 0.0)
        x_max = x_min + bw
        y_max = y_min + bh
        x_min = max(0.0, min(x_min, width - 1))
        y_min = max(0.0, min(y_min, height - 1))
        x_max = max(0.0, min(x_max, width))
        y_max = max(0.0, min(y_max, height))
        if x_max <= x_min or y_max <= y_min:
            return None
        return (x_min, y_min, x_max, y_max)

    def _infer_primary_label(self, instances: Sequence[InstanceRecord]) -> int:
        if not instances:
            return -1
        counter = Counter(instance.class_id for instance in instances)
        return counter.most_common(1)[0][0]

    def _validate_record(self, record: ImageRecord) -> bool:
        if not record.file_path.exists():
            message = f"Image not found: {record.file_path}"
            if self.skip_missing_images:
                self._log_issue(message)
                return False
            raise FileNotFoundError(message)
        try:
            with Image.open(record.file_path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError) as exc:
            message = f"Corrupted image skipped: {record.file_path} ({exc})"
            if self.skip_missing_images:
                self._log_issue(message)
                return False
            raise
        return True

    def _resolve_image_path(self, file_name: str) -> Path:
        candidate = Path(file_name)
        if candidate.is_absolute():
            return candidate
        return (self.image_dir / candidate).resolve()

    def _probe_image_size(self, path: Path) -> Optional[Tuple[int, int]]:
        if not path.exists():
            return None
        try:
            with Image.open(path) as img:
                return img.size
        except (UnidentifiedImageError, OSError):
            return None

    def _resolve_splits(
        self, split_ratios: Optional[Dict[str, float]], seed: int
    ) -> Dict[str, List[int]]:
        labels = [record.primary_label for record in self.records]
        builder = FracAtlasSplitBuilder(split_ratios, seed=seed)
        cached = self._maybe_read_split_cache(len(labels))
        if cached:
            return cached
        splits = builder.build(labels)
        self._maybe_write_split_cache(splits)
        return splits

    def _maybe_read_split_cache(self, expected_size: int) -> Optional[Dict[str, List[int]]]:
        if not self.persist_splits or not self.split_cache_path.exists():
            return None
        try:
            with self.split_cache_path.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
            splits = {key: value for key, value in cached.items() if isinstance(value, list)}
            total_indices = sum(len(indices) for indices in splits.values())
            if total_indices == expected_size:
                return {key: list(map(int, indices)) for key, indices in splits.items()}
        except (json.JSONDecodeError, OSError) as exc:
            self._log_issue(f"Failed to read split cache ({self.split_cache_path}): {exc}")
        return None

    def _maybe_write_split_cache(self, splits: Dict[str, List[int]]) -> None:
        if not self.persist_splits:
            return
        try:
            self.split_cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {key: list(map(int, indices)) for key, indices in splits.items()}
            with self.split_cache_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except OSError as exc:
            self._log_issue(f"Could not persist split cache: {exc}")

    # --------------------------------------------------------------------- #
    # Sample preparation
    # --------------------------------------------------------------------- #
    def _load_image(self, path: Path) -> Image.Image:
        with Image.open(path) as img:
            image = ImageOps.exif_transpose(img)
            image = image.convert("RGB")
        return image

    def _build_target(self, record: ImageRecord) -> Dict[str, Any]:
        labels = torch.tensor(
            [instance.class_id for instance in record.annotations],
            dtype=torch.long,
        )
        boxes = torch.tensor(
            [instance.bbox_xyxy for instance in record.annotations],
            dtype=torch.float32,
        )
        if boxes.ndim == 1:
            boxes = boxes.reshape(-1, 4)
        mask_images: List[Optional[Image.Image]] = []
        if "mask" in self.target_components:
            mask_images = [
                self._segmentation_to_mask(instance.segmentation, (record.width, record.height))
                for instance in record.annotations
            ]

        metadata = {
            "image_id": record.image_id,
            "file_name": record.file_name,
            "split": self.split,
            "original_size": (record.height, record.width),
            "num_instances": labels.shape[0],
            "has_annotations": bool(record.annotations),
            "targets_requested": tuple(sorted(self.target_components)),
        }
        return {
            "labels": labels,
            "boxes": boxes,
            "mask_images": mask_images,
            "metadata": metadata,
        }

    def _apply_transforms(
        self, image: Image.Image, target: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        target_h, target_w = self.image_size
        current_w, current_h = image.size
        scale_x = target_w / current_w
        scale_y = target_h / current_h
        if (current_w, current_h) != (target_w, target_h):
            image = image.resize((target_w, target_h), Image.BILINEAR)
        boxes = target["boxes"]
        if boxes.numel():
            boxes = boxes.clone()
            boxes[:, 0] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y
        flipped = False
        if self.enable_random_flip and torch.rand(1).item() < self.random_flip_prob:
            image = ImageOps.mirror(image)
            flipped = True
            if boxes.numel():
                x_min = target_w - boxes[:, 2]
                x_max = target_w - boxes[:, 0]
                boxes[:, 0] = x_min
                boxes[:, 2] = x_max
        target["boxes"] = boxes
        masks = self._prepare_masks(target.pop("mask_images", []), (target_h, target_w), flipped)
        target["masks"] = masks

        tensor = TF.pil_to_tensor(image).float().div(255.0)
        tensor = TF.normalize(tensor, self.normalize_mean, self.normalize_std)
        target["metadata"].update(
            {
                "scaled_size": (target_h, target_w),
                "scale_factors": (scale_y, scale_x),
                "flipped": flipped,
            }
        )
        return tensor, target

    def _prepare_masks(
        self, masks: Sequence[Optional[Image.Image]], size_hw: Tuple[int, int], flipped: bool
    ) -> torch.Tensor:
        target_h, target_w = size_hw
        if "mask" not in self.target_components or not masks:
            return torch.zeros((0, target_h, target_w), dtype=torch.float32)
        mask_tensors: List[torch.Tensor] = []
        for mask in masks:
            if mask is None:
                mask = Image.new("L", (target_w, target_h), 0)
            else:
                mask = mask.resize((target_w, target_h), Image.NEAREST)
            if flipped:
                mask = ImageOps.mirror(mask)
            array = np.array(mask, dtype=np.uint8, copy=False)
            mask_tensors.append(torch.from_numpy(array))
        if mask_tensors:
            stacked = torch.stack(mask_tensors, dim=0).float()
            stacked = (stacked > 0).float()
            return stacked
        return torch.zeros((0, target_h, target_w), dtype=torch.float32)

    def _segmentation_to_mask(
        self, segmentation: Optional[Any], size: Tuple[int, int]
    ) -> Optional[Image.Image]:
        width, height = size
        if not segmentation:
            return Image.new("L", (width, height), 0)
        mask_image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_image)
        if isinstance(segmentation, list):
            polygons = segmentation
            if polygons and isinstance(polygons[0], (int, float)):
                polygons = [polygons]
            for polygon in polygons:
                if len(polygon) >= 6:
                    draw.polygon(polygon, outline=1, fill=1)
        elif isinstance(segmentation, dict):
            try:
                counts = segmentation.get("counts")
                size_info = segmentation.get("size")
                if counts and size_info and len(size_info) == 2:
                    mask = self._decode_rle(segmentation)
                    if mask is not None:
                        mask_image = mask
            except AttributeError:
                pass
        mask_image = mask_image.point(lambda v: 255 if v > 0 else 0)
        return mask_image

    def _decode_rle(self, segmentation: Dict[str, Any]) -> Optional[Image.Image]:
        try:
            import pycocotools.mask as mask_utils  # type: ignore

            decoded = mask_utils.decode(segmentation)
            if decoded is None:
                return None
            array = decoded.astype(np.uint8) * 255
            height, width = array.shape
            return Image.fromarray(array.reshape((height, width)), mode="L")
        except ImportError:
            self._log_issue(
                "pycocotools not installed; unable to decode RLE segmentation. Returning empty mask."
            )
            size = segmentation.get("size", [self.image_size[0], self.image_size[1]])
            width, height = size[1], size[0]
            return Image.new("L", (width, height), 0)

    def _log_issue(self, message: str) -> None:
        self._validation_issues.append(message)
        LOGGER.warning(message)


__all__ = [
    "FracAtlasDataset",
    "fracatlas_collate_fn",
    "FracAtlasSplitBuilder",
    "FRACTURE_CLASSES",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect FracAtlas dataset statistics.")
    parser.add_argument("--root", required=True, help="Path to the FracAtlas image root.")
    parser.add_argument("--annotations", required=True, help="Path to COCO annotations json.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--image-size", type=int, default=1024, help="Output resolution (square).")
    args = parser.parse_args()

    dataset = FracAtlasDataset(
        root=args.root,
        annotation_file=args.annotations,
        split=args.split,
        image_size=args.image_size,
    )
    print(dataset.describe())
    print("Class distribution:")
    for name, count in dataset.get_class_distribution().items():
        print(f"  {name}: {count}")
