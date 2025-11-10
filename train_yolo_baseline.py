"""
Train a YOLOv8s-seg baseline on the FracAtlas dataset with multi-task objectives.

This script wraps Ultralytics' trainer with project-specific conveniences:
  * Converts the Roboflow export (7 classes) into the 6 canonical FracAtlas labels.
  * Creates a dedicated dataset YAML pointing at the prepared directory.
  * Applies the requested augmentation, optimizer, and LR schedule settings.
  * Enables TensorBoard logging and ensures best/last checkpoints are saved.

Usage:
    python train_yolo_baseline.py --config yolo_config.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import yaml
import torch
from ultralytics import YOLO

LOGGER = logging.getLogger("fracatlas.yolo.train")


def setup_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FracAtlas YOLOv8 baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("yolo_config.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Force regeneration of the prepared YOLO dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g., '0', '0,1', 'cpu'). Defaults to config.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def compute_signature(payload: Dict) -> str:
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def ensure_image_link(src: Path, dst: Path, strategy: str) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if strategy == "hardlink":
        try:
            os.link(src, dst)
            return
        except (AttributeError, OSError):
            LOGGER.warning("Hardlink failed for %s; falling back to copy.", src.name)
    shutil.copy2(src, dst)


def remap_label_file(src: Path, dst: Path, remap: Dict[int, int]) -> bool:
    if not src.exists():
        return False
    new_lines: List[str] = []
    for raw_line in src.read_text().splitlines():
        parts = raw_line.strip().split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        if cls not in remap:
            continue
        parts[0] = str(remap[cls])
        new_lines.append(" ".join(parts))
    if not new_lines:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return True


def prepare_fracatlas_dataset(config: Dict, force: bool = False) -> Path:
    dataset_cfg = config["dataset"]
    source_dir = Path(dataset_cfg["source_dir"]).resolve()
    prepared_dir = Path(dataset_cfg["prepared_dir"]).resolve()
    splits: Iterable[str] = dataset_cfg.get("splits", ("train", "valid", "test"))
    names: List[str] = dataset_cfg["class_names"]
    remap = {int(k): int(v) for k, v in dataset_cfg["class_remap"].items()}
    signature_payload = {
        "source": str(source_dir),
        "names": names,
        "remap": remap,
    }
    signature = compute_signature(signature_payload)
    marker = prepared_dir / ".prepared_signature"
    if marker.exists() and not force:
        if marker.read_text().strip() == signature:
            LOGGER.info("Prepared dataset already up to date at %s.", prepared_dir)
            return write_data_yaml(prepared_dir, names)
        LOGGER.info("Prepared dataset signature mismatch; rebuilding.")

    if prepared_dir.exists() and force:
        shutil.rmtree(prepared_dir)

    strategy = dataset_cfg.get("copy_strategy", "hardlink").lower()
    total_images = 0
    total_labels = 0
    for split in splits:
        src_images = source_dir / split / "images"
        src_labels = source_dir / split / "labels"
        dst_images = prepared_dir / split / "images"
        dst_labels = prepared_dir / split / "labels"
        if not src_images.exists():
            LOGGER.warning("Split '%s' missing at %s.", split, src_images)
            continue
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(src_images.glob("*")):
            if not image_path.is_file():
                continue
            dst_image_path = dst_images / image_path.name
            ensure_image_link(image_path, dst_image_path, strategy)
            total_images += 1
            label_src = src_labels / f"{image_path.stem}.txt"
            label_dst = dst_labels / f"{image_path.stem}.txt"
            if remap_label_file(label_src, label_dst, remap):
                total_labels += 1
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(signature, encoding="utf-8")
    LOGGER.info(
        "Prepared dataset at %s (images=%d, labels=%d).", prepared_dir, total_images, total_labels
    )
    return write_data_yaml(prepared_dir, names)


def write_data_yaml(prepared_dir: Path, class_names: List[str]) -> Path:
    data_yaml = prepared_dir / "fracatlas_data.yaml"
    payload = {
        "path": str(prepared_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": class_names,
    }
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    with data_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return data_yaml


def build_train_kwargs(cfg: Dict, data_yaml: Path, override_device: str | None) -> Dict:
    train_cfg = cfg["training"]
    aug_cfg = cfg.get("augmentation", {})
    checkpoint_cfg = cfg.get("checkpoint", {})
    kwargs = {
        "data": str(data_yaml),
        "epochs": train_cfg["epochs"],
        "imgsz": train_cfg["imgsz"],
        "batch": train_cfg["batch"],
        "optimizer": train_cfg.get("optimizer", "AdamW"),
        "lr0": train_cfg["lr0"],
        "lrf": train_cfg.get("lrf", 0.01),
        "cos_lr": bool(train_cfg.get("cos_lr", True)),
        "device": override_device or train_cfg.get("device", "auto"),
        "workers": train_cfg.get("workers", 8),
        "amp": bool(train_cfg.get("amp", True)),
        "patience": train_cfg.get("patience", 30),
        "pretrained": bool(train_cfg.get("pretrained", True)),
        "project": train_cfg.get("project", "runs/fracatlas"),
        "name": train_cfg.get("run_name", "yolov8s-seg-baseline"),
        "exist_ok": True,
        "val": True,
        "save": True,
        "save_period": checkpoint_cfg.get("save_period", -1),
        "resume": bool(train_cfg.get("resume", False)),
    }
    kwargs.update(aug_cfg)
    return kwargs


def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if args.device is None:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            args.device = "0"
        else:
            args.device = "cpu"
    data_yaml = prepare_fracatlas_dataset(cfg, force=args.rebuild_dataset)
    model_path = cfg["training"]["model"]
    LOGGER.info("Loading YOLO model weights from %s", model_path)
    model = YOLO(model_path)
    train_kwargs = build_train_kwargs(cfg, data_yaml, args.device)
    LOGGER.info("Starting training with parameters: %s", train_kwargs)
    results = model.train(**train_kwargs)
    run_dir = Path(train_kwargs["project"]) / train_kwargs["name"]
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    LOGGER.info("Training complete.")
    LOGGER.info("Best model: %s", best_path)
    LOGGER.info("Last checkpoint: %s", last_path)
    if isinstance(results, dict):
        LOGGER.info("Final metrics: %s", results)
    LOGGER.info(
        "Launch TensorBoard with: tensorboard --logdir %s",
        cfg.get("logging", {}).get("tensorboard_dir", "runs/fracatlas"),
    )


if __name__ == "__main__":
    main()
