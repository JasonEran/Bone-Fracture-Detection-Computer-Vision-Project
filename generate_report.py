"""
Generate a Markdown performance report for a YOLOv8 FracAtlas training run.

The script expects a Ultralytics run directory containing results.csv/args.yaml.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - plotting optional
    MATPLOTLIB_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a YOLOv8 training report.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to the Ultralytics run directory (contains results.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown output path (defaults to <run-dir>/performance_report.md).",
    )
    parser.add_argument(
        "--target-map",
        type=float,
        default=0.562,
        help="Target mAP@0.5 for comparison (FracAtlas baseline).",
    )
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records: List[Dict[str, float]] = []
        for row in reader:
            record: Dict[str, float] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    record[key] = float(value)
                except ValueError:
                    record[key] = value
            records.append(record)
    if not records:
        raise ValueError(f"No rows found in {path}")
    return records


def select_key(record: Dict[str, float], candidates: Sequence[str]) -> str:
    for key in candidates:
        if key in record:
            return key
    raise KeyError(f"None of the candidate keys found in record: {candidates}")


def maybe_plot(
    epochs: List[int],
    metric: List[float],
    losses: Dict[str, List[float]],
    output_path: Path,
) -> Optional[Path]:
    if not MATPLOTLIB_AVAILABLE:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, metric, label="mAP@0.5")
    for name, values in losses.items():
        plt.plot(epochs, values, label=name)
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def load_run_args(run_dir: Path) -> Dict:
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        return {}
    with args_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"Could not find results.csv under {run_dir}")
    records = read_csv(results_csv)
    metric_key = select_key(records[0], ["metrics/mAP50(B)", "metrics/mAP50"])
    metric_curve = [float(record.get(metric_key, 0.0)) for record in records]
    epochs = [int(record.get("epoch", idx)) for idx, record in enumerate(records)]
    best_idx = max(range(len(metric_curve)), key=lambda i: metric_curve[i])
    best_epoch = epochs[best_idx]
    best_metric = metric_curve[best_idx]
    status = "target met" if best_metric >= args.target_map else "below target"

    loss_keys = [key for key in records[0] if "loss" in key and key.startswith(("train/", "val/"))]
    loss_curves = {
        key: [float(record.get(key, 0.0)) for record in records] for key in loss_keys
    }
    plot_path = maybe_plot(
        epochs,
        metric_curve,
        loss_curves,
        run_dir / "performance_plot.png",
    )
    run_args = load_run_args(run_dir)
    output_path = args.output or (run_dir / "performance_report.md")
    lines = [
        "# FracAtlas YOLOv8 Baseline Report",
        f"- Run directory: `{run_dir}`",
        f"- Target mAP@0.5: {args.target_map:.3f}",
        f"- Best epoch: {best_epoch}",
        f"- Best mAP@0.5: {best_metric:.4f} ({status})",
        "",
        "## Metric Trajectory",
    ]
    if plot_path:
        lines.append(f"![Training curves]({plot_path.name})")
    else:
        lines.append("Matplotlib not available; skipping plot generation.")
    lines.append("")
    lines.append("## Loss Breakdown (last epoch)")
    if loss_keys:
        last_losses = {key: loss_curves[key][-1] for key in loss_keys}
        lines.append("```json")
        lines.append(json.dumps(last_losses, indent=2))
        lines.append("```")
    else:
        lines.append("No loss keys detected in results.csv.")
    lines.append("")
    lines.append("## Trainer Arguments")
    if run_args:
        lines.append("```yaml")
        lines.append(yaml.safe_dump(run_args, sort_keys=False))
        lines.append("```")
    else:
        lines.append("args.yaml not found; unable to display trainer arguments.")
    lines.append("")
    lines.append("## Next Steps")
    if best_metric < args.target_map:
        lines.append(
            "- Consider longer training, stronger augmentations, or hyperparameter search to reach the target."
        )
    else:
        lines.append("- Baseline meets or exceeds the FracAtlas target. Proceed to downstream evaluation.")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
