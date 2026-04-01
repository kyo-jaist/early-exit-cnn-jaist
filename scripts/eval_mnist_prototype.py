"""
Prototype evaluator for the corrected MNIST early-exit model.

This version measures:
- forced per-exit accuracy
- threshold-based exit ratio
- approximate per-exit inference time

Timing is still for development/debugging only.
Formal timing must later be rerun on unified HAKUSAN hardware.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.branchy_lenet_mnist import BranchyLeNetMNIST


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("eval_mnist_prototype")


def load_test_loader(batch_size: int) -> DataLoader:
    """Build MNIST test dataloader."""
    data_root = Path("data/mnist")
    data_root.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_dataset = datasets.MNIST(
        root=str(data_root),
        train=False,
        download=True,
        transform=transform,
    )

    LOGGER.info("Loaded MNIST test dataset with %d samples", len(test_dataset))

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def evaluate_forced_exit_accuracy(
    model: BranchyLeNetMNIST,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate forced accuracy for each exit independently."""
    model.eval()

    correct_1 = 0
    correct_2 = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            pred_1 = outputs["exit_1"].argmax(dim=1)
            pred_2 = outputs["exit_2"].argmax(dim=1)

            correct_1 += (pred_1 == labels).sum().item()
            correct_2 += (pred_2 == labels).sum().item()
            total += labels.size(0)

    return {
        "exit_1_accuracy": correct_1 / total,
        "exit_2_accuracy": correct_2 / total,
    }


def measure_per_exit_time(
    model: BranchyLeNetMNIST,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int = 5,
    measure_batches: int = 20,
) -> Dict[str, float]:
    """
    Measure approximate cumulative inference time in milliseconds per sample.

    exit_1 path:
        stage1 -> exit_1

    exit_2 path:
        stage1 -> stage2 -> exit_2
    """
    model.eval()

    measured_exit_1_times: List[float] = []
    measured_exit_2_times: List[float] = []

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)

            if batch_idx < warmup_batches:
                _ = model(images)
                continue

            if batch_idx >= warmup_batches + measure_batches:
                break

            # Measure exit_1 cumulative path.
            start_1 = time.perf_counter()
            x1 = model.features_stage1(images)
            out_1 = model.exit_1(x1)
            end_1 = time.perf_counter()

            # Measure exit_2 cumulative path.
            start_2 = time.perf_counter()
            x1 = model.features_stage1(images)
            x2 = model.features_stage2(x1)
            out_2 = model.exit_2(x2)
            end_2 = time.perf_counter()

            assert out_1.shape[0] == images.shape[0]
            assert out_2.shape[0] == images.shape[0]

            batch_size = images.shape[0]

            per_sample_1_ms = ((end_1 - start_1) * 1000.0) / batch_size
            per_sample_2_ms = ((end_2 - start_2) * 1000.0) / batch_size

            measured_exit_1_times.append(per_sample_1_ms)
            measured_exit_2_times.append(per_sample_2_ms)

    return {
        "exit_1_time_ms": sum(measured_exit_1_times) / len(measured_exit_1_times),
        "exit_2_time_ms": sum(measured_exit_2_times) / len(measured_exit_2_times),
    }


def simulate_early_exit_ratio_and_overall_accuracy(
    model: BranchyLeNetMNIST,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    """
    Simulate threshold-based early exit using exit_1 confidence.
    """
    model.eval()

    exit_1_count = 0
    exit_2_count = 0
    total = 0
    total_correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probs_1 = F.softmax(outputs["exit_1"], dim=1)
            conf_1, pred_1 = probs_1.max(dim=1)
            pred_2 = outputs["exit_2"].argmax(dim=1)

            use_exit_1 = conf_1 >= threshold
            final_pred = torch.where(use_exit_1, pred_1, pred_2)

            exit_1_count += use_exit_1.sum().item()
            exit_2_count += (~use_exit_1).sum().item()
            total_correct += (final_pred == labels).sum().item()
            total += labels.size(0)

    return {
        "threshold": threshold,
        "exit_1_ratio": exit_1_count / total,
        "exit_2_ratio": exit_2_count / total,
        "overall_accuracy": total_correct / total,
    }


def save_csv(rows: List[Dict[str, float]], output_path: Path) -> None:
    """Save rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info("Saved CSV to %s", output_path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MNIST early-exit prototype.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/mnist_branchy_style_last.pt",
        help="Path to saved checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Evaluation batch size.",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    LOGGER.info("Using device: %s", device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    LOGGER.info("Loading checkpoint from %s", checkpoint_path.resolve())

    model = BranchyLeNetMNIST(num_classes=10).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loader = load_test_loader(batch_size=args.batch_size)

    forced_acc = evaluate_forced_exit_accuracy(model, test_loader, device)
    timing = measure_per_exit_time(model, test_loader, device)

    thresholds = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
    threshold_rows: List[Dict[str, float]] = []

    for threshold in thresholds:
        sim_result = simulate_early_exit_ratio_and_overall_accuracy(
            model=model,
            loader=test_loader,
            device=device,
            threshold=threshold,
        )

        row = {
            "threshold": sim_result["threshold"],
            "exit_1_ratio": sim_result["exit_1_ratio"],
            "exit_2_ratio": sim_result["exit_2_ratio"],
            "overall_accuracy": sim_result["overall_accuracy"],
            "exit_1_accuracy_forced": forced_acc["exit_1_accuracy"],
            "exit_2_accuracy_forced": forced_acc["exit_2_accuracy"],
            "exit_1_time_ms_per_sample": timing["exit_1_time_ms"],
            "exit_2_time_ms_per_sample": timing["exit_2_time_ms"],
        }
        threshold_rows.append(row)

    csv_path = Path("results/mnist_branchy_style_eval.csv")
    save_csv(threshold_rows, csv_path)

    print("\nEVALUATION PROTOTYPE PASSED")
    print(f"checkpoint: {checkpoint_path}")
    print(f"csv: {csv_path}")
    print("\nPreview:")
    for row in threshold_rows:
        print(row)


if __name__ == "__main__":
    main()
