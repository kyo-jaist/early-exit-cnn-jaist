"""
Minimal end-to-end training script for the MNIST early-exit prototype.

This is the corrected version using a true early-exit architecture:
- exit_1 after stage1
- exit_2 after stage2
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.models.branchy_lenet_mnist import BranchyLeNetMNIST


LOGGER = logging.getLogger("train_mnist_prototype")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    LOGGER.info("Random seed set to %d", seed)


def build_dataloaders(batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    """Create MNIST train/test dataloaders."""
    data_root = Path("data/mnist")
    data_root.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Using MNIST data directory: %s", data_root.resolve())

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    LOGGER.info("Loading MNIST training dataset...")
    train_dataset = datasets.MNIST(
        root=str(data_root),
        train=True,
        download=True,
        transform=transform,
    )

    LOGGER.info("Loading MNIST test dataset...")
    test_dataset = datasets.MNIST(
        root=str(data_root),
        train=False,
        download=True,
        transform=transform,
    )

    LOGGER.info("Train dataset size: %d", len(train_dataset))
    LOGGER.info("Test dataset size: %d", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Evaluate accuracy for exit_1 and exit_2 independently."""
    model.eval()

    exit_1_correct = 0
    exit_2_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            pred_1 = outputs["exit_1"].argmax(dim=1)
            pred_2 = outputs["exit_2"].argmax(dim=1)

            exit_1_correct += (pred_1 == labels).sum().item()
            exit_2_correct += (pred_2 == labels).sum().item()
            total += labels.size(0)

    exit_1_acc = exit_1_correct / total
    exit_2_acc = exit_2_correct / total

    return exit_1_acc, exit_2_acc


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch_index: int,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()

    running_loss = 0.0
    total_samples = 0

    progress = tqdm(loader, desc=f"Epoch {epoch_index}", leave=True)

    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        # Weighted multi-exit training loss.
        loss_exit_1 = criterion(outputs["exit_1"], labels)
        loss_exit_2 = criterion(outputs["exit_2"], labels)
        loss = 0.3 * loss_exit_1 + 1.0 * loss_exit_2

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        avg_loss_so_far = running_loss / total_samples
        progress.set_postfix(loss=f"{avg_loss_so_far:.4f}")

    return running_loss / total_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MNIST early-exit prototype.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    LOGGER.info("Starting MNIST prototype training.")
    LOGGER.info("Arguments: %s", vars(args))

    set_seed(args.seed)

    device = torch.device("cpu")
    LOGGER.info("Using device: %s", device)

    train_loader, test_loader = build_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = BranchyLeNetMNIST(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        avg_train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch_index=epoch,
        )

        exit_1_acc, exit_2_acc = evaluate(
            model=model,
            loader=test_loader,
            device=device,
        )

        LOGGER.info(
            "Epoch %d finished | train_loss=%.6f | exit_1_acc=%.4f | exit_2_acc=%.4f",
            epoch,
            avg_train_loss,
            exit_1_acc,
            exit_2_acc,
        )

    checkpoint_path = results_dir / "mnist_branchy_style_last.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
        },
        checkpoint_path,
    )
    LOGGER.info("Saved checkpoint to %s", checkpoint_path.resolve())

    print("\nTRAINING PROTOTYPE PASSED")
    print(f"checkpoint: {checkpoint_path}")
    print("This confirms the corrected early-exit model and training loop run end-to-end.")


if __name__ == "__main__":
    main()
