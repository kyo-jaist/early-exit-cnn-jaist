"""
Smoke test for the first runnable MNIST early-exit prototype.

This script checks:
1. MNIST can be downloaded and loaded
2. A batch can be pulled from the dataloader
3. The BranchyLeNetMNIST model can run a forward pass on CPU
4. Output tensor shapes are correct
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.branchy_lenet_mnist import BranchyLeNetMNIST


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("smoke_test_mnist")


def main() -> None:
    # Always use CPU on this VM.
    device = torch.device("cpu")
    LOGGER.info("Using device: %s", device)

    # Keep paths explicit and project-local.
    data_root = Path("data/mnist")
    data_root.mkdir(parents=True, exist_ok=True)
    LOGGER.info("MNIST data directory: %s", data_root.resolve())

    # Standard MNIST tensor transform.
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    LOGGER.info("Loading MNIST test dataset...")
    test_dataset = datasets.MNIST(
        root=str(data_root),
        train=False,
        download=True,
        transform=transform,
    )

    LOGGER.info("Test dataset size: %d", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    images, labels = next(iter(test_loader))
    LOGGER.info("Batch images shape: %s", tuple(images.shape))
    LOGGER.info("Batch labels shape: %s", tuple(labels.shape))

    model = BranchyLeNetMNIST(num_classes=10).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(images.to(device))

    exit_1 = outputs["exit_1"]
    exit_2 = outputs["exit_2"]

    LOGGER.info("exit_1 logits shape: %s", tuple(exit_1.shape))
    LOGGER.info("exit_2 logits shape: %s", tuple(exit_2.shape))

    # Hard checks. If something is wrong, fail loudly.
    assert images.shape == (8, 1, 28, 28), f"Unexpected input shape: {images.shape}"
    assert exit_1.shape == (8, 10), f"Unexpected exit_1 shape: {exit_1.shape}"
    assert exit_2.shape == (8, 10), f"Unexpected exit_2 shape: {exit_2.shape}"

    print("\nSMOKE TEST PASSED")
    print("Input batch shape:", tuple(images.shape))
    print("exit_1 logits shape:", tuple(exit_1.shape))
    print("exit_2 logits shape:", tuple(exit_2.shape))


if __name__ == "__main__":
    main()
