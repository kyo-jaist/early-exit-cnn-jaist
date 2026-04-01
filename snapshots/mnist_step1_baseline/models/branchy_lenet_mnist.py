"""
LeNet-style early-exit model for MNIST.

This version is a true early-exit structure:
- exit_1 branches after stage1
- exit_2 branches after stage2

That makes the two exits differ in computational depth, which is necessary
for meaningful early-exit timing and exit-ratio experiments.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
from torch import nn


LOGGER = logging.getLogger(__name__)


class BranchyLeNetMNIST(nn.Module):
    """
    A small CNN with:
    - stage1: shallow shared feature extractor
    - exit_1: early classifier after stage1
    - stage2: deeper shared feature extractor
    - exit_2: final classifier after stage2
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # Input: [B, 1, 28, 28]
        # Output after stage1: [B, 5, 12, 12]
        self.features_stage1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Early-exit head uses shallow features.
        self.exit_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5 * 12 * 12, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

        # Input: [B, 5, 12, 12]
        # Output after stage2: [B, 10, 4, 4]
        self.features_stage2 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Final-exit head uses deeper features.
        self.exit_2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10 * 4 * 4, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

        LOGGER.info("Initialized BranchyLeNetMNIST with %d classes", num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass that returns logits for both exits.
        """
        LOGGER.debug("Input tensor shape: %s", tuple(x.shape))

        x_stage1 = self.features_stage1(x)
        LOGGER.debug("After stage1: %s", tuple(x_stage1.shape))

        exit_1_logits = self.exit_1(x_stage1)
        LOGGER.debug("exit_1 logits shape: %s", tuple(exit_1_logits.shape))

        x_stage2 = self.features_stage2(x_stage1)
        LOGGER.debug("After stage2: %s", tuple(x_stage2.shape))

        exit_2_logits = self.exit_2(x_stage2)
        LOGGER.debug("exit_2 logits shape: %s", tuple(exit_2_logits.shape))

        return {
            "exit_1": exit_1_logits,
            "exit_2": exit_2_logits,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    model = BranchyLeNetMNIST()
    dummy = torch.randn(4, 1, 28, 28)
    outputs = model(dummy)

    print("exit_1 shape =", tuple(outputs["exit_1"].shape))
    print("exit_2 shape =", tuple(outputs["exit_2"].shape))
