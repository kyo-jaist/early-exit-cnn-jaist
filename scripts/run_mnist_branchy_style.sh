#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source .venv/bin/activate
export PYTHONPATH=.

echo "===== STEP 1: SMOKE TEST ====="
python scripts/smoke_test_mnist.py

echo
echo "===== STEP 2: TRAIN ====="
python scripts/train_mnist_prototype.py \
  --epochs 1 \
  --batch-size 128 \
  --num-workers 0 \
  --lr 1e-3

echo
echo "===== STEP 3: EVAL ====="
python scripts/eval_mnist_prototype.py \
  --checkpoint results/mnist_branchy_style_last.pt \
  --batch-size 128

echo
echo "===== DONE ====="
echo "Checkpoint: results/mnist_branchy_style_last.pt"
echo "CSV: results/mnist_branchy_style_eval.csv"
