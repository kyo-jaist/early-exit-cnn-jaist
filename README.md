# early-exit-cnn-jaist

Modern Python 3 + PyTorch prototype project for early-exit CNN experiments inspired by BranchyNet.

## Current status

Step 1 baseline has been completed on the JAIST general VM.

Completed items:
- Python 3 virtual environment
- Official BranchyNet repository cloned as reference only
- MNIST branchy-style prototype implemented in PyTorch
- Smoke test script
- 1-epoch training prototype
- Evaluation prototype with CSV export
- Step 1 baseline snapshot
- GitHub repository initialized
- Git tag: `step1-mnist-baseline`

## Important policy

The current JAIST VM is only for:
- environment setup
- code development
- debugging
- prototype verification

Formal CPU/GPU timing results must later be rerun on unified HAKUSAN hardware.

## Project structure

- `src/`: source code
- `scripts/`: runnable scripts
- `requirements/`: environment freeze files
- `snapshots/`: saved baseline snapshots
- `refs/`: external reference repositories, not tracked in Git
- `data/`: local datasets, not tracked in Git
- `results/`: local outputs, not tracked in Git

## Step 1 runnable command

Run the Step 1 baseline pipeline with:

    cd ~/early_exit_project
    source .venv/bin/activate
    ./scripts/run_mnist_branchy_style.sh

## Current Step 1 outputs

Main files:
- `scripts/run_mnist_branchy_style.sh`
- `scripts/train_mnist_prototype.py`
- `scripts/eval_mnist_prototype.py`
- `results/mnist_branchy_style_last.pt`
- `results/mnist_branchy_style_eval.csv`

## Git baseline

- branch: `main`
- tag: `step1-mnist-baseline`