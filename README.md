# Drifting-Models-MNIST

This repository contains a compact MNIST reimplementation of the drifting-model pipeline.

## Workflow

1. Train the latent VAE backbone with `train_ae.py`.
2. Train the latent drift model with `train.py`.
3. Sample digits with `generate.py`.

## Training

```bash
python train_ae.py
python train.py --epochs 30
```

## Generation

```bash
python generate.py --digit 7 --num_samples 16 --output_path ./outputs/generated_digits.png
```

Checkpoints are written to `./checkpoints` and image grids to `./outputs`.
