# Training recipe

[← Back to FakeScope](../README.md#training)

This page records the training strategy and hyperparameters from the manuscript and supplementary material. Checkpoint-specific training scripts and data conversion commands have not yet been added to this repository.

## Strategy

Start from the corresponding pretrained multimodal model and apply supervised visual instruction tuning on FakeInstruct. The main manuscript specifies a frozen vision encoder, with full-parameter fine-tuning of the language model and modality projector. Each variant is trained for one epoch, and a single final checkpoint is used across evaluations. No image-level augmentation is applied during training.

## Reported hyperparameters

| Setting | FakeScope-M | FakeScope-L |
| :--- | :---: | :---: |
| Backbone | mPLUG-Owl2 · 7B | LLaVA-v1.5 · 7B |
| Vision encoder | CLIP-L/14 | CLIP-L/14 |
| Image resolution | 448 × 448 | 336 × 336 |
| Batch size (as listed in supplement) | 432 | 256 |
| Gradient accumulation | 2 | 1 |
| Peak learning rate | 2e-5 | 2e-5 |
| Learning-rate schedule | Cosine decay | Cosine decay |
| Warmup ratio | 0.03 | 0.03 |
| Weight decay | 0 | 0 |
| Optimizer | AdamW | AdamW |
| Precision | bfloat16 | bfloat16 |
| Epochs | 1 | 1 |
| Optimizer sharding | Enabled | Enabled |
| Activation checkpointing | Enabled | Enabled |

The main manuscript reports eight NVIDIA H20-NVLink 98GB GPUs and approximately 34 hours of training. The supplement’s batch-size row is reproduced as written; it does not specify whether that value is per-device or global. Do not treat it as a per-device command-line setting without checking the original training configuration.

## Before reproducing

[Request the training data by email](mailto:yixuan.li@bristol.ac.uk?subject=FakeInstruct%20training%20data%20access%20request), then use the accompanying documentation to resolve the annotation fields and image paths. Keep the original backbone’s image preprocessing and conversation formatting, and retain the evaluation data exclusions described in [dataset notes](datasets.md#evaluation-separation). Match the full training configuration before comparing a reproduction to the [reported results](results.md).

Sources: [FakeScope paper](https://arxiv.org/abs/2503.24267), “Implementation Details” and the supplementary “Hyper-parameters of Training” table; upstream [LLaVA](https://github.com/haotian-liu/LLaVA) and [mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2) runtimes.
