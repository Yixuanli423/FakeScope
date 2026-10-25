# Train FakeScope

[← Back to FakeScope](../README.md#training) · [Inference](inference.md) · [Training data](datasets.md)

The public launchers are based on the authors' FakeScope training scripts, which were adapted from Q-Instruct's **after** strategy: start from an instruction-tuned multimodal backbone, then fine-tune on FakeInstruct. See the original [Q-Instruct LLaVA guide](https://github.com/Q-Future/Q-Instruct/tree/main/scripts/llava_v1.5) and [mPLUG-Owl2 guide](https://github.com/Q-Future/Q-Instruct/tree/main/scripts/mplug_owl_2).

| Variant | Training launcher | Starting checkpoint |
| :--- | :--- | :--- |
| FakeScope-L | [`fake_instruct_llava_v1.5_7b.sh`](../scripts/train/fake_instruct_llava_v1.5_7b.sh) | `liuhaotian/llava-v1.5-7b` |
| FakeScope-M | [`fake_instruct_mplugowl2.sh`](../scripts/train/fake_instruct_mplugowl2.sh) | `MAGAer13/mplug-owl2-llama2-7b` |

## 1. Install the matching training environment

First complete the [backbone environment setup](quickstart.md#backbone-environments). This installs PyTorch before DeepSpeed or FlashAttention need it during setup. Keep the two environments separate, and run the following commands from the FakeScope repository root in the matching activated environment:

```bash
# FakeScope-L: LLAVA_ROOT points to the upstream checkout used during setup.
python -m pip install -r requirements/train-llava.txt -e "${LLAVA_ROOT}[train]"
python -m pip install flash-attn==2.5.2 --no-build-isolation
python -m pip check
```

```bash
# FakeScope-M: MPLUG_OWL2_ROOT points to mPLUG-Owl/mPLUG-Owl2.
python -m pip install -r requirements/train-mplugowl2.txt -e "${MPLUG_OWL2_ROOT}[train]"
python -m pip install flash-attn==2.5.2 --no-build-isolation
python -m pip check
```

The [requirements guide](../requirements/README.md) lists the per-backbone versions and CUDA build prerequisites. DeepSpeed is pinned to **0.12.6 for LLaVA** and **0.9.5 for mPLUG-Owl2**, following their upstream training extras. FlashAttention is installed separately after PyTorch and the build tools; it may require a compatible local CUDA toolkit and compiler.

The launchers use the matching backbone checkout's `scripts/zero3.json`; they do not substitute one backbone's ZeRO configuration for the other. Set `DEEPSPEED_CONFIG` to use your own configuration.

## 2. Prepare FakeInstruct

Request the training data download link by emailing [yixuan.li@bristol.ac.uk](mailto:yixuan.li@bristol.ac.uk?subject=FakeInstruct%20training%20data%20access%20request). Preserve the supplied annotations and image layout. An image record uses the backbone's conversation schema:

```json
{
  "id": "example-id",
  "image": "relative/path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat is the authenticity of this image?"},
    {"from": "gpt", "value": "<the supplied annotated response>"}
  ]
}
```

This is a schema illustration, not a training sample. `DATA_PATH` points to the JSON array of records, and `IMAGE_FOLDER` is the root against which each `image` path is resolved. The training schema (`image`, `conversations`) differs from the inference manifest (`image_id`).

**mPLUG-Owl2 requires `<|image|>` markers.** Q-Instruct's mPLUG-Owl2 guide calls out this conversion, and the upstream [token definition](https://github.com/X-PLUG/mPLUG-Owl/blob/main/mPLUG-Owl2/mplug_owl2/constants.py) uses `<|image|>`. If the supplied annotations still use LLaVA's `<image>` marker, create a separate copy:

```bash
# From the FakeScope repository root
python scripts/prepare_mplug_data.py \
  --input-file /absolute/path/to/FakeInstruct/FakeInstruct.json \
  --output-file /absolute/path/to/FakeInstruct/modified.json
```

The converter changes conversation text only, preserves image paths and other fields, and refuses to overwrite an existing file. Already-converted markers remain unchanged. It loads the annotation array into memory. If your data package already includes the mPLUG-Owl2 annotations, point `DATA_PATH` at that file directly.

## 3. Launch training

Run these commands from the FakeScope repository root in the matching environment. Paths are configurable; no private machine paths are embedded in the scripts.

### FakeScope-L

```bash
export LLAVA_ROOT=/absolute/path/to/LLaVA
export DATA_PATH=/absolute/path/to/FakeInstruct/FakeInstruct.json
export IMAGE_FOLDER=/absolute/path/to/FakeInstruct/Image
export NUM_GPUS=8

# Inspect the resolved command without starting DeepSpeed.
bash scripts/train/fake_instruct_llava_v1.5_7b.sh --dry-run

# Start training.
bash scripts/train/fake_instruct_llava_v1.5_7b.sh
```

### FakeScope-M

```bash
export MPLUG_OWL2_ROOT=/absolute/path/to/mPLUG-Owl/mPLUG-Owl2
export DATA_PATH=/absolute/path/to/FakeInstruct/modified.json
export IMAGE_FOLDER=/absolute/path/to/FakeInstruct/Image
export NUM_GPUS=8

bash scripts/train/fake_instruct_mplugowl2.sh --dry-run
bash scripts/train/fake_instruct_mplugowl2.sh
```

`--dry-run` checks the training entry point, ZeRO configuration, annotation file, and image directory, then prints the command. It does not import the models, download weights, or start a training job. It does not validate every annotation or image.

| Setting | How to customize it |
| :--- | :--- |
| Backbone directory | `LLAVA_ROOT` or `MPLUG_OWL2_ROOT` |
| Data | `DATA_PATH`, `IMAGE_FOLDER` |
| Starting checkpoint | `MODEL_PATH` |
| Output directory | `OUTPUT_DIR`; defaults to `checkpoints/llava-v1.5-7b-fakeinstruct` or `checkpoints/mplug-owl2-fakeinstruct` in this repository |
| GPU count | `NUM_GPUS`; if unset, DeepSpeed uses its default device discovery |
| Rendezvous port | `MASTER_PORT`; defaults to `29500` for LLaVA and `25901` for mPLUG-Owl2 |
| DeepSpeed configuration | `DEEPSPEED_CONFIG`; defaults to the respective backbone's `scripts/zero3.json` |
| Experiment logging | `REPORT_TO`; defaults to `wandb` for LLaVA and `tensorboard` for mPLUG-Owl2; set `none` to disable |
| Other trainer options | Append the upstream trainer arguments to the launcher |

For example, to reduce LLaVA's per-device batch size while keeping its effective batch size unchanged at the same GPU count:

```bash
bash scripts/train/fake_instruct_llava_v1.5_7b.sh \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8
```

Additional trainer arguments are appended after the defaults, so explicit overrides take precedence. The supplied defaults use large per-device batches. Choose a microbatch that fits your hardware; effective batch size is **per-device batch × GPU count × gradient accumulation**.

## Released launcher defaults

These values are preserved from the authors' supplied scripts, including the differences between the two backbones.

| Setting | FakeScope-M launcher | FakeScope-L launcher |
| :--- | :---: | :---: |
| Per-device training batch | 128 | 32 |
| Gradient accumulation | 1 | 1 |
| Epochs | 1 | 1 |
| Learning rate | 2e-5 | 2e-5 |
| Schedule | Cosine | Cosine |
| Warmup ratio | 0.02 | 0.03 |
| Weight decay | 0 | 0 |
| Precision | bfloat16 | bfloat16 |
| Maximum sequence length | 2048 | 2048 |
| Image aspect ratio | pad | pad |
| Group by modality length | True | False |
| Save every N steps | 800 | 50000 |
| Gradient checkpointing | True | True |
| `tune_visual_abstractor` | True | — |
| `freeze_vision_model` | **False** | Uses upstream LLaVA vision-encoder behavior |

At eight GPUs, these launcher settings imply effective batches of **1,024 for M** and **256 for L**. The mPLUG-Owl2 launcher explicitly leaves the vision model trainable, as in the supplied script. This release keeps those settings visible rather than silently replacing them with manuscript values.

## Manuscript recipe and reproduction scope

The main manuscript describes a frozen vision encoder and fine-tuning of the language model and modality projector, with no image-level augmentation. Its supplementary table reports:

| Setting | FakeScope-M in paper | FakeScope-L in paper |
| :--- | :---: | :---: |
| Input resolution | 448 × 448 | 336 × 336 |
| Batch size (as listed) | 432 | 256 |
| Gradient accumulation | 2 | 1 |
| Warmup ratio | 0.03 | 0.03 |
| Learning rate | 2e-5 | 2e-5 |
| Epochs | 1 | 1 |

The supplement does not specify whether its batch-size row is per-device or global. The main manuscript reports eight NVIDIA H20-NVLink 98GB GPUs and approximately 34 hours of training. The supplied mPLUG-Owl2 launcher differs in batch size, accumulation, warmup, and vision freezing; it should not be described as an exact reproduction of that table.

The public launchers are checked for shell syntax, argument forwarding, and dry-run configuration. Full GPU training and reproduction of the reported metrics have not been run in this release environment. Keep the upstream revision, resolved command, ZeRO configuration, data split, and training logs with any new checkpoint, and retain the [evaluation exclusions](datasets.md#evaluation-separation) before comparing with the [reported results](results.md).

Sources: [FakeScope paper](https://arxiv.org/abs/2503.24267), the authors' supplied training launchers, [Q-Instruct](https://github.com/Q-Future/Q-Instruct), and the upstream [LLaVA](https://github.com/haotian-liu/LLaVA) and [mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2) runtimes.
