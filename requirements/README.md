# Dependency environments

[← Back to FakeScope](../README.md) · [Installation](../docs/quickstart.md#backbone-environments) · [Training](../docs/training.md)

These files follow Q-Instruct's approach of installing each backbone in its own environment. The target is **Python 3.10 on Linux x86_64 with NVIDIA CUDA**. Choose one runtime per environment; the two PyTorch and Transformers stacks are incompatible with each other.

| File | Purpose |
| :--- | :--- |
| [`../requirements.txt`](../requirements.txt) | Shared dependencies; does not install a complete model runtime by itself |
| [`llava.txt`](llava.txt) | FakeScope-L inference, including shared dependencies |
| [`mplugowl2.txt`](mplugowl2.txt) | FakeScope-M inference, including shared dependencies |
| [`train-llava.txt`](train-llava.txt) | FakeScope-L runtime plus training dependencies |
| [`train-mplugowl2.txt`](train-mplugowl2.txt) | FakeScope-M runtime plus training dependencies |

The requirements install dependencies, not the `llava` or `mplug_owl2` source package. Install the matching upstream checkout in editable mode in the same pip command, as shown in the [setup guide](../docs/quickstart.md#backbone-environments).

## Core versions

| Package | FakeScope-L | FakeScope-M |
| :--- | :--- | :--- |
| PyTorch | 2.1.2 | 2.0.1 |
| torchvision | 0.16.2 | 0.15.2 |
| Transformers | 4.37.2 | 4.31.0 |
| tokenizers | 0.15.1 | 0.13.3 |
| accelerate | 0.21.0 | 0.21.0 |
| PEFT | 0.7.1 | 0.4.0 |
| bitsandbytes | 0.41.3 | 0.41.0 |
| Pydantic | 2.5.3 | 1.10.13 |
| DeepSpeed, for training | 0.12.6 | 0.9.5 |
| FlashAttention, separate training install | 2.5.2 | 2.5.2 |

NumPy is pinned to 1.26.4 for the older compiled dependencies. The shared Hugging Face Hub version and the LLaVA PEFT/bitsandbytes versions are compatibility choices for these legacy runtimes. The pinned Hub version uses `huggingface-cli login`; the newer `hf` download commands in the setup guide should run in a separate, current Hugging Face CLI environment.

## Training installation order

1. Install the selected inference requirements and upstream source package first, so PyTorch is available.
2. Install the corresponding training requirements and the upstream `[train]` extra.
3. Install `flash-attn==2.5.2` with `--no-build-isolation` after PyTorch, packaging, wheel, and Ninja are present.
4. Run `python -m pip check` in that environment.

FlashAttention is intentionally installed separately: its build imports PyTorch and may compile CUDA extensions. Compilation requires a compatible local CUDA toolkit (`nvcc`) and compiler, in addition to the GPU driver. Follow the upstream build guidance for your hardware; a successful dependency resolution does not verify CUDA binaries.

## Sources and validation

Q-Instruct does not provide a single root requirements file. Its training guides install the respective backbone's `.[train]` extra and FlashAttention. These profiles use:

- [Q-Instruct's mPLUG-Owl2 dependency declarations](https://github.com/Q-Future/Q-Instruct/blob/2a78134356b829da873ceb905542dead1efa8f6c/eval_scripts/mplug_owl_2/pyproject.toml), alongside its [LLaVA training guide](https://github.com/Q-Future/Q-Instruct/blob/2a78134356b829da873ceb905542dead1efa8f6c/scripts/llava_v1.5/README.md) and [mPLUG-Owl2 training guide](https://github.com/Q-Future/Q-Instruct/blob/2a78134356b829da873ceb905542dead1efa8f6c/scripts/mplug_owl_2/README.md).
- [LLaVA `pyproject.toml`](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/pyproject.toml).
- [mPLUG-Owl2 `pyproject.toml`](https://github.com/X-PLUG/mPLUG-Owl/blob/0f3068fdb47b77aedc71fc39f5735cd7aa35e8f9/mPLUG-Owl2/pyproject.toml).

These are source-based installation profiles, not an export of the authors' original training environment or a complete lockfile. Core versions are fixed; some supporting dependencies retain version ranges. Dependency resolution is checked for the target Python/platform, including DeepSpeed's declared source-distribution dependencies. CUDA extension builds, model imports on a GPU, and full training remain unverified in the release environment.
