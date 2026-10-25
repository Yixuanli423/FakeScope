# Inference setup

[← Back to FakeScope](../README.md#getting-started)

FakeScope uses the runtime of its underlying multimodal model. **FakeScope-M** and **FakeScope-L** require different packages and conversation templates. After setup, follow [Run FakeScope](inference.md) for the public inference launchers and LLaVA authenticity scorer.

## Model access and downloads

| Variant | Hugging Face repository | Conversation template |
| :--- | :--- | :--- |
| FakeScope-M | [FakeScope/mplug_owl_2_fakeinstruct_v3](https://huggingface.co/FakeScope/mplug_owl_2_fakeinstruct_v3) | `mplug_owl2` |
| FakeScope-L | [FakeScope/llava_1.5_fakeinstruct_v3](https://huggingface.co/FakeScope/llava_1.5_fakeinstruct_v3) | `llava_v1` |

Both repositories use **manual access approval**. Sign in on the relevant Hugging Face model page, request access, and wait for approval before downloading. Authenticate the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) with an account that has access:

```bash
hf auth login
```

Install the CLI following its official guide if `hf` is unavailable. A separate download environment avoids changing the older backbone runtimes’ dependencies.

From your FakeScope repository directory, download the variant you want to use:

```bash
# FakeScope-M
hf download FakeScope/mplug_owl_2_fakeinstruct_v3 \
  --local-dir checkpoints/mplug_owl2_fakeinstruct_v3

# FakeScope-L
hf download FakeScope/llava_1.5_fakeinstruct_v3 \
  --local-dir checkpoints/llava_1.5_fakeinstruct_v3
```

Keep the **`mplug_owl2`** spelling in the local FakeScope-M directory name: the upstream loader uses that substring to select its multimodal model implementation. The Hugging Face repository ID intentionally remains `mplug_owl_2_fakeinstruct_v3`.

The public file listings contain sharded model weights and tokenizer files for both variants. Model configuration files require approved access, so the runtime examples below have not been verified against those configurations or through end-to-end GPU inference.

## Backbone environments

The [requirements profiles](../requirements/README.md) follow Q-Instruct and the respective upstream backbone dependencies. Use **Python 3.10 on a Linux x86_64 CUDA machine**, with a separate environment for each model. Start from the FakeScope repository root in each example. The upstream revisions below match the dependency files.

### FakeScope-M: mPLUG-Owl2

```bash
python3.10 -m venv .venv/mplugowl2
source .venv/mplugowl2/bin/activate

export MPLUG_OWL2_ROOT=/absolute/path/to/mPLUG-Owl/mPLUG-Owl2
# For a new checkout; use your existing matching checkout if already cloned.
git clone https://github.com/X-PLUG/mPLUG-Owl.git "$(dirname "$MPLUG_OWL2_ROOT")"
git -C "$(dirname "$MPLUG_OWL2_ROOT")" checkout 0f3068fdb47b77aedc71fc39f5735cd7aa35e8f9

python -m pip install -r requirements/mplugowl2.txt -e "$MPLUG_OWL2_ROOT"
python -m pip check
bash scripts/infer_mplugowl2.sh --image-file /absolute/path/to/image.jpg
```

The paper uses 448 × 448 inputs for this variant. The launcher selects the `mplug_owl2` conversation template. See the upstream [mPLUG-Owl2 setup](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2).

### FakeScope-L: LLaVA-v1.5

In a separate shell, or after deactivating the other environment:

```bash
python3.10 -m venv .venv/llava
source .venv/llava/bin/activate

export LLAVA_ROOT=/absolute/path/to/LLaVA
# For a new checkout; use your existing matching checkout if already cloned.
git clone https://github.com/haotian-liu/LLaVA.git "$LLAVA_ROOT"
git -C "$LLAVA_ROOT" checkout c121f0432da27facab705978f83c4ada465e46fd

python -m pip install -r requirements/llava.txt -e "$LLAVA_ROOT"
python -m pip check
bash scripts/infer_llava.sh --image-file /absolute/path/to/image.jpg
```

The paper uses 336 × 336 inputs for this variant. The launcher selects the `llava_v1` conversation template. See [Run FakeScope](inference.md) for single-image and batch authenticity scoring, and [training setup](training.md#1-install-the-matching-training-environment) for the additional training requirements.

The root `requirements.txt` contains shared dependencies only. The two model-specific files include it automatically; do not combine their environments. The source checkout is required because these requirements alone do not install the backbone's Python package. The [dependency notes](../requirements/README.md) explain the selected versions and validation scope.

## Example queries

**Authenticity detection**

```text
What is the authenticity of this image?
```

**Evidence-based reasoning**

```text
Inspect this image carefully and evaluate its authenticity using visual evidence.
```

**Fine-grained analysis**

```text
Are the reflections consistent with the objects in this scene?
```

These are example user prompts, not precomputed model outputs. For numeric authenticity scores, use the [FakeScope-L scoring script](inference.md#2-score-a-single-image-with-fakescope-l) and check its documented scoring position; a free-form chat response is not a substitute for extracting the relevant logits. The upstream chat interfaces may sample responses, so repeated answers can differ.

## Training data

To obtain the training dataset download link, email [yixuan.li@bristol.ac.uk](mailto:yixuan.li@bristol.ac.uk?subject=FakeInstruct%20training%20data%20access%20request). See the [dataset notes](datasets.md) and [training recipe](training.md).

## Hardware reference

The paper evaluates on one NVIDIA A100 with 40GB memory. Actual memory usage depends on the runtime, precision, sequence length, and checkpoint. The paper does not establish a minimum memory requirement or report accuracy for a quantized release.
