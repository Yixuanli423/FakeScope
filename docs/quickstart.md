# Inference setup

[← Back to FakeScope](../README.md#getting-started)

FakeScope uses the runtime of its underlying multimodal model. **FakeScope-M** and **FakeScope-L** require different packages and conversation templates.

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

Use separate environments for the two runtimes. Follow each upstream repository’s supported Python, PyTorch, and CUDA versions.

### FakeScope-M: mPLUG-Owl2

Inside a dedicated environment:

```bash
git clone https://github.com/X-PLUG/mPLUG-Owl.git
cd mPLUG-Owl/mPLUG-Owl2
python -m pip install -e .
```

Set the absolute paths to your downloaded checkpoint and input image, then launch the upstream image-chat interface:

```bash
python -m mplug_owl2.serve.cli \
  --model-path /absolute/path/to/FakeScope/checkpoints/mplug_owl2_fakeinstruct_v3 \
  --image-file /absolute/path/to/image.jpg \
  --conv-mode mplug_owl2
```

The paper uses 448 × 448 inputs for this variant. See [mPLUG-Owl2 setup](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2) and its [model loader](https://github.com/X-PLUG/mPLUG-Owl/blob/main/mPLUG-Owl2/mplug_owl2/model/builder.py).

### FakeScope-L: LLaVA-v1.5

Inside a separate dedicated environment:

```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
python -m pip install -e .
```

Set your absolute checkpoint and image paths, and specify the LLaVA-v1.5 conversation template explicitly:

```bash
python -m llava.serve.cli \
  --model-path /absolute/path/to/FakeScope/checkpoints/llava_1.5_fakeinstruct_v3 \
  --image-file /absolute/path/to/image.jpg \
  --conv-mode llava_v1
```

The paper uses 336 × 336 inputs for this variant. See [LLaVA setup](https://github.com/haotian-liu/LLaVA) and its [image-chat interface](https://github.com/haotian-liu/LLaVA/blob/main/llava/serve/cli.py).

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

These are example user prompts, not precomputed model outputs. Numeric authenticity scores require the [token soft-scoring procedure](method.md#4-estimate-authenticity-through-token-soft-scoring); a free-form chat response is not a substitute for extracting the relevant logits. The upstream chat interfaces may sample responses, so repeated answers can differ.

## Training data

To obtain the training dataset download link, email [yixuan.li@bristol.ac.uk](mailto:yixuan.li@bristol.ac.uk?subject=FakeInstruct%20training%20data%20access%20request). See the [dataset notes](datasets.md) and [training recipe](training.md).

## Hardware reference

The paper evaluates on one NVIDIA A100 with 40GB memory. Actual memory usage depends on the runtime, precision, sequence length, and checkpoint. The paper does not establish a minimum memory requirement or report accuracy for a quantized release.
