# Run FakeScope

[← Back to FakeScope](../README.md#getting-started) · [Environment setup](quickstart.md) · [Token soft-scoring](method.md#4-estimate-authenticity-through-token-soft-scoring)

Ask for visual evidence in an image conversation, or export numerical authenticity scores with FakeScope-L.

| Entry point | Model | Output |
| :--- | :--- | :--- |
| [`infer_llava.sh`](../scripts/infer_llava.sh) | FakeScope-L | Interactive image conversation |
| [`infer_mplugowl2.sh`](../scripts/infer_mplugowl2.sh) | FakeScope-M | Interactive image conversation |
| [`eval_prob_llava.py`](../scripts/eval_prob_llava.py) | FakeScope-L | Four anchor logits, two soft scores, and run settings |

## Before you start

Request access to the [LLaVA checkpoint](https://huggingface.co/FakeScope/llava_1.5_fakeinstruct_v3) or the [mPLUG-Owl2 checkpoint](https://huggingface.co/FakeScope/mplug_owl_2_fakeinstruct_v3), download it, and install its upstream runtime using the [setup guide](quickstart.md). Use a separate environment for each backbone. The scoring script requires CUDA; the paper's reference hardware is one NVIDIA A100 40GB.

Run the following commands from the FakeScope repository root, with the matching environment activated. Replace example image paths with your own files.

## 1. Chat about an image

**FakeScope-L**

```bash
bash scripts/infer_llava.sh --image-file /absolute/path/to/image.jpg
```

**FakeScope-M**

```bash
bash scripts/infer_mplugowl2.sh --image-file /absolute/path/to/image.jpg
```

At the prompt, try:

```text
What is the authenticity of this image?
Inspect this image carefully and evaluate its authenticity using visual evidence.
Are the reflections consistent with the objects in this scene?
```

These launchers call the upstream [LLaVA image-chat CLI](https://github.com/haotian-liu/LLaVA/blob/main/llava/serve/cli.py) and [mPLUG-Owl2 image-chat CLI](https://github.com/X-PLUG/mPLUG-Owl/blob/main/mPLUG-Owl2/mplug_owl2/serve/cli.py), with the matching conversation template selected and a single CUDA device (`cuda:0`) by default. Follow-up questions retain the conversation; enter an empty line to exit. Responses use the upstream sampling defaults.

The default checkpoint directories are `checkpoints/llava_1.5_fakeinstruct_v3` and `checkpoints/mplug_owl2_fakeinstruct_v3`. Override them with `--model-path`; keep `llava` or `mplug_owl2`, respectively, in the checkpoint directory name because these upstream chat loaders use the name to select the model family. Other upstream arguments, such as `--device cuda:1`, `--max-new-tokens 512`, or `--load-8bit`, are forwarded unchanged. Set `PYTHON=/path/to/python` to choose an interpreter explicitly.

## 2. Score a single image with FakeScope-L

```bash
python scripts/eval_prob_llava.py \
  --model-path checkpoints/llava_1.5_fakeinstruct_v3 \
  --image-file /absolute/path/to/image.jpg \
  --answers-file results/image_scores.json
```

Local paths and HTTP(S) image URLs are supported. `--model-path` also accepts the Hugging Face repository ID `FakeScope/llava_1.5_fakeinstruct_v3` when your account has approved access.

The output is a JSON array. Each record retains the input fields and adds:

| Field | Meaning |
| :--- | :--- |
| `logit_fake`, `logit_Fake` | Raw next-token logits for the two fake anchors |
| `logit_real`, `logit_Real` | Raw next-token logits for the two real anchors |
| `prob_fake`, `prob_real` | Two-class softmax of the sums of the case-variant logits |

The calculation is:

```text
s_fake = logit_fake + logit_Fake
s_real = logit_real + logit_Real
[prob_fake, prob_real] = softmax([s_fake, s_real])
```

Higher `prob_fake` indicates a stronger fake score. These are token-based soft scores, not calibrated probabilities or benchmark metrics. The script does not use labels to select a prompt or compute a prediction.

## 3. Score an image collection

Create a JSON array, or a `.jsonl` file with one object per line. Each object needs an `image_id`:

```json
[
  {"image_id": "example_01.jpg"},
  {"image_id": "subfolder/example_02.png"}
]
```

The repository includes this [manifest template](../examples/images.json); supply your own images. Additional fields, including existing ground-truth labels, are copied to the output without interpreting their label convention.

```bash
python scripts/eval_prob_llava.py \
  --model-path checkpoints/llava_1.5_fakeinstruct_v3 \
  --input-file examples/images.json \
  --image-folder /absolute/path/to/your/images \
  --answers-file results/collection_scores.json
```

Relative image paths are resolved against `--image-folder` (the current directory by default), not against the manifest's directory. Absolute paths and HTTP(S) URLs are used directly. Images are processed sequentially; no training dataset is needed for inference.

Results are written after all images finish. The script stops on malformed records, missing images, decoding failures, or invalid anchor tokenization. Output directories are created automatically. Existing outputs require `--overwrite`; input manifests and images cannot be overwritten by output files.

## Scoring settings and provenance

The scorer is adapted from the authors' supplied `eval_prob_llava.py`. It preserves the authenticity question, `llava_v1` template, four anchor logits, and direct `image_processor.preprocess` path. It validates that each anchor is a single distinct non-special token, instead of assuming the tokenizer always inserts one beginning-of-sequence token.

**The default scores the first assistant token.** The supplied evaluator leaves the assistant turn empty before extracting logits. The paper describes the authenticity word in “This image is &lt;auth_token&gt;.” These are different scoring positions. To explicitly condition on that unfinished answer, use:

```bash
python scripts/eval_prob_llava.py \
  --model-path checkpoints/llava_1.5_fakeinstruct_v3 \
  --image-file /absolute/path/to/image.jpg \
  --assistant-prefix "This image is" \
  --answers-file results/prefix_scores.json
```

The prefix is appended to the open assistant turn without an end-of-turn separator. Prefix-conditioned scores have not been validated against the paper's reported results; do not mix scores from different positions in one evaluation.

**Precision is explicit.** The supplied evaluator forced 8-bit loading. This public entry point defaults to the upstream loader's float16 weights. Use `--load-8bit` to request the supplied evaluator's quantization choice, or `--load-4bit` for the upstream 4-bit option. Quantization requires the matching bitsandbytes/CUDA dependencies and can change scores. Set `--device cuda:1` to choose another GPU, or set `CUDA_VISIBLE_DEVICES` before launching; the script does not override it.

**Preprocessing is preserved.** The scorer passes RGB images directly to the checkpoint's image processor, without adding square padding. The interactive chat entry points follow their upstream preprocessing, which may use padding. Predictions from these paths are not guaranteed to be numerically interchangeable. Corrupt or truncated images are rejected instead of being silently accepted.

Each run also writes `<answers-file>.meta.json`, recording the rendered prompt, assistant prefix, anchor token IDs, precision, preprocessing, model path, device, and record count. Keep this file with your predictions. The JSON manifest remains unchanged.

## Validation scope

The release includes lightweight tests for token selection, score arithmetic, input/output handling, and command forwarding:

```bash
python -m unittest discover -s tests -v
```

These checks do not download weights. End-to-end GPU inference with the gated checkpoints has not been verified in the release environment. This release provides image chat for both backbones and logit export for LLaVA; it does not include an mPLUG-Owl2 scoring evaluator or the complete benchmark evaluation harness. See [evaluation scope](results.md#reproduction-status).
