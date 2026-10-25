# How FakeScope works

[← Back to FakeScope](../README.md)

## 1. Build forensic reasoning with ACoTI

**Anthropomorphic Chain-of-Thought Inference (ACoTI)** is the human–AI collaboration strategy used to construct FakeChain:

1. **Steer.** Human annotators identify visual evidence categories relevant to each image.
2. **Demonstrate.** A small set of human-written reasoning examples is selected to cover those categories and both authenticity labels.
3. **Enlighten.** The teacher model receives the target image, its known authenticity label, evidence categories, and demonstrations to produce structured reasoning.

The resulting annotation connects the appearance and location of visual evidence with an authenticity conclusion. This is a dataset-construction procedure: providing the known label here is distinct from asking FakeScope to predict an unknown label at inference.

## 2. Turn reasoning into visual instructions

FakeInstruct expands FakeChain into authenticity judgments, holistic reasoning, fine-grained visual questions, and extended discussions. These approximately 2M instructions teach a shared model to respond to different forensic requests.

## 3. Fine-tune a multimodal backbone

FakeScope has two 7B variants: **FakeScope-M** uses mPLUG-Owl2, while **FakeScope-L** uses LLaVA-v1.5. The vision encoder remains frozen while the language model and modality projector are fine-tuned. See the [training recipe](training.md).

## 4. Estimate authenticity through token soft-scoring

The paper uses a standardized exchange:

```text
USER: <Image> What is the authenticity of this image?
ASSISTANT: This image is <auth_token>.
```

At the `<auth_token>` position, let $z^{(v)}$ be the logit for vocabulary token $v$. For each class $c$ in `{fake, real}`, the paper aggregates the logits of its case-variant anchor tokens $T_c$:

$$s_c = \sum_{v \in T_c} z^{(v)}$$

It then applies a two-class softmax:

$$p(c) = \frac{\exp(s_c)}{\exp(s_{\mathrm{fake}}) + \exp(s_{\mathrm{real}})}$$

This operation uses the model’s logits, rather than asking it to write a confidence percentage in free text. It adds no trainable classifier head and requires no numerical-label supervision. The exact token IDs and insertion point depend on the tokenizer and conversation template, so a scoring implementation must verify both for each released checkpoint.

## Interpreting the outputs

The generated explanations make visual claims available for human inspection. They do not establish that the model internally used those exact claims to make its decision. Likewise, token soft-scoring provides a numerical authenticity score; calibration on a new distribution requires separate evaluation.

Source: [FakeScope paper](https://arxiv.org/abs/2503.24267), sections on FakeChain, FakeInstruct, and the FakeScope model.
