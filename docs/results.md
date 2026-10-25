# Evaluation results

[← Back to FakeScope](../README.md#results)

All numbers on this page are transcribed from the supplied FakeScope manuscript (`manuscript-TIFS-submission.tex`). They describe the paper’s experiments, not a new evaluation run. The public reference is [arXiv:2503.24267](https://arxiv.org/abs/2503.24267).

## Evaluation settings

| Benchmark | Purpose | Metric used below |
| :--- | :--- | :--- |
| FakeClass, from FakeBench | Authenticity judgment from textual responses | Accuracy (%) |
| FakeClue, from FakeBench | Interpretation and reasoning | Aggregated transparency score |
| FakeQA, from FakeBench | Open-ended fine-grained questions | Aggregated transparency score |
| FakeBench<sup>MCQ</sup> | Fine-grained visual attribution, 3,000 questions | Accuracy (%) |
| AIGCDetectBenchmark + AntifakePrompt | Generalization across image sources and generators | AP / mean AP (%) |
| WildRF + SynthWildX | Detection on in-the-wild social-media imagery | AP / mean AP (%) |

For FakeClue, **Interpret** uses the fault-finding setting where the label is supplied; **Reason** uses the inference setting where the model evaluates authenticity from the image. The aggregated transparency scores combine normalized lexical/semantic metrics and LLM-based judgments, as defined in the paper. They are not percentages.

## Multimodal forensic capabilities

| Model | FakeClass Acc. ↑ | Interpret ↑ | Reason ↑ | Analyze ↑ | MCQ Acc. ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| GPT-4V (teacher) | 78.03 | 0.573 | 0.460 | 0.477 | 66.60 |
| mPLUG-Owl2 | 71.78 | 0.423 | 0.296 | 0.368 | 65.97 |
| FakeScope-M | **99.68** | 0.638 | **0.614** | 0.540 | **78.50** |
| LLaVA-v1.5 | 57.70 | 0.468 | 0.311 | 0.394 | 61.83 |
| FakeScope-L | 98.30 | **0.640** | 0.610 | **0.553** | 77.74 |

Source tables: qualitative FakeClass detection, multimodal forensic capability comparisons, and forensic attribution capability.

## In-the-wild detection

AP (%), reported separately for each source or generator:

| Model | WildRF: Reddit | WildRF: Facebook | WildRF: X | SynthWildX: DALL·E 3 | SynthWildX: Firefly | SynthWildX: Midjourney | Mean |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FakeScope-M | 91.89 | 85.31 | 92.96 | 89.02 | 88.04 | 83.72 | **88.49** |
| FakeScope-L | 80.01 | 67.94 | 75.48 | 87.37 | 78.94 | 81.99 | 78.46 |

The mean is the paper’s reported average across these six subsets, not a pooled AP computed over all images.

## Generalization

| Model | Unseen subset mAP (%) ↑ | All 17 evaluation generators mAP (%) ↑ |
| :--- | ---: | ---: |
| FakeScope-M | **95.67** | 95.84 |
| FakeScope-L | 94.68 | **96.17** |

“Unseen” uses the paper’s generalization-table subset definition. The 17 evaluated generator/task columns include both seen and unseen cases, and should not be confused with the 17 generators used to build FakeChain. The paper also includes face manipulation and super-resolution as out-of-distribution cases. Its AIGCDetectBenchmark split removes ImageNet-derived source content to prevent overlap with training.

## Reproduction status

This repository documents the results and protocol and includes [inference scripts](inference.md) for image chat and LLaVA authenticity-logit export. The scorer preserves the supplied evaluator’s first-assistant-token position by default and records its settings. It does not include the complete benchmark evaluation harness, split manifests, or saved benchmark prediction files. Matching preprocessing, checkpoints, prompts, split exclusions, token handling, and metric aggregation is necessary before making a direct numerical comparison.
