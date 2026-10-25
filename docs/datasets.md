# FakeChain and FakeInstruct

[← Back to FakeScope](../README.md#datasets)

FakeChain provides structured forensic reasoning. FakeInstruct turns that knowledge into diverse image–instruction–response interactions for supervised fine-tuning.

## Request access

To obtain the **training dataset download link**, please send an access request to [yixuan.li@bristol.ac.uk](mailto:yixuan.li@bristol.ac.uk?subject=FakeInstruct%20training%20data%20access%20request).

Suggested email subject: **FakeInstruct training data access request**.

The tables below describe the paper’s datasets. Use the documentation provided with the data for its file layout, splits, and annotation fields.

## FakeChain

| Property | Description |
| :--- | :--- |
| Total images | 47,594 |
| AI-generated images | 23,797 |
| Real images | 23,797 |
| Synthetic sources | 17 generators |
| Real-image sources | DIV2K, RAISE, ImageNet |
| Conceptual record | Image, authenticity label, structured forensic reasoning |

The image/label/reasoning triple is the paper’s conceptual representation, not a claim about the released JSON field names.

### Generator coverage

ProGAN, StyleGAN, BigGAN, CogView2, ADM, IF, FuseDream, VQDM, Glide, Stable Diffusion, SDXL, FLUX.1, Firefly, DALL·E 2, DALL·E 3, Wukong, and Midjourney.

### Evidence categories

The human annotation process uses 16 reference categories:

| Visual appearance | Structure and geometry | Context and plausibility |
| :--- | :--- | :--- |
| Texture | Anatomy | Physics |
| Edge | Layout | Theme |
| Clarity | Symmetry | Content deficiency |
| Light & shadow | Reflection | Unrealistic |
| Overall hue | Perspective | |
| Distortion | Shape | |

The grouping above is for readability. The underlying categories come from the paper’s annotation protocol.

### Human–AI collaboration

The ACoTI pipeline combines human-selected visual evidence, a small set of human-written examples, and GPT-4V as the reasoning teacher. The manuscript describes 30 trained annotators, 100 reasoning exemplars (50 real and 50 fake), and typically 3–5 selected demonstrations per image. The known authenticity label is supplied during data construction.

## FakeInstruct

| Family | Rounded instruction count | Supervision |
| :--- | ---: | :--- |
| Absolute authenticity judgments | 47K | Explicit real/fake responses |
| Holistic reasoning | 95K | Visual evidence followed by an authenticity conclusion |
| Fine-grained trace evidence | 715K | Narrative and multiple-choice questions |
| Extended knowledge and discussions | 1,190K | Additional authenticity-related conversations |

These rounded counts total approximately 2M instructions. GPT-4o and Mistral are used to transform the source reasoning into varied instruction-following data.

## Evaluation separation

The paper evaluates on FakeBench, AntifakePrompt, AIGCDetectBenchmark, WildRF, and SynthWildX. Its AIGCDetectBenchmark evaluation excludes real ImageNet images and generated images whose source content traces to ImageNet. Preserve this source-level separation when comparing against the reported results.

## Source and terms

See the [paper and supplementary material](https://arxiv.org/abs/2503.24267) for source datasets, collection details, and prompt templates. Consult the terms provided with the data and the underlying image-source terms when downloading or redistributing it.
