#!/usr/bin/env python3
"""Export FakeScope-L authenticity logits for one image or a JSON/JSONL manifest.

Adapted from the authors' eval_prob_llava.py. See docs/inference.md for the
scoring position, preprocessing, quantization, and output format.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile


ANCHORS = ("fake", "Fake", "real", "Real")
QUESTION = "What is the authenticity of this image?"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-file", help="One image: local path or HTTP(S) URL.")
    source.add_argument("--input-file", type=Path, help="JSON array or .jsonl manifest.")
    parser.add_argument("--image-folder", type=Path, default=Path("."),
                        help="Root for relative image_id paths (default: current directory).")
    parser.add_argument("--model-path", default="FakeScope/llava_1.5_fakeinstruct_v3")
    parser.add_argument("--answers-file", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="CUDA device, e.g. cuda:1.")
    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument("--load-8bit", action="store_true")
    quantization.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--assistant-prefix", default="",
                        help='Optional unfinished answer, e.g. "This image is". '
                             'Default: score the first assistant token, as in the supplied script.')
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    return parser.parse_args(argv)


def is_url(value):
    return value.lower().startswith(("http://", "https://"))


def read_records(args):
    if args.image_file:
        records = [{"image_id": args.image_file}]
    else:
        with args.input_file.expanduser().open(encoding="utf-8") as source:
            if args.input_file.suffix.lower() == ".jsonl":
                records = [json.loads(line) for line in source if line.strip()]
            else:
                records = json.load(source)
    if not isinstance(records, list) or not records:
        raise ValueError("Input must be a nonempty JSON array or JSONL manifest.")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("image_id"), str):
            raise ValueError(f"Record {index} must contain a string image_id.")
        if not record["image_id"].strip():
            raise ValueError(f"Record {index} has an empty image_id.")
    return records


def resolve_image(image_id, image_folder):
    if is_url(image_id):
        return image_id
    image_path = Path(image_id).expanduser()
    if not image_path.is_absolute():
        image_path = image_folder.expanduser() / image_path
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return str(image_path.resolve())


def output_paths(args, image_sources):
    output = args.answers_file.expanduser().resolve()
    metadata = output.with_name(output.name + ".meta.json")
    inputs = {Path(source).resolve() for source in image_sources if not is_url(source)}
    if args.input_file:
        inputs.add(args.input_file.expanduser().resolve())
    for path in (output, metadata):
        if path in inputs:
            raise ValueError(f"Output would overwrite an input file: {path}")
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {path}. Use --overwrite to replace it.")
    return output, metadata


def anchor_token_ids(tokenizer):
    """Require one distinct, non-special token per anchor; never assume a BOS offset."""
    result = {}
    special_ids = set(tokenizer.all_special_ids)
    for anchor in ANCHORS:
        ids = tokenizer(anchor, add_special_tokens=False)["input_ids"]
        if len(ids) != 1 or ids[0] in special_ids:
            raise ValueError(f"Anchor {anchor!r} must be one non-special token; got {ids}.")
        result[anchor] = ids[0]
    if len(set(result.values())) != len(ANCHORS):
        raise ValueError("Case-variant anchors must have distinct token IDs.")
    return result


def build_prompt(template, image_token, assistant_prefix=""):
    conversation = template.copy()
    conversation.append_message(conversation.roles[0], image_token + "\n" + QUESTION)
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    prefix = assistant_prefix.strip()
    # Append to the open assistant turn, without inserting an end-of-turn token.
    return prompt + (" " + prefix if prefix else "")


def soft_scores(logits):
    """Sum case-variant logits, then apply a numerically stable two-class softmax."""
    values = [float(logits[anchor]) for anchor in ANCHORS]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite authenticity logits; check the model and precision.")
    fake, real = values[0] + values[1], values[2] + values[3]
    peak = max(fake, real)
    fake_exp, real_exp = math.exp(fake - peak), math.exp(real - peak)
    return {"prob_fake": fake_exp / (fake_exp + real_exp),
            "prob_real": real_exp / (fake_exp + real_exp)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as destination:
            json.dump(value, destination, ensure_ascii=False, indent=2, allow_nan=False)
            destination.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_image(source):
    from io import BytesIO
    from PIL import Image

    if is_url(source):
        import requests

        with requests.get(source, timeout=30) as response:
            response.raise_for_status()
            with Image.open(BytesIO(response.content)) as image:
                return image.convert("RGB")
    with Image.open(source) as image:
        return image.convert("RGB")


def main(args):
    records = read_records(args)
    sources = [resolve_image(item["image_id"], args.image_folder) for item in records]
    output_path, metadata_path = output_paths(args, sources)

    # Keep --help and input validation usable before installing the GPU runtime.
    import torch
    from llava.constants import (
        IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("This scoring entry point requires a CUDA-enabled LLaVA environment.")
    disable_torch_init()
    # Explicit model family also permits checkpoint directories with custom names.
    tokenizer, model, processor, _ = load_pretrained_model(
        args.model_path, None, "llava-v1.5-7b",
        load_8bit=args.load_8bit, load_4bit=args.load_4bit,
        device=str(device), device_map=str(device),
    )
    model.eval()
    token_ids = anchor_token_ids(tokenizer)
    image_token = DEFAULT_IMAGE_TOKEN
    if getattr(model.config, "mm_use_im_start_end", False):
        image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
    prefix = args.assistant_prefix.strip()
    prompt = build_prompt(conv_templates["llava_v1"], image_token, prefix)
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
    ).unsqueeze(0).to(device)

    predictions = []
    for index, (record, source) in enumerate(zip(records, sources), start=1):
        image = load_image(source)
        # Preserve the supplied evaluator's processor path (no added square padding).
        image_tensor = processor.preprocess(image, return_tensors="pt")["pixel_values"]
        image_tensor = image_tensor.to(device=device, dtype=torch.float16)
        with torch.inference_mode():
            next_logits = model(input_ids, images=image_tensor, use_cache=False)["logits"][0, -1]
        logits = {anchor: next_logits[token_id].float().item()
                  for anchor, token_id in token_ids.items()}
        result = dict(record)
        result.update({"logit_" + anchor: value for anchor, value in logits.items()})
        result.update(soft_scores(logits))
        predictions.append(result)
        print(f"[{index}/{len(records)}] {record['image_id']}", file=sys.stderr)

    metadata = {
        "model_path": args.model_path,
        "model_family": "llava-v1.5-7b",
        "conversation_template": "llava_v1",
        "question": QUESTION,
        "assistant_prefix": prefix,
        "scoring_position": "after_prefix" if prefix else "first_assistant_token",
        "rendered_prompt": prompt,
        "preprocessing": "image_processor.preprocess (no additional square padding)",
        "weight_precision": "8bit" if args.load_8bit else "4bit" if args.load_4bit else "float16",
        "anchor_token_ids": token_ids,
        "aggregation": "softmax([logit_fake + logit_Fake, logit_real + logit_Real])",
        "device": str(device),
        "count": len(predictions),
    }
    write_json(output_path, predictions)
    write_json(metadata_path, metadata)
    print(f"Saved {len(predictions)} predictions to {output_path}")
    print(f"Run settings: {metadata_path}")


if __name__ == "__main__":
    main(parse_args())
