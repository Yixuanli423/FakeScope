#!/usr/bin/env python3
"""Convert LLaVA <image> markers to mPLUG-Owl2 <|image|> in conversation text."""

import argparse
import copy
import json
from pathlib import Path

from eval_prob_llava import write_json


def convert_records(records):
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty JSON array of training records.")
    converted = copy.deepcopy(records)
    replacements = 0
    for index, record in enumerate(converted):
        if not isinstance(record, dict) or not isinstance(record.get("conversations"), list):
            raise ValueError(f"Record {index} must contain a conversations list.")
        for turn in record["conversations"]:
            if not isinstance(turn, dict) or not isinstance(turn.get("value"), str):
                raise ValueError(f"Record {index} has a conversation turn without a string value.")
            replacements += turn["value"].count("<image>")
            turn["value"] = turn["value"].replace("<image>", "<|image|>")
    return converted, replacements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    source = args.input_file.expanduser().resolve()
    destination = args.output_file.expanduser().resolve()
    if source == destination or destination.exists():
        parser.error("Choose a new output file; the source and existing files are preserved.")
    with source.open(encoding="utf-8") as handle:
        records = json.load(handle)
    converted, count = convert_records(records)
    write_json(destination, converted)
    print(f"Saved {len(converted)} records; replaced {count} image markers: {destination}")


if __name__ == "__main__":
    main()
