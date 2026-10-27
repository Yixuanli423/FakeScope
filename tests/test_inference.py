"""CPU-only checks; no model downloads or GPU inference are performed."""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_prob_llava as scorer
from prepare_mplug_data import convert_records


class Tokenizer:
    all_special_ids = [0, 1, 2]

    def __init__(self, tokens=None):
        self.tokens = tokens or dict(zip(scorer.ANCHORS, ([10], [11], [12], [13])))

    def __call__(self, text, add_special_tokens=True):
        ids = self.tokens[text]
        return {"input_ids": ([1] if add_special_tokens else []) + ids}


class Conversation:
    roles = ("USER", "ASSISTANT")

    def __init__(self):
        self.messages = []

    def copy(self):
        return Conversation()

    def append_message(self, role, message):
        self.messages.append((role, message))

    def get_prompt(self):
        return " ".join(role + ":" + (" " + message + "</s>" if message else "")
                        for role, message in self.messages)


class ScoringTests(unittest.TestCase):
    def test_case_variants_are_summed_before_softmax(self):
        scores = scorer.soft_scores({"fake": 2, "Fake": 3, "real": 1, "Real": 1})
        self.assertAlmostEqual(scores["prob_fake"], 1 / (1 + math.exp(-3)))
        self.assertAlmostEqual(sum(scores.values()), 1)

    def test_scores_are_stable_for_large_logits_and_ties(self):
        scores = scorer.soft_scores({"fake": 10000, "Fake": 10001, "real": -10000, "Real": 0})
        self.assertEqual(scores, {"prob_fake": 1, "prob_real": 0})
        self.assertEqual(scorer.soft_scores(dict.fromkeys(scorer.ANCHORS, -10000)),
                         {"prob_fake": 0.5, "prob_real": 0.5})
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            scorer.soft_scores(dict.fromkeys(scorer.ANCHORS, float("nan")))

    def test_token_ids_do_not_depend_on_bos(self):
        self.assertEqual(scorer.anchor_token_ids(Tokenizer()),
                         {"fake": 10, "Fake": 11, "real": 12, "Real": 13})

    def test_multitoken_special_and_duplicate_anchors_fail(self):
        for invalid in ([4, 5], [0], [11]):
            with self.subTest(ids=invalid):
                tokenizer = Tokenizer()
                tokenizer.tokens["fake"] = invalid
                with self.assertRaises(ValueError):
                    scorer.anchor_token_ids(tokenizer)

    def test_complete_sequences_preserve_split_words(self):
        tokenizer = Tokenizer()
        tokenizer.tokens["Fake"] = [383, 1296]
        self.assertEqual(scorer.anchor_token_sequences(tokenizer)["Fake"], [383, 1296])
        for bad in ([], [1, 4], [10]):
            tokenizer.tokens["Fake"] = bad
            with self.subTest(ids=bad), self.assertRaises(ValueError):
                scorer.anchor_token_sequences(tokenizer)

    def test_sequence_score_uses_conditional_second_token_and_caches_prompt(self):
        sequences = {"fake": [10], "Fake": [11, 15], "real": [12], "Real": [13]}
        calls = []
        def next_log_probs(prefix):
            calls.append(prefix)
            return {10: -1., 11: -2., 12: -3., 13: -4.} if not prefix else {15: -5.}
        values = scorer.sequence_log_likelihoods(sequences, next_log_probs)
        self.assertEqual(values, {"fake": -1., "Fake": -7., "real": -3., "Real": -4.})
        self.assertEqual(calls, [(), (11,)])
        self.assertAlmostEqual(scorer.soft_scores(values)["prob_fake"], 1/(1+math.exp(1)))

    def test_single_token_sequences_match_original_logit_aggregation(self):
        logits = {"fake": 2., "Fake": 3., "real": 1., "Real": 1.}
        normalizer = math.log(sum(math.exp(v) for v in logits.values()))
        sequences = scorer.anchor_token_sequences(Tokenizer())
        log_probs = {sequences[a][0]: value-normalizer for a,value in logits.items()}
        scores = scorer.soft_scores(scorer.sequence_log_likelihoods(sequences, lambda _: log_probs))
        self.assertAlmostEqual(scores["prob_fake"], scorer.soft_scores(logits)["prob_fake"])

    def test_sequence_policy_is_default_and_strict_remains_available(self):
        arguments = ["--image-file", "a.png", "--answers-file", "out.json"]
        self.assertEqual(scorer.parse_args(arguments).anchor_policy, "sequence")
        self.assertEqual(scorer.parse_args(arguments+["--anchor-policy", "strict"]).anchor_policy, "strict")

    def test_embedded_vision_index_detection(self):
        self.assertTrue(scorer.index_has_vision_weights({"weight_map": {
            "model.vision_tower.vision_tower.vision_model.embeddings.class_embedding": "shard.bin"}}))
        self.assertFalse(scorer.index_has_vision_weights({"weight_map": {"model.mm_projector.weight": "shard.bin"}}))

    def test_prefix_stays_inside_unfinished_assistant_turn(self):
        template = Conversation()
        plain = scorer.build_prompt(template, "<image>")
        conditioned = scorer.build_prompt(template, "<image>", " This image is ")
        self.assertTrue(plain.endswith("ASSISTANT:"))
        self.assertEqual(conditioned, plain + " This image is")
        self.assertNotIn("</s>", conditioned.split("ASSISTANT:")[-1])
        self.assertEqual(template.messages, [])


class InputOutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_json_and_jsonl_preserve_labels_and_extra_fields(self):
        records = [{"image_id": "sub/a.png", "authenticity": 0, "notes": "unchanged"}]
        for extension in ("json", "jsonl"):
            path = self.root / ("input." + extension)
            path.write_text(json.dumps(records if extension == "json" else records[0]))
            args = scorer.parse_args(["--input-file", str(path), "--answers-file", "out.json"])
            self.assertEqual(scorer.read_records(args), records)

    def test_invalid_manifest_fails_before_model_loading(self):
        for value in ({"image_id": "a.png"}, [], [{"image_id": 42}], [{"image_id": " "}]):
            source = self.root / "invalid.json"
            source.write_text(json.dumps(value))
            args = scorer.parse_args(["--input-file", str(source), "--answers-file", "out.json"])
            with self.assertRaises(ValueError):
                scorer.read_records(args)

    def test_image_paths_support_spaces_nested_paths_and_urls(self):
        folder = self.root / "images with spaces"
        folder.mkdir()
        image = folder / "a.png"
        image.touch()
        self.assertEqual(scorer.resolve_image("a.png", folder), str(image.resolve()))
        self.assertEqual(scorer.resolve_image(str(image), Path("ignored")), str(image.resolve()))
        self.assertEqual(scorer.resolve_image("https://example.org/a.png", folder),
                         "https://example.org/a.png")
        with self.assertRaises(FileNotFoundError):
            scorer.resolve_image("missing.png", folder)

    def test_output_never_overwrites_input_even_with_overwrite_flag(self):
        manifest = self.root / "input.json"
        manifest.write_text("[]")
        args = scorer.parse_args(["--input-file", str(manifest), "--answers-file", str(manifest),
                                  "--overwrite"])
        with self.assertRaises(ValueError):
            scorer.output_paths(args, [])

    def test_existing_metadata_also_requires_overwrite(self):
        output = self.root / "out.json"
        metadata = output.with_name("out.json.meta.json")
        metadata.touch()
        args = scorer.parse_args(["--image-file", "a.png", "--answers-file", str(output)])
        with self.assertRaises(FileExistsError):
            scorer.output_paths(args, [])
        args.overwrite = True
        self.assertEqual(scorer.output_paths(args, []), (output.resolve(), metadata.resolve()))

    def test_atomic_json_writer_keeps_existing_file_on_serialization_error(self):
        output = self.root / "nested" / "out.json"
        scorer.write_json(output, [{"image_id": "é.png"}])
        previous = output.read_bytes()
        with self.assertRaises(ValueError):
            scorer.write_json(output, {"bad": float("nan")})
        self.assertEqual(output.read_bytes(), previous)
        self.assertEqual(list(output.parent.iterdir()), [output])


class DataConversionTests(unittest.TestCase):
    def test_only_conversation_text_changes_and_source_is_preserved(self):
        record = {"image": "folder/<image>.jpg", "id": "same-id", "notes": "<image>",
                  "conversations": [{"from": "human", "value": "<image>\nDescribe it."},
                                    {"from": "gpt", "value": "An annotated response."}]}
        converted, count = convert_records([record])
        self.assertEqual(count, 1)
        self.assertEqual(converted[0]["conversations"][0]["value"], "<|image|>\nDescribe it.")
        self.assertEqual(converted[0]["image"], record["image"])
        self.assertEqual(converted[0]["notes"], record["notes"])
        self.assertEqual(converted[0]["conversations"][1], record["conversations"][1])
        self.assertTrue(record["conversations"][0]["value"].startswith("<image>"))
        self.assertEqual(convert_records(converted), (converted, 0))

    def test_malformed_training_records_fail(self):
        for value in ({}, [], [{}], [{"conversations": [{"value": 1}]}]):
            with self.assertRaises(ValueError):
                convert_records(value)


if __name__ == "__main__":
    unittest.main()
