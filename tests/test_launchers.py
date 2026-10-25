"""Exercise launchers with temporary fixtures; never run DeepSpeed or models."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fakescope test ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.backbone = self.root / "backbone source"
        for path in ("llava/train/train_mem.py", "mplug_owl2/train/train_mem.py", "scripts/zero3.json"):
            target = self.backbone / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}")
        self.annotations = self.root / "training data.json"
        self.annotations.write_text("[]")
        self.env = dict(os.environ)
        for key in ("MODEL_PATH", "OUTPUT_DIR", "DEEPSPEED_CONFIG", "REPORT_TO", "NUM_GPUS", "MASTER_PORT"):
            self.env.pop(key, None)
        self.env.update(LLAVA_ROOT=str(self.backbone), MPLUG_OWL2_ROOT=str(self.backbone),
                        DATA_PATH=str(self.annotations), IMAGE_FOLDER=str(self.root), NUM_GPUS="8")

    def training_command(self, variant, *extra, env=None):
        script = ROOT / "scripts/train" / ("fake_instruct_" + variant + ".sh")
        result = subprocess.run(["bash", str(script), "--dry-run", *extra],
                                env=env or self.env, text=True, capture_output=True, check=True)
        return shlex.split(result.stdout)

    def test_training_defaults_and_paths_with_spaces(self):
        for variant, batch, warmup, module in (
                ("llava_v1.5_7b", "32", "0.03", "llava"),
                ("mplugowl2", "128", "0.02", "mplug_owl2")):
            with self.subTest(variant=variant):
                command = self.training_command(variant)
                def value(flag):
                    return command[command.index(flag) + 1]
                self.assertEqual(command[0], "deepspeed")
                self.assertIn(str(self.backbone / module / "train/train_mem.py"), command)
                self.assertEqual(value("--data_path"), str(self.annotations))
                self.assertEqual(value("--image_folder"), str(self.root))
                self.assertEqual(value("--deepspeed"), str(self.backbone / "scripts/zero3.json"))
                self.assertEqual(value("--per_device_train_batch_size"), batch)
                self.assertEqual(value("--warmup_ratio"), warmup)
                self.assertEqual(value("--gradient_accumulation_steps"), "1")
                self.assertEqual(value("--num_gpus"), "8")
                if module == "mplug_owl2":
                    self.assertEqual(value("--freeze_vision_model"), "False")

    def test_user_overrides_are_forwarded_last(self):
        command = self.training_command("llava_v1.5_7b", "--per_device_train_batch_size", "4",
                                        "--gradient_accumulation_steps", "8")
        self.assertEqual(command[-4:], ["--per_device_train_batch_size", "4",
                                       "--gradient_accumulation_steps", "8"])

    def test_missing_data_fails_without_launching_deepspeed(self):
        self.env["DATA_PATH"] = str(self.root / "missing.json")
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            self.training_command("mplugowl2")
        self.assertIn("Required file not found", failure.exception.stderr)

    def test_custom_zero_config_output_and_logging(self):
        custom = self.root / "my zero.json"
        custom.write_text("{}")
        self.env.update(DEEPSPEED_CONFIG=str(custom), OUTPUT_DIR=str(self.root / "my output"),
                        REPORT_TO="none", MASTER_PORT="25999")
        command = self.training_command("mplugowl2")
        for flag, expected in (("--deepspeed", str(custom)), ("--report_to", "none"),
                               ("--output_dir", str(self.root / "my output")),
                               ("--master_port", "25999")):
            self.assertEqual(command[command.index(flag) + 1], expected)

    def test_chat_launchers_forward_image_and_checkpoint_without_splitting(self):
        # A Python executable stub captures arguments instead of importing a model.
        interpreter = self.root / "fake python"
        interpreter.write_text("#!" + sys.executable + "\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
        interpreter.chmod(0o755)
        for variant, module, template in (("llava", "llava.serve.cli", "llava_v1"),
                                           ("mplugowl2", "mplug_owl2.serve.cli", "mplug_owl2")):
            command = ["bash", str(ROOT / "scripts" / ("infer_" + variant + ".sh")),
                       "--image-file", "/images with spaces/photo.jpg",
                       "--model-path", "/models with spaces/custom"]
            result = subprocess.run(command, env=dict(self.env, PYTHON=str(interpreter)),
                                    text=True, capture_output=True, check=True)
            forwarded = json.loads(result.stdout)
            self.assertEqual(forwarded[:2], ["-m", module])
            self.assertEqual(forwarded[forwarded.index("--conv-mode") + 1], template)
            self.assertEqual(forwarded[-4:], command[-4:])


if __name__ == "__main__":
    unittest.main()
