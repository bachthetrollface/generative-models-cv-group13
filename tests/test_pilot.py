import csv
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from div2k import BatchSchedule, image_pairs, read_pair, training_split
from evaluate import main as evaluate_main, y_metrics
from models import Discriminator, SRResNet, VGGContent
from train import arguments, learning_rate, load_checkpoint, optimizer, save_checkpoint, update_batch


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_paper_architecture(self):
        g, d = SRResNet().eval(), Discriminator().eval()
        with torch.inference_mode():
            self.assertEqual(g(torch.zeros(1, 3, 5, 7)).shape, (1, 3, 20, 28))
            self.assertEqual(d(torch.zeros(1, 3, 96, 96)).shape, (1, 1))
        self.assertEqual(len(g.blocks), 16)
        self.assertEqual(sum(isinstance(m, nn.BatchNorm2d) for m in g.modules()), 33)
        convs = [m for m in d.features if isinstance(m, nn.Conv2d)]
        self.assertEqual([m.stride[0] for m in convs], [1, 2] * 4)
        self.assertTrue(all(m.kernel_size == (3, 3) for m in convs))

    def test_paper_loss_defaults_and_schedule(self):
        for stage, extra, weights in [("srresnet", [], (1, 0, 0)),
                                      ("srgan", ["--init-generator", "dummy"], (1, 0, 0.001)),
                                      ("srgan", ["--init-generator", "dummy", "--pixel-weight", "0",
                                                 "--vgg-weight", "1"], (0, 1, 0.001))]:
            with patch("sys.argv", ["train.py", "--out", "dummy", "--stage", stage, *extra]):
                args = arguments()
            self.assertEqual((args.pixel_weight, args.vgg_weight, args.adv_weight), weights)
        self.assertAlmostEqual(learning_rate(1e-4, 100_000, 99_999), 1e-4)
        self.assertAlmostEqual(learning_rate(1e-4, 100_000, 100_000), 1e-5)
        self.assertAlmostEqual(learning_rate(1e-4, 0, 500_000), 1e-4)

    def test_vgg_features_and_scaling_without_download(self):
        # Identity layers isolate preprocessing, feature amplitude, and stop-gradient behavior.
        for name, count in [("vgg22", 9), ("vgg54", 36)]:
            layers = nn.Sequential(*(nn.Identity() for _ in range(37)))
            with patch("models.vgg19", return_value=SimpleNamespace(features=layers)):
                vgg = VGGContent(name)
            self.assertEqual(len(vgg.features), count)
            x = torch.tensor([0.8, 0.0, -1.0]).view(1, 3, 1, 1).requires_grad_()
            y = torch.zeros_like(x, requires_grad=True)
            expected_rgb = torch.tensor([0.0, 0.5, 0.9]).view_as(x)
            expected = (expected_rgb - vgg.mean) / vgg.std / 12.75
            torch.testing.assert_close(vgg.encode(x), expected)
            vgg(x, y).backward()
            self.assertIsNotNone(x.grad)
            self.assertIsNone(y.grad)

    def test_paired_crops_and_sampler_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("hr", "lr"):
                (root / folder).mkdir()
            values = np.arange(30 * 25 * 3, dtype=np.uint8).reshape(30, 25, 3)
            Image.fromarray(values).save(root / "lr/0001x4.png")
            Image.fromarray(values.repeat(4, axis=0).repeat(4, axis=1)).save(root / "hr/0001.png")
            (pair,) = image_pairs(root / "hr", root / "lr")
            lr, hr = read_pair(pair, crop_lr=24)
            np.testing.assert_array_equal(np.asarray(hr)[::4, ::4], lr)
        schedule = list(BatchSchedule(750, 16, 5413, 0, 50))
        self.assertEqual(schedule[43:], list(BatchSchedule(750, 16, 5413, 43, 50)))
        self.assertEqual([len(batch) for batch in schedule[:47]], [16] * 46 + [14])
        self.assertEqual({key for batch in schedule[:47] for key, _ in batch}, set(range(750)))
        self.assertEqual(schedule[46:], list(BatchSchedule(750, 16, 5413, 46, 50)))
        self.assertEqual(schedule[47:], list(BatchSchedule(750, 16, 5413, 47, 50)))
        self.assertEqual([len(batch) for batch in BatchSchedule(3, 16, 0, 0, 2)], [3, 3])

    def test_luminance_metrics_and_border(self):
        target = torch.rand(3, 32, 32)
        altered = target.clone()
        altered[:, :4] = 0
        self.assertEqual(y_metrics(altered, target), (float("inf"), 1.0))

    def test_wild_variants_stay_with_their_hr_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hr, lr = root / "DIV2K_train_HR", root / "DIV2K_train_LR_wild"
            hr.mkdir()
            lr.mkdir()
            for index in range(1, 801):
                (hr / f"{index:04}.png").touch()
                for variant in range(1, 5):
                    (lr / f"{index:04}x4w{variant}.png").touch()
            train, validation = training_split(root, "realistic-wild")
            train_ids = {pair[1].stem for pair in train}
            validation_ids = {pair[1].stem for pair in validation}
            self.assertEqual((len(train), len(validation)), (3000, 200))
            self.assertEqual((len(train_ids), len(validation_ids)), (750, 50))
            self.assertFalse(train_ids & validation_ids)
            self.assertEqual(len({pair[0] for pair in train + validation}), 3200)

    def test_named_evaluation_writes_separate_results_for_each_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = {
                "div2k-bicubic": ("DIV2K_valid_HR", "DIV2K_valid_LR_bicubic/X4", "0801", "x4"),
                "div2k-unknown": ("DIV2K_valid_HR", "DIV2K_valid_LR_unknown/X4", "0801", "x4"),
                "div2k-realistic-mild": ("DIV2K_valid_HR", "DIV2K_valid_LR_mild", "0801", "x4m"),
                "div2k-realistic-difficult": ("DIV2K_valid_HR", "DIV2K_valid_LR_difficult", "0801", "x4d"),
                "div2k-realistic-wild": ("DIV2K_valid_HR", "DIV2K_valid_LR_wild", "0801", "x4w"),
                "urban100": ("Urban100_HR", "Urban100_LR_x4", "img_001", ""),
            }
            for index, (hr, lr, key, suffix) in enumerate(cases.values()):
                (root / hr).mkdir(parents=True, exist_ok=True)
                (root / lr).mkdir(parents=True)
                Image.new("RGB", (24, 32), "white").save(root / hr / f"{key}.png")
                Image.new("RGB", (6, 8), (index * 32,) * 3).save(root / lr / f"{key}{suffix}.png")
            output, checkpoint = root / "results", root / "checkpoint.pt"
            torch.save({"generator": {}}, checkpoint)
            argv = ["evaluate.py", "--test-sets", *cases, "--data", str(root),
                    "--output", str(output), "--checkpoint", str(checkpoint), "--device", "cpu", "--no-lpips"]
            with (patch("sys.argv", argv), patch("builtins.print"),
                  patch("evaluate.make_generator", return_value=nn.Upsample(scale_factor=4)) as create_model):
                evaluate_main()
            create_model.assert_called_once_with("srresnet")
            self.assertEqual({folder.name for folder in output.iterdir()}, set(cases))
            for name, (_, _, key, _) in cases.items():
                result = output / name
                with (result / "per-image.csv").open() as file:
                    self.assertEqual([row["id"] for row in csv.DictReader(file)], [key])
                self.assertIn("sr_psnr_y", json.loads((result / "mean-scores.json").read_text()))
                self.assertTrue((result / "preview.png").is_file())
                with Image.open(result / "sr" / f"{key}.png") as image:
                    self.assertEqual(image.size, (24, 32))

    def test_independent_loss_switches(self):
        x, target = torch.ones(1, 3, 3, 3), torch.zeros(1, 3, 3, 3)
        for pixel_weight, vgg_weight in [(1, 0), (0, 2), (0.5, 3)]:
            g = nn.Conv2d(3, 3, 1)
            args = SimpleNamespace(pixel_weight=pixel_weight, vgg_weight=vgg_weight)
            feature_loss = nn.MSELoss() if vgg_weight else None
            logs = update_batch(g, None, feature_loss, optimizer(g, 1e-4), None, x, target, args)
            self.assertEqual("pixel_mse" in logs, bool(pixel_weight))
            self.assertEqual("vgg_mse" in logs, bool(vgg_weight))
            self.assertNotIn("adversarial", logs)
            expected = pixel_weight * logs.get("pixel_mse", 0) + vgg_weight * logs.get("vgg_mse", 0)
            self.assertAlmostEqual(logs["g_loss"], expected, places=5)

    def test_checkpoint_contains_only_generator_parameters_and_buffers(self):
        torch.manual_seed(7)
        args = SimpleNamespace(pixel_weight=1.0, adv_weight=0.001)
        x, y = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
        g = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3))
        d = nn.Conv2d(3, 1, 1)
        opt_g, opt_d = optimizer(g, 1e-4), optimizer(d, 1e-4)
        update_batch(g, d, None, opt_g, opt_d, x, y, args)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run = Mock(id="unit")
            with patch.dict(os.environ, WANDB_CACHE_DIR=str(out / "cache"),
                            WANDB_DATA_DIR=str(out / "staging")):
                save_checkpoint(out, g, run, stage="srgan", step=1)
            run.log_artifact.assert_called_once()
            self.assertEqual(run.log_artifact.call_args.kwargs["aliases"], ["latest", "step-1"])
            artifact = run.log_artifact.call_args.args[0]
            self.assertEqual(artifact.name, "srgan-unit")
            self.assertEqual(artifact.metadata, {"step": 1, "stage": "srgan"})
            self.assertEqual(set(artifact.manifest.entries), {"training.pt"})
            saved = load_checkpoint(str(out / "latest.pt"), out)
            self.assertEqual(set(saved), {"generator"})
            torch.testing.assert_close(saved["generator"], g.state_dict(), rtol=0, atol=0)

    def test_t4_scripts_without_launching_training(self):
        root = Path(__file__).resolve().parents[1]
        env = {"PATH": os.environ["PATH"], "PYTHON": "/bin/echo"}
        def commands(script, *args, **options):
            output = subprocess.check_output(["bash", str(root / "scripts" / script), *args],
                                             env={**env, **options}, text=True)
            return [line.split() for line in output.splitlines()]

        with tempfile.TemporaryDirectory() as tmp:
            cnn, = commands("pilot_srresnet.sh", RUN_ROOT=tmp)
            warmup, gan = commands("pilot_srgan.sh", RUN_ROOT=tmp)
            for flags in (cnn, warmup, gan):
                self.assertEqual(flags[flags.index("--val-every") + 1], "50")
                self.assertIn("--lpips", flags)
                self.assertEqual(flags[flags.index("--device") + 1], "cuda")
            dp_cnn, = commands("pilot_srresnet.sh", "--data-parallel", RUN_ROOT=tmp)
            dp_warmup, dp_gan = commands("pilot_srgan.sh", "--data-parallel", RUN_ROOT=tmp)
            for flags in (dp_cnn, dp_warmup, dp_gan):
                self.assertIn("--data-parallel", flags)
                self.assertEqual(flags[flags.index("--batch-size") + 1], "16")
            pretrain_dir = Path(warmup[warmup.index("--out") + 1])
            self.assertNotEqual(cnn[cnn.index("--out") + 1], str(pretrain_dir))
            self.assertEqual(warmup[warmup.index("--steps") + 1], "200")
            self.assertEqual(gan[gan.index("--steps") + 1], "50")
            checkpoint = str(pretrain_dir / "latest.pt")
            self.assertEqual(gan[gan.index("--init-generator") + 1], checkpoint)
            pretrain_dir.mkdir()
            Path(checkpoint).touch()
            warmup, gan = commands("pilot_srgan.sh", "--no-lpips", RUN_ROOT=tmp)
            for flags in (warmup, gan):
                self.assertNotIn("--resume", flags)
                self.assertIn("--no-lpips", flags)
            subset_warmup, subset_gan = commands("pilot_srgan.sh", "--div2k-subset", "unknown", RUN_ROOT=tmp)
            for flags in (subset_warmup, subset_gan):
                self.assertEqual(flags[-2:], ["--div2k-subset", "unknown"])
            custom, = commands("pilot_srgan.sh", GENERATOR="my.pt", VAL_EVERY="5", EPOCHS="10")
            self.assertEqual(custom[custom.index("--val-every") + 1], "5")
            self.assertEqual(custom[custom.index("--init-generator") + 1], "my.pt")
            self.assertEqual(custom[custom.index("--epochs") + 1], "10")


if __name__ == "__main__":
    unittest.main()
