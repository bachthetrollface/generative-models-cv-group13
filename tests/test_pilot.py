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

from div2k import BatchSchedule, image_pairs, read_pair
from evaluate import y_metrics
from models import Discriminator, SRResNet, VGGContent
from train import (
    arguments, learning_rate, load_checkpoint, optimizer, restore_training_state,
    resume_history, save_checkpoint, update_batch,
)


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
        self.assertEqual(len({key for batch in schedule[:46] for key, _ in batch}), 46 * 16)

    def test_luminance_metrics_and_border(self):
        target = torch.rand(3, 32, 32)
        altered = target.clone()
        altered[:, :4] = 0
        self.assertEqual(y_metrics(altered, target), (float("inf"), 1.0))

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

    def test_checkpoint_resume_with_toy_networks(self):
        # Exercise the trainer functions on tiny tensors with both Adam optimizers.
        torch.manual_seed(7)
        args = SimpleNamespace(pixel_weight=1.0, adv_weight=0.001)
        x, y = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
        g, d = nn.Conv2d(3, 3, 1), nn.Conv2d(3, 1, 1)
        opt_g, opt_d = optimizer(g, 1e-4), optimizer(d, 1e-4)
        update_batch(g, d, None, opt_g, opt_d, x, y, args)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run = Mock()
            state = dict(stage="srgan", step=1, run_id="unit", cpu_rng=torch.get_rng_state(),
                         generator=g.state_dict(), discriminator=d.state_dict(),
                         opt_g=opt_g.state_dict(), opt_d=opt_d.state_dict())
            with patch.dict(os.environ, WANDB_CACHE_DIR=str(out / "cache"),
                            WANDB_DATA_DIR=str(out / "staging")):
                save_checkpoint(out, state, run)
            run.log_artifact.assert_called_once()
            self.assertEqual(run.log_artifact.call_args.kwargs["aliases"], ["latest", "step-1"])
            self.assertIn("training.pt", run.log_artifact.call_args.args[0].manifest.entries)
            saved = load_checkpoint(str(out / "latest.pt"), out)
            expected = update_batch(g, d, None, opt_g, opt_d, x, y, args)
            other_g, other_d = nn.Conv2d(3, 3, 1), nn.Conv2d(3, 1, 1)
            other_g_opt, other_d_opt = optimizer(other_g, 1e-4), optimizer(other_d, 1e-4)
            restore_training_state(saved, other_g, other_d, other_g_opt, other_d_opt, torch.device("cpu"))
            replay = update_batch(other_g, other_d, None, other_g_opt, other_d_opt, x, y, args)
            self.assertEqual(expected, replay)
            for a, b in zip([*g.parameters(), *d.parameters()], [*other_g.parameters(), *other_d.parameters()]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_resume_discards_metrics_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "metrics.jsonl"
            history.write_bytes(b'{"train/step":1}\n{"train/step":2}\n{"partial":')
            resume_history(history, 1)
            self.assertEqual(history.read_bytes(), b'{"train/step":1}\n')

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
            pretrain_dir = Path(warmup[warmup.index("--out") + 1])
            self.assertNotEqual(cnn[cnn.index("--out") + 1], str(pretrain_dir))
            self.assertEqual(warmup[warmup.index("--steps") + 1], "200")
            self.assertEqual(gan[gan.index("--steps") + 1], "50")
            checkpoint = str(pretrain_dir / "latest.pt")
            self.assertEqual(gan[gan.index("--init-generator") + 1], checkpoint)
            pretrain_dir.mkdir()
            Path(checkpoint).touch()
            retry, _ = commands("pilot_srgan.sh", "--no-lpips", RUN_ROOT=tmp)
            self.assertEqual(retry[retry.index("--resume") + 1], checkpoint)
            self.assertIn("--no-lpips", retry)
            custom, = commands("pilot_srgan.sh", GENERATOR="my.pt", VAL_EVERY="5")
            self.assertEqual(custom[custom.index("--val-every") + 1], "5")
            self.assertEqual(custom[custom.index("--init-generator") + 1], "my.pt")
            resumed, = commands("pilot_srgan.sh", EPOCHS="10", RESUME="saved.pt")
            self.assertIn("--epochs", resumed)
            self.assertNotIn("--init-generator", resumed)
            self.assertIn("--resume", resumed)


if __name__ == "__main__":
    unittest.main()
