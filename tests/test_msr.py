import copy
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from div2k import TrainPatches
from evaluate import main as evaluate_main
from models import MSRResNet, VGGContent, VGGStyleDiscriminator
from train import arguments, learning_rate, save_checkpoint, train


class MSRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def launcher_args(self, name, *flags, kaggle=False, **options):
        root = Path(__file__).resolve().parents[1]
        env = {"PATH": os.environ["PATH"], "PYTHON": "/bin/echo", **options}
        prefix = "kaggle" if kaggle else "run"
        output = subprocess.check_output(["bash", f"scripts/{prefix}_{name}.sh", *flags],
                                         cwd=root, env=env, text=True)
        stages = []
        for line in output.splitlines():
            with patch("sys.argv", line.split()):
                stages.append(arguments())
        return stages

    def test_launchers_use_full_recipes_and_matching_pretraining(self):
        standalone, = self.launcher_args("msrresnet")
        pretrain, gan = self.launcher_args("msrgan", "--div2k-subset=unknown", "--no-lpips")
        for args in (standalone, pretrain, gan):
            self.assertEqual((args.arch, args.adam_beta2, args.ema_decay), ("msrresnet", 0.99, 0.999))
        for args in (standalone, pretrain):
            self.assertEqual((args.steps, args.lr, args.lr_cosine_period, args.lr_min),
                             (1_000_000, 2e-4, 250_000, 1e-7))
            self.assertEqual((args.pixel_weight, args.vgg_weight, args.adv_weight), (1, 0, 0))
        self.assertEqual((gan.steps, gan.lr, gan.lr_milestones, gan.lr_gamma),
                         (400_000, 1e-4, [50_000, 100_000, 200_000, 300_000], 0.5))
        self.assertEqual((gan.pixel_weight, gan.vgg_weight, gan.adv_weight, gan.vgg_layer),
                         (0.01, 1, 0.005, "conv5_4"))
        for args in (pretrain, gan):
            self.assertEqual(args.div2k_subset, "unknown")
            self.assertFalse(args.lpips)
        self.assertNotEqual(standalone.out, pretrain.out)
        self.assertEqual(Path(gan.init_generator), pretrain.out / "latest.pt")
        supplied, = self.launcher_args("msrgan", GENERATOR="my.pt", EPOCHS="2")
        self.assertEqual((supplied.init_generator, supplied.steps, supplied.epochs), ("my.pt", None, 2))

    def test_kaggle_launchers(self):
        standalone, = self.launcher_args("msrresnet", kaggle=True)
        self.assertEqual(standalone.out, Path("runs/msrresnet"))
        self.assertIn(str(standalone.data), ("/kaggle/input/df2k-ost",
                                             "/kaggle/input/datasets/thaihoa1476050/df2k-ost"))
        self.assertEqual(standalone.dataset, "df2k-ost")
        self.assertIsNone(standalone.val_images)
        self.assertTrue(standalone.test_kaggle)
        pretrain, gan = self.launcher_args("msrgan", "--data-parallel", kaggle=True,
                                           DATA_ROOT="/custom/input", RUN_ROOT="/custom/output")
        for args in (pretrain, gan):
            self.assertEqual(args.data, Path("/custom/input"))
            self.assertTrue(args.data_parallel)
        self.assertFalse(pretrain.test_kaggle)
        self.assertTrue(gan.test_kaggle)
        self.assertEqual(gan.init_generator, "/custom/output/msrgan-pretrain/latest.pt")
        supplied, = self.launcher_args("msrgan", kaggle=True, GENERATOR="/input/generator.pt")
        self.assertEqual(supplied.init_generator, "/input/generator.pt")

    def test_schedule_boundaries(self):
        for step, expected in [(0, 2e-4), (125_000, (2e-4 + 1e-7) / 2), (250_000, 1e-7),
                               (250_001, 2e-4), (750_000, 1e-7)]:
            self.assertAlmostEqual(learning_rate(2e-4, None, step, cosine_period=250_000, minimum=1e-7),
                                   expected, places=12)
        milestones = [50_000, 100_000, 200_000, 300_000]
        for index, step in enumerate(milestones):
            for count, power in [(step - 1, index), (step, index + 1)]:
                self.assertEqual(learning_rate(1e-4, None, count, milestones=milestones, gamma=0.5),
                                 1e-4 * 0.5 ** power)

    def test_training_pairs_keep_rgb_range_and_alignment_after_augmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            hr, lr = Path(tmp) / "hr.png", Path(tmp) / "lr.png"
            pixels = np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype=np.uint8)
            pixels[:, :, 0] = 255
            Image.fromarray(pixels).save(lr)
            Image.fromarray(pixels.repeat(4, axis=0).repeat(4, axis=1)).save(hr)
            patches = TrainPatches([("pair", hr, lr)], msr=True)
            for seed in range(8):
                inputs, target = patches[0, seed]
                self.assertEqual(target.shape, (3, 128, 128))
                self.assertTrue((inputs[0] == 1).all())
                torch.testing.assert_close(inputs, target[:, ::4, ::4])

    def test_vgg_l1_before_relu_and_discriminator_shape(self):
        features = nn.Sequential(*(nn.Identity() for _ in range(35)), nn.ReLU())
        with patch("models.vgg19", return_value=SimpleNamespace(features=features)):
            loss = VGGContent("conv5_4", msr=True)
        x = torch.zeros(1, 3, 8, 8, requires_grad=True)
        y = torch.ones_like(x, requires_grad=True)
        torch.testing.assert_close(loss.encode(x), (x - loss.mean) / loss.std)
        distance = loss(x, y)
        torch.testing.assert_close(distance, loss.std.reciprocal().mean())
        distance.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNone(y.grad)
        with torch.no_grad():
            self.assertEqual(VGGStyleDiscriminator()(torch.rand(1, 3, 128, 128)).shape, (1, 1))

    def test_checkpoint_evaluation_uses_averaged_rgb_generator(self):
        model = MSRResNet()
        self.assertEqual(len(model.body), 16)
        self.assertFalse(any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules()))
        average = copy.deepcopy(model)
        with torch.no_grad():
            average.conv_last.weight.zero_()
            average.conv_last.bias.copy_(torch.tensor([0.2, 0, -0.2]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hr, lr = root / "Urban100_HR", root / "Urban100_LR_x4"
            hr.mkdir()
            lr.mkdir()
            for folder, size in [(hr, (20, 28)), (lr, (5, 7))]:
                Image.new("RGB", size, (30, 120, 240)).save(folder / "img_001.png")
            save_checkpoint(root, model, None, stage="msrgan", step=1, arch="msrresnet", ema=average)
            output = root / "scores"
            argv = ["evaluate.py", "--data", tmp, "--test-sets", "urban100", "--no-lpips", "--device", "cpu",
                    "--checkpoint", str(root / "latest.pt"), "--output", str(output)]
            with patch("sys.argv", argv), patch("builtins.print"):
                evaluate_main()
            with Image.open(output / "urban100/sr/img_001.png") as image:
                self.assertEqual(image.size, (20, 28))
                self.assertEqual(image.getpixel((0, 0)), (81, 120, 189))

    def test_gan_rejects_original_generator_before_loading_perceptual_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save({"generator": {}}, root / "original.pt")
            args, = self.launcher_args("msrgan", "--device", "cpu", GENERATOR=str(root / "original.pt"))
            args.out = root / "gan"
            with patch("train.training_split", return_value=([], [])), self.assertRaisesRegex(ValueError, "uses srresnet"):
                train(args)


if __name__ == "__main__":
    unittest.main()
