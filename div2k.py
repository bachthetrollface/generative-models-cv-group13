"""Pair original image IDs first, then sample aligned ×4 crops."""
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def image_pairs(hr_dir, lr_dir):
    hr = {p.stem: p for p in sorted(Path(hr_dir).glob("*.png"))}
    lr = {}
    for path in sorted(Path(lr_dir).glob("*.png")):
        key = path.stem.removesuffix("x4")
        if key in lr:
            raise ValueError(f"Duplicate LR ID {key} in {lr_dir}")
        lr[key] = path
    if not hr or hr.keys() != lr.keys():
        raise ValueError(f"Expected matching HR/LR PNGs in {hr_dir} and {lr_dir}. "
                         f"HR={len(hr)}, LR={len(lr)}, unmatched={sorted(hr.keys() ^ lr.keys())[:8]}")
    return [(key, hr[key], lr[key]) for key in sorted(hr)]


SPLIT_FILE = Path(__file__).resolve().parent / "splits" / "div2k_train_val.txt"


def training_split(root):
    pairs = image_pairs(root / "DIV2K_train_HR", root / "DIV2K_train_LR_bicubic" / "X4")
    expected = {f"{i:04d}" for i in range(1, 801)}
    if {p[0] for p in pairs} != expected:
        raise ValueError("Expected all 800 official DIV2K train IDs 0001–0800.")

    assignments = {}
    for number, line in enumerate(SPLIT_FILE.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.partition("#")[0].split()
        if not fields:
            continue
        if len(fields) != 2 or fields[0] not in ("train", "validation"):
            raise ValueError(f"{SPLIT_FILE}:{number}: expected a split name and image ID")
        split, image_id = fields
        if image_id not in expected or image_id in assignments:
            raise ValueError(f"{SPLIT_FILE}:{number}: unknown or duplicate image ID {image_id}")
        assignments[image_id] = split
    if assignments.keys() != expected or list(assignments.values()).count("validation") != 50:
        raise ValueError(f"{SPLIT_FILE} must assign all 800 IDs to 750 training and 50 validation images.")
    return ([p for p in pairs if assignments[p[0]] == "train"],
            [p for p in pairs if assignments[p[0]] == "validation"])


def read_pair(pair, crop_lr=0, rng=None):
    key, hr_path, lr_path = pair
    with Image.open(hr_path) as hr, Image.open(lr_path) as lr:
        w, h = lr.size
        # Official LR images correspond to HR cropped to a multiple of four.
        if hr.width // 4 != w or hr.height // 4 != h:
            raise ValueError(f"Misaligned ×4 pair {key}: HR {hr.size}, LR {lr.size}")
        width, height = w, h
        left = top = 0
        if crop_lr:
            if rng is not None and min(w, h) < crop_lr:
                raise ValueError(f"{key} too small for a {crop_lr}-pixel LR crop")
            width, height = min(w, crop_lr), min(h, crop_lr)
            if rng is None:
                left, top = (w - width) // 2, (h - height) // 2
            else:
                left, top = rng.randrange(w - width + 1), rng.randrange(h - height + 1)
        box = (left, top, left + width, top + height)
        lr = lr.crop(box).convert("RGB")
        hr = hr.crop(tuple(4 * x for x in box)).convert("RGB")
    return lr, hr


def bgr_tensor(image):
    rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    return rgb.flip(0).float().div_(255)


class TrainPatches(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, key):
        index, crop_seed = key
        lr, hr = read_pair(self.pairs[index], crop_lr=24, rng=random.Random(crop_seed))
        return bgr_tensor(lr), bgr_tensor(hr).mul_(2).sub_(1)


class BatchSchedule:
    """Step-addressed crops make mid-epoch resumes independent of worker prefetch."""
    def __init__(self, size, batch_size, seed, start, stop):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.start, self.stop = start, stop

    def __iter__(self):
        per_epoch = math.ceil(self.size / self.batch_size)
        current_epoch = None
        for step in range(self.start, self.stop):
            epoch, batch = divmod(step, per_epoch)
            if epoch != current_epoch:
                indices = list(range(self.size))
                random.Random(f"{self.seed}/{epoch}").shuffle(indices)
                current_epoch = epoch
            offset = batch * self.batch_size
            yield [(i, f"{self.seed}/{epoch}/{i}") for i in indices[offset:offset + self.batch_size]]

    def __len__(self):
        return self.stop - self.start
