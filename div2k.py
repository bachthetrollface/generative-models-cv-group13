"""Pair original image IDs first, then sample aligned ×4 crops."""
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DIV2K_SUBSETS = {
    "bicubic": ("bicubic", "x4"),
    "unknown": ("unknown", "x4"),
    "realistic-mild": ("mild", "x4m"),
    "realistic-difficult": ("difficult", "x4d"),
    "realistic-wild": ("wild", "x4w"),
}
SPLIT_FILE = Path(__file__).resolve().parent / "splits" / "div2k_train_val.txt"


def image_pairs(hr_dir, lr_dir, *, lr_suffix="x4", variants=()):
    hr = {p.stem: p for p in sorted(Path(hr_dir).glob("*.png"))}
    if variants:
        hr = {f"{key}{lr_suffix}{variant}": path
              for key, path in hr.items() for variant in variants}
        lr_suffix = ""
    lr = {}
    for path in sorted(Path(lr_dir).glob("*.png")):
        key = path.stem.removesuffix(lr_suffix)
        if key in lr:
            raise ValueError(f"Duplicate LR ID {key} in {lr_dir}")
        lr[key] = path
    if not hr or hr.keys() != lr.keys():
        raise ValueError(f"Expected matching HR/LR PNGs in {hr_dir} and {lr_dir}. "
                         f"HR={len(hr)}, LR={len(lr)}, unmatched={sorted(hr.keys() ^ lr.keys())[:8]}")
    return [(key, hr[key], lr[key]) for key in sorted(hr)]


def div2k_pairs(root, split, subset):
    track, suffix = DIV2K_SUBSETS[subset]
    hr_dir = root / f"DIV2K_{split}_HR"
    lr_dir = root / f"DIV2K_{split}_LR_{track}"
    if subset in ("bicubic", "unknown"):
        lr_dir /= "X4"
    variants = range(1, 5) if subset == "realistic-wild" and split == "train" else ()
    return image_pairs(hr_dir, lr_dir, lr_suffix=suffix, variants=variants)


def training_split(root, subset="bicubic"):
    pairs = div2k_pairs(root, "train", subset)
    expected = {f"{i:04d}" for i in range(1, 801)}
    if {hr_path.stem for _, hr_path, _ in pairs} != expected:
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
    return ([p for p in pairs if assignments[p[1].stem] == "train"],
            [p for p in pairs if assignments[p[1].stem] == "validation"])


def read_pair(pair, crop_lr=0, rng=None):
    key, hr_path, lr_path = pair
    with Image.open(hr_path) as hr, Image.open(lr_path) as lr:
        w, h = lr.size
        # Use the nominal ×4 grid, cropping the HR dimensions to a multiple of four.
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


def rgb_tensor(image):
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div_(255)


def bgr_tensor(image):
    return rgb_tensor(image).flip(0)


class TrainPatches(Dataset):
    def __init__(self, pairs, *, msr=False):
        self.pairs = pairs
        self.msr = msr

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, key):
        index, crop_seed = key
        rng = random.Random(crop_seed)
        lr, hr = read_pair(self.pairs[index], crop_lr=32 if self.msr else 24, rng=rng)
        if self.msr:
            for transform in (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM,
                              Image.Transpose.TRANSPOSE):
                if rng.random() < 0.5:
                    lr, hr = lr.transpose(transform), hr.transpose(transform)
            return rgb_tensor(lr), rgb_tensor(hr)
        return bgr_tensor(lr), bgr_tensor(hr).mul_(2).sub_(1)


class BatchSchedule:
    """Deterministic batches and crop seeds indexed by update."""
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
