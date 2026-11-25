"""Pair original image IDs first, then sample aligned ×4 crops."""
from itertools import zip_longest
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


def urban100_pairs(root):
    kaggle_dir = root / "Urban 100" / "X4 Urban100" / "X4"
    if not kaggle_dir.is_dir():
        return image_pairs(root / "Urban100_HR", root / "Urban100_LR_x4")
    pairs = []
    for index in range(1, 101):
        key = f"img_{index:03d}"
        hr = kaggle_dir / "HIGH x4 URban100" / f"{key}_SRF_4_HR.png"
        lr = kaggle_dir / "LOW x4 URban100" / f"{key}_SRF_4_LR.png"
        for path in (hr, lr):
            if not path.is_file():
                raise ValueError(f"Missing Urban100 ×4 evaluation image: {path}")
        pairs.append((key, hr, lr))
    return pairs


def div2k_pairs(root, split, subset):
    track, suffix = DIV2K_SUBSETS[subset]
    hr_dir = root / f"DIV2K_{split}_HR"
    lr_dir = root / f"DIV2K_{split}_LR_{track}"
    if subset in ("bicubic", "unknown"):
        lr_dir /= "X4"
    variants = range(1, 5) if subset == "realistic-wild" and split == "train" else ()
    return image_pairs(hr_dir, lr_dir, lr_suffix=suffix, variants=variants)


def kaggle_hr_pairs(root, folder):
    directory = root / folder
    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise ValueError(f"No DF2K + OST images in {directory}. Point --data at the extracted Kaggle root.")
    return [(p.stem, p, None) for p in paths]


def training_split(root, subset="bicubic", *, dataset="div2k"):
    if dataset == "df2k-ost":
        if subset != "bicubic":
            raise ValueError("DF2K + OST supports bicubic ×4. Use --dataset div2k for paired DIV2K tracks.")
        pairs = kaggle_hr_pairs(root, "train/DIV2K/DIV2K_train_HR")
    else:
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
    train = [p for p in pairs if assignments[p[1].stem] == "train"]
    valid = [p for p in pairs if assignments[p[1].stem] == "validation"]
    if dataset == "df2k-ost":
        validation_groups = [valid]
        for source in ("Flickr2K", "OST"):
            manifest = SPLIT_FILE.with_name(f"{source.lower()}_validation.txt")
            held_out = manifest.read_text(encoding="utf-8").splitlines()
            samples = {key: (f"{source}/{key}", path, None)
                       for key, path, _ in kaggle_hr_pairs(root, f"train/{source}")}
            validation_groups.append([samples.pop(key) for key in held_out if key in samples])
            train.extend(samples.values())
        # A short pilot validation takes a prefix, so alternate sources.
        valid = [pair for group in zip_longest(*validation_groups) for pair in group if pair is not None]
    return train, valid


def read_pair(pair, crop_lr=0, rng=None):
    key, hr_path, lr_path = pair
    with Image.open(hr_path) as hr:
        w, h = hr.width // 4, hr.height // 4
        # Use the nominal ×4 grid, cropping the HR dimensions to a multiple of four.
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
        hr_box = tuple(4 * x for x in box)
        if lr_path is None:
            # Resize with the surrounding image as filter context, including at crop edges.
            aligned = hr.crop((0, 0, 4 * w, 4 * h)).convert("RGB")
            lr = aligned.resize((width, height), Image.Resampling.BICUBIC, box=hr_box)
        else:
            with Image.open(lr_path) as original:
                if original.size != (w, h):
                    raise ValueError(f"Misaligned ×4 pair {key}: HR {hr.size}, LR {original.size}")
                lr = original.crop(box).convert("RGB")
        hr = hr.crop(hr_box).convert("RGB")
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
