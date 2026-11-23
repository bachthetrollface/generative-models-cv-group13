"""Evaluate paired ×4 PNGs with the same protocol and generate visual comparisons."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from div2k import DIV2K_SUBSETS, bgr_tensor, div2k_pairs, image_pairs, read_pair
from models import SRResNet


def device_for(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def to_image(rgb):
    array = rgb.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def y_metrics(pred, target):
    # BT.601 studio-range Y, expressed on the 0..1 image scale.
    weights = pred.new_tensor([65.481, 128.553, 24.966]).view(3, 1, 1) / 255
    def luminance(rgb):
        return ((rgb * weights).sum(0) + 16 / 255)[None, None, 4:-4, 4:-4]
    x, y = luminance(pred), luminance(target)
    mse = (x - y).square().mean().item()
    psnr = -10 * math.log10(mse) if mse else math.inf
    if min(x.shape[-2:]) < 11:
        raise ValueError("PSNR/SSIM evaluation needs at least 19 HR pixels per side.")
    axis = torch.arange(-5, 6, dtype=x.dtype, device=x.device)
    kernel = torch.softmax(-axis.square() / (2 * 1.5 ** 2), dim=0)
    window = torch.outer(kernel, kernel)[None, None]
    def blur(image):
        return F.conv2d(image, window)
    ux, uy = blur(x), blur(y)
    vx, vy = blur(x.square()) - ux.square(), blur(y.square()) - uy.square()
    cov = blur(x * y) - ux * uy
    ssim = ((2 * ux * uy + 0.01 ** 2) * (2 * cov + 0.03 ** 2)
            / ((ux.square() + uy.square() + 0.01 ** 2) * (vx + vy + 0.03 ** 2)))
    return psnr, ssim.mean().item()


def save_preview(path, images):
    panels = []
    for label, image in images:
        # Identical central HR crop at native resolution for all methods.
        w, h = image.size
        cw, ch = min(w, 256), min(h, 256)
        left, top = (w - cw) // 2, (h - ch) // 2
        panels.append((label, image.crop((left, top, left + cw, top + ch))))
    out = Image.new("RGB", (cw * len(panels), ch + 20), "white")
    draw = ImageDraw.Draw(out)
    for i, (label, panel) in enumerate(panels):
        draw.text((i * cw + 4, 3), label, fill="black")
        out.paste(panel, (i * cw, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def make_lpips(device):
    import lpips
    return lpips.LPIPS(net="alex", version="0.1", verbose=False).eval().to(device)


@torch.inference_mode()
def evaluate(generator, pairs, device, crop_lr=0, preview=None, perceptual=None, *, sr_dir=None):
    generator.eval()
    if sr_dir is not None:
        sr_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, pair in enumerate(pairs):
        lr, hr = read_pair(pair, crop_lr=crop_lr)
        inputs = bgr_tensor(lr).unsqueeze(0).to(device)
        target = bgr_tensor(hr).flip(0).to(device)
        reference = lr.resize(hr.size, Image.Resampling.BICUBIC)
        baseline = bgr_tensor(reference).flip(0).to(device)
        prediction = ((generator(inputs)[0].flip(0) + 1) * 0.5).clamp(0, 1)
        prediction = (prediction * 255).round() / 255  # evaluate exportable 8-bit images
        row = {"id": pair[0]}
        for name, rgb in (("sr", prediction), ("bicubic", baseline)):
            row[f"{name}_psnr_y"], row[f"{name}_ssim_y"] = y_metrics(rgb, target)
            if perceptual is not None:
                def scaled(image):
                    return image[None, :, 4:-4, 4:-4] * 2 - 1
                row[f"{name}_lpips"] = perceptual(scaled(rgb), scaled(target)).item()
        rows.append(row)
        show_preview = index == 0 and preview is not None
        if sr_dir is not None or show_preview:
            sr_image = to_image(prediction)
            if sr_dir is not None:
                sr_image.save(sr_dir / f"{pair[0]}.png")
            if show_preview:
                save_preview(preview, [("Bicubic", reference), ("Model", sr_image), ("HR", hr)])
    return {k: float(np.mean([row[k] for row in rows])) for k in rows[0] if k != "id"}, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    div2k_test_sets = {f"div2k-{subset}": subset for subset in DIV2K_SUBSETS}
    p.add_argument("--test-sets", nargs="+", action="extend", choices=[*div2k_test_sets, "urban100"],
                   help="Evaluate named test sets and save each under its own output directory")
    p.add_argument("--data", type=Path, default=Path("data"), help="Root directory for named test sets")
    p.add_argument("--hr", type=Path, help="HR directory for a custom test set")
    p.add_argument("--lr", type=Path, help="LR directory for a custom test set")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    if args.test_sets and (args.hr is not None or args.lr is not None):
        p.error("Use --test-sets or a custom --hr and --lr pair.")
    if not args.test_sets and (args.hr is None or args.lr is None):
        p.error("Supply --test-sets or both --hr and --lr.")

    try:
        if args.test_sets:
            jobs = []
            for name in dict.fromkeys(args.test_sets):
                if name == "urban100":
                    pairs = image_pairs(args.data / "Urban100_HR", args.data / "Urban100_LR_x4")
                else:
                    pairs = div2k_pairs(args.data, "valid", div2k_test_sets[name])
                jobs.append((name, pairs, args.output / name))
        else:
            jobs = [(None, image_pairs(args.hr, args.lr), args.output)]
    except ValueError as exc:
        p.error(str(exc))

    device = device_for(args.device)
    generator = SRResNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    generator.load_state_dict(checkpoint["generator"])
    perceptual = make_lpips(device) if args.lpips else None
    for name, pairs, output in jobs:
        scores, rows = evaluate(generator, pairs, device, preview=output / "preview.png",
                                perceptual=perceptual, sr_dir=output / "sr")
        with (output / "per-image.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output / "mean-scores.json").write_text(json.dumps(scores, indent=2))
        print(f"{name} {scores}" if name else scores, flush=True)


if __name__ == "__main__":
    main()
