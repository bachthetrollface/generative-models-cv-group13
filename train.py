"""Train ×4 SRResNet and MSRResNet generators on DF2K + OST or paired DIV2K images."""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import time
import uuid

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import wandb

from div2k import DIV2K_SUBSETS, BatchSchedule, TrainPatches, kaggle_hr_pairs, training_split
from evaluate import KAGGLE_TEST_SETS, device_for, evaluate, evaluate_checkpoint, make_lpips
from models import Discriminator, VGGContent, VGGStyleDiscriminator, make_generator


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=["srresnet", "srgan"], required=True)
    p.add_argument("--arch", choices=["srresnet", "msrresnet"], default="srresnet",
                   help="Generator family. The run_msr*.sh scripts supply the BasicSR optimization settings.")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--dataset", choices=["df2k-ost", "div2k"],
                   help="Training data source (default: df2k-ost for bicubic, div2k for other tracks)")
    p.add_argument("--div2k-subset", choices=DIV2K_SUBSETS, default="bicubic",
                   help="DIV2K ×4 degradation track")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--init-generator", help="Initialize a fresh run from generator weights, using a path or wandb:// reference")
    duration = p.add_mutually_exclusive_group()
    duration.add_argument("--steps", type=int, help="Number of stage updates")
    duration.add_argument("--epochs", type=int, help="Number of stage epochs")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=5413)
    p.add_argument("--lr", type=float, default=1e-4)
    schedule = p.add_mutually_exclusive_group()
    schedule.add_argument("--lr-drop-step", type=int, help="Updates before a single LR drop, 0 disables")
    schedule.add_argument("--lr-milestones", type=int, nargs="+", help="Completed updates at each LR drop")
    schedule.add_argument("--lr-cosine-period", type=int, help="Updates in each cosine restart cycle")
    p.add_argument("--lr-gamma", type=float, default=0.1, help="LR multiplier at each drop")
    p.add_argument("--lr-min", type=float, default=0.0, help="Minimum LR of a cosine cycle")
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--ema-decay", type=float, default=0.0, help="Generator EMA for evaluation, 0 disables")
    p.add_argument("--pixel-weight", type=float, default=1.0, help="Pixel loss weight")
    p.add_argument("--vgg-weight", type=float, default=0.0, help="VGG feature loss weight")
    p.add_argument("--vgg-layer", choices=["vgg22", "vgg54", "conv5_4"], default="vgg54")
    p.add_argument("--adv-weight", type=float, help="Adversarial weight, default 0 for SRResNet and 0.001 for SRGAN")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--data-parallel", action="store_true", help="Split training batches across visible CUDA GPUs on one machine")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--test-kaggle", action="store_true",
                   help="Evaluate the final checkpoint on both Kaggle test folders after training")
    p.add_argument("--val-images", type=int, help="Limit validation to this many held-out HR images (omit for the full split)")
    p.add_argument("--val-crop-lr", type=int, default=64, help="Central validation crop in LR pixels, 0 for full images")
    p.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True, help="Also compute LPIPS-Alex on validation images")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.getenv("WANDB_MODE", "offline"))
    p.add_argument("--project", default=os.getenv("WANDB_PROJECT", "div2k-sr-group13"))
    p.add_argument("--entity", default=os.getenv("WANDB_ENTITY"))
    args = p.parse_args()
    if args.dataset is None:
        args.dataset = "df2k-ost" if args.div2k_subset == "bicubic" else "div2k"
    if args.dataset == "df2k-ost" and args.div2k_subset != "bicubic":
        p.error("DF2K + OST supports bicubic ×4. Use --dataset div2k for other tracks.")
    gan = args.stage == "srgan"
    if args.adv_weight is None:
        args.adv_weight = 0.001 if gan else 0.0
    if args.lr_drop_step is None and args.lr_milestones is None and args.lr_cosine_period is None:
        args.lr_drop_step = 100_000 if gan else 0
    weights = [args.pixel_weight, args.vgg_weight, args.adv_weight]
    if any(w < 0 or not math.isfinite(w) for w in weights) or not any(weights):
        p.error("Enable at least one positive loss weight. All weights must be finite and nonnegative.")
    if gan and not args.init_generator:
        p.error("SRGAN needs --init-generator to initialize the generator.")
    if min(args.batch_size, args.log_every, args.save_every, args.val_every) < 1 or args.workers < 0:
        p.error("Batch size and logging intervals must be positive, workers nonnegative.")
    if args.lr <= 0 or not math.isfinite(args.lr) or (args.lr_drop_step or 0) < 0:
        p.error("Learning rate must be positive and finite; the drop step must be nonnegative.")
    if args.lr_milestones and (min(args.lr_milestones) < 1 or args.lr_milestones != sorted(set(args.lr_milestones))):
        p.error("LR milestones must be positive and strictly increasing.")
    if args.lr_cosine_period is not None and (args.lr_cosine_period < 1 or not 0 <= args.lr_min <= args.lr):
        p.error("Cosine scheduling needs a positive period and a minimum between zero and the initial LR.")
    if not 0 < args.lr_gamma <= 1 or not 0 <= args.adam_beta2 < 1 or not 0 <= args.ema_decay < 1:
        p.error("Use 0 < lr-gamma <= 1 and 0 <= adam-beta2, ema-decay < 1.")
    if (args.val_images is not None and args.val_images < 1) or 0 < args.val_crop_lr < 5 or args.val_crop_lr < 0:
        p.error("Use a positive validation image count and a validation crop of 0 or at least 5 LR pixels.")
    return args


def load_checkpoint(source, out):
    if source.startswith("wandb://"):
        artifact = wandb.Api().artifact(source.removeprefix("wandb://"), type="model")
        folder = artifact.download(root=str(out / "downloads" / artifact.id))
        source = Path(folder) / "training.pt"
    return torch.load(source, weights_only=True, map_location="cpu")


def optimizer(model, lr, beta1=0.9, beta2=0.999):
    return torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))


def learning_rate(initial, drop_step, completed_steps, *, milestones=None, gamma=0.1,
                  cosine_period=None, minimum=0):
    if cosine_period:
        # BasicSR includes the minimum at each cycle boundary before restarting.
        phase = (completed_steps - 1) % cosine_period + 1 if completed_steps else 0
        return minimum + (initial - minimum) * (1 + math.cos(math.pi * phase / cosine_period)) / 2
    drops = milestones or ([drop_step] if drop_step else [])
    return initial * gamma ** sum(step <= completed_steps for step in drops)


def update_discriminator(discriminator, opt_d, target, prediction):
    discriminator.train().requires_grad_(True)
    opt_d.zero_grad(set_to_none=True)
    real_logits = discriminator(target)
    fake_logits = discriminator(prediction.detach())
    d_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
    d_loss.backward()
    opt_d.step()
    return dict(d_loss=d_loss.item(), d_real=real_logits.detach().sigmoid().mean().item(),
                d_fake=fake_logits.detach().sigmoid().mean().item())


def update_batch(generator, discriminator, vgg, opt_g, opt_d, lr_image, target, args, *, msr=False):
    generator.train()
    opt_g.zero_grad(set_to_none=True)
    prediction = generator(lr_image)
    logs = {}
    if discriminator is not None:
        if not msr:
            logs.update(update_discriminator(discriminator, opt_d, target, prediction))
        # BasicSR updates G first with D's batch statistics active.
        discriminator.train(msr).requires_grad_(False)
    loss = prediction.new_zeros(())
    if args.pixel_weight:
        pixel = F.l1_loss(prediction, target) if msr else F.mse_loss(prediction, target)
        loss = loss + args.pixel_weight * pixel
        logs["pixel_l1" if msr else "pixel_mse"] = pixel.item()
    if vgg is not None:
        feature = vgg(prediction, target)
        loss = loss + args.vgg_weight * feature
        logs["vgg_l1" if msr else "vgg_mse"] = feature.item()
    if discriminator is not None:
        adversarial = F.softplus(-discriminator(prediction)).mean()
        loss = loss + args.adv_weight * adversarial
        logs["adversarial"] = adversarial.item()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite generator loss")
    loss.backward()
    opt_g.step()
    if msr and discriminator is not None:
        logs.update(update_discriminator(discriminator, opt_d, target, prediction))
    logs["g_loss"] = loss.item()
    return logs


def save_checkpoint(out, generator, run, *, stage, step, arch="srresnet", ema=None):
    path = out / "latest.pt"
    temporary = out / "latest.pt.tmp"
    checkpoint = {"generator": generator.state_dict()}
    if arch != "srresnet":
        checkpoint["arch"] = arch
    if ema is not None:
        checkpoint["generator_ema"] = ema.state_dict()
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)
    if run is not None:
        artifact = wandb.Artifact(f"{stage}-{run.id}", type="model",
                                  metadata={"step": step, "stage": stage})
        artifact.add_file(str(path), name="training.pt", policy="mutable")
        run.log_artifact(artifact, aliases=["latest", f"step-{step}"])


def train(args):
    msr = args.arch == "msrresnet"
    experiment = ("m" + args.stage) if msr else args.stage
    train_pairs, valid_pairs = training_split(args.data, args.div2k_subset, dataset=args.dataset)
    valid_ids = set(list(dict.fromkeys(pair[1] for pair in valid_pairs))[:args.val_images])
    validation_sample = [pair for pair in valid_pairs if pair[1] in valid_ids]
    steps_per_epoch = math.ceil(len(train_pairs) / args.batch_size)
    settings = {k: getattr(args, k) for k in (
        "stage", "batch_size", "seed", "lr", "lr_drop_step", "pixel_weight",
        "vgg_weight", "vgg_layer", "adv_weight", "div2k_subset", "dataset")}
    split = {"train": [p[0] for p in train_pairs], "validation": [p[0] for p in valid_pairs]}
    stop = args.steps
    if args.epochs is not None:
        stop = args.epochs * steps_per_epoch
    if stop is None:
        stop = {"srresnet": 1_000_000, "msrresnet": 1_000_000, "srgan": 200_000, "msrgan": 400_000}[experiment]
    if stop <= 0:
        raise ValueError("Training requires a positive number of updates.")
    existing = args.out / "latest.pt"
    if existing.exists():
        raise ValueError(f"{existing} already exists. Select a fresh output directory.")
    history_path = args.out / "metrics.jsonl"
    if history_path.exists() and history_path.stat().st_size:
        raise ValueError(f"{history_path} already records a run. Select a fresh output directory.")

    device = device_for(args.device)
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    if args.data_parallel and (gpu_count < 2 or args.batch_size < gpu_count):
        raise ValueError("--data-parallel needs at least two visible CUDA GPUs and at least one sample per GPU.")
    torch.manual_seed(args.seed)
    generator = make_generator(args.arch).to(device)
    ema = copy.deepcopy(generator).eval().requires_grad_(False) if args.ema_decay else None
    if args.init_generator:
        checkpoint = load_checkpoint(args.init_generator, args.out)
        source_arch = checkpoint.get("arch", "srresnet")
        if source_arch != args.arch:
            raise ValueError(f"Generator checkpoint uses {source_arch}, but --arch is {args.arch}.")
        generator.load_state_dict(checkpoint["generator"])
        if ema is not None:
            ema.load_state_dict(checkpoint.get("generator_ema", checkpoint["generator"]))
        del checkpoint
    discriminator_class = VGGStyleDiscriminator if msr else Discriminator
    discriminator = discriminator_class().to(device) if args.adv_weight else None
    vgg = VGGContent(args.vgg_layer, msr=msr).to(device) if args.vgg_weight else None
    perceptual = make_lpips(device) if args.lpips else None
    opt_g = optimizer(generator, args.lr, args.adam_beta1, args.adam_beta2)
    opt_d = optimizer(discriminator, args.lr, args.adam_beta1, args.adam_beta2) if discriminator is not None else None

    # Keep the underlying modules for single-image validation and portable checkpoints.
    training_generator = generator
    training_discriminator = discriminator
    if args.data_parallel:
        training_generator = torch.nn.DataParallel(generator)
        if discriminator is not None:
            training_discriminator = torch.nn.DataParallel(discriminator)

    schedules = BatchSchedule(len(train_pairs), args.batch_size, args.seed, 0, stop)
    loader = DataLoader(TrainPatches(train_pairs, msr=msr), batch_sampler=schedules, num_workers=args.workers,
                        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(args.seed))
    (args.out / "split.json").write_text(json.dumps(split, indent=2))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(total_steps=stop, steps_per_epoch=steps_per_epoch, device_used=str(device))
    if device.type == "cuda":
        config["training_gpu_count"] = gpu_count if args.data_parallel else 1
        config["gpu_name"] = torch.cuda.get_device_name(device)
        torch.cuda.reset_peak_memory_stats(device)
    (args.out / "config-from-step-0.json").write_text(json.dumps(config, indent=2))
    run = None
    if args.wandb_mode != "disabled":
        run = wandb.init(project=args.project, entity=args.entity, name=args.out.name,
                         id=uuid.uuid4().hex[:12], dir=str(args.out), mode=args.wandb_mode, config=settings)
        run.config.update(config, allow_val_change=True)
        run.define_metric("train/step")
        run.define_metric("*", step_metric="train/step")
    print(f"{experiment}: {stop} updates on {device}, output {args.out}", flush=True)
    with history_path.open("a", buffering=1) as history:
        for step, (lr_image, target) in enumerate(loader, 1):
            batch_start = time.perf_counter()
            rate = learning_rate(args.lr, args.lr_drop_step, step - 1, milestones=args.lr_milestones,
                                 gamma=args.lr_gamma, cosine_period=args.lr_cosine_period, minimum=args.lr_min)
            for opt in (opt_g, opt_d):
                if opt:
                    for group in opt.param_groups:
                        group["lr"] = rate
            losses = update_batch(training_generator, training_discriminator, vgg, opt_g, opt_d,
                                  lr_image.to(device), target.to(device), args, msr=msr)
            if ema is not None:
                with torch.no_grad():
                    for average, current in zip(ema.parameters(), generator.parameters()):
                        average.lerp_(current, 1 - args.ema_decay)
                    for average, current in zip(ema.buffers(), generator.buffers()):
                        average.copy_(current)
            values = {"train/step": step}
            if step % args.log_every == 0 or step in (1, stop):
                values.update({f"train/{k}": v for k, v in losses.items()})
                values.update({"train/lr": rate, "train/epoch": step / steps_per_epoch})
                values["train/update_seconds"] = time.perf_counter() - batch_start
                if device.type == "cuda":
                    values["train/peak_gpu_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            preview = None
            if step % args.val_every == 0 or step == stop:
                preview = args.out / "previews" / f"step-{step:07d}.png"
                scores, _ = evaluate(ema if ema is not None else generator, validation_sample, device,
                                     args.val_crop_lr, preview, perceptual, arch=args.arch)
                values.update({f"val/{k}": v for k, v in scores.items()})
            if step % args.save_every == 0 or step == stop:
                save_checkpoint(args.out, generator, run, stage=experiment, step=step, arch=args.arch, ema=ema)
            if len(values) > 1:
                print(json.dumps(values), file=history)
                print(json.dumps(values), flush=True)
                if run:
                    if preview:
                        values["val/preview"] = wandb.Image(str(preview), caption="Bicubic | model | HR")
                    run.log(values)
    if run:
        run.finish()


if __name__ == "__main__":
    args = arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        test_jobs = []
        if args.test_kaggle:
            test_jobs = [(name, kaggle_hr_pairs(args.data, folder), args.out / "test" / name)
                         for name, folder in KAGGLE_TEST_SETS.items()]
        train(args)
        if test_jobs:
            evaluate_checkpoint(args.out / "latest.pt", test_jobs, args.device, args.lpips)
