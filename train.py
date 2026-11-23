"""Train ×4 SRResNet-MSE and SRGAN-MSE on paired DIV2K PNGs."""
import argparse
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

from div2k import BatchSchedule, TrainPatches, training_split
from evaluate import device_for, evaluate, make_lpips
from models import Discriminator, SRResNet, VGGContent


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=["srresnet", "srgan"], required=True)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, required=True)
    start = p.add_mutually_exclusive_group()
    start.add_argument("--resume", help="Full checkpoint path or wandb://entity/project/artifact:alias")
    start.add_argument("--init-generator", help="Load generator weights only, from a path or wandb:// reference")
    duration = p.add_mutually_exclusive_group()
    duration.add_argument("--steps", type=int, help="Total stage updates, including completed updates")
    duration.add_argument("--epochs", type=int, help="Total stage epochs, including completed epochs")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=5413)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-drop-step", type=int, help="Updates at initial LR before a single 10× drop; 0 disables")
    p.add_argument("--pixel-weight", type=float, default=1.0, help="Pixel MSE weight")
    p.add_argument("--vgg-weight", type=float, default=0.0, help="VGG feature MSE weight")
    p.add_argument("--vgg-layer", choices=["vgg22", "vgg54"], default="vgg54")
    p.add_argument("--adv-weight", type=float, help="Adversarial weight, default 0 for SRResNet and 0.001 for SRGAN")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--data-parallel", action="store_true", help="Split training batches across visible CUDA GPUs on one machine")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--val-images", type=int, default=8, help="First N held-out IDs (1–50)")
    p.add_argument("--val-crop-lr", type=int, default=64, help="Central validation crop in LR pixels, 0 for full images")
    p.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True, help="Also compute LPIPS-Alex on validation images")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.getenv("WANDB_MODE", "offline"))
    p.add_argument("--project", default=os.getenv("WANDB_PROJECT", "div2k-sr-group13"))
    p.add_argument("--entity", default=os.getenv("WANDB_ENTITY"))
    args = p.parse_args()
    gan = args.stage == "srgan"
    if args.adv_weight is None:
        args.adv_weight = 0.001 if gan else 0.0
    if args.lr_drop_step is None:
        args.lr_drop_step = 100_000 if gan else 0
    weights = [args.pixel_weight, args.vgg_weight, args.adv_weight]
    if any(w < 0 or not math.isfinite(w) for w in weights) or not any(weights):
        p.error("Enable at least one positive loss weight. All weights must be finite and nonnegative.")
    if gan and not (args.init_generator or args.resume):
        p.error("SRGAN needs --init-generator from SRResNet or --resume from SRGAN.")
    if min(args.batch_size, args.log_every, args.save_every, args.val_every) < 1 or args.workers < 0:
        p.error("Batch size and logging intervals must be positive, workers nonnegative.")
    if args.lr <= 0 or not math.isfinite(args.lr) or args.lr_drop_step < 0:
        p.error("Learning rate must be positive and finite; the drop step must be nonnegative.")
    if not 1 <= args.val_images <= 50 or 0 < args.val_crop_lr < 5 or args.val_crop_lr < 0:
        p.error("Use 1–50 validation images and a validation crop of 0 or at least 5 LR pixels.")
    return args


def load_checkpoint(source, out):
    if source.startswith("wandb://"):
        artifact = wandb.Api().artifact(source.removeprefix("wandb://"), type="model")
        folder = artifact.download(root=str(out / "downloads" / artifact.id))
        source = Path(folder) / "training.pt"
    return torch.load(source, weights_only=True, map_location="cpu")


def optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))


def learning_rate(initial, drop_step, completed_steps):
    return initial * (0.1 if drop_step and completed_steps >= drop_step else 1)


def restore_training_state(checkpoint, generator, discriminator, opt_g, opt_d, device):
    generator.load_state_dict(checkpoint["generator"])
    opt_g.load_state_dict(checkpoint["opt_g"])
    if discriminator is not None:
        discriminator.load_state_dict(checkpoint["discriminator"])
        opt_d.load_state_dict(checkpoint["opt_d"])
    torch.set_rng_state(checkpoint["cpu_rng"])
    if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
        torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
    if device.type == "mps" and checkpoint.get("mps_rng") is not None:
        torch.mps.set_rng_state(checkpoint["mps_rng"])


def update_batch(generator, discriminator, vgg, opt_g, opt_d, lr_image, target, args):
    generator.train()
    opt_g.zero_grad(set_to_none=True)
    prediction = generator(lr_image)
    logs = {}
    if discriminator is not None:
        discriminator.train().requires_grad_(True)
        opt_d.zero_grad(set_to_none=True)
        real_logits = discriminator(target)
        fake_logits = discriminator(prediction.detach())
        d_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
        d_loss.backward()
        opt_d.step()
        logs.update(d_loss=d_loss.item(), d_real=real_logits.detach().sigmoid().mean().item(),
                    d_fake=fake_logits.detach().sigmoid().mean().item())
        discriminator.eval().requires_grad_(False)  # freeze D weights and running statistics for G
    loss = prediction.new_zeros(())
    if args.pixel_weight:
        pixel = F.mse_loss(prediction, target)
        loss = loss + args.pixel_weight * pixel
        logs["pixel_mse"] = pixel.item()
    if vgg is not None:
        feature = vgg(prediction, target)
        loss = loss + args.vgg_weight * feature
        logs["vgg_mse"] = feature.item()
    if discriminator is not None:
        adversarial = F.softplus(-discriminator(prediction)).mean()
        loss = loss + args.adv_weight * adversarial
        logs["adversarial"] = adversarial.item()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite generator loss")
    loss.backward()
    opt_g.step()
    logs["g_loss"] = loss.item()
    return logs


def save_checkpoint(out, state, run):
    path = out / "latest.pt"
    temporary = out / "latest.pt.tmp"
    torch.save(state, temporary)
    os.replace(temporary, path)
    if run is not None:
        artifact = wandb.Artifact(f"{state['stage']}-{state['run_id']}", type="model",
                                  metadata={"step": state["step"], "stage": state["stage"]})
        artifact.add_file(str(path), name="training.pt", policy="mutable")
        run.log_artifact(artifact, aliases=["latest", f"step-{state['step']}"])


def resume_history(path, step):
    """Discard metrics after the restored checkpoint, including a partial last line."""
    if not path.exists():
        return
    with path.open("rb+") as history:
        while True:
            offset = history.tell()
            line = history.readline()
            if not line:
                break
            try:
                if json.loads(line)["train/step"] <= step:
                    continue
            except (ValueError, KeyError):
                pass
            history.truncate(offset)
            break


def train(args):
    train_pairs, valid_pairs = training_split(args.data)
    steps_per_epoch = math.ceil(len(train_pairs) / args.batch_size)
    settings = {k: getattr(args, k) for k in (
        "stage", "batch_size", "seed", "lr", "lr_drop_step", "pixel_weight",
        "vgg_weight", "vgg_layer", "adv_weight")}
    split = {"train": [p[0] for p in train_pairs], "validation": [p[0] for p in valid_pairs]}
    checkpoint = load_checkpoint(args.resume, args.out) if args.resume else None
    start = 0
    sampler_offset = 0
    if checkpoint:
        if checkpoint["settings"] != settings or checkpoint["split"] != split:
            raise ValueError("Resume settings or DIV2K split changed. Use --init-generator for a new experiment.")
        start = checkpoint["step"]
        if "sampler_offset" in checkpoint:
            sampler_offset = checkpoint["sampler_offset"]
        elif len(train_pairs) % args.batch_size:
            # Older checkpoints skipped one tail batch per completed epoch.
            sampler_offset = start // (len(train_pairs) // args.batch_size)
    stop = args.steps
    if args.epochs is not None:
        stop = args.epochs * steps_per_epoch - sampler_offset
    if stop is None:
        stop = 200_000 if args.stage == "srgan" else 1_000_000
    existing = args.out / "latest.pt"
    if args.resume and existing.is_file() and Path(args.resume).resolve() == existing.resolve() and stop == start:
        print(f"Checkpoint already reached {stop} updates: {existing}")
        return
    if stop <= start:
        raise ValueError(f"Total training target {stop} must exceed completed steps {start}.")
    if existing.exists() and (not args.resume or Path(args.resume).resolve() != existing.resolve()):
        raise ValueError(f"{existing} already exists. Resume this file or select a fresh output directory.")
    history_path = args.out / "metrics.jsonl"
    if not args.resume and history_path.exists() and history_path.stat().st_size:
        raise ValueError(f"{history_path} already records a run. Select a fresh output directory.")

    device = device_for(args.device)
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    if args.data_parallel and (gpu_count < 2 or args.batch_size < gpu_count):
        raise ValueError("--data-parallel needs at least two visible CUDA GPUs and at least one sample per GPU.")
    torch.manual_seed(args.seed)
    generator = SRResNet().to(device)
    discriminator = Discriminator().to(device) if args.adv_weight else None
    vgg = VGGContent(args.vgg_layer).to(device) if args.vgg_weight else None
    perceptual = make_lpips(device) if args.lpips else None
    opt_g = optimizer(generator, args.lr)
    opt_d = optimizer(discriminator, args.lr) if discriminator is not None else None
    run_id = uuid.uuid4().hex[:12]
    if checkpoint:
        restore_training_state(checkpoint, generator, discriminator, opt_g, opt_d, device)
        run_id = checkpoint["run_id"]
    elif args.init_generator:
        generator.load_state_dict(load_checkpoint(args.init_generator, args.out)["generator"])
    del checkpoint

    # Keep the underlying modules for single-image validation and portable checkpoints.
    training_generator = generator
    training_discriminator = discriminator
    if args.data_parallel:
        training_generator = torch.nn.DataParallel(generator)
        if discriminator is not None:
            training_discriminator = torch.nn.DataParallel(discriminator)

    schedules = BatchSchedule(len(train_pairs), args.batch_size, args.seed, start + sampler_offset, stop + sampler_offset)
    loader = DataLoader(TrainPatches(train_pairs), batch_sampler=schedules, num_workers=args.workers,
                        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(args.seed))
    (args.out / "split.json").write_text(json.dumps(split, indent=2))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(total_steps=stop, steps_per_epoch=steps_per_epoch, device_used=str(device))
    if device.type == "cuda":
        config["training_gpu_count"] = gpu_count if args.data_parallel else 1
        config["gpu_name"] = torch.cuda.get_device_name(device)
        torch.cuda.reset_peak_memory_stats(device)
    (args.out / f"config-from-step-{start}.json").write_text(json.dumps(config, indent=2))
    run = None
    if args.wandb_mode != "disabled":
        run = wandb.init(project=args.project, entity=args.entity, name=args.out.name,
                         id=run_id, resume="allow" if args.resume and args.wandb_mode == "online" else None,
                         dir=str(args.out), mode=args.wandb_mode, config=settings)
        run.config.update(config, allow_val_change=True)
        run.define_metric("train/step")
        run.define_metric("*", step_metric="train/step")
    print(f"{args.stage}: {start} -> {stop} updates on {device}, output {args.out}", flush=True)
    if args.resume:
        resume_history(history_path, start)
    with history_path.open("a", buffering=1) as history:
        for step, (lr_image, target) in enumerate(loader, start + 1):
            batch_start = time.perf_counter()
            rate = learning_rate(args.lr, args.lr_drop_step, step - 1)
            for opt in (opt_g, opt_d):
                if opt:
                    for group in opt.param_groups:
                        group["lr"] = rate
            losses = update_batch(training_generator, training_discriminator, vgg, opt_g, opt_d,
                                  lr_image.to(device), target.to(device), args)
            values = {"train/step": step}
            if step % args.log_every == 0 or step in (start + 1, stop):
                values.update({f"train/{k}": v for k, v in losses.items()})
                values.update({"train/lr": rate, "train/epoch": (step + sampler_offset) / steps_per_epoch})
                values["train/update_seconds"] = time.perf_counter() - batch_start
                if device.type == "cuda":
                    values["train/peak_gpu_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            preview = None
            if step % args.val_every == 0 or step == stop:
                preview = args.out / "previews" / f"step-{step:07d}.png"
                scores, _ = evaluate(generator, valid_pairs[:args.val_images], device,
                                     args.val_crop_lr, preview, perceptual)
                values.update({f"val/{k}": v for k, v in scores.items()})
            if step % args.save_every == 0 or step == stop:
                state = dict(format_version=1, stage=args.stage, settings=settings, config=config, split=split,
                             step=step, sampler_offset=sampler_offset, run_id=run_id, generator=generator.state_dict(),
                             opt_g=opt_g.state_dict(), cpu_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                             mps_rng=torch.mps.get_rng_state() if device.type == "mps" else None,
                             discriminator=discriminator.state_dict() if discriminator is not None else None,
                             opt_d=opt_d.state_dict() if opt_d is not None else None)
                save_checkpoint(args.out, state, run)
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
        train(args)
