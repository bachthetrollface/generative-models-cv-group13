# SRResNet, SRGAN, MSRResNet and MSRGAN on DF2K + OST ×4

All four launchers default to bicubic ×4 training on the Kaggle DF2K + OST images. The SRResNet pilot trains from scratch for 250 pixel-MSE updates. The SRGAN pilot runs its own 200-update MSE pretraining stage and initializes the GAN from those weights for 50 updates. The dedicated MSRResNet and MSRGAN launchers use the full BasicSR ×4 training schedules.

## Run the pilot

On a Linux machine with an NVIDIA T4, use Python 3.11 or 3.12 and a driver compatible with the CUDA 13.0 PyTorch wheels. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert torch.version.cuda == "13.0"'
wandb login
bash scripts/download_df2k_ost.sh
# Standalone SRResNet experiment
bash scripts/pilot_srresnet.sh
# Independent SRGAN experiment, including its own generator pretraining
bash scripts/pilot_srgan.sh
```

The pilot scripts default to CUDA and use one T4 per process by default. Set `PRETRAIN_STEPS` to adjust the GAN’s private pretraining budget or set `GENERATOR` to initialize the GAN from an existing generator checkpoint. Use a fresh `RUN_ROOT` to rerun training. Choose a GPU with `CUDA_VISIBLE_DEVICES=0`. The trainer records GPU name, peak PyTorch memory allocation, and update time. Batch size 16 is the paper setting. If memory requires a smaller batch, set the same `BATCH_SIZE` for both pilots and record the change. The scripts honor `PYTHON`, `DATA_ROOT`, `RUN_ROOT`, `DEVICE`, `WORKERS`, `BATCH_SIZE`, `STEPS`, `EPOCHS`, and the standard W&B project and entity variables. The first VGG-based run downloads the torchvision VGG19 weights into `.cache/torch`. [PyTorch wheels](https://pytorch.org/get-started/previous-versions/)

The Kaggle download script extracts version 1 into `data/`. Set `DATA_ROOT` on the download and pilot commands to use another directory. On Kaggle, skip the download and set `DATA_ROOT` to the attached dataset directory containing `train/` and `test/`. [DF2K + OST dataset card](https://www.kaggle.com/datasets/thaihoa1476050/df2k-ost)

The loader reads HR PNGs from `train/DIV2K/DIV2K_train_HR/`, `train/Flickr2K/`, and `train/OST/`. It trims the right and bottom edges to multiples of four and uses Pillow bicubic downsampling to generate aligned LR inputs. Internal validation uses the same downsampling. For official benchmark LR pixels, use the paired DIV2K evaluation workflow below.

For a DIV2K-only bicubic run, use `bash scripts/download_div2k.sh` and pass `--dataset div2k` to the pilot. The DIV2K download script defaults to the official bicubic ×4 track. It accepts one or more tracks from `bicubic`, `unknown`, `realistic-mild`, `realistic-difficult`, and `realistic-wild`, and downloads the corresponding LR training and validation archives with their shared HR archives. Set `DATA_ROOT` to use a different data directory. [Official DIV2K archives](https://data.vision.ee.ethz.ch/cvl/DIV2K/#data-access)

Use `--div2k-subset` to select the training track. Both pilot launchers accept it, including the SRGAN generator pretraining stage. For example, an unknown ×4 pilot uses:

```bash
bash scripts/download_div2k.sh unknown
RUN_ROOT=runs/unknown bash scripts/pilot_srgan.sh --div2k-subset unknown
```

All training stages read [the fixed DIV2K split](splits/div2k_train_val.txt) and record it in `split.json`. It assigns 750 HR image IDs to training and 50 to internal validation. Wild keeps all four LR training variants of each HR image in the same partition, for 3,000 training pairs and 200 internal validation pairs. DF2K + OST uses the same 50/800 validation fraction for each source, rounded to the nearest image with halves rounded up. The training/validation counts are 750/50 for DIV2K, 2,484/166 for Flickr2K, and 8,422/562 for OST. The [Flickr2K](splits/flickr2k_validation.txt) and [OST](splits/ost_validation.txt) manifests list held-out HR stems in a fixed order, selected by ranking the SHA-256 hash of each `source/stem`. All other training-folder IDs go to training. Both Kaggle test folders are excluded from training and internal validation. The official validation IDs 0801–0900 are used as project test images for each track. To switch tracks from a trained generator, initialize a fresh run from its weights with `--init-generator` (or `GENERATOR` in the SRGAN launcher).

## Run MSR training in a Kaggle notebook

Enable a GPU accelerator, attach the DF2K + OST and Urban100 datasets, and switch the notebook working directory to the repository root. Use the notebook's GPU-enabled PyTorch and torchvision. Install the additional training packages in the notebook kernel if needed. Enable Internet for the first LPIPS and VGG19 weight downloads, or supply cached weights through `TORCH_HOME`.

```python
%pip install wandb==0.19.11 lpips==0.1.4
```

Run the commands in separate notebook cells to train MSRResNet and then fine-tune its checkpoint with MSRGAN:

```bash
!bash scripts/kaggle_msrresnet.sh
!GENERATOR=runs/msrresnet/latest.pt bash scripts/kaggle_msrgan.sh
```

Attach `harshraone/urban100` as a second notebook input. Both launchers detect its short or owner-qualified mount. For a custom mount, set `URBAN100_ROOT` to the directory containing `Urban 100/`. Urban100 is used only for final evaluation. [Kaggle Urban100 dataset](https://www.kaggle.com/datasets/harshraone/urban100)

After training, each Kaggle launcher reloads its final checkpoint and evaluates the averaged generator on all 100 images in `test/DIV2K_valid/`, 300 in `test/OutdoorSceneTest300/`, and the 100 supplied ×4 LR/HR pairs from the attached Urban100 dataset. Under the model’s run directory, `test/kaggle-div2k/`, `test/ost300/`, and `test/urban100/` contain separate scores and super-resolved images.

For an independent MSRGAN run, use `!bash scripts/kaggle_msrgan.sh`. It runs its own MSRResNet pretraining stage. To start from a checkpoint attached from a previous notebook, set `GENERATOR` to its mounted path.

Both launchers detect the short and owner-qualified Kaggle dataset paths and save runs under `runs/` inside the cloned repository. Override these with `DATA_ROOT` and `RUN_ROOT`. They use the full training schedules and all 778 validation images by default. Set `STEPS` and `PRETRAIN_STEPS` for shorter runs, or pass `--data-parallel` when using two GPUs. W&B logging defaults to disabled on Kaggle. Set `WANDB_MODE=online` with W&B credentials to enable it.

## Train and evaluate MSRResNet and MSRGAN

After the environment and Kaggle dataset setup above, run these commands from the repository root. Passing the standalone MSRResNet checkpoint to MSRGAN lets both experiments share the same pretraining run.

```bash
bash scripts/run_msrresnet.sh
GENERATOR=runs/msrresnet/latest.pt bash scripts/run_msrgan.sh

bash scripts/download_urban100.sh
for model in msrresnet msrgan; do
  python evaluate.py --checkpoint "runs/$model/latest.pt" \
    --test-sets kaggle-div2k ost300 urban100 --output "runs/eval/$model"
done
```

Running `bash scripts/run_msrgan.sh` by itself first trains an MSRResNet in `runs/msrgan-pretrain/`, then fine-tunes the GAN in `runs/msrgan/`. A shorter two-stage run can use `RUN_ROOT=runs/msr-smoke PRETRAIN_STEPS=2 STEPS=2 bash scripts/run_msrgan.sh`. Schedule boundaries always count optimizer updates at their original scale.

| Stage | Generator loss on RGB images at the [0, 1] intensity scale | Learning rate | Default updates |
|---|---|---|---:|
| MSRResNet | Pixel L1 | Adam at 2e-4, four 250,000-update cosine cycles with minimum 1e-7 | 1,000,000 |
| MSRGAN | 0.01 × pixel L1 + 1 × VGG19 feature L1 at conv5_4 before ReLU + 0.005 × vanilla GAN loss | Both Adam optimizers start at 1e-4 and halve the rate after 50k, 100k, 200k and 300k updates | 400,000 |

Both stages use Adam betas (0.9, 0.99), generator EMA decay 0.999, and aligned 32×32 LR / 128×128 HR crops with random flips and transpositions. The generator has 16 residual blocks without batch normalization and adds a learned residual to bilinearly upsampled input. MSRGAN uses a VGG-style discriminator and updates the generator and discriminator once per batch. These settings follow BasicSR's published ×4 configurations. [MSRResNet configuration](https://github.com/XPixelGroup/BasicSR/blob/master/options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml) · [MSRGAN configuration](https://github.com/XPixelGroup/BasicSR/blob/master/options/train/SRResNet_SRGAN/train_MSRGAN_x4.yml)

Edit `scripts/msr_common.sh` to change the MSR recipes. It selects `--arch msrresnet` and supplies stage-specific optimizer and loss arguments to `train.py`. It sources `scripts/pilot_common.sh` for data, batch size, device, logging, validation and duration overrides. Both families use the project's fixed DIV2K split, epoch sampler, and PSNR-Y, SSIM-Y and LPIPS evaluation protocol.

MSR checkpoints store the architecture, the current generator weights for fine-tuning, and the averaged generator weights for evaluation. The GAN initializes both sets from the pretraining checkpoint. Evaluation selects the architecture and averaged weights automatically. Existing SRResNet and SRGAN checkpoints also work with the same evaluation command.

## Train a single experiment on two T4s

On a machine with two T4s, select both devices and pass `--data-parallel` to the experiment you want to run.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/pilot_srresnet.sh --data-parallel
CUDA_VISIBLE_DEVICES=0,1 bash scripts/pilot_srgan.sh --data-parallel
```

Each command runs one process. The default global batch of 16 splits into eight images per GPU. SRGAN uses both GPUs for its private generator pretraining and adversarial stage. BatchNorm computes statistics per replica, so results can differ from a single-GPU batch of 16. Validation, optional VGG feature loss, W&B logging, and checkpoint writing use the primary GPU or host process. Checkpoint parameter names are compatible with single-GPU evaluation and generator initialization. The logged peak GPU memory is for the primary GPU.

## What the pilot measures

Both stages use up to 16 distinct training pairs per batch, random 24×24 LR / 96×96 HR crops on the nominal ×4 grid, and Adam at 1e-4. The generator uses 16 residual blocks, BatchNorm, PReLU, two PixelShuffle stages, and a linear three-channel output in BGR order. The discriminator uses eight 3×3 convolutions and a 1024-unit dense layer. Both networks follow Figure 4. The paper reports training on 350,000 ImageNet images. The default pilot uses DF2K + OST. [Ledig et al., architecture and training](https://arxiv.org/html/1609.04802v5#S2)

| Stage | Default loss | Full paper schedule |
|---|---|---|
| SRResNet-MSE | Pixel MSE on HR intensities scaled to [−1, 1] | 1,000,000 updates at 1e-4 |
| SRGAN-MSE | Pixel MSE + 0.001 × non-saturating adversarial generator loss | 100,000 updates at 1e-4, then 100,000 at 1e-5 |

LR inputs use [0, 1]. Each VGG feature tensor is divided by 12.75 before MSE. VGG receives RGB standardized for the torchvision ImageNet weights. The GAN alternates one discriminator and one generator update. We compute binary cross-entropy in logit space. Implementation choices include PyTorch initialization, per-channel PReLU, Adam β₂ = 0.999, mean loss reduction across batch and channels, and frozen discriminator running statistics during the generator update. The pilot ends before the paper's GAN learning-rate drop. [Losses and training details](https://arxiv.org/html/1609.04802v5#S2.SS2)

The pilot uses the SRGAN-MSE variant from section 3.4 of the paper. For a later SRGAN-VGG54 comparison, reuse the same generator checkpoint with `RUN_ROOT=runs/gan-vgg54 GENERATOR=runs/t4-pilot/srgan-pretrain/latest.pt bash scripts/pilot_srgan.sh --pixel-weight 0 --vgg-weight 1`. The trainer also supports `--vgg-layer vgg22|vgg54` and `--adv-weight`. A zero weight disables that component. Calling the trainer directly without a duration uses the full schedule for the selected stage. [Paper loss variants](https://arxiv.org/html/1609.04802v5#S3.SS4)

## Metrics, images, and checkpoints

Under `runs/t4-pilot/`, the independent experiments write to `srresnet/` and `srgan/`. SRGAN saves its pretrained generator weights in `srgan-pretrain/`. Read `metrics.jsonl` for step, learning rate, epoch progress, individual active losses, and discriminator real/fake scores. SRResNet and SRGAN both validate every 50 updates. Both also validate at the final update. Set `VAL_EVERY` to override the interval. Validation logs bicubic and model PSNR/SSIM on BT.601 luminance after shaving four HR pixels per edge. SSIM uses an 11×11 Gaussian window. Each validation pass covers the full active holdout by default (778 HR images for DF2K + OST or 50 for DIV2K alone). The wild track evaluates all four LR variants for each held-out HR image. Validation uses centered 256×256 HR crops and saves a comparison under `previews/`. Pass `--val-crop-lr 0` for full-size images. For a quicker check, pass `--val-images 8` or set `VAL_IMAGES=8` on a launcher. Validation and full-image evaluation report LPIPS-Alex v0.1 for both bicubic and model outputs by default. The metric downloads its pretrained backbone on first use. Pass `--no-lpips` to disable it.

The T4 scripts default to online W&B logging, including images and versioned checkpoint artifacts. Set `WANDB_PROJECT` and optionally `WANDB_ENTITY` before starting. On an offline machine, set `WANDB_MODE=offline` and skip login. To upload an offline run later, use `wandb sync <offline-run-directory>` on the directory printed by W&B and keep the local artifact staging directory until sync completes. Use `WANDB_MODE=disabled` for local checkpoints, images, and JSONL only.

The trainer atomically replaces `latest.pt` at each save interval and at the final update. The SRResNet and SRGAN pilots save only the generator parameters and buffers, including BatchNorm statistics. W&B versions contain the same weights under `training.pt` and carry `latest` and `step-N` aliases. Both new and older checkpoints can supply generator weights for evaluation or `--init-generator`.

```bash
# Run a fresh SRResNet experiment for 500 updates.
RUN_ROOT=runs/srresnet-500 STEPS=500 bash scripts/pilot_srresnet.sh

# Run a fresh GAN for 10 epochs using existing pretraining weights.
RUN_ROOT=runs/srgan-10-epochs EPOCHS=10 GENERATOR=runs/t4-pilot/srgan-pretrain/latest.pt \
  bash scripts/pilot_srgan.sh
```

Each epoch uses every training pair once. With a batch size of 16, DF2K + OST takes 729 updates per epoch (11,656 images). DIV2K-only takes 47 updates for the 750-pair tracks and 188 for wild. Each sampled pair gets a fresh crop each epoch. To initialize a fresh GAN from weights in W&B, use an authenticated account and set `GENERATOR=wandb://ENTITY/PROJECT/ARTIFACT:latest`.

## Evaluate full images

To evaluate both held-out Kaggle sets, select `kaggle-div2k` (100 DIV2K validation images) and `ost300` (300 images in `test/OutdoorSceneTest300/`). Both use the same Pillow bicubic ×4 downsampling as training.

```bash
python evaluate.py --data "${DATA_ROOT:-data}" --test-sets kaggle-div2k ost300 \
  --checkpoint runs/t4-pilot/srgan/latest.pt --output runs/eval/srgan
```

To evaluate an existing checkpoint on the Kaggle Urban100 mount, run from the repository root with the mount path shown in the notebook Input pane:

```bash
python evaluate.py --test-sets urban100 --urban100-root /kaggle/input/urban100 \
  --checkpoint runs/msrresnet/latest.pt --output runs/msrresnet/test
```

For the official bicubic ×4 DIV2K pairs, first run `bash scripts/download_div2k.sh --valid-only bicubic`.

Run `bash scripts/download_urban100.sh` to fetch the 100 Urban100 HR and bicubic ×4 LR pairs. It uses `DATA_ROOT` (default `data`) and extracts the images into `Urban100_HR/` and `Urban100_LR_x4/`, ready for the command below. [Urban100 archives](https://huggingface.co/datasets/eugenesiow/Urban100/tree/57309de03db02ac12ee09ad6127e7b815c211328/data)

Pass one or more names to `--test-sets` to evaluate a checkpoint across datasets in a single command. Named datasets are read from `--data` (default `data`). Each set gets its own folder under `--output`, with `sr/` PNG exports, a per-image CSV, mean scores, and a comparison preview. Each PNG is four times the LR width and height and uses the HR image ID as its filename.

```bash
python evaluate.py --checkpoint runs/t4-pilot/srgan/latest.pt \
  --test-sets div2k-bicubic urban100 --output runs/eval/srgan --lpips

bash scripts/download_div2k.sh --valid-only unknown realistic-mild realistic-difficult realistic-wild
python evaluate.py --checkpoint runs/t4-pilot/srgan/latest.pt \
  --test-sets div2k-unknown div2k-realistic-mild div2k-realistic-difficult div2k-realistic-wild \
  --output runs/eval/srgan --lpips
```

For custom pairs, pass `--hr` and `--lr` in place of `--test-sets`. Custom LR filenames can use the same stem as the HR image or add `x4`. Results go directly under `--output`.

The realistic tracks can introduce spatial shifts and scaling. Training crops and evaluation scores use the project's nominal ×4 grid. The NTIRE 2018 leaderboard protocol included a translation search. [NTIRE 2018 challenge report](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Timofte_NTIRE_2018_Challenge_CVPR_2018_paper.pdf)

## Unit tests

Run `python -m unittest discover -s tests -v` in the pinned environment. Tests use synthetic images and toy networks to check checkpoint contents, loss settings, pairing, metrics, and launcher arguments. They run locally with mocked W&B logging.
