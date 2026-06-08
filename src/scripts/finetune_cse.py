#!/usr/bin/env python3
"""
Fine-tune MVSplat (from re10k.ckpt) on the CARLA CSE training scenes.

Builds the model through the repo exactly like `render_cse.py`, loads the
pretrained weights, then runs a simple photometric (MSE + LPIPS) training loop
over K-nearest-view interpolation samples (odd->odd self-supervision). Saves
checkpoints that `render_cse.py` can load directly (same ModelWrapper format).

Example:
  python -m src.scripts.finetune_cse \
      --data_dir   datasets/carla_cse/train \
      --checkpoint re10k.ckpt \
      --out        checkpoints/carla_ft \
      --num_context_views 6 --steps 8000

Then evaluate the fine-tuned model with the SAME zero-shot pipeline:
  python -m src.scripts.render_cse --data_dir datasets/carla_cse/test \
      --checkpoint checkpoints/carla_ft/finetune_008000.ckpt \
      --out outputs/mvsplat_ft --num_context_views 6
"""
import argparse
import math
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.scripts.render_cse import build_cfg
from src.model.encoder import get_encoder
from src.model.decoder import get_decoder
from src.model.model_wrapper import ModelWrapper
from src.dataset.dataset_carla_cse import DatasetCarlaCSETrain


def to_device(batch, device):
    for sub in ("context", "target"):
        for k, v in batch[sub].items():
            if torch.is_tensor(v):
                batch[sub][k] = v.to(device)
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="datasets/carla_cse/train")
    ap.add_argument("--checkpoint", required=True, help="init weights (e.g. re10k.ckpt)")
    ap.add_argument("--out", default="checkpoints/carla_ft")
    ap.add_argument("--experiment", default="re10k")
    ap.add_argument("--num_context_views", type=int, default=6)
    ap.add_argument("--image_shape", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--lpips_weight", type=float, default=0.05)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--near", type=float, default=None)
    ap.add_argument("--far", type=float, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model the same way as render_cse, then load the pretrained weights.
    _, cfg = build_cfg(args.experiment, args.checkpoint, args.image_shape,
                       args.num_context_views)
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    wrapper = ModelWrapper.load_from_checkpoint(
        args.checkpoint,
        optimizer_cfg=cfg.optimizer, test_cfg=cfg.test, train_cfg=cfg.train,
        encoder=encoder, encoder_visualizer=encoder_visualizer, decoder=decoder,
        losses=[], step_tracker=None, strict=False,
    ).to(device)
    wrapper.train()
    encoder, decoder, data_shim = wrapper.encoder, wrapper.decoder, wrapper.data_shim

    # Frozen LPIPS-VGG as a perceptual loss term.
    import lpips as lpips_pkg
    lpips_net = lpips_pkg.LPIPS(net="vgg").to(device)
    for p in lpips_net.parameters():
        p.requires_grad_(False)
    lpips_net.eval()

    dataset = DatasetCarlaCSETrain(
        args.data_dir, tuple(args.image_shape), args.num_context_views,
        near=args.near, far=args.far,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)
    print(f"[data] {len(dataset)} samples from {args.data_dir} "
          f"({args.num_context_views} context views)")

    opt = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def save(step):
        p = out / f"finetune_{step:06d}.ckpt"
        torch.save({"state_dict": wrapper.state_dict(),
                    "pytorch-lightning_version": pl.__version__,
                    "global_step": step}, p)
        print(f"\n[ckpt] saved {p}")

    step, done = 0, False
    pbar = tqdm(total=args.steps, desc="finetune")
    while not done:
        for batch in loader:
            batch = to_device(batch, device)
            batch = data_shim(batch)
            for grp in opt.param_groups:
                grp["lr"] = args.lr * min(1.0, step / max(1, args.warmup))

            gaussians = encoder(batch["context"], step, deterministic=False)
            b, v, _, h, w = batch["target"]["image"].shape
            out_dec = decoder.forward(
                gaussians,
                batch["target"]["extrinsics"], batch["target"]["intrinsics"],
                batch["target"]["near"], batch["target"]["far"], (h, w),
                depth_mode=None,
            )
            pred = out_dec.color.reshape(b * v, 3, h, w)
            gt = batch["target"]["image"].reshape(b * v, 3, h, w)
            mse = F.mse_loss(pred, gt)
            lp = lpips_net(pred * 2 - 1, gt * 2 - 1).mean()
            loss = mse + args.lpips_weight * lp

            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
            pbar.update(1)
            if step % args.log_every == 0:
                psnr = -10 * math.log10(mse.item() + 1e-12)
                pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f}",
                                 lpips=f"{lp.item():.3f}")
            if step % args.save_every == 0 or step >= args.steps:
                save(step)
            if step >= args.steps:
                done = True
                break
    pbar.close()
    print("[done] fine-tuning complete")


if __name__ == "__main__":
    main()
