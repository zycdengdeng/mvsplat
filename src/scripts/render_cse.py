#!/usr/bin/env python3
"""
Render MVSplat target views for the CARLA CSE benchmark and export them as PNGs
named exactly as the target names in `test.txt`, so they can be scored with the
shared `tools/eval_cse.py` (apples-to-apples with GS-Net / 3DGS).

Pipeline:
  runs/cse_scenes/<id>/  --tools/colmap_to_pixelsplat.py-->  datasets/carla_cse/test/
  datasets/carla_cse/test/  --THIS SCRIPT-->  <out>/<id>/renders/<target_name>.png
  <out>/<id>/renders/      --tools/eval_cse.py-->  PSNR / SSIM / LPIPS

For each TARGET view we feed the 2 nearest SOURCE views (re10k weights are 2-view)
to MVSplat's own encoder->gaussians, then render with MVSplat's own decoder at the
GT target resolution. The checkpoint and model are built through the repo's own
config + ModelWrapper, so weights load exactly as in `src.main`.

Example:
  python -m src.scripts.render_cse \
      --data_dir datasets/carla_cse/test \
      --checkpoint checkpoints/re10k.ckpt \
      --out outputs/mvsplat \
      --experiment re10k
"""
import argparse
from pathlib import Path

import hydra
import torch
from tqdm import tqdm

from src.config import load_typed_root_config
from src.global_cfg import set_cfg
from src.misc.image_io import save_image
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.model.model_wrapper import ModelWrapper
from src.dataset.dataset_carla_cse import DatasetCarlaCSE


def build_cfg(experiment: str, checkpoint: str, image_shape, num_context_views: int):
    """Compose the repo's Hydra config (same groups as `src.main` test mode)."""
    overrides = [
        f"+experiment={experiment}",
        "mode=test",
        "dataset/view_sampler=evaluation",
        # Build the encoder for the actual number of context views we feed. The
        # released weights are 2-view, but num_views only drives cross-view
        # attention grouping (no weight-shape dependency), so K-view inference
        # works and just changes coverage.
        f"dataset.view_sampler.num_context_views={num_context_views}",
        f"checkpointing.load={checkpoint}",
        f"dataset.image_shape=[{image_shape[0]},{image_shape[1]}]",
        "wandb.mode=disabled",
    ]
    with hydra.initialize(version_base=None, config_path="../../config"):
        cfg_dict = hydra.compose(config_name="main", overrides=overrides)
    set_cfg(cfg_dict)  # encoder reads get_cfg() (mode, num_context_views) at init
    return cfg_dict, load_typed_root_config(cfg_dict)


def load_model(cfg, checkpoint, device):
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    model = ModelWrapper.load_from_checkpoint(
        checkpoint,
        optimizer_cfg=cfg.optimizer,
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        encoder=encoder,
        encoder_visualizer=encoder_visualizer,
        decoder=decoder,
        losses=[],  # rendering only: no loss needed (avoids LPIPS/VGG weight download)
        step_tracker=None,
        strict=False,
    )
    return model.eval().to(device)


def batchify(views, device):
    return {
        k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v)
        for k, v in views.items()
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="dir with the converted .torch chunks + index.json + cse_meta.json")
    ap.add_argument("--checkpoint", required=True, help="MVSplat .ckpt (e.g. checkpoints/re10k.ckpt)")
    ap.add_argument("--out", default="outputs/mvsplat", help="output root; renders go to <out>/<id>/renders")
    ap.add_argument("--experiment", default="re10k", help="repo experiment config (re10k / acid)")
    ap.add_argument("--num_context_views", type=int, default=2)
    ap.add_argument("--image_shape", type=int, nargs=2, default=[256, 256],
                    help="encoder input H W (re10k weights -> 256 256)")
    ap.add_argument("--near", type=float, default=None, help="override near for ALL scenes")
    ap.add_argument("--far", type=float, default=None, help="override far for ALL scenes")
    ap.add_argument("--default_near", type=float, default=0.5)
    ap.add_argument("--default_far", type=float, default=150.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_dict, cfg = build_cfg(args.experiment, args.checkpoint, args.image_shape,
                              args.num_context_views)
    model = load_model(cfg, args.checkpoint, device)
    encoder, decoder = model.encoder, model.decoder

    dataset = DatasetCarlaCSE(
        args.data_dir,
        image_shape=tuple(args.image_shape),
        num_context_views=args.num_context_views,
        near=args.near, far=args.far,
        default_near=args.default_near, default_far=args.default_far,
    )

    out_root = Path(args.out)
    per_scene = {}
    for item in tqdm(dataset, desc="rendering CSE targets"):
        scene = item["scene"]
        context = batchify(item["context"], device)
        target = batchify(item["target"], device)

        gaussians = encoder(context, 0, deterministic=False)
        h, w = target["image"].shape[-2:]
        output = decoder.forward(
            gaussians,
            target["extrinsics"],
            target["intrinsics"],
            target["near"],
            target["far"],
            (h, w),
            depth_mode=None,
        )
        color = output.color[0, 0]  # [3, H, W] in [0,1]

        renders_dir = out_root / scene / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)
        save_image(color, renders_dir / item["target_name"])
        per_scene[scene] = per_scene.get(scene, 0) + 1

    print("\n[done] rendered target views per scene:")
    for scene, n in sorted(per_scene.items()):
        print(f"  {scene}: {n} -> {out_root / scene / 'renders'}")
    print(f"\nNext: score with tools/eval_cse.py, e.g.\n"
          f"  python tools/eval_cse.py --multi \\\n" +
          " \\\n".join(
              f"    {out_root}/{s}/renders:runs/cse_scenes/{s}" for s in sorted(per_scene)
          ) +
          f" \\\n    --out {out_root}/cse_scores.json")


if __name__ == "__main__":
    main()
