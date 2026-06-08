# MVSplat zero-shot on the CARLA CSE benchmark

End-to-end pipeline to evaluate **MVSplat (official RealEstate10K weights)** on the
CARLA **CSE (Cross-Sensor)** benchmark and score it with the shared `eval_cse.py`,
so the numbers are directly comparable to GS-Net and 3DGS.

```
runs/cse_scenes/<id>/  ─(1) tools/colmap_to_pixelsplat.py─►  datasets/carla_cse/test/
datasets/carla_cse/test/  ─(2) src/scripts/render_cse.py──►  outputs/mvsplat/<id>/renders/<target_name>.png
outputs/mvsplat/<id>/renders/  ─(3) tools/eval_cse.py─────►  PSNR / SSIM / LPIPS
```

For each of the 60 **target** views per scene, the 2 nearest **source** views are fed
to MVSplat's own encoder→Gaussians, then rendered with MVSplat's own decoder at the
**GT target resolution**. Each even (target) camera sits angularly between two odd
(source) cameras, so this is close to interpolation — the favorable regime for
MVSplat's 2-view setup.

## Files added here

| File | Role |
|---|---|
| `tools/colmap_to_pixelsplat.py` | (1) Convert `runs/cse_scenes/<id>/` COLMAP scenes → pixelSplat `.torch` chunks. Also estimates per-scene `near`/`far` from the source `points3D`. |
| `src/dataset/dataset_carla_cse.py` | `DatasetCarlaCSE` + `ViewSamplerCSE` (one item per target view; context = 2 nearest source views). Reuses the repo's pose/crop conventions. |
| `src/scripts/render_cse.py` | (2) Loads the checkpoint via the repo's `ModelWrapper`, renders every target view, exports PNGs named exactly per `test.txt`. |
| `tools/eval_cse.py` | (3) The shared scorer (unchanged). PSNR / SSIM(11×11 Gaussian) / LPIPS-vgg. |

## Prerequisites

- The 5 test scenes at `runs/cse_scenes/110 … 510` (COLMAP text model + 120 images + `test.txt`).
- MVSplat env (see main `README.md`; PyTorch 2.1.2, a CUDA GPU).
- Checkpoint `checkpoints/re10k.ckpt` (RealEstate10K, the main zero-shot point) from
  the [pretrained models](https://drive.google.com/drive/folders/14_E_5R6ojOWnLSrSVLVEMHnTiKsfddjU).
- For scoring: `pip install lpips` (or place the repo's `lpipsPyTorch` on the path).

## Step 1 — convert the scenes

```bash
python tools/colmap_to_pixelsplat.py \
    --scenes runs/cse_scenes/110 runs/cse_scenes/210 runs/cse_scenes/310 \
             runs/cse_scenes/410 runs/cse_scenes/510 \
    --out    datasets/carla_cse/test
```
Sanity: each scene must print `60 src / 60 tgt`. It also prints the estimated
`near`/`far` (from `points3D`); these are written into `cse_meta.json`.

## Step 2 — render & export target views

```bash
python -m src.scripts.render_cse \
    --data_dir   datasets/carla_cse/test \
    --checkpoint checkpoints/re10k.ckpt \
    --out        outputs/mvsplat \
    --experiment re10k
```
Produces `outputs/mvsplat/<id>/renders/<target_name>.png` — 60 PNGs per scene, named
exactly as in `test.txt`, at the GT resolution.

`near`/`far` come from `cse_meta.json` (the `points3D` estimate). If `points3D` was
missing you can override globally, e.g. `--near 0.5 --far 150`, and adjust.

> If the GT images are smaller than 256 px on a side, lower the encoder input with
> e.g. `--image_shape 128 128` (the `re10k` weights were trained at 256×256, so keep
> 256 when possible).

## Step 3 — score (shared metrics)

```bash
python tools/eval_cse.py --multi \
  outputs/mvsplat/110/renders:runs/cse_scenes/110 \
  outputs/mvsplat/210/renders:runs/cse_scenes/210 \
  outputs/mvsplat/310/renders:runs/cse_scenes/310 \
  outputs/mvsplat/410/renders:runs/cse_scenes/410 \
  outputs/mvsplat/510/renders:runs/cse_scenes/510 \
  --out outputs/mvsplat/cse_scores.json
```
Reports the per-seq + 5-seq-average PSNR/SSIM/LPIPS table.

## Expectations & framing (for the rebuttal)

RealEstate10K weights are trained on **small-baseline indoor** interpolation. The
CARLA ring is wider-baseline outdoor, so **zero-shot numbers are expected to be
clearly below GS-Net** — that gap *is* the indoor→outdoor transfer evidence. The
fair, in-domain comparison would come from a CARLA fine-tuned run (RGB+pose only),
which is a separate follow-up.

## Fine-tuning on CARLA (in-domain comparison)

The zero-shot numbers above use the RealEstate10K weights as-is. For the fair,
in-domain comparison, fine-tune from `re10k.ckpt` on the CARLA **training** scenes
(sequences 101–109 … 501–509; the 5 test scenes 110/210/310/410/510 are held out).

1. Convert the training scenes (their COLMAP models are binary; the converter
   auto-detects `.bin`):
   ```bash
   python tools/colmap_to_pixelsplat.py \
       --scenes /path/to/input_output/101_dense /path/to/input_output/102_dense ... \
       --out datasets/carla_cse/train
   ```
   Each prints `60 src / 0 tgt` (training scenes have no `test.txt`).

2. Fine-tune (photometric MSE + LPIPS, K-nearest-view self-supervision):
   ```bash
   python -m src.scripts.finetune_cse \
       --data_dir   datasets/carla_cse/train \
       --checkpoint re10k.ckpt \
       --out        checkpoints/carla_ft \
       --num_context_views 6 --steps 8000
   ```

3. Evaluate the fine-tuned checkpoint with the SAME pipeline (Steps 2–3 above),
   pointing `--checkpoint` at `checkpoints/carla_ft/finetune_008000.ckpt` and using
   the same `--num_context_views`. Report alongside the zero-shot numbers.

> Train and test with the **same** `--num_context_views` (e.g. 6) for consistency.

## Acceptance checklist

- [ ] Converter prints `60 src / 60 tgt` for all 5 scenes.
- [ ] `renders/` has exactly 60 PNGs per scene, names == `test.txt`, shape == GT.
- [ ] `eval_cse.py` runs without "no overlap"/shape errors and prints the table.
- [ ] Sanity-check 2–3 rendered images visually (geometry roughly aligned, not noise).
- [ ] Save `cse_scores.json` and the printed table.
