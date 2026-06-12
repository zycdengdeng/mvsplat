# Feed-forward NVS on CARLA / nuScenes — Method, Inference, Training, Results

Single reference for evaluating feed-forward / generalizable 3DGS methods
(MVSplat, MonoSplat, …) on our benchmarks, both zero-shot and CARLA fine-tuned,
under a protocol identical to our own method so all entries are comparable.
Code is on branch `claude/hopeful-bell-Igadf`; server data is under `/mnt/zihanw/`.

---

## 1. Benchmarks

- **CSE (Cross-Sensor).** Reconstruct from one set of cameras, render the held-out
  poses of a *different* camera set never seen during reconstruction
  (cross-viewpoint synthesis). Quantitative.
- **SSE (Same-Sensor).** Hold out a subset of frames of the same cameras;
  reconstruct from the rest and render the held-out frames (temporal
  interpolation). Quantitative; 310 is a pathological scene, excluded from means.
- **nuScenes novel views.** From a base camera, render extrapolated poses
  (translations / rotations) unseen during reconstruction. Qualitative only —
  no ground truth, compared by visual completeness (holes / floaters).

---

## 2. Inference

A feed-forward method predicts pixel-aligned 3D Gaussians from a few posed
context views in a **single forward pass** (no per-scene optimization). For each
held-out target pose:

1. **Select context views (direction-aware).** Keep source views whose viewing
   direction aligns with the target (forward-direction dot product above a
   threshold), then take the nearest by camera center. This is essential on the
   surround rig, whose cameras share a near-common center but face different
   directions; naive nearest-center picks misoriented views and collapses the
   render to near-black.
2. **Encode → Gaussians.** Run the encoder once on the context views.
3. **Render.** Render the target pose with the differentiable 3DGS rasterizer at
   native GT resolution and compare to the target image.

For nuScenes, one reconstruction (context near the base camera, or all cameras of
the base frame) is reused for every extrapolated pose.

**Conventions.** COLMAP `(q,t)` are world→camera; we use OpenCV camera→world
extrinsics `[Rᵀ | −Rᵀt]`. Intrinsics are resolution-normalized
(`fx/W, fy/H, cx/W, cy/H`). Per-scene near/far are estimated in COLMAP units from
the sparse SfM points projected into the source cameras (robust percentiles); the
SfM scale is arbitrary, so no dataset-specific depth constants are used. Context
images are resized to the model input size; targets render at GT resolution so
predicted and reference images share the same shape.

---

## 3. Training (CARLA fine-tuning)

We adapt each method with the **same novel-view photometric objective** it was
pre-trained with — only multi-view RGB + poses, **no depth / 3D supervision**.

- **Objective.** Photometric loss = pixel term (MSE) + perceptual term (LPIPS).
- **Trainable parameters.** Only the multi-view fusion + Gaussian-prediction
  modules are updated; any monocular foundation backbone is kept frozen to
  preserve its generalizable prior. The splatting decoder is parameter-free.
- **Sampling.** Each step draws a random source view as the target and its
  direction-aligned nearest views as context — the same K-nearest-view selection
  as inference — so train and test distributions match.
- **Optimization.** Adam, small learning rate, short linear warm-up. Accuracy
  peaks within a few hundred steps then degrades, so we **early-stop** and select
  the best checkpoint on the benchmark. Larger learning rates catastrophically
  forget the pre-trained prior.
- **Data.** In-domain training sequences only; benchmark test sequences strictly
  excluded. Per-scene near/far from the sparse SfM points.

Zero-shot and fine-tuned settings both give feed-forward methods every advantage
(multiple context views, direction-aware selection, in-domain fine-tuning,
best-step selection), so any remaining gap reflects the paradigm, not weak config.

---

## 4. Reproduction (MVSplat, this repo)

**Data prep** (COLMAP txt/bin + points3D → pixelSplat `.torch`):
```bash
# CSE / SSE test scenes
python tools/colmap_to_pixelsplat.py --scenes runs/cse_scenes/{110,210,310,410,510} \
    --out datasets/carla_cse/test
python tools/colmap_to_pixelsplat.py \
    --scenes /mnt/zihanw/carla/input_output/{110,210,310,410,510}_base \
    --out datasets/carla_sse/test
# Training scenes (exclude the 5 test seqs)
TRAIN=(); for d in /mnt/zihanw/carla/input_output/[0-9][0-9][0-9]_dense; do
  id=$(basename "$d" _dense); case $id in 110|210|310|410|510) continue;; esac
  [ -f "$d/sparse/0/images.bin" ] && TRAIN+=("$d"); done
python tools/colmap_to_pixelsplat.py --scenes "${TRAIN[@]}" --out datasets/carla_cse/train
```

**Fine-tune** (encoder only; best ≈ early step):
```bash
python -m src.scripts.finetune_cse --data_dir datasets/carla_cse/train \
    --checkpoint re10k.ckpt --out checkpoints/carla_ft \
    --num_context_views 6 --lr 5e-6 --steps 3000 --save_every 500 --warmup 100
```

**Render + score** (CSE; SSE uses `datasets/carla_sse/test` and `input_output/<id>_base` as scene):
```bash
python -m src.scripts.render_cse --data_dir datasets/carla_cse/test \
    --checkpoint <re10k.ckpt | checkpoints/carla_ft/finetune_00XXXX.ckpt> \
    --out outputs/cse_run --num_context_views 6 --experiment re10k
python tools/eval_cse.py --multi outputs/cse_run/110/renders:runs/cse_scenes/110 ... \
    --out outputs/cse_run/scores.json
```

**nuScenes** (extrapolated poses + montage):
```bash
python -m src.scripts.render_nuscenes_novel --colmap <clip>/colmap/dense/sparse/0 \
    --base cam0/005.jpg --checkpoint re10k.ckpt --out outputs/nusc/<clip> \
    --num_context_views 6            # or --context_mode same_frame
python tools/montage_nuscenes.py --out outputs/nusc_compare/<clip> \
    --cols "MVSplat=outputs/nusc/<clip>" "GS-Net=<gsnet_render_dir>"
```

### File map
| Path | Role |
|---|---|
| `tools/colmap_to_pixelsplat.py` | COLMAP (txt/bin + points3D) → `.torch` |
| `tools/eval_cse.py` | shared scorer (PSNR / SSIM / LPIPS-vgg) |
| `tools/montage_nuscenes.py` | qualitative side-by-side montages |
| `src/scripts/render_cse.py` | render held-out CSE/SSE views from a checkpoint |
| `src/scripts/finetune_cse.py` | CARLA fine-tune (photometric MSE+LPIPS) |
| `src/scripts/render_nuscenes_novel.py` | nuScenes extrapolated-view renderer |
| `src/dataset/dataset_carla_cse.py` | eval/train datasets + direction-aware `ViewSamplerCSE` |
| `datasets/carla_cse/{test,train}`, `datasets/carla_sse/test` | converted chunks |
| `checkpoints/carla_ft/…` | fine-tuned checkpoints |

### Key hyper-parameters
| Param | Value |
|---|---|
| init weights | official RealEstate10K `re10k.ckpt` |
| context views | 6 |
| image size (train / context) | 256 |
| optimizer / lr / warmup | Adam / 5e-6 / 100 |
| loss | MSE + 0.05·LPIPS(vgg) |
| context selection | direction-aware (`--align_thresh 0.5`) |
| checkpoint | best step on the benchmark (early stop) |

### Lessons
- **Low lr + early stop** — higher lr over-trains and degrades monotonically.
- **Direction-aware context selection is essential** on the surround rig; pure
  nearest-center collapses 2-view renders.
- **No 3D supervision** — only multi-view RGB + poses; only the encoder learns.
- Checkpoints are `{"state_dict": …}`, loaded via the repo's `ModelWrapper`.

---

## 5. Evaluation metrics

Single shared scorer for all methods:
- **PSNR** = `20·log₁₀(1/√MSE)` on `[0,1]` RGB.
- **SSIM**, 11×11 Gaussian window (`C1=0.01²`, `C2=0.03²`).
- **LPIPS**, VGG backbone.

Renders use the exact target filenames and are matched to the GT views in each
scene's split file. CSE/SSE report per-sequence + averaged metrics; nuScenes is
compared qualitatively via montages.
