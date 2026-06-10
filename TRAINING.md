# MVSplat on CARLA — Fine-tuning Details (for migration / write-up)

How the CARLA fine-tuned MVSplat was produced, end to end, with exact paths,
hyper-parameters, and design choices. All code lives in this repo (branch
`claude/hopeful-bell-Igadf`); server-specific data paths are under `/mnt/zihanw/`.

---

## 0. TL;DR

- **Init weights:** MVSplat official RealEstate10K checkpoint `re10k.ckpt`.
- **Data:** 43 CARLA training sequences (101–509, the 5 test seqs 110/210/310/410/510
  excluded), 60 odd-camera source views each, with COLMAP poses.
- **Objective:** pure photometric self-supervision (MSE + 0.05·LPIPS). **Only RGB +
  camera poses are needed — no depth, no 3D ground truth.**
- **Recipe:** per step pick a random source view as the *target*, its 6 direction-aligned
  nearest views as *context*; encoder → 3D Gaussians → decoder renders the target at
  256×256 → loss vs the real target image. Adam, **lr 5e-6**, 3000 steps, **early-stop
  best ≈ 1000 steps**.
- **Best checkpoint:** `checkpoints/carla_ft_v2/finetune_001000.ckpt` (CSE 17.96 PSNR).

Only the **encoder** is updated (the splatting decoder is parameter-free).

---

## 1. What is being trained (paradigm)

MVSplat is **feed-forward**: given a few posed input images it predicts pixel-aligned
3D Gaussians in a single forward pass (no per-scene optimization). Fine-tuning adapts
the encoder to CARLA appearance/geometry using the same novel-view photometric loss
the model was pre-trained with:

```
context (K posed views) --encoder--> Gaussians --decoder(target pose)--> rendered RGB
loss = MSE(rendered, target_GT) + 0.05 * LPIPS_vgg(rendered, target_GT)
```

Training distribution mirrors the test protocol: **K-nearest-view interpolation**
(odd→odd at train time; odd→even (CSE) / frame-holdout (SSE) at test time).

---

## 2. Data preparation

### 2.1 Source (on the server)
`/mnt/zihanw/carla/input_output/<id>_dense/`  for the 43 training sequences:
```
<id>_dense/
  images/1.png … 60.png            # 60 odd-camera source views
  sparse/0/cameras.bin             # COLMAP intrinsics (PINHOLE, binary)
  sparse/0/images.bin              # world->camera poses (binary)
  sparse/0/points3D.ply            # SfM points (used only to estimate near/far)
```
Training seqs = 101–109, 201–209, 301–309, 401–409, 501–509 **minus** 102 & 109
(no COLMAP model) and **minus** the 5 SSE/CSE test seqs 110/210/310/410/510.

### 2.2 Convert to pixelSplat `.torch` chunks
`tools/colmap_to_pixelsplat.py` reads COLMAP (txt **or** bin), tolerates truncated
images, and writes the format MVSplat/DepthSplat consume.

```bash
cd <repo>
TRAIN=()
for d in /mnt/zihanw/carla/input_output/[0-9][0-9][0-9]_dense; do
  id=$(basename "$d" _dense)
  case $id in 110|210|310|410|510) continue;; esac        # drop test seqs
  [ -f "$d/sparse/0/images.bin" ] && TRAIN+=("$d")
done
python tools/colmap_to_pixelsplat.py --scenes "${TRAIN[@]}" --out datasets/carla_cse/train
```
Output: `datasets/carla_cse/train/{000000.torch … , index.json, cse_meta.json}`
(each scene prints `60 src / 0 tgt`; `cse_meta.json` stores per-scene `near/far`,
camera centers, and image names).

---

## 3. Training

### 3.1 Command (the final/best run, "v2")
```bash
python -m src.scripts.finetune_cse \
    --data_dir   datasets/carla_cse/train \
    --checkpoint re10k.ckpt \
    --out        checkpoints/carla_ft_v2 \
    --num_context_views 6 \
    --lr 5e-6 --steps 3000 --save_every 500 --warmup 100
```
~30 min on one A100; saves `finetune_000500.ckpt … finetune_003000.ckpt`.

### 3.2 Hyper-parameters
| Param | Value | Note |
|---|---|---|
| init weights | `re10k.ckpt` | MVSplat official RealEstate10K |
| context views (K) | 6 | feed-forward input count |
| image size | 256×256 | encoder input & render res at train time |
| optimizer | Adam | updates encoder params only |
| learning rate | **5e-6** | higher (2e-5) catastrophically forgets |
| warmup | 100 steps | linear 0→lr |
| steps | 3000 | **best ≈ 1000 (early stop)** |
| loss | MSE + 0.05·LPIPS(vgg) | photometric only |
| batch size | 1 | raise if GPU allows |
| context selection | direction-aware | `align_thresh 0.5` |

### 3.3 What happens per step (code: `src/scripts/finetune_cse.py`)
1. `build_cfg(...)` composes the repo's Hydra config (mode=test so the unimatch
   backbone isn't re-downloaded), `num_context_views=6`.
2. `ModelWrapper.load_from_checkpoint(re10k.ckpt, losses=[], strict=False)` → loads
   encoder+decoder weights; `wrapper.train()`.
3. Dataset yields `{context, target}`; `encoder(context) → gaussians`;
   `decoder.forward(gaussians, target pose, near, far, (256,256)) → color`.
4. `loss = F.mse_loss + 0.05 * lpips_vgg(pred*2-1, gt*2-1)`; Adam step.
5. Checkpoints saved as `{"state_dict": wrapper.state_dict(), ...}` — directly
   loadable by `render_cse.py`.

### 3.4 Training data sampling (code: `DatasetCarlaCSETrain` in
`src/dataset/dataset_carla_cse.py`)
- One item per (scene, anchor view); the **anchor is the target**.
- **Context = 6 nearest views that face the same direction** as the anchor
  (forward-dot > `align_thresh`=0.5, then nearest by camera center; anchor excluded).
  This is the key fix — pure nearest-center picks misoriented cameras on the CARLA
  rig (all cameras share ~one center) and collapses to near-black renders.
- Context & target both resized to 256×256 (`crop_shim`); `near/far` from
  `cse_meta.json` (per-scene SfM-point estimate).

---

## 4. Evaluation (same scripts produce the reported numbers)
```bash
# render the held-out target views with a fine-tuned ckpt
python -m src.scripts.render_cse --data_dir datasets/carla_cse/test \
    --checkpoint checkpoints/carla_ft_v2/finetune_001000.ckpt \
    --out outputs/cse_ftv2_001000 --num_context_views 6 --experiment re10k
# score (shared metrics)
python tools/eval_cse.py --multi \
    outputs/cse_ftv2_001000/110/renders:runs/cse_scenes/110 ... --out scores.json
```
Best fine-tuned CSE = **17.96 / 0.674 / 0.405** (PSNR/SSIM/LPIPS) at step 1000,
vs zero-shot 6-view 17.28 and GS-Net 19.89 / 3DGS baseline 18.00.

---

## 5. File map (all in this repo unless noted)
| Path | Role |
|---|---|
| `src/scripts/finetune_cse.py` | fine-tune driver (loss loop, optimizer, ckpt save) |
| `src/dataset/dataset_carla_cse.py` | `DatasetCarlaCSETrain` (train) + `DatasetCarlaCSE` (eval) + `ViewSamplerCSE` (direction-aware) |
| `src/scripts/render_cse.py` | render held-out views from a checkpoint |
| `tools/colmap_to_pixelsplat.py` | COLMAP (txt/bin + points3D) → `.torch` |
| `tools/eval_cse.py` | shared scorer (PSNR/SSIM/LPIPS-vgg) |
| `re10k.ckpt` | init weights (MVSplat official, downloaded) |
| `datasets/carla_cse/train/` | converted training chunks |
| `checkpoints/carla_ft_v2/finetune_00XXXX.ckpt` | fine-tuned checkpoints |
| `/mnt/zihanw/carla/input_output/<id>_dense/` | raw CARLA training scenes (server) |

---

## 6. Lessons / gotchas (so the migration reproduces)
- **lr 5e-6 + early stop.** lr 2e-5 over-trains and degrades monotonically
  (CSE 16.36 → 16.05 over 2k→8k); 5e-6 peaks ≈ step 1000 then plateaus/declines.
- **Direction-aware context selection is essential** on the CARLA rig (`align_thresh
  0.5`). Pure nearest-center collapses 2-view to near-black (SSE 4.51 → 7.61,
  CSE 2-view 10.36 → 15.72 after the fix).
- **No 3D supervision** — only multi-view RGB + poses.
- **Only the encoder learns**; the CUDA splatting decoder has no trainable params.
- Checkpoints are plain `{"state_dict": ...}` and load via the repo's `ModelWrapper`.
