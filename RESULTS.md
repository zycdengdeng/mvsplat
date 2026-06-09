# CARLA / nuScenes Feed-Forward NVS Benchmarks — Results & Reproduction

Unified evaluation of feed-forward NVS methods on our benchmarks, scored with the
**same** `tools/eval_cse.py` (PSNR / SSIM-11x11 / LPIPS-vgg) for apples-to-apples
comparison with GS-Net / 3DGS.

## Benchmarks

| Benchmark | Type | Task | Eval |
|---|---|---|---|
| **CSE** (Cross-Sensor) | quantitative | reconstruct from odd cams → render held-out **even cams** (5 seqs: 110/210/310/410/510) | PSNR/SSIM/LPIPS, 5-seq avg |
| **SSE** (Same-Sensor) | quantitative | reconstruct from 48 frames → render held-out **frames 4,9** | PSNR/SSIM/LPIPS, avg **excl. 310** |
| **nuScenes novel** | qualitative | render 20 extrapolated poses off a base camera | none (visual holes/floaters) |

## Reusable infrastructure (method-agnostic)

| File | Role |
|---|---|
| `tools/colmap_to_pixelsplat.py` | COLMAP scene (txt/bin, +points3D for near/far) → pixelSplat `.torch` |
| `tools/eval_cse.py` | shared scorer (used by ALL methods) |
| `tools/montage_nuscenes.py` | side-by-side qualitative montages |
| `src/scripts/render_cse.py` | MVSplat CSE/SSE renderer (`--num_context_views`, `--align_thresh`) |
| `src/scripts/finetune_cse.py` | MVSplat CARLA fine-tune (MSE+LPIPS) |
| `src/scripts/render_nuscenes_novel.py` | MVSplat nuScenes extrapolated-view renderer |

> Context selection is **direction-aware** (`--align_thresh`, forward-dot filter then
> nearest-center). `-1` reverts to legacy pure nearest-center.

## MVSplat results

### CSE (5-seq avg)
| Config | PSNR | SSIM | LPIPS | notes |
|---|---|---|---|---|
| zero-shot, 2-view | 10.36 | 0.467 | 0.543 | re10k, legacy sel. |
| zero-shot, 6-view | 16.61 | 0.669 | 0.454 | re10k, legacy sel. |
| CARLA fine-tune, 6-view (best @500) | **17.07** | 0.655 | 0.429 | lr5e-6 |
| CARLA fine-tune lr2e-5 @2k/4k/8k | 16.36 / 16.16 / 16.05 | | | over-trains |

### SSE (4-seq avg, excl. 310)
| Config | PSNR | SSIM | LPIPS | notes |
|---|---|---|---|---|
| zero-shot, 2-view | 4.51 → **7.61** | 0.142 → 0.402 | 0.637 → 0.567 | legacy → direction-aware |
| zero-shot, 6-view | 13.68 | 0.613 | 0.489 | legacy sel. |
| CARLA fine-tune, 6-view | 14.62 | 0.610 | 0.457 | legacy sel. |

> SSE is a forced fit for feed-forward (temporal forward-baseline interpolation);
> report 6-view; treat as supplementary.

### nuScenes (qualitative)
re10k & acid, each with **2-view** and **same-frame (6 surround cams)** context.
Output: `outputs/nusc_*/<clip>/{00_orig..19_*}.png`; compare via `montage_nuscenes.py`.

## Reference (ours)
| Method | CSE PSNR | SSE |
|---|---|---|
| GS-Net + 3DGS | 19.89 | +1.69 vs 3DGS |
| 3DGS baseline | 18.00 | — |

## Other feed-forward methods (same eval口径)
| Method | CSE PSNR | SSIM | LPIPS | views |
|---|---|---|---|---|
| MonoSplat | 10.75 | 0.524 | 0.571 | (TBD) |
| DepthSplat | TBD | | | |

To add a method: convert the scenes with `colmap_to_pixelsplat.py`, render the held-out
views with that method into `<m>/<id>/renders/<exact test.txt name>.png`, then score with
`tools/eval_cse.py --multi`. Same data, same scorer = comparable.

## Status
- [x] MVSplat CSE (zero-shot 2/6-view, fine-tuned)
- [x] MVSplat SSE (zero-shot 2/6-view, fine-tuned)
- [x] MVSplat nuScenes (re10k/acid, 2-view/same-frame)
- [x] MonoSplat CSE
- [ ] nuScenes montages + GS-Net side-by-side
- [ ] (optional) re-run CSE/SSE with consistent direction-aware selection for final numbers
- [ ] DepthSplat (sister task)
