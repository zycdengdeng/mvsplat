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

All numbers below use the **direction-aware** context selection (`--align_thresh 0.5`),
the final unified protocol. (Legacy pure-nearest numbers were superseded.)

### CSE (5-seq avg)
| Config | PSNR | SSIM | LPIPS |
|---|---|---|---|
| zero-shot, 2-view | 15.72 | 0.643 | 0.492 |
| zero-shot, 6-view | 17.28 | 0.671 | 0.451 |
| CARLA fine-tune v2, 6-view @500 | 17.77 | 0.669 | 0.412 |
| **CARLA fine-tune v2, 6-view @1000 (best)** | **17.96** | 0.674 | 0.405 |
| CARLA fine-tune v2, 6-view @1500 | 17.94 | 0.668 | 0.409 |

### SSE (4-seq avg, excl. 310)
| Config | PSNR | SSIM | LPIPS |
|---|---|---|---|
| zero-shot, 2-view | 7.61 | 0.402 | 0.567 |
| zero-shot, 6-view | 14.86 | 0.663 | 0.475 |
| CARLA fine-tune v2, 6-view | _TBD_ | | |

> SSE is a forced fit for feed-forward (temporal forward-baseline interpolation);
> 6-view is the representative number; treat SSE as supplementary.

### nuScenes (qualitative)
re10k & acid, each with **2-view** and **same-frame (6 surround cams)** context.
Output: `outputs/nusc_*/<clip>/{00_orig..19_*}.png`; compare via `montage_nuscenes.py`.

## Reference (ours)
| Method | CSE PSNR | SSIM | LPIPS |
|---|---|---|---|
| 3DGS (SfM init) | 18.06 | 0.739 | 0.272 |
| GS-Net + 3DGS (ours) | 19.75 | 0.741 | 0.266 |

## Consolidated CSE comparison (paper口径)
| Method | Type | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| 3DGS (SfM init) | per-scene opt | 18.06 | 0.739 | 0.272 |
| **GS-Net + 3DGS (ours)** | per-scene opt | **19.75** | **0.741** | **0.266** |
| MVSplat (zero-shot, 6-view) | feed-forward | 17.28 | 0.671 | 0.451 |
| MonoSplat (zero-shot, 6-view) | feed-forward | 16.70 | 0.694 | 0.498 |
| MVSplat (CARLA fine-tuned) | feed-forward | 17.96 | 0.674 | 0.405 |
| MonoSplat (CARLA fine-tuned) | feed-forward | 18.41 | 0.703 | 0.455 |

> Feed-forward methods get every advantage (6 context views, direction-aware
> selection, in-domain fine-tuning, best-step selection) yet stay below GS-Net;
> the SSIM/LPIPS gap (0.67-0.70 / 0.41-0.50 vs 0.74 / 0.27) is large even where
> PSNR is close.

## Other feed-forward methods (same eval口径)
| Method | CSE PSNR | SSIM | LPIPS | views |
|---|---|---|---|---|
| MonoSplat (zero-shot) | 16.70 | 0.694 | 0.498 | 6 |
| MonoSplat (CARLA fine-tuned) | 18.41 | 0.703 | 0.455 | 6 |
| DepthSplat | TBD | | | |

To add a method: convert the scenes with `colmap_to_pixelsplat.py`, render the held-out
views with that method into `<m>/<id>/renders/<exact test.txt name>.png`, then score with
`tools/eval_cse.py --multi`. Same data, same scorer = comparable.

## Status
- [x] MVSplat CSE (zero-shot 2/6-view, fine-tuned v2) — direction-aware, final
- [x] MVSplat SSE (zero-shot 2/6-view) — direction-aware, final
- [ ] MVSplat SSE fine-tune v2 @1000 (run with the best CSE ckpt)
- [x] MVSplat nuScenes (re10k/acid, 2-view/same-frame)
- [x] MonoSplat CSE
- [ ] nuScenes montages + GS-Net side-by-side
- [ ] DepthSplat (sister task)
