# Feed-forward NVS baselines — Method, Inference, Training

How we evaluate feed-forward / generalizable 3D Gaussian Splatting methods
(e.g. MVSplat, MonoSplat) on our CARLA and nuScenes benchmarks, both zero-shot
and after in-domain fine-tuning, under a protocol identical to the one used for
our own method so that all entries are directly comparable.

---

## 1. Benchmarks

- **CSE (Cross-Sensor).** Reconstruct a scene from one set of cameras and render
  the held-out poses of a *different* set of cameras that were never observed
  during reconstruction (cross-camera / cross-viewpoint synthesis). Quantitative.
- **SSE (Same-Sensor).** Hold out a subset of frames of the same cameras;
  reconstruct from the remaining frames and render the held-out frames (temporal
  interpolation). Quantitative.
- **nuScenes novel views.** Take a base camera and render a set of extrapolated
  poses (translations and rotations) that the reconstruction never saw.
  Qualitative only — no ground truth, compared by visual completeness
  (holes / floaters) under extrapolation.

---

## 2. Inference

A feed-forward method predicts pixel-aligned 3D Gaussians from a few posed input
("context") views in a **single forward pass**, with no per-scene optimization.
For each held-out target pose we:

1. **Select context views (direction-aware).** Among the available source views,
   keep those whose viewing direction aligns with the target (forward-direction
   dot product above a threshold), then take the nearest by camera center.
   Direction-aware selection is essential on the surround camera rig, whose
   cameras share a near-common center but face different directions; a naive
   nearest-center rule picks misoriented views and collapses the render.
2. **Encode → Gaussians.** Run the method's encoder once on the selected context
   views to obtain the scene's 3D Gaussians.
3. **Render.** Render the target pose with the differentiable 3DGS rasterizer at
   the native ground-truth resolution and compare against the target image
   (for quantitative benchmarks).

For the nuScenes qualitative benchmark, one reconstruction (from context near the
base camera, or from all cameras of the base frame) is reused to render every
extrapolated pose.

**Conventions.** COLMAP `(q, t)` are world→camera; we form OpenCV camera→world
extrinsics `[Rᵀ | −Rᵀt]`. Intrinsics are resolution-normalized
(`fx/W, fy/H, cx/W, cy/H`) so they are resolution-independent. Per-scene
near/far are estimated in COLMAP units by projecting the sparse SfM points into
the source cameras and taking robust depth percentiles; the SfM scale is
arbitrary, so no dataset-specific depth constants are assumed. Context images are
resized to the model's input resolution; targets are rendered at GT resolution so
predicted and reference images share the same shape.

---

## 3. Fine-tuning

We adapt each feed-forward method to the target domain using the **same
novel-view photometric objective** it was pre-trained with — no depth or 3D
supervision, only multi-view RGB with known poses.

- **Objective.** A photometric loss between the rendered target and the real
  target image, combining a pixel term (MSE) and a perceptual term (LPIPS).
- **Trainable parameters.** Only the multi-view fusion and Gaussian-prediction
  modules are updated; any monocular foundation backbone is kept frozen to
  preserve its generalizable prior and avoid overfitting the small in-domain set.
  (The splatting decoder is parameter-free.)
- **Sampling.** Each step draws a random source view as the target and its
  direction-aligned nearest views as context — i.e. the same K-nearest-view
  selection used at inference — so training and test distributions match.
- **Optimization.** Adam with a small learning rate and a short linear warm-up.
  Accuracy peaks within a few hundred steps and then degrades; we therefore
  use **early stopping** and select the best checkpoint on the target benchmark.
  Larger learning rates catastrophically forget the pre-trained prior.
- **Data.** In-domain training sequences only; the benchmark test sequences are
  strictly excluded. Per-scene near/far come from the sparse SfM points.
- **Checkpoint selection.** The best step is chosen on the benchmark — the same
  protocol for every fine-tuned method, keeping the comparison apples-to-apples.

Both the zero-shot and fine-tuned settings give the feed-forward methods every
advantage (multiple context views, direction-aware selection, in-domain
fine-tuning, best-step selection), so any remaining gap reflects a property of
the paradigm rather than weak configuration.

---

## 4. Evaluation

All methods (ours and the feed-forward baselines) are scored by a single shared
script so the numbers are comparable:

- **PSNR** = `20·log₁₀(1/√MSE)` on `[0,1]` RGB, per image then averaged.
- **SSIM** with an 11×11 Gaussian window (`C1 = 0.01²`, `C2 = 0.03²`).
- **LPIPS** with a VGG backbone.

Each method writes its renders under the exact target filenames, so they are
matched against the ground-truth views listed in each scene's split file. CSE and
SSE report per-sequence and averaged metrics; nuScenes is compared qualitatively
via side-by-side montages at each extrapolated pose.
