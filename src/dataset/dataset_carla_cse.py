"""
CARLA CSE (Cross-Sensor) dataset + view sampler for MVSplat zero-shot evaluation.

This is a self-contained reader for the pixelSplat `.torch` chunks produced by
`tools/colmap_to_pixelsplat.py`. It deliberately does NOT register itself in the
repo's strongly-typed Hydra/dacite config unions; instead it is driven directly by
`src/scripts/render_cse.py`, which keeps the change small and avoids touching the
training/eval config machinery.

Conventions are kept identical to `DatasetRE10k`:
  * intrinsics are the 3x3 NORMALIZED K (fx,fy,cx,cy already divided by W/H),
  * extrinsics are camera-to-world (= inverse of the stored world->camera w2c),
  * context images are rescaled+center-cropped to the model input size via the
    repo's own `crop_shim` (so intrinsics stay consistent),
  * target images are kept at their native GT resolution so the rendered output
    matches the GT shape that `eval_cse.py` asserts on.

ViewSamplerCSE: enumerate one item per TARGET view; the context is the
`num_context_views` nearest SOURCE views by Euclidean distance between camera
centers (source = the 60 odd cameras; target = the 60 even cameras).
"""
from io import BytesIO
from pathlib import Path

import json
import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from PIL import Image
from torch.utils.data import Dataset

from .shims.crop_shim import rescale_and_crop


def convert_poses(poses: torch.Tensor):
    """[N,18] cameras row -> (extrinsics c2w [N,4,4], intrinsics normalized [N,3,3]).

    Identical math to DatasetRE10k.convert_poses (kept standalone on purpose)."""
    b, _ = poses.shape
    intrinsics = repeat(torch.eye(3, dtype=torch.float32), "h w -> b h w", b=b).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return w2c.inverse(), intrinsics


class ViewSamplerCSE:
    """For each target view, pick the `num_context_views` context source views.

    Selection is direction-aware: keep sources whose viewing direction aligns with
    the target (forward-dot > align_thresh), then take the nearest by camera center.
    This matters a lot on the CARLA rig where all cameras share ~one center, so a
    pure nearest-center rule can pick a camera pointing a different way (→ the
    context never sees what the target sees → near-black render). Set align_thresh
    to -1 to disable the filter (pure nearest-center, legacy behavior).
    """

    def __init__(self, num_context_views: int = 2, align_thresh: float = 0.5):
        self.num_context_views = num_context_views
        self.align_thresh = align_thresh

    def targets(self, is_target):
        return [int(t) for t in np.where(np.asarray(is_target, dtype=bool))[0]]

    def context_for(self, t, centers, forwards, is_target):
        is_target = np.asarray(is_target, dtype=bool)
        src = np.where(~is_target)[0]
        assert len(src) >= self.num_context_views, (
            f"need >= {self.num_context_views} source views, got {len(src)}")
        keep = src[(forwards[src] @ forwards[t]) > self.align_thresh]
        if len(keep) < self.num_context_views:
            keep = src  # too few aligned -> relax to all sources
        d = np.linalg.norm(centers[keep] - centers[t], axis=1)
        return [int(c) for c in keep[np.argsort(d)[: self.num_context_views]]]


class DatasetCarlaCSE(Dataset):
    """One example == one target view (with its nearest source views as context)."""

    def __init__(
        self,
        data_dir,
        image_shape=(256, 256),
        num_context_views: int = 2,
        near: float | None = None,
        far: float | None = None,
        default_near: float = 0.5,
        default_far: float = 150.0,
        align_thresh: float = 0.5,
    ):
        self.data_dir = Path(data_dir)
        self.image_shape = tuple(image_shape)
        self.num_context_views = num_context_views
        self.cli_near, self.cli_far = near, far
        self.default_near, self.default_far = default_near, default_far
        self.to_tensor = tf.ToTensor()
        self.sampler = ViewSamplerCSE(num_context_views, align_thresh)

        self.index = json.load(open(self.data_dir / "index.json"))
        self.meta = json.load(open(self.data_dir / "cse_meta.json"))

        # Lazily cache loaded chunks (one scene per chunk by default).
        self._chunk_cache: dict[str, dict] = {}

        # One item per target view; context is chosen per-item in __getitem__
        # (needs orientation, available only after loading the poses).
        self.items = []
        for scene_key in self.index:
            for t in self.sampler.targets(self.meta[scene_key]["is_target"]):
                self.items.append((scene_key, t))

    def scene_near_far(self, scene_key: str):
        if self.cli_near is not None and self.cli_far is not None:
            return float(self.cli_near), float(self.cli_far)
        m = self.meta[scene_key]
        near = m.get("near", None)
        far = m.get("far", None)
        if near is None or far is None:
            return self.default_near, self.default_far
        return float(near), float(far)

    def _load_scene(self, scene_key: str) -> dict:
        if scene_key not in self._chunk_cache:
            chunk_path = self.data_dir / self.index[scene_key]
            chunk = torch.load(chunk_path)
            for s in chunk:
                self._chunk_cache[s["key"]] = s
        return self._chunk_cache[scene_key]

    def _decode(self, raw: torch.Tensor):
        return self.to_tensor(Image.open(BytesIO(raw.numpy().tobytes())).convert("RGB"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        scene_key, t = self.items[i]
        scene = self._load_scene(scene_key)
        m = self.meta[scene_key]
        extrinsics, intrinsics = convert_poses(scene["cameras"])

        # direction-aware context selection (orientation from the loaded poses)
        centers = extrinsics[:, :3, 3].numpy()
        forwards = (extrinsics[:, :3, :3] @ torch.tensor([0.0, 0.0, 1.0])).numpy()
        ctx = self.sampler.context_for(t, centers, forwards, m["is_target"])

        # ----- context: aligned-nearest source views, resized to model input -----
        ctx_imgs = torch.stack([self._decode(scene["images"][c]) for c in ctx])
        ctx_K = intrinsics[ctx].clone()
        ctx_imgs, ctx_K = rescale_and_crop(ctx_imgs, ctx_K, self.image_shape)
        ctx_ext = extrinsics[ctx]

        # ----- target: kept at native GT resolution (no crop) -----
        tgt_img = self._decode(scene["images"][t])  # [3,H,W] GT
        tgt_ext = extrinsics[t : t + 1]
        tgt_K = intrinsics[t : t + 1].clone()

        near, far = self.scene_near_far(scene_key)
        nc = len(ctx)
        return {
            "scene": scene_key,
            "target_name": m["names"][t],
            "context": {
                "image": ctx_imgs,
                "intrinsics": ctx_K,
                "extrinsics": ctx_ext,
                "near": torch.full((nc,), near, dtype=torch.float32),
                "far": torch.full((nc,), far, dtype=torch.float32),
                "index": torch.tensor(ctx, dtype=torch.int64),
            },
            "target": {
                "image": tgt_img.unsqueeze(0),
                "intrinsics": tgt_K,
                "extrinsics": tgt_ext,
                "near": torch.full((1,), near, dtype=torch.float32),
                "far": torch.full((1,), far, dtype=torch.float32),
                "index": torch.tensor([t], dtype=torch.int64),
            },
        }


class DatasetCarlaCSETrain(Dataset):
    """Training pairs from CARLA source-only scenes (odd cameras with COLMAP poses).

    One item == one anchor view used as the TARGET, with its `num_context_views`
    nearest views as context. This mirrors the K-nearest interpolation used at test
    time (odd->even), here as odd->odd self-supervision. Both context and target are
    resized to the model input size; the loss is photometric (handled by the caller).
    """

    def __init__(
        self,
        data_dir,
        image_shape=(256, 256),
        num_context_views: int = 6,
        near: float | None = None,
        far: float | None = None,
        default_near: float = 0.5,
        default_far: float = 150.0,
        align_thresh: float = 0.5,
    ):
        self.data_dir = Path(data_dir)
        self.image_shape = tuple(image_shape)
        self.num_context_views = num_context_views
        self.align_thresh = align_thresh
        self.cli_near, self.cli_far = near, far
        self.default_near, self.default_far = default_near, default_far
        self.to_tensor = tf.ToTensor()
        self.index = json.load(open(self.data_dir / "index.json"))
        self.meta = json.load(open(self.data_dir / "cse_meta.json"))
        self._chunk_cache: dict[str, dict] = {}
        self.items = []
        for scene_key in self.index:
            n = len(self.meta[scene_key]["names"])
            for a in range(n):
                self.items.append((scene_key, a))

    scene_near_far = DatasetCarlaCSE.scene_near_far
    _load_scene = DatasetCarlaCSE._load_scene
    _decode = DatasetCarlaCSE._decode

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        scene_key, a = self.items[i]
        scene = self._load_scene(scene_key)
        extrinsics, intrinsics = convert_poses(scene["cameras"])

        # context = direction-aligned views nearest to the anchor (anchor excluded)
        centers = extrinsics[:, :3, 3]
        forwards = extrinsics[:, :3, :3] @ torch.tensor([0.0, 0.0, 1.0])
        d = (centers - centers[a]).norm(dim=1)
        aligned = (forwards @ forwards[a]) > self.align_thresh
        aligned[a] = False
        if int(aligned.sum()) < self.num_context_views:
            aligned = torch.ones_like(aligned)
            aligned[a] = False
        d = torch.where(aligned, d, torch.full_like(d, float("inf")))
        k = min(self.num_context_views, int(aligned.sum()))
        ctx = torch.topk(d, k, largest=False).indices

        ctx_imgs = torch.stack([self._decode(scene["images"][int(c)]) for c in ctx])
        ctx_K = intrinsics[ctx].clone()
        ctx_imgs, ctx_K = rescale_and_crop(ctx_imgs, ctx_K, self.image_shape)

        tgt_img = self._decode(scene["images"][a]).unsqueeze(0)
        tgt_K = intrinsics[a : a + 1].clone()
        tgt_img, tgt_K = rescale_and_crop(tgt_img, tgt_K, self.image_shape)

        near, far = self.scene_near_far(scene_key)
        return {
            "scene": scene_key,
            "context": {
                "image": ctx_imgs,
                "intrinsics": ctx_K,
                "extrinsics": extrinsics[ctx],
                "near": torch.full((k,), near, dtype=torch.float32),
                "far": torch.full((k,), far, dtype=torch.float32),
                "index": ctx.to(torch.int64),
            },
            "target": {
                "image": tgt_img,
                "intrinsics": tgt_K,
                "extrinsics": extrinsics[a : a + 1],
                "near": torch.full((1,), near, dtype=torch.float32),
                "far": torch.full((1,), far, dtype=torch.float32),
                "index": torch.tensor([a], dtype=torch.int64),
            },
        }
