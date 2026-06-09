#!/usr/bin/env python3
"""
Render MVSplat at the nuScenes "novel extrapolated view" qualitative benchmark.

Pick a base camera (e.g. FRONT cam0/005.jpg) from a clip's COLMAP model, build a
MVSplat reconstruction from the K nearest forward-facing real views, then render
the 20 extrapolated novel poses (translations / rotations off the base pose) for a
side-by-side qualitative comparison with GS-Net / baseline. No GT, no metrics.

Novel poses follow NUSCENES_NOVEL_VIEW.md Section 3 (OpenCV camera frame):
  C  = -R^T t ; C' = C + R^T dpos ; R' = Rx(dpitch) Ry(dyaw) R ; t' = -R' C'
or are loaded verbatim from the provided poses JSON (Section 4) via --poses_json.

Example:
  python -m src.scripts.render_nuscenes_novel \
      --colmap /mnt/zihanw/gsnet_nusc/348_clip_09/colmap/dense/sparse/0 \
      --base cam0/005.jpg --checkpoint re10k.ckpt \
      --out outputs/nusc/348_clip_09 --num_context_views 6
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import colmap_to_pixelsplat as cmap  # noqa: E402

from src.scripts.render_cse import build_cfg, load_model  # noqa: E402
from src.dataset.shims.crop_shim import rescale_and_crop   # noqa: E402
from src.misc.image_io import save_image                   # noqa: E402

# tag, (dx,dy,dz) meters in camera frame, yaw deg (about y/down), pitch deg (about x/right)
OFFSETS = [
    ("00_orig", (0, 0, 0), 0, 0),
    ("01_left1", (-1, 0, 0), 0, 0),
    ("02_left2", (-2, 0, 0), 0, 0),
    ("03_right1", (1, 0, 0), 0, 0),
    ("04_right2", (2, 0, 0), 0, 0),
    ("05_up1", (0, -1, 0), 0, 0),
    ("06_up2", (0, -2, 0), 0, 0),
    ("07_fwd2", (0, 0, 2), 0, 0),
    ("08_fwd4", (0, 0, 4), 0, 0),
    ("09_back2", (0, 0, -2), 0, 0),
    ("10_yawL10", (0, 0, 0), -10, 0),
    ("11_yawL25", (0, 0, 0), -25, 0),
    ("12_yawR10", (0, 0, 0), 10, 0),
    ("13_yawR25", (0, 0, 0), 25, 0),
    ("14_pitchUp10", (0, 0, 0), 0, -10),
    ("15_pitchDn10", (0, 0, 0), 0, 10),
    ("16_left2_yawR15", (-2, 0, 0), 15, 0),
    ("17_up1_pitchDn10", (0, -1, 0), 0, 10),
    ("18_fwd3_left1", (-1, 0, 3), 0, 0),
    ("19_right2_yawL15", (2, 0, 0), -15, 0),
]


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def k_norm(cam):
    fx, fy, cx, cy = cmap.intrinsics_from_camera(cam)
    w, h = cam["width"], cam["height"]
    K = torch.eye(3, dtype=torch.float32)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx / w, fy / h, cx / w, cy / h
    return K


def c2w_from(R, t):
    w2c = torch.eye(4, dtype=torch.float32)
    w2c[:3, :3] = torch.tensor(R, dtype=torch.float32)
    w2c[:3, 3] = torch.tensor(t, dtype=torch.float32)
    return torch.inverse(w2c)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap", required=True, help="path to <clip>/.../sparse/0")
    ap.add_argument("--base", default="cam0/005.jpg", help="base image name in images.bin")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images_dir", default=None,
                    help="dir with the actual images (default: <sparse>/../../images)")
    ap.add_argument("--poses_json", default=None,
                    help="use exported world_to_cam poses instead of computing them")
    ap.add_argument("--experiment", default="re10k")
    ap.add_argument("--num_context_views", type=int, default=6)
    ap.add_argument("--image_shape", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--fwd_thresh", type=float, default=0.3,
                    help="keep context views whose forward dir dot base > this")
    ap.add_argument("--near", type=float, default=None)
    ap.add_argument("--far", type=float, default=None)
    ap.add_argument("--default_near", type=float, default=0.5)
    ap.add_argument("--default_far", type=float, default=150.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, cfg = build_cfg(args.experiment, args.checkpoint, args.image_shape,
                       args.num_context_views)
    model = load_model(cfg, args.checkpoint, device)
    encoder, decoder = model.encoder, model.decoder

    sparse = Path(args.colmap)
    cameras = cmap.read_cameras(sparse)
    images = cmap.read_images(sparse)
    names = [im["name"] for im in images]
    if args.base not in names:
        raise SystemExit(f"base '{args.base}' not in model; e.g. {names[:5]}")
    bi = names.index(args.base)
    Rs = [cmap.qvec2rotmat(im["qvec"]) for im in images]
    ts = [im["tvec"] for im in images]
    centers = np.array([-R.T @ t for R, t in zip(Rs, ts)])
    fwds = np.array([R.T @ np.array([0, 0, 1.0]) for R in Rs])

    Rb, tb = Rs[bi], ts[bi]
    base_cam = cameras[images[bi]["camera_id"]]
    W, H = base_cam["width"], base_cam["height"]
    fx, fy, cx, cy = cmap.intrinsics_from_camera(base_cam)
    bc, bfwd = centers[bi], fwds[bi]

    # context = forward-aligned views nearest to the base camera (base included)
    dot = fwds @ bfwd
    dist = np.linalg.norm(centers - bc, axis=1)
    cand = np.where(dot > args.fwd_thresh)[0]
    ctx_idx = cand[np.argsort(dist[cand])][: args.num_context_views]
    print(f"[context] {len(ctx_idx)} views: {[names[j] for j in ctx_idx]}")

    images_dir = Path(args.images_dir) if args.images_dir else sparse.parents[1] / "images"
    to_tensor = tf.ToTensor()
    ctx_imgs = torch.stack([
        to_tensor(Image.open(images_dir / names[j]).convert("RGB")) for j in ctx_idx])
    ctx_K = torch.stack([k_norm(cameras[images[j]["camera_id"]]) for j in ctx_idx])
    ctx_imgs, ctx_K = rescale_and_crop(ctx_imgs, ctx_K, tuple(args.image_shape))
    ctx_ext = torch.stack([c2w_from(Rs[j], ts[j]) for j in ctx_idx])

    # near/far from the SfM points projected into the context cameras
    if args.near is not None and args.far is not None:
        near, far = args.near, args.far
    else:
        xyz = cmap.read_points3D_xyz(sparse)
        near, far = cmap.estimate_near_far(xyz, [(Rs[j], ts[j]) for j in ctx_idx])
        if near is None:
            near, far = args.default_near, args.default_far
    print(f"[bounds] near={near:.3f} far={far:.3f}")

    k = len(ctx_idx)
    context = {
        "image": ctx_imgs.unsqueeze(0).to(device),
        "intrinsics": ctx_K.unsqueeze(0).to(device),
        "extrinsics": ctx_ext.unsqueeze(0).to(device),
        "near": torch.full((1, k), near, dtype=torch.float32).to(device),
        "far": torch.full((1, k), far, dtype=torch.float32).to(device),
    }
    # One feed-forward reconstruction, reused for every novel pose.
    gaussians = encoder(context, 0, deterministic=False)

    # Build the 20 novel poses (compute from base, or load the provided JSON).
    novel = []
    if args.poses_json:
        for v in json.load(open(args.poses_json))["views"]:
            novel.append((v["tag"], np.array(v["world_to_cam"], dtype=np.float64),
                          v["fx"], v["fy"], v["cx"], v["cy"], v["width"], v["height"]))
    else:
        for tag, dpos, yaw, pitch in OFFSETS:
            Cp = bc + Rb.T @ np.array(dpos, dtype=np.float64)
            Rp = Rx(np.deg2rad(pitch)) @ Ry(np.deg2rad(yaw)) @ Rb
            tp = -Rp @ Cp
            w2c = np.eye(4)
            w2c[:3, :3] = Rp
            w2c[:3, 3] = tp
            novel.append((tag, w2c, fx, fy, cx, cy, W, H))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dumped = {"base_image": args.base, "context": [names[j] for j in ctx_idx],
              "near": near, "far": far, "views": []}
    for tag, w2c, fxx, fyy, cxx, cyy, Wt, Ht in novel:
        ext = torch.inverse(torch.tensor(w2c, dtype=torch.float32))[None, None].to(device)
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fxx / Wt, fyy / Ht, cxx / Wt, cyy / Ht
        K = K[None, None].to(device)
        nf = torch.full((1, 1), near, dtype=torch.float32).to(device)
        ff = torch.full((1, 1), far, dtype=torch.float32).to(device)
        out = decoder.forward(gaussians, ext, K, nf, ff, (int(Ht), int(Wt)), depth_mode=None)
        save_image(out.color[0, 0], out_dir / f"{tag}.png")
        dumped["views"].append({"tag": tag, "world_to_cam": w2c.tolist()})
    json.dump(dumped, open(out_dir / "poses.json", "w"), indent=2)
    print(f"[done] {len(novel)} novel views -> {out_dir}")


if __name__ == "__main__":
    main()
