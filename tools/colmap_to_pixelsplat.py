#!/usr/bin/env python3
"""
Convert a CARLA CSE COLMAP scene (runs/cse_scenes/<id>/) into the pixelSplat
`.torch` chunk format used by BOTH MVSplat and DepthSplat.

pixelSplat / MVSplat chunk schema (per scene dict):
  {
    "key":        str,                      # scene id (e.g. "110")
    "cameras":    float tensor [N, 18],     # [fx, fy, cx, cy, 0, 0,  w2c(3x4 row-major)]
                                            #   intrinsics NORMALIZED by image size
                                            #   extrinsics = world->camera, OpenCV convention
    "images":     list[uint8 tensor],       # each = raw JPEG bytes of one frame
    "url":        str,
    "timestamps": long tensor [N],
  }
Also emits cse_meta.json: per scene the frame names, is_target (name in test.txt),
camera centers (for choosing context views and naming the rendered outputs), and
near/far depth bounds estimated from the source SfM points (points3D) when present.
"""
import argparse, io, json, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def read_cameras_txt(path):
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()
            cameras[int(tok[0])] = dict(
                model=tok[1], width=int(tok[2]), height=int(tok[3]),
                params=list(map(float, tok[4:])))
    return cameras


def read_images_txt(path):
    images = []
    with open(path) as f:
        lines = list(f)
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        tok = line.split()
        images.append(dict(
            name=tok[9],
            qvec=np.array(list(map(float, tok[1:5])), dtype=np.float64),
            tvec=np.array(list(map(float, tok[5:8])), dtype=np.float64),
            camera_id=int(tok[8])))
        i += 2  # skip the 2D-points line
    return images


def read_points3D_xyz(sparse_dir):
    """Read the 3D point cloud XYZ from a COLMAP model.
    Tries points3D.txt (text model), then points3D.ply (e.g. exported SfM cloud),
    then points3D.bin (binary model). Returns an [M, 3] float64 array, or None if
    no point file is present."""
    txt = Path(sparse_dir) / "points3D.txt"
    if txt.exists():
        xyz = []
        with open(txt) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tok = line.split()
                xyz.append([float(tok[1]), float(tok[2]), float(tok[3])])
        return np.array(xyz, dtype=np.float64) if xyz else None
    ply = Path(sparse_dir) / "points3D.ply"
    if ply.exists():
        try:
            from plyfile import PlyData
            v = PlyData.read(str(ply))["vertex"]
            xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]),
                            np.asarray(v["z"])], axis=1).astype(np.float64)
            return xyz if len(xyz) else None
        except Exception as e:  # noqa: BLE001
            print(f"[warn] failed to parse points3D.ply: {e}")
            return None
    binp = Path(sparse_dir) / "points3D.bin"
    if binp.exists():
        try:
            import struct
            xyz = []
            with open(binp, "rb") as f:
                num = struct.unpack("<Q", f.read(8))[0]
                for _ in range(num):
                    f.read(8)                       # point3D_id (uint64)
                    x, y, z = struct.unpack("<ddd", f.read(24))
                    f.read(3)                       # rgb (3 x uint8)
                    f.read(8)                       # error (double)
                    track_len = struct.unpack("<Q", f.read(8))[0]
                    f.read(track_len * 8)           # track (track_len x 2 x int32)
                    xyz.append([x, y, z])
            return np.array(xyz, dtype=np.float64) if xyz else None
        except Exception as e:  # noqa: BLE001
            print(f"[warn] failed to parse points3D.bin: {e}")
            return None
    return None


def estimate_near_far(xyz, source_cams, lo=5.0, hi=95.0):
    """Project the SfM points into every source camera and take robust depth
    percentiles. `source_cams` is a list of (R, t) world->camera pairs.
    Returns (near, far) or (None, None) if no valid depths."""
    if xyz is None or len(xyz) == 0 or not source_cams:
        return None, None
    depths = []
    for R, t in source_cams:
        z = (xyz @ R.T + t)[:, 2]          # camera-space depth
        depths.append(z[z > 1e-6])
    if not depths:
        return None, None
    depths = np.concatenate(depths)
    if depths.size == 0:
        return None, None
    near = float(np.percentile(depths, lo))
    far = float(np.percentile(depths, hi))
    near = max(near, 1e-3)
    if far <= near:
        far = near * 100.0
    return near, far


def intrinsics_from_camera(cam):
    p, model = cam["params"], cam["model"]
    if model == "PINHOLE":
        return p[0], p[1], p[2], p[3]
    if model == "SIMPLE_PINHOLE":
        return p[0], p[0], p[1], p[2]
    return p[0], (p[1] if len(p) > 1 else p[0]), \
           (p[2] if len(p) > 2 else cam["width"]/2.0), \
           (p[3] if len(p) > 3 else cam["height"]/2.0)


def jpeg_bytes_tensor(image_path, jpeg_quality=95):
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8).copy()
    return torch.from_numpy(data), img.size


def convert_scene(scene_dir, jpeg_quality=95):
    scene_dir = Path(scene_dir)
    scene_id = scene_dir.name
    sparse = scene_dir / "sparse" / "0"
    cameras = read_cameras_txt(sparse / "cameras.txt")
    images = read_images_txt(sparse / "images.txt")
    test_txt = sparse / "test.txt"
    target_names = {ln.strip() for ln in open(test_txt) if ln.strip()} if test_txt.exists() else set()
    img_dir = scene_dir / "images"

    cam_rows, image_tensors, names, is_target, cam_centers, timestamps = [], [], [], [], [], []
    source_cams = []  # (R, t) for source (non-target) views, used for depth bounds
    for idx, im in enumerate(images):
        cam = cameras[im["camera_id"]]
        W, H = cam["width"], cam["height"]
        fx, fy, cx, cy = intrinsics_from_camera(cam)
        R = qvec2rotmat(im["qvec"]); t = im["tvec"]
        w2c_3x4 = np.concatenate([R, t.reshape(3, 1)], axis=1)
        cam_rows.append(np.concatenate([
            np.array([fx/W, fy/H, cx/W, cy/H, 0.0, 0.0]), w2c_3x4.reshape(-1)]))
        cam_centers.append((-R.T @ t).tolist())
        tens, (iw, ih) = jpeg_bytes_tensor(img_dir / im["name"], jpeg_quality)
        if (iw, ih) != (W, H):
            print(f"[warn] {im['name']}: image {iw}x{ih} != cameras.txt {W}x{H}")
        image_tensors.append(tens)
        tgt = im["name"] in target_names
        names.append(im["name"]); is_target.append(tgt); timestamps.append(idx)
        if not tgt:
            source_cams.append((R, t))

    # Estimate depth bounds from the source SfM points (same COLMAP units as poses).
    near, far = estimate_near_far(read_points3D_xyz(sparse), source_cams)

    scene_dict = {
        "key": scene_id,
        "cameras": torch.tensor(np.stack(cam_rows, 0), dtype=torch.float32),
        "images": image_tensors, "url": "",
        "timestamps": torch.tensor(timestamps, dtype=torch.long)}
    meta = {
        "scene_id": scene_id, "width": cameras[images[0]["camera_id"]]["width"],
        "height": cameras[images[0]["camera_id"]]["height"],
        "names": names, "is_target": is_target, "cam_centers": cam_centers,
        "near": near, "far": far,
        "num_source": int(sum(1 for x in is_target if not x)),
        "num_target": int(sum(is_target))}
    return scene_dict, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene"); ap.add_argument("--scenes", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk_size", type=int, default=1)
    ap.add_argument("--jpeg_quality", type=int, default=95)
    args = ap.parse_args()
    scene_dirs = ([args.scene] if args.scene else []) + (args.scenes or [])
    assert scene_dirs
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    index, meta_all, chunk, cidx = {}, {}, [], 0

    def flush():
        nonlocal chunk, cidx
        if not chunk:
            return
        fname = f"{cidx:06d}.torch"; torch.save(chunk, out / fname)
        for s in chunk:
            index[s["key"]] = fname
        print(f"[ok] {fname}: {len(chunk)} scene(s)"); chunk = []; cidx += 1

    for sd in scene_dirs:
        scene_dict, meta = convert_scene(sd, args.jpeg_quality)
        nf = (f"near={meta['near']:.3f} far={meta['far']:.3f}"
              if meta["near"] is not None else "near/far=auto-fallback")
        print(f"[scene {scene_dict['key']}] {len(scene_dict['images'])} frames "
              f"({meta['num_source']} src / {meta['num_target']} tgt)  {nf}")
        chunk.append(scene_dict); meta_all[scene_dict["key"]] = meta
        if len(chunk) >= args.chunk_size:
            flush()
    flush()
    json.dump(index, open(out / "index.json", "w"), indent=2)
    json.dump(meta_all, open(out / "cse_meta.json", "w"), indent=2)
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
