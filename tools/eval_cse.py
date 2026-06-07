#
# Standalone CSE evaluator: score a method's rendered target views against GT.
# Metrics match the 3DGS repo exactly (PSNR / SSIM 11x11-Gaussian / LPIPS-vgg),
# so numbers are directly comparable to ours regardless of which method produced
# the renders. Use this single script for ALL methods (ours and baselines).
#
# Inputs: a folder of rendered target images + a folder of GT target images,
# matched by filename. For our CARLA CSE scenes the GT target images are the
# `images/<name>` for every `name` in `sparse/0/test.txt` (the 60 even views).
#
# Usage:
#   # score one sequence
#   python eval_cse.py --renders <your_method>/110/renders --gt <gt>/110/gt
#   # or auto-pull GT from a cse_scenes COLMAP scene (uses test.txt):
#   python eval_cse.py --renders <your_method>/110/renders --scene runs/cse_scenes/110
#   # average over the 5 sequences (each a renders dir + scene):
#   python eval_cse.py --multi <method>/110:runs/cse_scenes/110 \
#                              <method>/210:runs/cse_scenes/210 ...
#
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as tf

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----- PSNR (identical to repo utils/image_utils.py) -----
def psnr(a, b):
    mse = ((a - b) ** 2).view(a.shape[0], -1).mean(1, keepdim=True)
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).mean()


# ----- SSIM (identical to repo utils/loss_utils.py: 11x11 Gaussian, C1/C2) -----
def _gauss(w, sigma):
    g = torch.tensor([math.exp(-(x - w // 2) ** 2 / (2 * sigma ** 2)) for x in range(w)])
    return g / g.sum()


def _window(w, ch):
    _1 = _gauss(w, 1.5).unsqueeze(1)
    _2 = (_1 @ _1.t()).float().unsqueeze(0).unsqueeze(0)
    return _2.expand(ch, 1, w, w).contiguous()


def ssim(img1, img2, w=11):
    ch = img1.size(-3)
    win = _window(w, ch).type_as(img1).to(img1.device)
    mu1 = F.conv2d(img1, win, padding=w // 2, groups=ch)
    mu2 = F.conv2d(img2, win, padding=w // 2, groups=ch)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(img1 * img1, win, padding=w // 2, groups=ch) - mu1_sq
    s2 = F.conv2d(img2 * img2, win, padding=w // 2, groups=ch) - mu2_sq
    s12 = F.conv2d(img1 * img2, win, padding=w // 2, groups=ch) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return m.mean()


# ----- LPIPS-vgg. Prefer the repo's lpipsPyTorch (exact match to our numbers);
#       fall back to the pip `lpips` package (net='vgg', [0,1]->[-1,1]). -----
def get_lpips():
    try:
        from lpipsPyTorch import lpips as _l
        return (lambda a, b: _l(a, b, net_type="vgg")), "lpipsPyTorch(vgg)"
    except Exception:
        import lpips as _lp
        net = _lp.LPIPS(net="vgg").to(DEV).eval()
        return (lambda a, b: net(a * 2 - 1, b * 2 - 1).mean()), "lpips-pkg(vgg)"


def _load(path):
    return tf.to_tensor(Image.open(path)).unsqueeze(0)[:, :3].to(DEV)


def score_dir(renders_dir, gt_dir, lpips_fn):
    names = sorted(set(os.listdir(renders_dir)) & set(os.listdir(gt_dir)))
    names = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg"))]
    assert names, f"no matching filenames between {renders_dir} and {gt_dir}"
    ps, ss, ls = [], [], []
    with torch.no_grad():
        for n in names:
            r, g = _load(os.path.join(renders_dir, n)), _load(os.path.join(gt_dir, n))
            assert r.shape == g.shape, f"size mismatch on {n}: {r.shape} vs {g.shape}"
            ps.append(psnr(r, g).item())
            ss.append(ssim(r, g).item())
            ls.append(lpips_fn(r, g).item())
    return (sum(ps) / len(ps), sum(ss) / len(ss), sum(ls) / len(ls), len(names))


def gt_from_scene(scene):
    """GT target dir = the images/ entries listed in sparse/0/test.txt.
    Returns a dict name->path so the caller can match against renders."""
    sp = os.path.join(scene, "sparse", "0")
    tt = os.path.join(sp, "test.txt")
    names = [l.strip() for l in open(tt) if l.strip()]
    img_dir = os.path.join(scene, "images")
    return {n: os.path.join(img_dir, n) for n in names}


def score_against_scene(renders_dir, scene, lpips_fn):
    """Like score_dir but GT comes from the COLMAP scene's test.txt."""
    gt = gt_from_scene(scene)
    rend = {n: os.path.join(renders_dir, n) for n in os.listdir(renders_dir)
            if n.lower().endswith((".png", ".jpg", ".jpeg"))}
    names = sorted(set(gt) & set(rend))
    assert names, (f"no overlap between renders ({len(rend)}) and test.txt ({len(gt)}). "
                   f"Render files must be named EXACTLY as the target names in test.txt.")
    ps, ss, ls = [], [], []
    with torch.no_grad():
        for n in names:
            r, g = _load(rend[n]), _load(gt[n])
            assert r.shape == g.shape, f"size mismatch on {n}: {r.shape} vs {g.shape}"
            ps.append(psnr(r, g).item())
            ss.append(ssim(r, g).item())
            ls.append(lpips_fn(r, g).item())
    return (sum(ps) / len(ps), sum(ss) / len(ss), sum(ls) / len(ls), len(names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", help="dir of rendered target images")
    ap.add_argument("--gt", help="dir of GT target images (matched by filename)")
    ap.add_argument("--scene", help="cse_scenes/<id> COLMAP scene; GT pulled from test.txt")
    ap.add_argument("--multi", nargs="+",
                    help="space-separated 'renders_dir:scene_dir' pairs; reports per-seq + avg")
    ap.add_argument("--out", default=None, help="optional json output path")
    args = ap.parse_args()
    lpips_fn, lp_name = get_lpips()
    print(f"[lpips backend] {lp_name}")

    rows = []
    if args.multi:
        for pair in args.multi:
            rd, sc = pair.split(":")
            p, s, l, n = score_against_scene(rd, sc, lpips_fn)
            rows.append((os.path.basename(sc.rstrip("/")), p, s, l, n))
    else:
        if args.scene:
            p, s, l, n = score_against_scene(args.renders, args.scene, lpips_fn)
            rows.append((os.path.basename(args.scene.rstrip("/")), p, s, l, n))
        else:
            p, s, l, n = score_dir(args.renders, args.gt, lpips_fn)
            rows.append(("seq", p, s, l, n))

    print("\n| Seq | PSNR | SSIM | LPIPS | #views |")
    print("|-----|------|------|-------|--------|")
    for name, p, s, l, n in rows:
        print(f"| {name} | {p:.2f} | {s:.3f} | {l:.3f} | {n} |")
    if len(rows) > 1:
        ap_, as_, al_ = (sum(r[i] for r in rows) / len(rows) for i in (1, 2, 3))
        print(f"| **Avg** | **{ap_:.2f}** | **{as_:.3f}** | **{al_:.3f}** | |")
    if args.out:
        json.dump({"backend": lp_name,
                   "per_seq": [{"seq": r[0], "PSNR": r[1], "SSIM": r[2],
                                "LPIPS": r[3], "views": r[4]} for r in rows]},
                  open(args.out, "w"), indent=2)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
