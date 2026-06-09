#!/usr/bin/env python3
"""
Build side-by-side comparison montages for the nuScenes novel-view benchmark.

For every novel-pose tag (00_orig ... 19_*) shared across the given method dirs,
horizontally concatenate the renders (each labelled) into one image. Purely for
the qualitative figure (no GT / no metrics for the extrapolated views).

Example:
  python tools/montage_nuscenes.py --out outputs/nusc_compare/348 \
    --cols "MVSplat-2view=outputs/nusc_2view/348_clip_09" \
           "MVSplat-6env=outputs/nusc_allframe/348_clip_09" \
           "MVSplat-acid=outputs/nusc_acid_allframe/348_clip_09" \
           "GS-Net=/mnt/zihanw/nusc_novel/348_clip_09_front_d15000"
"""
import argparse
import os
from pathlib import Path

from PIL import Image, ImageDraw


def labelled(img, text, bar=28):
    out = Image.new("RGB", (img.width, img.height + bar), (0, 0, 0))
    out.paste(img, (0, bar))
    ImageDraw.Draw(out).text((5, 7), text, fill=(255, 255, 255))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", nargs="+", required=True, help="label=dir pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=360, help="row height per panel")
    args = ap.parse_args()

    cols = [c.split("=", 1) for c in args.cols]
    sets = [set(f for f in os.listdir(d) if f.lower().endswith(".png")) for _, d in cols]
    tags = sorted(set.intersection(*sets)) if sets else []
    assert tags, f"no common .png tags across {[d for _, d in cols]}"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for t in tags:
        panels = []
        for lab, d in cols:
            im = Image.open(Path(d) / t).convert("RGB")
            r = args.height / im.height
            im = im.resize((max(1, int(im.width * r)), args.height))
            panels.append(labelled(im, lab))
        W, H = sum(p.width for p in panels), max(p.height for p in panels)
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        x = 0
        for p in panels:
            canvas.paste(p, (x, 0))
            x += p.width
        canvas.save(out / t)
    print(f"[done] {len(tags)} montages -> {out}")


if __name__ == "__main__":
    main()
