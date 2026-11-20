"""
Convert external matting datasets into the MicroMat-3K evaluation format used by
eval/eval_loader.py:

    data/<OUT>/data_list.txt      "<img> <matte> <prompt_json> <seg>" per line (paths relative to data/<OUT>)
    data/<OUT>/matte/<tag>/*.png   alpha matte, single channel uint8
    data/<OUT>/prompt/<tag>/*.json {"point": [[x, y, label], ...], "bbox": [x1, y1, x2, y2]}
    data/<OUT>/meta.json           generation settings

<tag> is the value to pass as --dataset-data-type (the loader keeps lines containing it).
The <seg> column is unused by the loader and only written as a placeholder.

Prompt rules mimic MicroMat-3K (measured on its provided prompts):
  - bbox: min-max of the foreground (alpha > BOX_THR), each coordinate shifted by
          uniform noise in [-5%, +5%] of the box width/height, clipped to the image.
  - points: #positive and #negative sampled from MicroMat-3K's empirical distributions;
          positives uniformly from alpha > 0.5, negatives uniformly from alpha < NEG_THR.
All randomness is seeded per image (seed + file name), so prompts are reproducible.

Usage:
    python script/prepare_dataset.py --dataset AM2K
    python script/prepare_dataset.py --dataset PPM100
    python script/prepare_dataset.py --dataset AIM500
    python script/prepare_dataset.py --dataset T460 --bg-dir data/raw/VOC2012/JPEGImages --num-bgs 20
"""
import argparse
import json
import math
import os
import zlib

import numpy as np
from PIL import Image
from tqdm import tqdm

RAW = "data/raw"

# MicroMat-3K empirical counts (all 3,000 prompts)
POS_COUNTS = {1: 829, 2: 619, 3: 452, 4: 347, 5: 231, 6: 133, 7: 98, 8: 100, 9: 53, 10: 49,
              11: 33, 12: 20, 13: 16, 14: 10, 16: 4, 17: 3, 18: 2, 20: 1}
NEG_COUNTS = {0: 883, 1: 672, 2: 426, 3: 285, 4: 198, 5: 168, 6: 97, 7: 102, 8: 57, 9: 34,
              10: 26, 11: 23, 12: 7, 13: 7, 14: 2, 15: 2, 16: 6, 17: 3, 19: 1, 20: 1}

BOX_THR = 0.1    # foreground for bbox (robust to faint noise, e.g. JPEG mattes)
POS_THR = 0.5    # positive clicks
NEG_THR = 0.02   # negative clicks (background); avoids clicking on semi-transparent foreground
BOX_NOISE = 0.05


def load_alpha(path):
    """Return alpha as uint8 HxW. Handles L / RGB / RGBA mattes."""
    im = Image.open(path)
    a = np.array(im)
    if a.ndim == 2:
        return a.astype(np.uint8)
    rgb = a[..., :3]
    if a.shape[2] == 4 and rgb.min() == rgb.max() and a[..., 3].min() != a[..., 3].max():
        return a[..., 3].astype(np.uint8)  # alpha stored only in A channel
    return np.array(im.convert("L"), dtype=np.uint8)


def sample_count(rng, counts):
    k = np.array(list(counts.keys()))
    p = np.array(list(counts.values()), dtype=float)
    return int(rng.choice(k, p=p / p.sum()))


def sample_points(rng, mask, n):
    ys, xs = np.nonzero(mask)
    if n == 0 or len(xs) == 0:
        return []
    idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    return [[int(xs[i]), int(ys[i])] for i in idx]


def make_prompt(alpha_u8, seed, name):
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    a = alpha_u8.astype(np.float32) / 255.0
    h, w = a.shape

    fg_box = a > BOX_THR
    if not fg_box.any():
        raise ValueError(f"{name}: empty foreground")
    ys, xs = np.nonzero(fg_box)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    bw, bh = x2 - x1, y2 - y1
    noise = rng.uniform(-BOX_NOISE, BOX_NOISE, size=4) * np.array([bw, bh, bw, bh])
    box = np.array([x1, y1, x2, y2], dtype=float) + noise
    box = [int(np.clip(round(box[0]), 0, w - 1)), int(np.clip(round(box[1]), 0, h - 1)),
           int(np.clip(round(box[2]), 0, w - 1)), int(np.clip(round(box[3]), 0, h - 1))]

    pos_region = a > POS_THR
    if not pos_region.any():  # e.g. fully semi-transparent object: use its most opaque pixels
        pos_region = a >= a.max() * 0.9
    pos = sample_points(rng, pos_region, sample_count(rng, POS_COUNTS))
    neg = sample_points(rng, a < NEG_THR, sample_count(rng, NEG_COUNTS))
    points = [p + [1] for p in pos] + [p + [0] for p in neg]
    order = rng.permutation(len(points))
    points = [points[i] for i in order]
    return {"point": points, "bbox": box}


def collect(dataset, args):
    """Yield (name, img_path, alpha_path, tag). img paths are relative to the repo root."""
    if dataset == "AM2K":
        d = f"{RAW}/AM-2K/validation"
        for f in sorted(os.listdir(f"{d}/original")):
            s = os.path.splitext(f)[0]
            yield s, f"{d}/original/{f}", f"{d}/mask/{s}.png", "am2k"
    elif dataset == "PPM100":
        d = f"{RAW}/PPM-100/PPM-100"
        for f in sorted(os.listdir(f"{d}/image")):
            yield os.path.splitext(f)[0], f"{d}/image/{f}", f"{d}/matte/{f}", "ppm100"
    elif dataset == "AIM500":
        d = f"{RAW}/AIM-500/AIM-500"
        meta = json.load(open(f"{d}/aim_category_type.json"))
        for f in sorted(os.listdir(f"{d}/original")):
            s = os.path.splitext(f)[0]
            # one tag per category, e.g. "aimcat-transparent"; type (SO/STM/NS) is kept in meta.json
            yield s, f"{d}/original/{f}", f"{d}/mask/{s}.png", f"aimcat-{meta[s]['category'].lower()}"
    elif dataset == "T460":
        d = f"{RAW}/Transparent-460/Transparent-460/Test"
        fgs = [l.strip() for l in open(f"{d}/transparent-460-test-fg-names.txt") if l.strip()]
        bgs = [l.strip() for l in open(f"{d}/transparent-460-test-bg-names.txt") if l.strip()]
        for i, fg in enumerate(fgs):
            s = os.path.splitext(fg)[0]
            for j in range(args.num_bgs):  # official list: 20 bgs per fg; we may use the first k
                yield f"{s}_{j}", (f"{d}/fg/{fg}", bgs[i * 20 + j]), f"{d}/alpha/{s}.png", "t460"
    else:
        raise ValueError(dataset)


def composite(fg_path, bg_path, alpha_u8):
    """Same rule as the official Composition_code.py (bg upscaled to cover fg if needed,
    top-left crop, I = a*F + (1-a)*B with integer truncation), vectorised."""
    fg = Image.open(fg_path).convert("RGB")
    w, h = fg.size
    bg = Image.open(bg_path).convert("RGB")
    bw, bh = bg.size
    ratio = max(w / bw, h / bh)
    if ratio > 1:
        bg = bg.resize((math.ceil(bw * ratio), math.ceil(bh * ratio)), Image.BICUBIC)
    bg = np.array(bg.crop((0, 0, w, h)), dtype=np.float64)
    fg = np.array(fg, dtype=np.float64)
    a = (alpha_u8.astype(np.float64) / 255.0)[..., None]
    out = np.where(a >= 1, fg, np.where(a > 0, a * fg + (1 - a) * bg, bg))
    return Image.fromarray(np.trunc(out).astype(np.uint8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["AM2K", "PPM100", "AIM500", "T460"])
    p.add_argument("--out", default=None, help="output dir name under data/ (default: dataset name)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bg-dir", default=None, help="T460 only: Pascal VOC JPEGImages dir")
    p.add_argument("--num-bgs", type=int, default=20, help="T460 only: backgrounds per fg (1-20)")
    args = p.parse_args()

    out = os.path.join("data", args.out or args.dataset)
    lines, tags = [], {}
    for name, img, alpha_path, tag in tqdm(list(collect(args.dataset, args)), desc=args.dataset):
        alpha = load_alpha(alpha_path)
        os.makedirs(f"{out}/matte/{tag}", exist_ok=True)
        os.makedirs(f"{out}/prompt/{tag}", exist_ok=True)
        matte_rel = f"matte/{tag}/{name}.png"
        Image.fromarray(alpha).save(f"{out}/{matte_rel}")

        if args.dataset == "T460":
            fg_path, bg_name = img
            os.makedirs(f"{out}/img", exist_ok=True)
            img = f"{out}/img/{name}.png"
            if not os.path.exists(img):
                composite(fg_path, os.path.join(args.bg_dir, bg_name), alpha).save(img)
        else:
            w, h = Image.open(img).size
            assert (h, w) == alpha.shape, f"size mismatch: {name}"

        prompt_rel = f"prompt/{tag}/{name}.json"
        with open(f"{out}/{prompt_rel}", "w") as f:
            json.dump(make_prompt(alpha, args.seed, name), f)

        img_rel = os.path.relpath(img, out)
        lines.append(f"{img_rel} {matte_rel} {prompt_rel} seg/{tag}/{name}.png")
        tags[tag] = tags.get(tag, 0) + 1

    with open(f"{out}/data_list.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    meta = {"dataset": args.dataset, "seed": args.seed, "num_images": len(lines), "tags": tags,
            "box_thr": BOX_THR, "box_noise": BOX_NOISE, "pos_thr": POS_THR, "neg_thr": NEG_THR}
    if args.dataset == "T460":
        meta["num_bgs"] = args.num_bgs
    if args.dataset == "AIM500":
        meta["aim_category_type"] = json.load(
            open(f"{RAW}/AIM-500/AIM-500/aim_category_type.json"))
    with open(f"{out}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {len(lines)} samples to {out}  tags={tags}")
    print(f"--dataset-valset {os.path.basename(out)} --dataset-data-type {','.join(tags)}")


if __name__ == "__main__":
    main()
