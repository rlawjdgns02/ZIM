"""
Convert P3M-500-NP (P3M-10k validation, non-privacy / unblurred faces) into the evaluation
format used by eval/eval_loader.py.

Prompt generation is NOT reimplemented here: load_alpha / make_prompt and all thresholds are
imported from script/prepare_dataset.py, so the rules are identical to AM2K / PPM100 / AIM500
(bbox: fg min-max +-5% noise; points: MicroMat-3K count distributions; per-image seed).

Output (default D:/zim_data/P3M500NP, i.e. --data-root D:/zim_data --dataset-valset P3M500NP):
    data_list.txt              "<img> <matte> <prompt_json> <seg>", paths relative to the output dir
    matte/p3mnp/<name>.png     alpha matte, uint8
    prompt/p3mnp/<name>.json   {"point": [[x, y, label], ...], "bbox": [x1, y1, x2, y2]}
    meta.json
Images are not copied; data_list.txt points at the original JPGs with a relative path, so the
output dir must be on the same drive as the dataset.

Usage:
    python script/prepare_p3m.py --p3m-root D:/datasets/P3M-10k/P3M-10k --out D:/zim_data/P3M500NP
"""
import argparse
import json
import os
import sys

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_dataset import (BOX_NOISE, BOX_THR, NEG_THR, POS_THR,  # noqa: E402
                             load_alpha, make_prompt)

TAG = "p3mnp"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--p3m-root", default="D:/datasets/P3M-10k/P3M-10k")
    p.add_argument("--out", default="D:/zim_data/P3M500NP")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    src = os.path.join(args.p3m_root, "validation", "P3M-500-NP")
    names = [l.strip() for l in open(os.path.join(args.p3m_root, "P3M-500-NP_list.txt")) if l.strip()]
    names = sorted(os.path.splitext(n)[0] for n in names)

    out = args.out
    os.makedirs(f"{out}/matte/{TAG}", exist_ok=True)
    os.makedirs(f"{out}/prompt/{TAG}", exist_ok=True)

    lines = []
    for name in tqdm(names, desc="P3M-500-NP"):
        img = os.path.join(src, "original_image", f"{name}.jpg")
        alpha = load_alpha(os.path.join(src, "mask", f"{name}.png"))
        w, h = Image.open(img).size
        assert (h, w) == alpha.shape, f"size mismatch: {name}"

        matte_rel = f"matte/{TAG}/{name}.png"
        Image.fromarray(alpha).save(f"{out}/{matte_rel}")
        prompt_rel = f"prompt/{TAG}/{name}.json"
        with open(f"{out}/{prompt_rel}", "w") as f:
            json.dump(make_prompt(alpha, args.seed, name), f)

        img_rel = os.path.relpath(img, out).replace("\\", "/")
        assert " " not in img_rel, "data_list.txt is space-separated"
        lines.append(f"{img_rel} {matte_rel} {prompt_rel} seg/{TAG}/{name}.png")

    with open(f"{out}/data_list.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    meta = {"dataset": "P3M-500-NP", "seed": args.seed, "num_images": len(lines), "tags": {TAG: len(lines)},
            "box_thr": BOX_THR, "box_noise": BOX_NOISE, "pos_thr": POS_THR, "neg_thr": NEG_THR,
            "src": os.path.abspath(src)}
    with open(f"{out}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {len(lines)} samples to {out}")
    print(f"--data-root {os.path.dirname(os.path.abspath(out))} "
          f"--dataset-valset {os.path.basename(out)} --dataset-data-type {TAG}")


if __name__ == "__main__":
    main()
