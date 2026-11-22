"""
Qualitative check for the face-blur experiment: run ZIM and SAM (ViT-B) on the original and the
face-blurred version of selected images with the SAME prompt json used in the evaluation, and save
one comparison panel per image:

    row 1: original image (+prompt) | ZIM on original | SAM on original | GT
    row 2: blurred image  (+prompt) | ZIM on blurred  | SAM on blurred  | |ZIM blur - ZIM orig|

Like eval/evaluator.py, SAM output is the binary mask from SamPredictor and ZIM output is its
continuous matte. Panels are written as JPG to --out (default D:/zim_data/outputs/blur_vis).

Usage:
    python script/visualize_blur.py --names p_99a33b12 p_ccb611f0 --prompt bbox
"""
import argparse
import os

import cv2
import json
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from zim_anything import ZimPredictor, zim_model_registry


def lookup(root, name):
    for l in open(os.path.join(root, "data_list.txt")).read().splitlines():
        img, matte, prompt, _ = l.split(" ")
        if os.path.splitext(os.path.basename(matte))[0] == name:
            return [os.path.normpath(os.path.join(root, x)) for x in (img, matte, prompt)]
    raise KeyError(name)


def run(pred, img, prm, kind, prompt):
    pred.set_image(img)
    kw = dict(point_coords=None, point_labels=None, box=None, multimask_output=False)
    if prompt == "bbox":
        box = np.array(prm["bbox"], dtype=np.float32)
        kw["box"] = box[None] if kind == "sam" else box
    else:
        pts = np.array(prm["point"], dtype=np.float32)
        kw["point_coords"], kw["point_labels"] = pts[:, :2], pts[:, 2]
    masks, _, _ = pred.predict(**kw)
    return np.clip(masks[0].astype(np.float32), 0, 1)


def mse(pred, gt):
    return float(((pred - gt) ** 2).mean() * 1e3)


def draw_prompt(img_bgr, prm, prompt, s):
    out = img_bgr.copy()
    if prompt == "bbox":
        x1, y1, x2, y2 = [int(v * s) for v in prm["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        for x, y, lab in prm["point"]:
            cv2.circle(out, (int(x * s), int(y * s)), 5, (0, 255, 0) if lab == 1 else (0, 0, 255), -1)
    return out


def label(tile, text):
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(tile, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--prompt", default="bbox", choices=["bbox", "point"])
    p.add_argument("--orig-root", default="D:/zim_data/P3M500NP")
    p.add_argument("--blur-root", default="D:/zim_data/P3M500NP_blur")
    p.add_argument("--zim", default="results/zim_vit_b_2043")
    p.add_argument("--sam", default="results/sam_vit_b_01ec64.pth")
    p.add_argument("--out", default="D:/zim_data/outputs/blur_vis")
    p.add_argument("--height", type=int, default=360)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    zim = ZimPredictor(zim_model_registry["vit_b"](checkpoint=args.zim).cuda())
    sam = SamPredictor(sam_model_registry["vit_b"](checkpoint=args.sam).cuda())

    for name in args.names:
        o_img, matte, prm_path = lookup(args.orig_root, name)
        b_img, matte_b, prm_b = lookup(args.blur_root, name)
        assert os.path.samefile(matte, matte_b) and os.path.samefile(prm_path, prm_b)
        prm = json.load(open(prm_path))
        gt = cv2.imread(matte, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        res, rows = {}, []
        for tag, path in [("orig", o_img), ("blur", b_img)]:
            bgr = cv2.imread(path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with torch.inference_mode():
                res[tag] = {"zim": run(zim, rgb, prm, "zim", args.prompt),
                            "sam": run(sam, rgb, prm, "sam", args.prompt), "bgr": bgr}
        s = args.height / gt.shape[0]
        sz = (round(gt.shape[1] * s), args.height)
        g = lambda a: cv2.cvtColor(cv2.resize((a * 255).astype(np.uint8), sz), cv2.COLOR_GRAY2BGR)
        for tag in ["orig", "blur"]:
            r = res[tag]
            last = (label(g(gt), "GT") if tag == "orig" else
                    label(g(np.abs(res["blur"]["zim"] - res["orig"]["zim"])), "|ZIM blur - ZIM orig|"))
            rows.append(np.hstack([
                label(draw_prompt(cv2.resize(r["bgr"], sz), prm, args.prompt, s), f"{tag} image"),
                label(g(r["zim"]), f"ZIM {tag}  MSE {mse(r['zim'], gt):.1f}"),
                label(g(r["sam"]), f"SAM {tag}  MSE {mse(r['sam'], gt):.1f}"),
                last]))
        out = os.path.join(args.out, f"{name}_{args.prompt}.jpg")
        cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(out)


if __name__ == "__main__":
    main()
