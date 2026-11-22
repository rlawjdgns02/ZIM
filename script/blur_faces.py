"""
Build a face-blurred copy of a prepared dataset (e.g. P3M500NP from script/prepare_p3m.py).

  1. Detect faces with OpenCV YuNet (cv2.FaceDetectorYN, score >= --score-thr).
  2. Keep only images with >= 1 detected face.
  3. Gaussian-blur only the face regions: an ellipse around each detected box (box enlarged by
     --expand on every side), blended with a feathered mask so no hard rectangular edge is added.
     Blur strength scales with face size: sigma = max(face_w, face_h) * --sigma-ratio.
  4. Save the blurred image as PNG (lossless), so every pixel outside the blurred faces is
     bit-identical to the decoded original.

The ground-truth matte and prompt files are NOT copied: data_list.txt of the blurred set points
at the source dataset's matte/ and prompt/ files, so both versions use exactly the same GT and
prompts, and the per-image CSVs can be joined by file name.

Output (default D:/zim_data/P3M500NP_blur):
    data_list.txt           "<img> <matte> <prompt_json> <seg>"  (matte/prompt -> ../<src>/...)
    img/<tag>/<name>.png    blurred image
    faces.json              per-image detections and blurred-area statistics
    meta.json

Usage:
    python script/blur_faces.py --src D:/zim_data/P3M500NP --out D:/zim_data/P3M500NP_blur \
        --model D:/zim_data/models/face_detection_yunet_2023mar.onnx
"""
import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm


def detect_faces(detector, img_bgr, det_size, score_thr):
    """Run YuNet on a copy resized so the long side is det_size; return boxes in original pixels."""
    h, w = img_bgr.shape[:2]
    s = det_size / max(h, w)
    small = cv2.resize(img_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    detector.setInputSize((small.shape[1], small.shape[0]))
    detector.setScoreThreshold(score_thr)
    _, faces = detector.detect(small)
    out = []
    if faces is not None:
        for f in faces:
            x, y, fw, fh = (f[:4] / s).tolist()
            out.append({"box": [round(x, 1), round(y, 1), round(fw, 1), round(fh, 1)],
                        "score": round(float(f[14]), 4)})
    return out


def blur_faces(img, faces, expand, sigma_ratio):
    """Return (blurred image, blend mask in [0,1])."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    total = np.zeros((h, w), np.float32)
    for f in faces:
        x, y, fw, fh = f["box"]
        cx, cy = x + fw / 2, y + fh / 2
        ax, ay = fw * (0.5 + expand), fh * (0.5 + expand)          # ellipse semi-axes
        sigma = max(fw, fh) * sigma_ratio
        pad = int(np.ceil(3 * sigma))
        x0, y0 = max(int(cx - ax) - pad, 0), max(int(cy - ay) - pad, 0)
        x1, y1 = min(int(np.ceil(cx + ax)) + pad, w), min(int(np.ceil(cy + ay)) + pad, h)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = out[y0:y1, x0:x1]
        blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)

        m = np.zeros(crop.shape[:2], np.float32)
        cv2.ellipse(m, (round(cx - x0), round(cy - y0)), (max(round(ax), 1), max(round(ay), 1)),
                    0, 0, 360, 1.0, -1)
        feather = max(min(ax, ay) * 0.15, 1.0)                     # soft edge, ~15% of the radius
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=feather)[..., None]
        out[y0:y1, x0:x1] = crop * (1 - m) + blurred * m
        total[y0:y1, x0:x1] = np.maximum(total[y0:y1, x0:x1], m[..., 0])
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="D:/zim_data/P3M500NP", help="dataset made by prepare_*.py")
    p.add_argument("--out", default="D:/zim_data/P3M500NP_blur")
    p.add_argument("--model", default="D:/zim_data/models/face_detection_yunet_2023mar.onnx")
    p.add_argument("--det-size", type=int, default=640, help="long side used for detection")
    p.add_argument("--score-thr", type=float, default=0.9)
    p.add_argument("--expand", type=float, default=0.2, help="box enlargement per side (fraction)")
    p.add_argument("--sigma-ratio", type=float, default=0.125, help="sigma = max(face w,h) * ratio")
    p.add_argument("--detect-only", action="store_true", help="only count detections, write nothing")
    args = p.parse_args()

    detector = cv2.FaceDetectorYN.create(args.model, "", (320, 320), args.score_thr)
    src_name = os.path.basename(os.path.normpath(args.src))
    lines = [l.split(" ") for l in open(os.path.join(args.src, "data_list.txt")).read().splitlines() if l]

    records, out_lines, n_det = {}, [], 0
    for img_rel, matte_rel, prompt_rel, seg_rel in tqdm(lines, desc="faces", mininterval=5):
        name = os.path.splitext(os.path.basename(matte_rel))[0]
        tag = matte_rel.split("/")[1]
        img = cv2.imread(os.path.join(args.src, img_rel), cv2.IMREAD_COLOR)
        faces = detect_faces(detector, img, args.det_size, args.score_thr)
        rec = {"num_faces": len(faces), "faces": faces}
        if faces:
            n_det += 1
        if faces and not args.detect_only:
            blurred, mask = blur_faces(img, faces, args.expand, args.sigma_ratio)
            alpha = cv2.imread(os.path.join(args.src, matte_rel), cv2.IMREAD_GRAYSCALE)
            region = mask > 0.5
            trans = (alpha > 5) & (alpha < 250)                        # semi-transparent GT pixels
            fg = alpha > 127
            rec.update({
                "blur_area_ratio": float(region.mean()),               # of the whole image
                "blur_area_of_fg": float((region & fg).sum() / max(fg.sum(), 1)),
                "blur_on_transition": float((region & trans).sum() / max(trans.sum(), 1)),
            })
            os.makedirs(os.path.join(args.out, "img", tag), exist_ok=True)
            out_img = f"img/{tag}/{name}.png"
            cv2.imwrite(os.path.join(args.out, out_img), blurred)
            out_lines.append(f"{out_img} ../{src_name}/{matte_rel} ../{src_name}/{prompt_rel} {seg_rel}")
        records[name] = rec

    print(f"faces detected in {n_det} / {len(lines)} images "
          f"(det_size={args.det_size}, score_thr={args.score_thr})")
    if args.detect_only:
        return

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "data_list.txt"), "w") as f:
        f.write("\n".join(out_lines) + "\n")
    with open(os.path.join(args.out, "faces.json"), "w") as f:
        json.dump(records, f, indent=1)
    meta = {"src": os.path.abspath(args.src), "num_src_images": len(lines), "num_images": len(out_lines),
            "detector": os.path.basename(args.model), "det_size": args.det_size,
            "score_thr": args.score_thr, "expand": args.expand, "sigma_ratio": args.sigma_ratio,
            "blend": "elliptical mask, feather sigma = 0.15 * min semi-axis", "format": "png"}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {len(out_lines)} blurred images to {args.out}")


if __name__ == "__main__":
    main()
