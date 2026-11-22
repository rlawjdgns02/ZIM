"""
Paired comparison: original vs face-blurred images (same GT, same prompts), per model / prompt.

Inputs: per_image.csv from eval/main_eval.py for the original and blurred runs, and faces.json
from script/blur_faces.py. Only images present in the blurred run (faces detected) are paired.

Writes:
    <out-md>   markdown tables (included in the report)
    <out-csv>  one row per (model, prompt, image): original and blurred scores side by side

Statistics follow the E1/E7 reports: Wilcoxon signed-rank (two-sided), bootstrap 95% CI of the
mean difference (5,000 resamples, seed 0), image-level rates, failure = MSE > 100.

Usage:
    python script/compare_blur.py \
        --orig D:/zim_data/outputs/P3M500NP_vit_b/per_image.csv \
        --blur D:/zim_data/outputs/P3M500NP_blur_vit_b/per_image.csv \
        --faces D:/zim_data/P3M500NP_blur/faces.json \
        --out-md D:/zim_data/outputs/p3m_blur_compare.md \
        --out-csv reports/P3M500NP_face_blur_per_image.csv
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, wilcoxon

METRICS = ["mse", "sad", "mae", "grad", "conn"]
MODELS = ["zim", "sam"]
PROMPTS = ["bbox", "point"]
PROMPT_KO = {"bbox": "박스", "point": "점"}
FAIL = 100.0


def load(path):
    d = {}
    for r in csv.DictReader(open(path, newline="")):
        d[(r["model"], r["prompt"], r["fname"])] = {**r, **{m: float(r[m]) for m in METRICS}}
    return d


def boot_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def wil(a, b):
    try:
        return wilcoxon(a, b).pvalue
    except ValueError:  # all differences zero
        return float("nan")


def fp(p):
    if not np.isfinite(p):
        return "–"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"{m}×10<sup>{int(e)}</sup>"
    return f"{p:.3f}"


def table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    return s + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orig", default="D:/zim_data/outputs/P3M500NP_vit_b/per_image.csv")
    p.add_argument("--blur", default="D:/zim_data/outputs/P3M500NP_blur_vit_b/per_image.csv")
    p.add_argument("--faces", default="D:/zim_data/P3M500NP_blur/faces.json")
    p.add_argument("--out-md", default="D:/zim_data/outputs/p3m_blur_compare.md")
    p.add_argument("--out-csv", default="reports/P3M500NP_face_blur_per_image.csv")
    args = p.parse_args()

    orig, blur = load(args.orig), load(args.blur)
    faces = json.load(open(args.faces))
    names = sorted({k[2] for k in blur})
    n_src = len({k[2] for k in orig})
    md = []

    # ---- dataset / detection summary
    nf = [faces[os.path.splitext(n)[0]]["num_faces"] for n in names]
    stats = {k: np.array([faces[os.path.splitext(n)[0]][k] for n in names])
             for k in ["blur_area_ratio", "blur_area_of_fg", "blur_on_transition"]}
    md.append("### 얼굴 검출 및 블러 영역\n")
    md.append(table(["항목", "값"], [
        ["원본 이미지 수", n_src],
        ["얼굴 검출 이미지 수 (블러 버전)", f"{len(names)} ({len(names) / n_src:.1%})"],
        ["얼굴 1개 / 2개 이상", f"{nf.count(1)} / {sum(x >= 2 for x in nf)}"],
        ["블러 영역 / 이미지 (중앙값, p90)",
         f"{np.median(stats['blur_area_ratio']):.1%}, {np.percentile(stats['blur_area_ratio'], 90):.1%}"],
        ["블러 영역 / 전경 (중앙값, p90)",
         f"{np.median(stats['blur_area_of_fg']):.1%}, {np.percentile(stats['blur_area_of_fg'], 90):.1%}"],
        ["블러가 덮은 정답 반투명 픽셀 비율 (중앙값, p90)",
         f"{np.median(stats['blur_on_transition']):.1%}, {np.percentile(stats['blur_on_transition'], 90):.1%}"],
    ]))

    # ---- original, full 500 (context): face vs no-face subsets
    md.append("\n### 원본 전체 500장: 얼굴 검출 여부별 MSE 평균 (참고)\n")
    rows = []
    face_set = set(names)
    for mo in MODELS:
        for pr in PROMPTS:
            all_ = [v["mse"] for k, v in orig.items() if k[0] == mo and k[1] == pr]
            f_ = [v["mse"] for k, v in orig.items() if k[0] == mo and k[1] == pr and k[2] in face_set]
            nf_ = [v["mse"] for k, v in orig.items() if k[0] == mo and k[1] == pr and k[2] not in face_set]
            rows.append([mo.upper(), PROMPT_KO[pr], f"{np.mean(all_):.2f} (N={len(all_)})",
                         f"{np.mean(f_):.2f} (N={len(f_)})", f"{np.mean(nf_):.2f} (N={len(nf_)})"])
    md.append(table(["모델", "프롬프트", "전체", "얼굴 검출", "미검출"], rows))

    # ---- paired: original vs blurred, all metrics
    paired = {}
    md.append(f"\n### 원본 vs 블러 쌍 비교 (N={len(names)})\n")
    md.append("Δ = 블러 − 원본 (양수 = 블러 후 오차 증가 = 성능 하락). 모든 지표는 낮을수록 좋음.\n")
    rows = []
    for mo in MODELS:
        for pr in PROMPTS:
            for m in METRICS:
                a = np.array([orig[(mo, pr, n)][m] for n in names])
                b = np.array([blur[(mo, pr, n)][m] for n in names])
                paired[(mo, pr, m)] = (a, b)
                d = b - a
                lo, hi = boot_ci(d)
                rows.append([mo.upper(), PROMPT_KO[pr], m.upper(),
                             f"{a.mean():.2f}", f"{b.mean():.2f}", f"{d.mean():+.2f}",
                             f"{(b.mean() / a.mean() - 1):+.1%}" if a.mean() > 0 else "–",
                             f"{np.median(a):.2f}", f"{np.median(b):.2f}", f"{np.median(d):+.3f}",
                             f"[{lo:+.2f}, {hi:+.2f}]",
                             f"{(d > 0).mean():.0%} / {(d < 0).mean():.0%}", fp(wil(a, b))])
    md.append(table(["모델", "프롬프트", "지표", "원본 평균", "블러 평균", "Δ 평균", "Δ 평균 (%)",
                     "원본 중앙값", "블러 중앙값", "Δ 중앙값", "Δ 평균 95% CI",
                     "나빠짐 / 좋아짐", "Wilcoxon p"], rows))

    # ---- failures
    md.append(f"\n### 실패(MSE > {FAIL:.0f}) 이미지 수\n")
    rows = []
    for mo in MODELS:
        for pr in PROMPTS:
            a, b = paired[(mo, pr, "mse")]
            fa, fb = a > FAIL, b > FAIL
            rows.append([mo.upper(), PROMPT_KO[pr], int(fa.sum()), int(fb.sum()),
                         int((~fa & fb).sum()), int((fa & ~fb).sum())])
    md.append(table(["모델", "프롬프트", "원본 실패", "블러 실패", "새로 실패", "실패 해소"], rows))

    # ---- does ZIM > SAM still hold?
    md.append("\n### ZIM vs SAM 격차: 원본 vs 블러 (MSE)\n")
    rows = []
    for pr in PROMPTS:
        for tag, idx in [("원본", 0), ("블러", 1)]:
            z = paired[("zim", pr, "mse")][idx]
            s = paired[("sam", pr, "mse")][idx]
            rows.append([PROMPT_KO[pr], tag, f"{z.mean():.2f}", f"{s.mean():.2f}",
                         f"{s.mean() / z.mean():.2f}×", f"{np.median(z):.2f}", f"{np.median(s):.2f}",
                         f"{(z < s).mean():.1%}", fp(wil(z, s))])
    md.append(table(["프롬프트", "버전", "ZIM 평균", "SAM 평균", "SAM÷ZIM", "ZIM 중앙값", "SAM 중앙값",
                     "ZIM 승률", "Wilcoxon p"], rows))
    md.append("\n블러로 인한 오차 증가량 비교 (같은 이미지에서 ΔZIM vs ΔSAM, MSE):\n\n")
    rows = []
    for pr in PROMPTS:
        dz = np.subtract(*paired[("zim", pr, "mse")][::-1])
        ds = np.subtract(*paired[("sam", pr, "mse")][::-1])
        rows.append([PROMPT_KO[pr], f"{dz.mean():+.2f}", f"{ds.mean():+.2f}",
                     f"{(np.abs(dz) > np.abs(ds)).mean():.1%}", fp(wil(np.abs(dz), np.abs(ds)))])
    md.append(table(["프롬프트", "ΔZIM 평균", "ΔSAM 평균", "|ΔZIM| > |ΔSAM| 비율", "Wilcoxon p (|Δ|)"], rows))

    # ---- dose-response: blur coverage vs change
    md.append("\n### 블러 범위와 MSE 변화의 관계 (Spearman ρ)\n")
    rows = []
    for mo in MODELS:
        for pr in PROMPTS:
            a, b = paired[(mo, pr, "mse")]
            d = b - a
            r = [mo.upper(), PROMPT_KO[pr]]
            for k in ["blur_area_of_fg", "blur_on_transition"]:
                rho, pv = spearmanr(stats[k], d)
                r.append(f"{rho:+.2f} (p={pv:.3f})")
            hi = stats["blur_on_transition"] >= np.percentile(stats["blur_on_transition"], 75)
            r += [f"{d[~hi].mean():+.2f}", f"{d[hi].mean():+.2f}"]
            rows.append(r)
    md.append(table(["모델", "프롬프트", "ρ(블러/전경, ΔMSE)", "ρ(블러/반투명, ΔMSE)",
                     "ΔMSE 평균: 반투명 겹침 하위 75%", "ΔMSE 평균: 상위 25%"], rows))

    # ---- largest changes
    md.append("\n### MSE 변화가 큰 이미지 (박스 프롬프트, |Δ| 상위 5)\n")
    rows = []
    for mo in MODELS:
        a, b = paired[(mo, "bbox", "mse")]
        d = b - a
        for i in np.argsort(-np.abs(d))[:5]:
            fn = os.path.splitext(names[i])[0]
            rows.append([mo.upper(), names[i], f"{a[i]:.2f}", f"{b[i]:.2f}", f"{d[i]:+.2f}",
                         faces[fn]["num_faces"], f"{faces[fn]['blur_on_transition']:.1%}"])
    md.append(table(["모델", "이미지", "원본 MSE", "블러 MSE", "Δ", "얼굴 수", "블러/반투명"], rows))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "prompt", "fname", "num_faces", "blur_area_ratio", "blur_area_of_fg",
                    "blur_on_transition", "ratio", "scale"]
                   + [f"{m}_orig" for m in METRICS] + [f"{m}_blur" for m in METRICS])
        for mo in MODELS:
            for pr in PROMPTS:
                for n in names:
                    fn = os.path.splitext(n)[0]
                    o, bl = orig[(mo, pr, n)], blur[(mo, pr, n)]
                    w.writerow([mo, pr, n, faces[fn]["num_faces"],
                                f"{faces[fn]['blur_area_ratio']:.4f}", f"{faces[fn]['blur_area_of_fg']:.4f}",
                                f"{faces[fn]['blur_on_transition']:.4f}", f"{float(o['ratio']):.4f}", o["scale"]]
                               + [f"{o[m]:.4f}" for m in METRICS] + [f"{bl[m]:.4f}" for m in METRICS])
    print(f"wrote {args.out_md} and {args.out_csv} (paired N={len(names)})")


if __name__ == "__main__":
    main()
