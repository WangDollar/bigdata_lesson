"""翻转课堂投屏版：从 outputs/homework 里已保存的结果画投屏用的大字号图，再把两份 HTML 打印成 PDF。

不重跑任何分析，所有数字都来自 analysis.py 已写出的 CSV / results.json。
用法：
    /opt/homebrew/Caskroom/miniconda/base/envs/mtest/bin/python build_slides.py
"""
import json
import logging
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2] / "report" / "slides"  # 投屏版 html/pdf 和 figs 所在
OUT = Path(__file__).resolve().parents[2] / "outputs" / "homework"
FIG = HERE / "figs"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

RAW, BONF, BH = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, BAND = "#e1e0d9", "#c3c2b7", "#e1e0d9"
VALID = "#6250d6"

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams.update({
    "font.family": "Hiragino Sans GB", "axes.unicode_minus": False, "font.size": 15,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": BASE,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)


def fig_toy():
    p = np.array([0.001, 0.009, 0.020, 0.040, 0.200])
    i = np.arange(1, 6)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    xs = np.repeat(np.arange(0.5, 6.0), 2)[1:-1]
    ys = np.repeat(i * 0.01, 2)
    ax.plot(xs, ys, color=BH, lw=3, label="BH 阈值：排名 i × 0.05/5")
    ax.plot([0.5, 5.5], [0.01, 0.01], color=BONF, lw=3, label="Bonferroni 阈值：0.05/5 = 0.01")
    top = 0.064
    shown = np.minimum(p, top)
    rejected = p <= i * 0.01
    ax.scatter(i[rejected], shown[rejected], s=110, color=INK, zorder=3, edgecolor="white", lw=2)
    ax.scatter(i[~rejected], shown[~rejected], s=110, facecolor="white", edgecolor=INK, lw=2, zorder=3)
    for k in range(5):
        txt = f"{p[k]:.3f}" if p[k] < top else "0.200 ↑"
        ax.annotate(txt, (i[k], shown[k]), xytext=(10, 4) if k == 0 else (10, -18) if p[k] < top else (10, -4),
                    textcoords="offset points", fontsize=14, color=INK2)
    ax.set_xlim(0.5, 5.6)
    ax.set_ylim(0, 0.068)
    ax.set_xticks(i)
    ax.set_xlabel("p 值从小到大的排名 i")
    ax.set_ylabel("p 值")
    ax.legend(loc="upper left", fontsize=13.5)
    save(fig, "s03-toy.png")


def fig_null(res):
    runs = pd.read_csv(OUT / "simA-all-null-runs.csv")
    r = runs.loc[runs.method == "raw", "R"]
    v = res["simA"]["variation"]["raw"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.hist(r, bins=np.arange(r.min() - 0.5, r.max() + 1.5), color=RAW, edgecolor="white", lw=0.6)
    ax.axvline(v["mean_R"], color=INK, ls="--", lw=2)
    ax.text(0.02, 0.97, f"平均 {v['mean_R']:.1f}（理论 50）\n90% 落在 {v['p05_R']:.0f}–{v['p95_R']:.0f}",
            transform=ax.transAxes, va="top", fontsize=14, color=INK2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.set_xlabel("一次实验里 raw p ≤ 0.05 的个数")
    ax.set_ylabel("实验次数（共 1,000 次）")
    save(fig, "s04-null.png")


def fig_ratio():
    c4 = pd.read_csv(OUT / "C4-replication.csv")
    c4["ratio"] = c4.delta_AML_minus_ALL / c4.discovery_delta
    c4 = c4.sort_values("ratio")
    y = np.arange(len(c4))
    ok = c4.replicated.to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.axvline(1, color=INK3, lw=1.5, ls="--")
    ax.scatter(c4.ratio[ok], y[ok], s=120, color=VALID, zorder=3, label="验证通过（7）")
    ax.scatter(c4.ratio[~ok], y[~ok], s=120, facecolor="white", edgecolor=VALID, lw=2, zorder=3, label="未通过（3）")
    ax.set_yticks(y, c4.probe, fontsize=12.5)
    ax.set_xlim(0, 1.5)
    ax.set_xlabel("验证效应 ÷ 发现效应（< 1 即缩水）")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=13)
    save(fig, "s07-ratio.png")


def fig_perm(res):
    perm = pd.read_csv(OUT / "adv1-permutation-null.csv")
    a = res["adv1_permutation"]
    mu, sd = a["independent_reference"]["mean"], a["independent_reference"]["sd"]
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    ax.axvspan(mu - 2 * sd, mu + 2 * sd, color=BAND, label=f"若检验相互独立：{mu:.0f} ± {sd:.0f}")
    ax.hist(perm.raw, bins=np.arange(80, 1760, 40), color=RAW, edgecolor="white", lw=0.6, label="打乱标签 1,000 次")
    ax.axvline(a["observed_raw"], color=INK, lw=3)
    ax.text(a["observed_raw"] - 20, ax.get_ylim()[1] * 0.55, f"实际标签\n{a['observed_raw']:,}",
            ha="right", fontsize=15, color=INK)
    ax.set_xlim(0, 1800)
    ax.set_xlabel("raw p ≤ 0.05 的探针数")
    ax.set_ylabel("打乱次数")
    ax.legend(loc="upper center", fontsize=13.5)
    save(fig, "s08-perm.png")


def fig_pvalues(res):
    p = pd.read_csv(OUT / "C1-discovery-all-probes.csv").raw
    m, pi0 = len(p), res["adv2_storey"]["pi0"]
    flat = m * 0.05
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    ax.axvspan(0.5, 1.0, color="#f1f0ea", zorder=0)
    ax.text(0.75, 950, "p > 0.5：\n几乎都来自\n真零假设", ha="center", va="center", fontsize=14, color=INK2)
    ax.hist(p, bins=np.arange(0, 1.0001, 0.05), color=RAW, edgecolor="white", lw=0.6)
    ax.axhline(flat, color=INK3, ls=":", lw=2.5, label=f"若全部无差异：每格 {flat:.0f}")
    ax.axhline(pi0 * flat, color=INK, ls="--", lw=2, label=f"实际平坦高度：每格约 {pi0 * flat:.0f}")
    ax.set_xlabel("原始 p 值（每格宽 0.05）")
    ax.set_ylabel("探针数")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, 1.0), fontsize=13.5)
    save(fig, "s09-pvalues.png")


def fig_overlap(res):
    a = res["adv1_permutation"]
    k = np.array([int(x) for x in a["raw_median_by_overlap"]])
    med = np.array(list(a["raw_median_by_overlap"].values()))
    n = np.array([a["n_by_overlap"][str(x)] for x in k])
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.vlines(a["overlap_expected"], 140, 1200, color=INK3, ls="--", lw=1.5)
    ax.text(a["overlap_expected"] + 0.08, 1150, "随机期望 3.18", fontsize=13, color=INK3)
    ax.plot(k, med, color=INK, lw=2.5, marker="o", ms=9)
    for x, y_, c in zip(k, med, n):
        ax.text(x, 45, f"{c} 次", ha="center", fontsize=12, color=INK3)
    ax.set_ylim(0, 1250)
    ax.set_xlabel("假 AML 组（11 人）里混入的真 AML 人数")
    ax.set_ylabel("raw 显著数（中位数）")
    save(fig, "sB-overlap.png")


def to_pdf(html, pdf):
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={profile}", f"--print-to-pdf={pdf}", html.as_uri()],
                       check=True, capture_output=True)
    print("已导出", pdf)


def main():
    FIG.mkdir(exist_ok=True)
    res = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    fig_toy()
    fig_null(res)
    fig_ratio()
    fig_perm(res)
    fig_pvalues(res)
    fig_overlap(res)
    to_pdf(HERE / "投屏版.html", HERE / "投屏版.pdf")
    to_pdf(HERE / "讲稿.html", HERE / "讲稿_A4.pdf")


if __name__ == "__main__":
    main()
