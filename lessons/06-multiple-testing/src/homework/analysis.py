"""第五课作业：多重检验校正 —— 全部计算与作图。

用法（在 conda 环境中，任意工作目录均可）：
    python analysis.py

数据：lessons/06-multiple-testing/data/leukemia.csv（课程提供，运行前做 SHA-256 校验）
输出：output/   结果表（CSV）与 results.json（报告引用的全部数字）
      figures/  报告用图（PNG）

主线计算函数（adjust、welch_table、simulate 等）与课程 Notebook
notebooks/multiple-test-lab.ipynb 保持一致，保证同一协议、同一结果。
本作业新增两项进阶：
  进阶① 置换检验：打乱 train 的 ALL/AML 标签，保留探针之间的真实相关结构，
         看“标签与表达无关”时会出现多少显著结果、BH/Bonferroni 多常出错。
  进阶② Storey π0 与 q-value：估计 7,129 个探针中真零假设的比例，
         回答“大概有多少差异是真的”；并用置换结果检查这个估计量。
"""
from pathlib import Path
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
LESSON = HERE.parents[1]  # lessons/06-multiple-testing
DATA = LESSON / "data" / "leukemia.csv"
OUT = LESSON / "outputs" / "homework"
FIG = LESSON / "report" / "figures"
EXPECTED_SHA256 = "915bce00295ac470f6bd970a4032730a170a080642a5da53ab0bcb0545156f19"

# ---- 预先固定的协议（运行前写定，不按结果修改）----
SEED = 42        # 与课程一致；不换种子挑结果
REPEATS = 1000   # 实验 A/B 每种设置的重复次数
N_PERM = 1000    # 进阶①置换次数
LAMBDA = 0.5     # 进阶② Storey π0 的预设 λ
ALPHA = 0.05

META = ["sample_id", "original_split", "diagnosis"]
METHODS = ["raw", "bonferroni", "bh"]


# =====================================================================
# 主线计算函数：与课程 Notebook 一致
# =====================================================================

def load_data():
    df = pd.read_csv(DATA)
    x = df.drop(columns=META)
    if df.shape != (72, 7132) or not np.isfinite(x.to_numpy()).all():
        raise ValueError("Unexpected shape or nonfinite expression values")
    if not df.sample_id.is_unique or not x.columns.is_unique:
        raise ValueError("Duplicate identifiers")
    return df


def adjust(p):
    """All input p-values constitute ONE family; do not prefilter on p."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected a nonempty, finite p-value vector in [0, 1]")
    m = len(p)
    order = np.argsort(p, kind="stable")
    ranks = np.arange(1, m + 1)
    sorted_adjusted = np.minimum.accumulate((p[order] * m / ranks)[::-1])[::-1]
    bh = np.empty(m)
    bh[order] = np.minimum(sorted_adjusted, 1)
    return pd.DataFrame({"raw": p, "bonferroni": np.minimum(m * p, 1), "bh": bh})


def welch_table(df, probes=None):
    """Two-sided Welch tests, AML minus ALL; pointwise 95% CI."""
    if probes is None:
        probes = list(df.drop(columns=META).columns)
    if not len(probes):
        raise ValueError("No probes to test")
    a = df.loc[df.diagnosis.eq("AML"), probes].to_numpy(float)
    b = df.loc[df.diagnosis.eq("ALL"), probes].to_numpy(float)
    if min(len(a), len(b)) < 2:
        raise ValueError("Each group needs at least two observations")
    test = stats.ttest_ind(a, b, axis=0, equal_var=False)
    ci = test.confidence_interval(confidence_level=0.95)
    delta = a.mean(axis=0) - b.mean(axis=0)
    pooled = np.sqrt(((len(a)-1)*a.var(axis=0, ddof=1) + (len(b)-1)*b.var(axis=0, ddof=1)) / (len(a)+len(b)-2))
    if (pooled == 0).any():
        raise ValueError("Zero pooled SD: investigate instead of silently dropping a test")
    result = pd.DataFrame({"probe": probes, "n_AML": len(a), "n_ALL": len(b),
                           "mean_AML": a.mean(axis=0), "mean_ALL": b.mean(axis=0),
                           "delta_AML_minus_ALL": delta, "ci95_low": ci.low,
                           "ci95_high": ci.high, "cohen_d": delta/pooled})
    return pd.concat([result, adjust(test.pvalue)], axis=1)


def counts(table, alpha=ALPHA):
    return {method: int((table[method] <= alpha).sum()) for method in METHODS}


def freeze_candidates(table, path, limit=10):
    chosen = table.loc[table.bh <= .05].sort_values(["raw", "probe"]).head(limit)
    locked = chosen[["probe", "delta_AML_minus_ALL"]].rename(columns={"delta_AML_minus_ALL": "discovery_delta"})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement after viewing replication.
    with path.open("x", encoding="utf-8", newline="") as f:
        locked.to_csv(f, index=False)
    return locked


def replicate(candidate_path):
    locked = pd.read_csv(candidate_path)
    if locked.empty:
        return pd.DataFrame(columns=["probe", "replicated"])
    if not locked.probe.is_unique or len(locked) > 10:
        raise ValueError("Expected at most ten unique, previously fixed candidates")
    df = load_data()
    out = welch_table(df.loc[df.original_split.eq("test")], list(locked.probe))
    out = out.merge(locked, on="probe", validate="one_to_one")
    out["same_direction"] = np.sign(out.delta_AML_minus_ALL) == np.sign(out.discovery_delta)
    out["replicated"] = (out.bonferroni <= .05) & out.same_direction
    return out


def simulate(repeats=1000, seed=42, delta=0.0, signals=0, m=1000, n_all=27, n_aml=11):
    """Independent Gaussian teaching data, NOT measured Golub expression."""
    if repeats < 2 or not 0 <= signals <= m or (signals > 0 and delta == 0):
        raise ValueError("Invalid simulation settings")
    rng = np.random.default_rng(seed)
    truth = np.arange(m) < signals
    rows = []
    for run in range(repeats):
        a = rng.normal(size=(n_aml, m))
        b = rng.normal(size=(n_all, m))
        a[:, truth] += delta
        p = stats.ttest_ind(a, b, axis=0, equal_var=True).pvalue
        adjusted = adjust(p)
        for method in METHODS:
            rejected = adjusted[method].to_numpy() <= .05
            r = int(rejected.sum())
            v = int((rejected & ~truth).sum())
            s = int((rejected & truth).sum())
            rows.append({"run": run, "method": method, "R": r, "V": v,
                         "S": s, "FDP": v/max(r, 1), "any_false": int(v > 0),
                         "power": s/signals if signals else np.nan})
    return pd.DataFrame(rows)


def summarize_simulation(runs):
    rows = []
    for method, d in runs.groupby("method", sort=False):
        rows.append({"method": method, "repeats": len(d), "mean_R": d.R.mean(),
                     "mean_V": d.V.mean(), "FWER_estimate": d.any_false.mean(),
                     "FWER_MCSE": d.any_false.std(ddof=1)/np.sqrt(len(d)),
                     "FDR_estimate": d.FDP.mean(),
                     "FDR_MCSE": d.FDP.std(ddof=1)/np.sqrt(len(d)),
                     "mean_power": d.power.mean()})
    return pd.DataFrame(rows)


def variation(runs):
    return runs.groupby("method", sort=False).R.agg(
        mean_R="mean", sd_R="std",
        p05_R=lambda x: x.quantile(.05), p95_R=lambda x: x.quantile(.95))


# =====================================================================
# 进阶②：Storey π0 与 q-value
# =====================================================================

def storey_pi0(p, lam=LAMBDA, cap=True):
    """π0(λ) = #{p > λ} / (m(1-λ))：p 值分布在 λ 右侧的“平坦高度”。"""
    p = np.asarray(p, dtype=float)
    est = (p > lam).sum() / (len(p) * (1 - lam))
    return min(1.0, est) if cap else est


def storey_q(p, pi0):
    """q_(i) = min(1, min_{j>=i} π0·m·p_(j)/j)，即 π0 × 未截断的 BH adjusted p。"""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    stepped = np.minimum.accumulate((pi0 * p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(stepped, 1)
    return q


# =====================================================================
# 进阶①：置换检验（打乱 train 的 ALL/AML 标签）
# =====================================================================

def permutation_null(train, n_perm=N_PERM, seed=SEED, lam_grid=None):
    x = train.drop(columns=META).to_numpy(float)
    true_aml = train.diagnosis.eq("AML").to_numpy()
    n, n_aml = len(true_aml), int(true_aml.sum())
    rng = np.random.default_rng(seed)
    rows, pi0_rows = [], []
    for b in range(n_perm):
        fake_aml = np.zeros(n, dtype=bool)
        fake_aml[rng.permutation(n)[:n_aml]] = True
        p = stats.ttest_ind(x[fake_aml], x[~fake_aml], axis=0, equal_var=False).pvalue
        adj = adjust(p)
        rows.append({"perm": b,
                     "true_AML_in_fake_AML": int((fake_aml & true_aml).sum()),
                     **{method: int((adj[method] <= ALPHA).sum()) for method in METHODS},
                     "pi0": storey_pi0(p)})
        if lam_grid is not None:
            pi0_rows.append([storey_pi0(p, lam, cap=False) for lam in lam_grid])
    return pd.DataFrame(rows), np.array(pi0_rows)


# =====================================================================
# 作图（统一配色：方法颜色在所有图中固定）
# =====================================================================

INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"raw": "#2a78d6", "bonferroni": "#eb6834", "bh": "#1baf7a"}
LABEL = {"raw": "原始 p", "bonferroni": "Bonferroni", "bh": "BH"}
DISC_COLOR, VALID_COLOR = INK2, "#6250d6"
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#256abf", "#184f95"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Hiragino Sans GB", "STHeiti", "PingFang SC", "Arial Unicode MS",
                        "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "text.color": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.titlesize": 11, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8.5, "savefig.dpi": 200,
})


def fig_funnel(stages, path):
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    ax.set_axis_off()
    k = len(stages)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, k)
    for i, (count, title, note) in enumerate(stages):
        y = k - i - 1
        color = RAMP[i]
        ax.add_patch(Rectangle((0.0, y + 0.12), 1.0, 0.76, facecolor="#f6f9fe", edgecolor=color, linewidth=1.4))
        ax.add_patch(Rectangle((0.0, y + 0.12), 0.2, 0.76, facecolor=color, edgecolor=color, linewidth=1.4))
        ax.text(0.1, y + 0.5, f"{count:,}", ha="center", va="center", fontsize=20, fontweight="bold",
                color=INK if i < 3 else "white")
        ax.text(0.225, y + 0.66, title, ha="left", va="center", fontsize=11.5, fontweight="bold", color=INK)
        ax.text(0.225, y + 0.32, note, ha="left", va="center", fontsize=9, color=INK2)
        if i < k - 1:
            ax.annotate("", xy=(0.1, y + 0.02 - 0.12), xytext=(0.1, y + 0.12),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_sim_a(null_runs, summary, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained",
                               gridspec_kw={"width_ratios": [1.6, 1]})
    r = null_runs.loc[null_runs.method.eq("raw"), "R"]
    a.hist(r, bins=np.arange(r.min() - .5, r.max() + 1.5), color=COLOR["raw"], edgecolor="white", linewidth=0.5)
    a.axvline(r.mean(), color=INK, lw=1.2, ls="--")
    lo, hi = r.quantile([.05, .95])
    a.set_ylim(0, a.get_ylim()[1] * 1.25)
    a.text(0.98, 0.97, f"虚线：平均 {r.mean():.1f}（理论 1000×0.05 = 50）\n90% 的实验落在 {lo:.0f}–{hi:.0f}",
           transform=a.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
    a.grid(axis="x", visible=False)
    a.set(title="实验 A：全部无差异时，每次实验的原始显著数", xlabel="一次实验中 raw p ≤ 0.05 的个数", ylabel="重复次数（共 1,000 次）")
    s = summary.set_index("method")
    xs = np.arange(len(METHODS))
    vals = [s.loc[m, "FWER_estimate"] for m in METHODS]
    b.bar(xs, vals, width=0.58, color=[COLOR[m] for m in METHODS])
    for x, v in zip(xs, vals):
        b.text(x, v + 0.025, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
    b.axhline(ALPHA, color=MUTED, ls="--", lw=1)
    b.text(len(METHODS) - 0.55, ALPHA + 0.025, "0.05", fontsize=8, color=MUTED, ha="right")
    b.set_xticks(xs, [LABEL[m] for m in METHODS])
    b.grid(axis="x", visible=False)
    b.set_ylim(0, 1.12)
    b.set(title="至少出现 1 个假发现的比例（FWER）", ylabel="1,000 次中的比例")
    fig.savefig(path)
    plt.close(fig)


def fig_sim_b(signal_summary, path):
    metrics = [("mean_power", "平均检出力（100 个真信号被检出的比例）"),
               ("FDR_estimate", "FDR（各次错误发现比例的平均）"),
               ("FWER_estimate", "FWER（至少错 1 个的比例）")]
    scen = [("signal_1.0", "δ = 1.0"), ("signal_0.5", "δ = 0.5")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), layout="constrained", sharey=True)
    w = 0.25
    for ax, (col, title) in zip(axes, metrics):
        for j, m in enumerate(METHODS):
            xs = np.arange(len(scen)) + (j - 1) * (w + 0.02)
            vals = [signal_summary.loc[signal_summary.scenario.eq(sc) & signal_summary.method.eq(m), col].iloc[0]
                    for sc, _ in scen]
            ax.bar(xs, vals, width=w, color=COLOR[m], label=LABEL[m])
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.015, f"{v:.3f}" if v < 0.1 else f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
        if col != "mean_power":
            ax.axhline(ALPHA, color=MUTED, ls="--", lw=1)
        ax.set_xticks(range(len(scen)), [lab for _, lab in scen])
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.1)
    axes[0].set_ylabel("1,000 次重复的平均 / 比例")
    handles = [Rectangle((0, 0), 1, 1, color=COLOR[m]) for m in METHODS]
    fig.legend(handles, [LABEL[m] for m in METHODS], loc="outside upper center", ncol=3)
    fig.savefig(path)
    plt.close(fig)


def fig_pvalues(p, pi0, lam_grid, pi0_grid, perm_pi0_grid, path):
    m = len(p)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9), layout="constrained",
                               gridspec_kw={"width_ratios": [1.4, 1]})
    a.hist(p, bins=np.linspace(0, 1, 21), color=COLOR["raw"], edgecolor="white", linewidth=0.6)
    all_null, flat = m / 20, pi0 * m / 20
    a.axhline(all_null, color=MUTED, ls=":", lw=1.3, label=f"若全部无差异：每格约 {all_null:.0f}")
    a.axhline(flat, color=INK, ls="--", lw=1.2, label=f"按 π0 = {pi0:.2f} 的平坦高度：每格约 {flat:.0f}")
    a.legend(loc="upper right")
    a.grid(axis="x", visible=False)
    a.set(title=f"train 全部 {m:,} 个探针的原始 p 值分布", xlabel="原始 p 值（每格宽 0.05）", ylabel="探针数")
    lo, med, hi = np.percentile(perm_pi0_grid, [5, 50, 95], axis=0)
    b.fill_between(lam_grid, lo, hi, color=GRID, label="打乱标签 1,000 次（5%–95%）")
    b.plot(lam_grid, med, color=MUTED, lw=1.2)
    b.plot(lam_grid, pi0_grid, color=COLOR["raw"], lw=2, marker="o", ms=4, label="实际标签")
    b.plot([LAMBDA], [pi0], "o", color=INK, ms=8, zorder=3)
    b.annotate(f"预设 λ = {LAMBDA}：π0 = {pi0:.3f}", (LAMBDA, pi0), xytext=(8, -16),
               textcoords="offset points", fontsize=8.5, color=INK)
    b.set(title="Storey π0(λ)：对 λ 的敏感性", xlabel="λ", ylabel="π0(λ) 估计（未截断）", ylim=(0, 1.25))
    b.legend(loc="lower left")
    fig.savefig(path)
    plt.close(fig)


def fig_validation(val, path):
    n, ncol = len(val), 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.5, 2.25 * nrow + 0.7), layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    for i, (ax, (_, r)) in enumerate(zip(axes, val.iterrows())):
        ax.axvline(0, color=BASE, lw=1, ls="--")
        ax.plot([r.disc_low, r.disc_high], [1, 1], color=DISC_COLOR, lw=2, solid_capstyle="round")
        ax.plot(r.discovery_delta, 1, "o", color=DISC_COLOR, ms=7)
        ax.plot([r.ci95_low, r.ci95_high], [0, 0], color=VALID_COLOR, lw=2, solid_capstyle="round")
        ax.plot(r.delta_AML_minus_ALL, 0, "o", color=VALID_COLOR, ms=7)
        lo = min(r.disc_low, r.ci95_low, 0)
        hi = max(r.disc_high, r.ci95_high, 0)
        pad = (hi - lo) * 0.1
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.6, 1.6)
        ax.set_yticks([1, 0], ["发现", "验证"] if i % ncol == 0 else ["", ""])
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7.5)
        status = "通过" if r.replicated else "未通过"
        ax.set_title(f"{r.probe}\n{status}｜验证/发现 = {r.ratio:.2f}", fontsize=8.5)
    for ax in axes[n:]:
        ax.set_visible(False)
    handles = [Line2D([], [], color=DISC_COLOR, lw=2, marker="o", ms=6),
               Line2D([], [], color=VALID_COLOR, lw=2, marker="o", ms=6)]
    fig.legend(handles, ["发现部分 train（ALL 27 / AML 11）：均值差与逐项 95% CI",
                         "留出部分 test（ALL 20 / AML 14）：均值差与逐项 95% CI"],
               loc="outside upper center", ncol=2)
    fig.supxlabel("AML－ALL 平均表达差（原始强度尺度；虚线 = 0）", fontsize=9, color=INK2)
    fig.savefig(path)
    plt.close(fig)


def fig_permutation(perm, observed_raw, m, n_total, n_aml, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.0), layout="constrained",
                               gridspec_kw={"width_ratios": [1.3, 1]})
    r = perm.raw
    exp, sd = m * ALPHA, np.sqrt(m * ALPHA * (1 - ALPHA))
    a.axvspan(exp - 2 * sd, exp + 2 * sd, color=GRID, zorder=0,
              label=f"若 {m:,} 个检验相互独立：{exp:.0f} ± {2 * sd:.0f}")
    a.hist(r, bins=40, color=COLOR["raw"], edgecolor="white", linewidth=0.5, label=f"打乱标签 {len(perm):,} 次")
    a.axvline(observed_raw, color=INK, lw=1.6)
    a.text(observed_raw, a.get_ylim()[1] * 0.6, f"实际标签 {observed_raw:,} ", ha="right", fontsize=8.5, color=INK)
    a.set_xlim(0, max(r.max(), observed_raw) * 1.06)
    a.grid(axis="x", visible=False)
    a.set(title="进阶①：打乱 ALL/AML 标签后，每次的原始显著数", xlabel="raw p ≤ 0.05 的探针数", ylabel="置换次数")
    a.legend(loc="upper left", bbox_to_anchor=(0.42, 1.0))
    jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(perm))
    b.scatter(perm.true_AML_in_fake_AML + jitter, r, s=9, color=COLOR["raw"], alpha=0.35, linewidths=0)
    med = perm.groupby("true_AML_in_fake_AML").raw.median()
    b.plot(med.index, med.values, color=INK, lw=1.5, marker="o", ms=5, label="中位数")
    b.axvline(n_aml * n_aml / n_total, color=MUTED, ls="--", lw=1)
    b.set(title="与“假 AML 组里混入的真 AML 人数”的关系",
          xlabel=f"假 AML 组（{n_aml} 人）中真正的 AML 人数（虚线 = 随机期望）", ylabel="raw p ≤ 0.05 的探针数")
    b.legend(loc="upper left")
    fig.savefig(path)
    plt.close(fig)


# =====================================================================
# 主流程
# =====================================================================

def jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (pd.Series, pd.Index)):
        return o.tolist()
    return str(o)


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}
    if tuple(int(x) for x in scipy.__version__.split(".")[:2]) < (1, 11):
        raise SystemExit(f"需要 scipy>=1.11（当前 {scipy.__version__}），见 requirements.txt")
    print("环境:", versions)

    res = {"environment": versions,
           "protocol": {"seed": SEED, "repeats": REPEATS, "n_perm": N_PERM, "lambda": LAMBDA, "alpha": ALPHA}}

    # ---------- C0 数据 ----------
    sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA256:
        raise SystemExit("数据文件与课程快照不同")
    data = load_data()
    split = pd.crosstab(data.original_split, data.diagnosis)
    res["data"] = {"sha256": sha, "shape": list(data.shape), "n_probes": int(data.shape[1] - len(META)),
                   "split": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in split.to_dict("index").items()}}
    print("数据:", res["data"]["shape"], res["data"]["split"])

    # ---------- 实验 A：全零 ----------
    null_runs = simulate(repeats=REPEATS, seed=SEED, delta=0, signals=0)
    null_runs.to_csv(OUT / "simA-all-null-runs.csv", index=False)
    null_var = variation(null_runs)
    null_summary = summarize_simulation(null_runs)
    null_summary.to_csv(OUT / "simA-summary.csv", index=False)
    res["simA"] = {"first_run_R": null_runs.loc[null_runs.run.eq(0)].set_index("method").R.to_dict(),
                   "variation": null_var.to_dict("index"),
                   "summary": null_summary.set_index("method").to_dict("index")}
    fig_sim_a(null_runs, null_summary, FIG / "fig2-simA-all-null.png")
    print("实验 A 完成")

    # ---------- 实验 B：100 个真信号 ----------
    parts = []
    for label, delta in [("signal_1.0", 1.0), ("signal_0.5", .5)]:
        runs = simulate(repeats=REPEATS, seed=SEED, delta=delta, signals=100)
        runs.to_csv(OUT / f"simB-{label}-runs.csv", index=False)
        s = summarize_simulation(runs)
        s.insert(0, "scenario", label)
        parts.append(s)
    signal_summary = pd.concat(parts, ignore_index=True)
    signal_summary.to_csv(OUT / "simB-summary.csv", index=False)
    res["simB"] = {sc: d.drop(columns="scenario").set_index("method").to_dict("index")
                   for sc, d in signal_summary.groupby("scenario")}
    fig_sim_b(signal_summary, FIG / "fig3-simB-signal.png")
    print("实验 B 完成")

    # ---------- C1 完整发现家族 ----------
    train = data.loc[data.original_split.eq("train")]
    table = welch_table(train)
    np.testing.assert_allclose(table.bh, stats.false_discovery_control(table.raw, method="bh"))
    m = len(table)
    c = counts(table)
    k = c["bh"]
    p_sorted = np.sort(table.raw.to_numpy())
    res["C1"] = {"m": m, "counts": c, "bonferroni_raw_threshold": ALPHA / m,
                 "expected_raw_if_all_null": m * ALPHA,
                 "bh_k": k, "bh_p_at_k": float(p_sorted[k - 1]) if k else None,
                 "bh_threshold_at_k": k * ALPHA / m if k else None,
                 "n_direction_among_bh": {"AML_higher": int(((table.bh <= ALPHA) & (table.delta_AML_minus_ALL > 0)).sum()),
                                          "AML_lower": int(((table.bh <= ALPHA) & (table.delta_AML_minus_ALL < 0)).sum())}}
    print("C1 发现数:", c)

    # ---------- C2 错误流程：先选前 20 再校正 ----------
    top20 = table.sort_values(["raw", "probe"]).head(20).copy()
    wrong = adjust(top20.raw)
    top20["WRONG_bh_after_selecting_top20"] = wrong.bh.to_numpy()
    top20["WRONG_bonf_after_selecting_top20"] = wrong.bonferroni.to_numpy()
    top20.to_csv(OUT / "C2-selection-bias-top20.csv", index=False)
    first = top20.iloc[0]
    res["C2"] = {"example": {"probe": first.probe, "raw": first.raw, "full_bonferroni": first.bonferroni,
                             "full_bh": first.bh, "wrong_bh": first.WRONG_bh_after_selecting_top20,
                             "wrong_bonferroni": first.WRONG_bonf_after_selecting_top20},
                 "top20_sig_full": {"bonferroni": int((top20.bonferroni <= ALPHA).sum()), "bh": int((top20.bh <= ALPHA).sum())},
                 "top20_sig_wrong": {"bonferroni": int((top20.WRONG_bonf_after_selecting_top20 <= ALPHA).sum()),
                                     "bh": int((top20.WRONG_bh_after_selecting_top20 <= ALPHA).sum())}}

    # ---------- C3 锁定候选（在看 test 之前保存）----------
    chosen = table.loc[table.bh.le(ALPHA)].sort_values(["raw", "probe"]).head(10)
    locked_path = OUT / "locked-candidates.csv"
    if locked_path.exists():
        locked = pd.read_csv(locked_path)
        if locked.probe.tolist() != chosen.probe.tolist():
            raise ValueError("已有锁定名单不同；不得按验证结果覆盖名单")
        np.testing.assert_allclose(locked.discovery_delta, chosen.delta_AML_minus_ALL)
    else:
        locked = freeze_candidates(table, locked_path)
    res["C3"] = {"locked_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(locked_path.stat().st_mtime)),
                 "candidates": chosen[["probe", "n_AML", "n_ALL", "mean_AML", "mean_ALL", "delta_AML_minus_ALL",
                                       "ci95_low", "ci95_high", "cohen_d", "raw", "bonferroni", "bh"]].to_dict("records")}

    # ---------- C4 留出验证 ----------
    rep = replicate(locked_path)
    rep.to_csv(OUT / "C4-replication.csv", index=False)
    val = rep.merge(chosen[["probe", "ci95_low", "ci95_high"]].rename(columns={"ci95_low": "disc_low", "ci95_high": "disc_high"}),
                    on="probe", validate="one_to_one")
    val["ratio"] = val.delta_AML_minus_ALL / val.discovery_delta
    val = val.set_index("probe").loc[chosen.probe].reset_index()
    res["C4"] = {"n_locked": len(rep), "n_replicated": int(rep.replicated.sum()),
                 "n_same_direction": int(rep.same_direction.sum()),
                 "n_shrunk": int((val.ratio.abs() < 1).sum()),
                 "median_ratio": float(val.ratio.median()),
                 "rows": val[["probe", "discovery_delta", "disc_low", "disc_high", "delta_AML_minus_ALL",
                              "ci95_low", "ci95_high", "raw", "bonferroni", "same_direction", "replicated",
                              "ratio"]].to_dict("records")}
    fig_validation(val, FIG / "fig5-validation.png")
    print("C4 验证通过:", res["C4"]["n_replicated"], "/", res["C4"]["n_locked"])

    # ---------- 进阶② Storey π0 / q-value ----------
    lam_grid = np.round(np.arange(0.05, 0.951, 0.05), 2)
    p = table.raw.to_numpy()
    pi0 = storey_pi0(p)
    pi0_grid = np.array([storey_pi0(p, lam, cap=False) for lam in lam_grid])
    table["storey_q"] = storey_q(p, pi0)
    table.to_csv(OUT / "C1-discovery-all-probes.csv", index=False)

    # ---------- 进阶① 置换检验 ----------
    perm, perm_pi0_grid = permutation_null(train, lam_grid=lam_grid)
    perm.to_csv(OUT / "adv1-permutation-null.csv", index=False)
    n_total, n_aml = len(train), int(train.diagnosis.eq("AML").sum())
    res["adv1_permutation"] = {
        "n_perm": len(perm),
        "raw": {"mean": perm.raw.mean(), "sd": perm.raw.std(ddof=1), "median": perm.raw.median(),
                "p05": perm.raw.quantile(.05), "p95": perm.raw.quantile(.95), "max": int(perm.raw.max())},
        "independent_reference": {"mean": m * ALPHA, "sd": float(np.sqrt(m * ALPHA * (1 - ALPHA)))},
        "observed_raw": c["raw"],
        "perm_p_raw_count": (1 + int((perm.raw >= c["raw"]).sum())) / (len(perm) + 1),
        "share_any_bonferroni": float((perm.bonferroni > 0).mean()),
        "share_any_bh": float((perm.bh > 0).mean()),
        "bh": {"mean": perm.bh.mean(), "p95": perm.bh.quantile(.95), "max": int(perm.bh.max())},
        "bonferroni": {"mean": perm.bonferroni.mean(), "max": int(perm.bonferroni.max())},
        "overlap_expected": n_aml * n_aml / n_total,
        "spearman_overlap_vs_raw": float(stats.spearmanr(perm.true_AML_in_fake_AML, perm.raw).statistic),
        "spearman_dist_vs_raw": float(stats.spearmanr(
            (perm.true_AML_in_fake_AML - n_aml * n_aml / n_total).abs(), perm.raw).statistic),
        "raw_median_by_overlap": perm.groupby("true_AML_in_fake_AML").raw.median().to_dict(),
        "n_by_overlap": perm.true_AML_in_fake_AML.value_counts().sort_index().to_dict(),
    }
    near = perm.true_AML_in_fake_AML.between(2, 4)
    res["adv1_permutation"]["share_any_by_overlap"] = {
        group: {"n": int(sel.sum()),
                "bonferroni": float((perm.bonferroni[sel] > 0).mean()),
                "bh": float((perm.bh[sel] > 0).mean())}
        for group, sel in [("overlap_2_4", near), ("overlap_0_1_or_5_7", ~near)]
    }
    fig_permutation(perm, c["raw"], m, n_total, n_aml, FIG / "fig6-permutation.png")
    print("进阶① 完成")

    res["adv2_storey"] = {
        "lambda": LAMBDA, "pi0": pi0, "est_non_null": m * (1 - pi0),
        "pi0_grid": dict(zip(lam_grid.tolist(), pi0_grid.tolist())),
        "pi0_range_lambda_0.2_0.8": [float(pi0_grid[(lam_grid >= .2) & (lam_grid <= .8)].min()),
                                     float(pi0_grid[(lam_grid >= .2) & (lam_grid <= .8)].max())],
        "n_q_le_0.05": int((table.storey_q <= ALPHA).sum()), "n_bh_le_0.05": c["bh"],
        "perm_pi0": {"median": float(perm.pi0.median()), "p05": float(perm.pi0.quantile(.05)),
                     "min": float(perm.pi0.min()), "share_below_real": float((perm.pi0 <= pi0).mean())},
    }
    fig_pvalues(p, pi0, lam_grid, pi0_grid, perm_pi0_grid, FIG / "fig4-pvalues-pi0.png")
    print("进阶② π0 =", round(pi0, 4), "q≤0.05:", res["adv2_storey"]["n_q_le_0.05"])

    # ---------- 手算例子核对（作业可选题）----------
    toy = adjust([0.001, 0.009, 0.020, 0.040, 0.200])
    res["toy_example"] = toy.to_dict("list")

    # ---------- 漏斗图（主线）----------
    fig_funnel([
        (m, "检验家族：train 中全部探针", f"38 份样本（ALL 27 / AML 11），每个探针一次双侧 Welch t 检验；看结果前定好"),
        (c["raw"], "原始 p ≤ 0.05", f"不控制多重性：即使全部无差异，平均也会有约 {m * ALPHA:.0f} 个偶然显著"),
        (c["bh"], "BH adjusted p ≤ 0.05（控制 FDR）", "长期平均错误发现比例 ≤ 5%；不保证这一批里每个都是真的"),
        (c["bonferroni"], "Bonferroni ≤ 0.05（控制 FWER）", "整个家族“至少错 1 个”的概率 ≤ 5%；代价是检出力更低"),
        (len(rep), "预先锁定的候选", "BH 显著者按 raw p 取前 10 个，在打开 test 之前保存名单与方向"),
        (res["C4"]["n_replicated"], "留出样本验证通过", "test（ALL 20 / AML 14）候选家族 Bonferroni ≤ 0.05 且方向一致；非外部验证"),
    ], FIG / "fig1-funnel.png")

    res["elapsed_seconds"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=jsonable), encoding="utf-8")
    print(f"全部完成，用时 {res['elapsed_seconds']} 秒；结果见 {OUT}，图见 {FIG}")


if __name__ == "__main__":
    main()
