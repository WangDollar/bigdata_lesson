"""第一篇：单棵决策树的问题——偏差与方差（CTG）。

唯一改变的变量：max_depth。数据、划分、随机种子、指标全部固定。
  实验 1 深度扫描：开发集（train+validation）按 recording_id 分组 5 折 CV，看训练折 vs 验证折表现。
  实验 2 稳定性：固定 train / validation，对 train 做 30 次 bootstrap 重训，看同一批验证片段的预测变化。
              主分析：按录音分层重抽样（含病理录音 / 不含病理录音各自有放回抽同样多个）。
              对照：按录音不分层重抽样、按行重抽样。

运行：python src/part1_single_tree.py
只读 data/ctg/student/，不读测试标签。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "ctg" / "student"
OUT = BASE / "outputs" / "part1"
SEED = 1010
DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, None]
N_BOOT = 30
LABEL, GROUP = "pathologic", "recording_id"

# chart tokens (dataviz reference palette, light mode)
INK, INK2, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": ["PingFang SC"], "axes.unicode_minus": False, "font.size": 11,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False,
})


def load():
    train = pd.read_csv(DATA / "train.csv")
    valid = pd.read_csv(DATA / "validation.csv")
    feats = [c for c in train.columns if c not in {"sample_id", GROUP, LABEL}]
    return train, valid, feats


def tree(depth):
    return DecisionTreeClassifier(max_depth=depth, random_state=SEED)


def dlabel(depth):
    return "不限" if depth is None else str(depth)


def scores(model, X, y):
    p = model.predict_proba(X)[:, 1]
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            "bal_acc": balanced_accuracy_score(y, model.predict(X))}


# ---------- 实验 1：深度扫描（分组 CV） ----------
def depth_sweep(dev, feats):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(dev, dev[LABEL], dev[GROUP]))  # 同一组折用于所有深度
    rows = []
    for d in DEPTHS:
        for k, (a, b) in enumerate(folds):
            m = tree(d).fit(dev.iloc[a][feats], dev.iloc[a][LABEL])
            for part, idx in (("train", a), ("valid", b)):
                rows.append({"depth": dlabel(d), "fold": k, "part": part, "n_leaves": m.get_n_leaves(),
                             "fitted_depth": m.get_depth(), **scores(m, dev.iloc[idx][feats], dev.iloc[idx][LABEL])})
    long = pd.DataFrame(rows)
    summary = (long.groupby(["depth", "part"], sort=False)
               .agg(pr_auc=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
                    bal_acc=("bal_acc", "mean"), bal_acc_sd=("bal_acc", "std"),
                    roc_auc=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
                    n_leaves=("n_leaves", "mean"), fitted_depth=("fitted_depth", "mean"))
               .reset_index())
    return long, summary


def paired_vs(long, ref="4"):
    # 5 折共用同一划分：逐折求 “ref 深度 − 其他深度”，扣掉各折共同的难易差异
    v = long[long.part == "valid"]
    rows = []
    for metric in ("pr_auc", "bal_acc"):
        wide = v.pivot(index="fold", columns="depth", values=metric)
        for d in wide.columns.drop(ref):
            diff = wide[ref] - wide[d]
            rows.append({"metric": metric, "comparison": f"{ref} vs {d}", "mean_diff": diff.mean(),
                         "ref_wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()), "n_folds": len(diff)})
    return pd.DataFrame(rows)


# ---------- 实验 2：bootstrap 重训稳定性 ----------
SCHEMES = ("stratified", "recording", "row")  # 第一个是主分析


def resamples(train, scheme, rng):
    if scheme == "row":
        return [rng.integers(0, len(train), len(train)) for _ in range(N_BOOT)]
    by_rec = train.groupby(GROUP).indices
    recs = np.array(list(by_rec))
    if scheme == "recording":
        strata = [recs]
    else:  # stratified：含病理录音、不含病理录音各自有放回抽同样多个，病理录音数每次固定
        has_pos = train.groupby(GROUP)[LABEL].max().reindex(recs).to_numpy() == 1
        strata = [recs[has_pos], recs[~has_pos]]
    return [np.concatenate([by_rec[r] for s in strata for r in rng.choice(s, len(s), replace=True)])
            for _ in range(N_BOOT)]


def stability(train, valid, feats):
    Xv, yv = valid[feats], valid[LABEL].to_numpy()
    rows, roots = [], []
    for scheme in SCHEMES:
        samples = resamples(train, scheme, np.random.default_rng(SEED))  # 所有深度共享同一批重抽样
        for d in DEPTHS:
            probs, perf = [], []
            for b, idx in enumerate(samples):
                m = tree(d).fit(train.iloc[idx][feats], train.iloc[idx][LABEL])
                probs.append(m.predict_proba(Xv)[:, 1])
                perf.append(average_precision_score(yv, probs[-1]))
                if d == DEPTHS[0]:  # 根节点切分与 max_depth 无关，记录一次即可
                    roots.append({"scheme": scheme, "resample": b, "root_feature": feats[m.tree_.feature[0]],
                                  "threshold": m.tree_.threshold[0], "n_segments": len(idx),
                                  "n_pathologic": int(train[LABEL].to_numpy()[idx].sum())})
            probs = np.asarray(probs)
            share = (probs > 0.5).mean(axis=0)            # 30 次中判为病理的比例
            flipped = (share > 0) & (share < 1)            # 至少翻转过一次
            # Brier 分解（逐个验证片段）：mean_b (p_b - y)^2 = (p̄ - y)^2 + var_b(p)  →  偏差² + 方差
            bias2 = (probs.mean(axis=0) - yv) ** 2
            var = probs.var(axis=0)
            decomp = {f"{name}{suffix}": vals[mask].mean()
                      for suffix, mask in (("", slice(None)), ("_pathologic", yv == 1), ("_other", yv == 0))
                      for name, vals in (("bias2", bias2), ("variance", var))}
            rows.append({"scheme": scheme, "depth": dlabel(d), **decomp,
                         "pred_sd_mean": probs.std(axis=0).mean(),
                         "flip_rate": flipped.mean(),
                         "flip_rate_pathologic": flipped[yv == 1].mean(),
                         "flip_rate_other": flipped[yv == 0].mean(),
                         "pairwise_disagreement": (2 * share * (1 - share) * N_BOOT / (N_BOOT - 1)).mean(),
                         "valid_pr_auc_mean": np.mean(perf), "valid_pr_auc_sd": np.std(perf, ddof=1)})
    return pd.DataFrame(rows), pd.DataFrame(roots)


# ---------- 图 ----------
def end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=10)


def plot_depth(summary, path):
    xs = np.arange(len(DEPTHS))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, (key, title) in zip(axes, [("pr_auc", "PR-AUC"), ("bal_acc", "Balanced accuracy")]):
        tr = summary[summary.part == "train"]
        va = summary[summary.part == "valid"]
        ax.plot(xs, tr[key], color=ORANGE, marker="o", ms=5, label="训练折")
        ax.plot(xs, va[key], color=BLUE, marker="o", ms=5, label="验证折（均值 ± 折间 SD）")
        ax.fill_between(xs, va[key] - va[key + "_sd"], va[key] + va[key + "_sd"], color=BLUE, alpha=0.10, lw=0)
        end_label(ax, xs[-1], tr[key].iloc[-1], "训练")
        end_label(ax, xs[-1], va[key].iloc[-1], "验证")
        ax.set_xticks(xs, [dlabel(d) for d in DEPTHS])
        ax.set_xlim(-0.3, len(DEPTHS) - 0.2)
        ax.set_xlabel("max_depth")
        ax.set_title(title, loc="left", fontsize=12, fontweight=600)
    axes[0].legend(loc="lower right", fontsize=9.5)
    fig.suptitle("实验 1　树越深：训练折与验证折的差距（CTG 开发集，按录音分组 5 折 CV）", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bias_variance(stab, path):
    s = stab[stab.scheme == SCHEMES[0]]
    xs = np.arange(len(DEPTHS))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(xs, s.bias2, width=0.55, color=BLUE, label="偏差²：30 棵树的平均预测离真值多远", edgecolor=SURF, linewidth=2)
    ax.bar(xs, s.variance, bottom=s.bias2, width=0.55, color=ORANGE, label="方差：30 棵树彼此差多少", edgecolor=SURF, linewidth=2)
    for x, b, v in zip(xs, s.bias2, s.variance):
        ax.annotate(f"方差占 {v / (b + v):.0%}", (x, b + v), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=INK2, fontsize=9.5)
    ax.set_xticks(xs, [dlabel(d) for d in DEPTHS])
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Brier 分数（越低越好）")
    ax.set_ylim(0, s.bias2.add(s.variance).max() * 1.15)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("实验 2　Brier = 偏差² + 方差：树越深，偏差²持续下降，方差占比从约 1/10 升到约 2/5（30 次按录音分层重抽样）",
                 loc="left", fontsize=12, fontweight=600)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_instability(stab, roots, path):
    xs = np.arange(len(DEPTHS))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    rec = roots[roots.scheme == SCHEMES[0]]
    counts = rec.root_feature.value_counts().sort_values()
    ax.barh(counts.index, counts.values, color=BLUE, height=0.5)
    for i, (feat, v) in enumerate(counts.items()):
        th = rec.loc[rec.root_feature == feat, "threshold"]
        span = f"阈值 {th.min():g}" if th.min() == th.max() else f"阈值 {th.min():g}–{th.max():g}"
        ax.annotate(f"{v} 次（{span}）", (v, i), xytext=(4, 0), textcoords="offset points", va="center", color=INK2, fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, counts.max() * 1.9)
    ax.set_title("根节点用了哪个特征", loc="left", fontsize=12, fontweight=600)

    ax = axes[1]
    s = stab[stab.scheme == SCHEMES[0]]
    for key, color, name in (("flip_rate_pathologic", ORANGE, "病理片段（34 个）"), ("flip_rate_other", BLUE, "其余片段（390 个）")):
        ax.plot(xs, s[key] * 100, color=color, marker="o", ms=5, label=name)
    ax.set_xticks(xs, [dlabel(d) for d in DEPTHS])
    ax.set_ylim(0, 100)
    ax.set_xlabel("max_depth")
    ax.set_title("30 次重训中预测类别变过的片段比例（%）", loc="left", fontsize=12, fontweight=600)
    ax.legend(loc="center right", fontsize=9.5)
    fig.suptitle("实验 2　不稳定在哪里：第一个问题就会变，翻转集中在病理片段（30 次按录音分层重抽样）",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, valid, feats = load()
    dev = pd.concat([train, valid], ignore_index=True)

    long, summary = depth_sweep(dev, feats)
    long.to_csv(OUT / "depth_cv_folds.csv", index=False)
    summary.to_csv(OUT / "depth_cv_summary.csv", index=False)
    paired = paired_vs(long)
    paired.to_csv(OUT / "depth_paired_vs4.csv", index=False)

    stab, roots = stability(train, valid, feats)
    stab.to_csv(OUT / "stability.csv", index=False)
    roots.to_csv(OUT / "root_splits.csv", index=False)

    example = tree(2).fit(train[feats], train[LABEL])
    (OUT / "tree_depth2.txt").write_text(export_text(example, feature_names=feats, show_weights=True, decimals=3), encoding="utf-8")

    plot_depth(summary, OUT / "fig1_depth_curve.png")
    plot_bias_variance(stab, OUT / "fig2_bias_variance.png")
    plot_instability(stab, roots, OUT / "fig3_instability.png")

    pd.set_option("display.width", 200)
    cols = ["depth", "n_leaves", "pr_auc", "pr_auc_sd", "bal_acc", "bal_acc_sd", "roc_auc", "roc_auc_sd"]
    print("== 实验 1：训练折 ==\n", summary[summary.part == "train"][cols].round(3).to_string(index=False))
    print("== 实验 1：验证折 ==\n", summary[summary.part == "valid"][cols].round(3).to_string(index=False))
    print("== 实验 1：逐折配对（深度 4 − 其他）==\n", paired.round(3).to_string(index=False))
    print("== 实验 2：稳定性 ==\n", stab.round(3).to_string(index=False))
    print("== 每次重抽样的病理片段数 ==\n", roots.groupby("scheme").n_pathologic.describe()[["mean", "std", "min", "max"]].to_string())
    print("== 根节点 ==\n", roots.groupby(["scheme", "root_feature"]).threshold.agg(["size", "min", "max"]).to_string())
    print("== 深度 2 示例树（train 全部 1279 段）==\n", export_text(example, feature_names=feats, show_weights=True, decimals=3))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
