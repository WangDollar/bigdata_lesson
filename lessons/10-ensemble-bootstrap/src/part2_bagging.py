"""第二篇：Bagging 路线——平均能降什么？随机森林多了什么？（CTG）

对比 2-1  只改变基学习器深度（树桩 / 深度 4 / 不限），每种比较：单树、按行 bagging、按录音分层 bagging。
对比 2-2  只改变每个节点的候选特征数 max_features（21=bagging, 10, 4=RF 默认, 2, 1），按行 / 按录音分层两种抽样各跑一遍。
判类阈值  balanced accuracy 同时报告 0.5 阈值与“训练集病理比例”阈值（BA 的理论最优阈值，事先固定、不在评价折上调）。

bagging / RF 用同一个手写实现（Bag），保证“按行 vs 按录音”只差抽样单位；树的设置与平均方式完全相同。

测量面板
  表现   开发集按录音分组 5 折 CV（与第一篇同一组折）：PR-AUC、balanced accuracy；与深度 4 单树逐折配对。
  稳定性 两层设计：外层 30 次按录音分层重抽样 train（= 换一批训练数据，与第一篇相同），
         内层在每份数据上训练一个 100 棵树的集成。固定 424 个验证片段：
           Brier = 偏差² + 方差（方差 = 外层 30 次之间集成预测的波动）、病理片段翻转率；
           σ²  = 单棵成员树的方差；ρ = (集成方差 − σ²/B) / (σ²(1 − 1/B))，即“平均也去不掉的方差占比”。

运行：python src/part2_bagging.py
只读 data/ctg/student/，不读测试标签。
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

from part1_single_tree import (BASE, BLUE, GRID, GROUP, INK, INK2, LABEL, ORANGE, SEED, SURF, dlabel, load, plt)

OUT = BASE / "outputs" / "part2"
N_OUTER, N_TREES = 30, 100
N_FEATS = 21


# ---------- 重抽样 ----------
def recording_sampler(groups, y, stratified=True):
    """返回 draw(rng) -> (行索引, 抽样单位编号)。分层：含病理录音 / 不含病理录音各自有放回抽同样多个。"""
    order = pd.Series(np.arange(len(groups))).groupby(np.asarray(groups)).indices   # 键已排序，与第一篇一致
    recs = np.array(list(order))
    has_pos = np.array([y[order[r]].max() == 1 for r in recs])
    strata = [recs[has_pos], recs[~has_pos]] if stratified else [recs]

    def draw(rng):
        picks = [r for s in strata for r in rng.choice(s, len(s), replace=True)]
        idx = np.concatenate([order[r] for r in picks])
        unit = np.concatenate([np.full(len(order[r]), u) for u, r in enumerate(picks)])  # 重复抽到的录音算不同单位
        return idx, unit
    return draw


def proba1(model, X):
    p = model.predict_proba(X)
    return p[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(X))


# ---------- 模型 ----------
class Single:
    def __init__(self, depth):
        self.depth = depth

    def fit(self, X, y, groups, seed):
        self.tree = DecisionTreeClassifier(max_depth=self.depth, random_state=SEED).fit(X, y)
        return self

    def tree_probs(self, X):
        return proba1(self.tree, X)[None, :]


class Bag:
    """bagging / 随机森林：每棵树用一份 bootstrap 样本；max_features<21 时每个节点只看随机的部分特征（= RF）。"""
    def __init__(self, depth=None, max_features=None, unit="row"):
        self.depth, self.max_features, self.unit = depth, max_features, unit

    def fit(self, X, y, groups, seed):
        rng = np.random.default_rng(seed)
        draw = recording_sampler(groups, y) if self.unit == "recording" else None
        self.trees = []
        for _ in range(N_TREES):
            idx = rng.integers(0, len(y), len(y)) if draw is None else draw(rng)[0]
            t = DecisionTreeClassifier(max_depth=self.depth, max_features=self.max_features,
                                       random_state=int(rng.integers(2**31 - 1)))
            self.trees.append(t.fit(X[idx], y[idx]))
        return self

    def tree_probs(self, X):
        return np.array([proba1(t, X) for t in self.trees])


def configs():
    rows = []
    for d in (1, 4, None):
        rows += [("2-1", f"单树·{dlabel(d)}", lambda d=d: Single(d), d, "—", N_FEATS),
                 ("2-1", f"按行bagging·{dlabel(d)}", lambda d=d: Bag(d, None, "row"), d, "按行", N_FEATS),
                 ("2-1", f"按录音bagging·{dlabel(d)}", lambda d=d: Bag(d, None, "recording"), d, "按录音分层", N_FEATS)]
    rows += [("基线", f"单树·{d}", lambda d=d: Single(d), d, "—", N_FEATS) for d in (5, 6)]
    for unit, tag, cn in (("row", "按行", "按行"), ("recording", "按录音", "按录音分层")):
        rows += [("2-2", f"{tag}RF·mf={k}", lambda k=k, u=unit: Bag(None, k, u), None, cn, k) for k in (10, 4, 2, 1)]
    return pd.DataFrame(rows, columns=["block", "name", "make", "depth", "unit", "max_features"])


# ---------- 表现：分组 5 折 CV ----------
def cv_one(make, dev, feats, folds):
    X, y, g = dev[feats].to_numpy(), dev[LABEL].to_numpy(), dev[GROUP].to_numpy()
    out = []
    for k, (a, b) in enumerate(folds):
        p = make().fit(X[a], y[a], g[a], SEED).tree_probs(X[b]).mean(axis=0)
        prev = y[a].mean()                                   # 训练折病理比例，事先固定的阈值
        row = {"fold": k, "pr_auc": average_precision_score(y[b], p), "roc_auc": roc_auc_score(y[b], p)}
        for tag, thr in (("", 0.5), ("_prev", prev)):
            pred = p > thr
            row[f"bal_acc{tag}"] = balanced_accuracy_score(y[b], pred)
            row[f"sens{tag}"] = pred[y[b] == 1].mean()
            row[f"spec{tag}"] = 1 - pred[y[b] == 0].mean()
        out.append(row)
    return out


# ---------- 稳定性：两层设计 ----------
def outer_one(make, k, idx, unit, Xtr, ytr, Xv):
    return make().fit(Xtr[idx], ytr[idx], unit, seed=SEED * 1000 + k).tree_probs(Xv)   # (B, N)


def stability_one(T, yv, prev):
    """T: (K, B, N) 每次外层重抽样、每棵树对验证片段的预测。"""
    K, B, _ = T.shape
    E = T.mean(axis=1)                                   # (K, N) 集成预测
    pos = yv == 1
    v_ens = E.var(axis=0)                                # 外层之间的集成方差
    bias2 = (E.mean(axis=0) - yv) ** 2
    row = {"bias2": bias2.mean(), "variance": v_ens.mean(), "brier": bias2.mean() + v_ens.mean(),
           "variance_pathologic": v_ens[pos].mean(),
           "valid_pr_auc_mean": np.mean([average_precision_score(yv, e) for e in E]),
           "valid_pr_auc_sd": np.std([average_precision_score(yv, e) for e in E], ddof=1)}
    for tag, thr in (("", 0.5), ("_prev", prev)):
        share = (E > thr).mean(axis=0)
        row[f"flip_rate_pathologic{tag}"] = ((share > 0) & (share < 1))[pos].mean()
    if B > 1:
        flat = T.reshape(K * B, -1)
        sigma2 = flat.var(axis=0)
        for suffix, m in (("", slice(None)), ("_pathologic", pos)):
            s2, ve = sigma2[m].mean(), v_ens[m].mean()
            row[f"sigma2{suffix}"] = s2
            row[f"rho{suffix}"] = (ve - s2 / B) / (s2 * (1 - 1 / B))
        row["tree_pr_auc"] = np.mean([average_precision_score(yv, t) for t in flat])
        row["tree_brier"] = ((flat - yv) ** 2).mean()
    return row


# ---------- 图 ----------
def style_panel(ax, title):
    ax.set_title(title, loc="left", fontsize=12, fontweight=600)


def plot_averaging(stab, cvsum, cvlong, path):
    order = [f"{kind}·{dlabel(d)}" for d in (1, 4, None) for kind in ("单树", "按行bagging", "按录音bagging")]
    xs = np.array([g * 4 + j for g in range(3) for j in range(3)], dtype=float)
    short = ["单树", "按行", "按录音"] * 3
    s = stab.set_index("name").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    ax.bar(xs, s.bias2, width=0.75, color=BLUE, edgecolor=SURF, linewidth=2, label="偏差²")
    ax.bar(xs, s.variance, bottom=s.bias2, width=0.75, color=ORANGE, edgecolor=SURF, linewidth=2, label="方差")
    for x, b, v in zip(xs, s.bias2, s.variance):
        ax.annotate(f"{b + v:.3f}", (x, b + v), xytext=(0, 3), textcoords="offset points", ha="center", color=INK2, fontsize=9)
    ax.set_ylabel("验证集 Brier（越低越好）")
    ax.legend(loc="upper right", fontsize=9.5)
    style_panel(ax, "Brier = 偏差² + 方差（30 次换训练数据）")

    ax = axes[1]
    for x, name in zip(xs, order):
        folds = cvlong.loc[cvlong.name == name, "pr_auc"]
        ax.scatter(np.full(len(folds), x), folds, s=18, color=GRID, edgecolor=INK2, linewidth=0.6, zorder=2)
        m = cvsum.loc[name, "pr_auc"]
        ax.plot([x - 0.32, x + 0.32], [m, m], color=INK, lw=2.2, zorder=3)
        ax.annotate(f"{m:.2f}", (x + 0.36, m), va="center", color=INK2, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.12", fc=SURF, ec="none"), zorder=4)
    ax.set_ylabel("PR-AUC（点 = 各折，横线 = 均值）")
    style_panel(ax, "分组 5 折 CV 的 PR-AUC")

    for ax in axes:
        ax.set_xticks(xs, short, fontsize=9.5)
        ax.set_xlim(xs[0] - 0.8, xs[-1] + 0.9)
        for g, lab in enumerate(["树桩（深度 1）", "深度 4", "深度不限"]):
            ax.annotate(lab, (g * 4 + 1, 0), xycoords=("data", "axes fraction"), xytext=(0, -34),
                        textcoords="offset points", ha="center", color=INK, fontsize=10.5)
    fig.suptitle("对比 2-1　平均能降什么：bagging 主要去掉方差，几乎去不掉偏差", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_good_different(stab, cvsum, path):
    ks = (10, 4, 2, 1)
    series = [("按行抽样", BLUE, ["按行bagging·不限"] + [f"按行RF·mf={k}" for k in ks]),
              ("按录音分层抽样", ORANGE, ["按录音bagging·不限"] + [f"按录音RF·mf={k}" for k in ks])]
    labels = ["21\nbagging", "10", "4\nRF 默认", "2", "1"]
    xs = np.arange(len(labels))
    st, cv = stab.set_index("name"), cvsum
    panels = [(lambda n: st.loc[n, "tree_pr_auc"], "“好”：单棵成员树的 PR-AUC", "{:.2f}"),
              (lambda n: st.loc[n, "rho"], "“不同”：ρ（平均去不掉的方差占比）", "{:.2f}"),
              (lambda n: cv.loc[n, "pr_auc"], "整体：CV PR-AUC（与阈值无关）", "{:.3f}"),
              (lambda n: cv.loc[n, "bal_acc_prev"], "整体：CV balanced accuracy（阈值 = 训练集病理比例）", "{:.3f}")]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))
    for ax, (get, title, fmt) in zip(axes.flat, panels):
        vals = {name: np.array([get(n) for n in names]) for name, _, names in series}
        a, b = vals[series[0][0]], vals[series[1][0]]
        for name, color, _ in series:
            v_all = vals[name]
            ax.plot(xs, v_all, color=color, marker="o", ms=6, label=name)
            other = b if name == series[0][0] else a
            for x, v, o in zip(xs, v_all, other):
                up = v >= o if name == series[1][0] else v > o      # 每个位置：较大者标上方，较小者标下方
                ax.annotate(fmt.format(v), (x, v), xytext=(0, 7 if up else -15), textcoords="offset points",
                            ha="center", color=INK2, fontsize=8.5, bbox=dict(boxstyle="round,pad=0.1", fc=SURF, ec="none"))
        ax.set_xticks(xs, labels, fontsize=9.5)
        ax.margins(y=0.18)
        style_panel(ax, title)
    axes[0, 0].legend(loc="upper right", fontsize=9.5)
    for ax in axes[1]:
        ax.set_xlabel("每个节点的候选特征数 max_features")
    fig.suptitle("对比 2-2　好而不同：候选特征越少，树越弱，但也越不相关（深度不限，100 棵）",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    train, valid, feats = load()
    dev = pd.concat([train, valid], ignore_index=True)
    cfg = configs()

    # 表现
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(dev, dev[LABEL], dev[GROUP]))
    cv_rows = Parallel(n_jobs=-1)(delayed(cv_one)(c.make, dev, feats, folds) for c in cfg.itertuples())
    cvlong = pd.concat([pd.DataFrame(r).assign(name=n) for r, n in zip(cv_rows, cfg.name)], ignore_index=True)
    cvsum = cvlong.groupby("name", sort=False).agg(pr_auc=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
                                                   bal_acc=("bal_acc", "mean"), roc_auc=("roc_auc", "mean"),
                                                   bal_acc_prev=("bal_acc_prev", "mean"), sens=("sens", "mean"),
                                                   sens_prev=("sens_prev", "mean"), spec_prev=("spec_prev", "mean"))
    ref = cvlong[cvlong.name == "单树·4"].set_index("fold")
    paired = []
    for n in cfg.name:
        cur = cvlong[cvlong.name == n].set_index("fold")
        for m in ("pr_auc", "bal_acc", "bal_acc_prev"):
            diff = cur[m] - ref[m]
            paired.append({"name": n, "metric": m, "mean_diff_vs_tree4": diff.mean(), "wins_vs_tree4": int((diff > 0).sum()),
                           "ties": int((diff == 0).sum())})
    paired = pd.DataFrame(paired)
    print(f"CV 完成 {time.time() - t0:.0f}s")

    # 稳定性（两层）
    Xtr, ytr, Xv, yv = train[feats].to_numpy(), train[LABEL].to_numpy(), valid[feats].to_numpy(), valid[LABEL].to_numpy()
    draw = recording_sampler(train[GROUP].to_numpy(), ytr)
    rng = np.random.default_rng(SEED)                 # 与第一篇同一批外层重抽样
    outer = [draw(rng) for _ in range(N_OUTER)]
    stab_rows = []
    for c in cfg.itertuples():
        T = np.asarray(Parallel(n_jobs=-1)(delayed(outer_one)(c.make, k, idx, unit, Xtr, ytr, Xv)
                                           for k, (idx, unit) in enumerate(outer)))
        stab_rows.append({"name": c.name, **stability_one(T, yv, ytr.mean())})
        print(f"  {c.name:<16} 完成 {time.time() - t0:.0f}s")
    stab = pd.DataFrame(stab_rows)

    table = cfg.drop(columns="make").merge(cvsum.reset_index(), on="name").merge(stab, on="name")
    table.to_csv(OUT / "part2_summary.csv", index=False)
    cvlong.to_csv(OUT / "cv_folds.csv", index=False)
    paired.to_csv(OUT / "paired_vs_tree4.csv", index=False)
    plot_averaging(stab, cvsum, cvlong, OUT / "fig1_what_averaging_removes.png")
    plot_good_different(stab, cvsum, OUT / "fig2_good_and_different.png")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    cols = ["block", "name", "pr_auc", "roc_auc", "bal_acc", "sens", "bal_acc_prev", "sens_prev", "spec_prev",
            "brier", "bias2", "variance", "flip_rate_pathologic", "flip_rate_pathologic_prev", "rho", "tree_pr_auc"]
    print(table[cols].round(4).to_string(index=False))
    print(paired.pivot(index="name", columns="metric", values=["mean_diff_vs_tree4", "wins_vs_tree4"]).round(3)
          .reindex(cfg.name).to_string())
    print(f"Wrote {OUT}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
