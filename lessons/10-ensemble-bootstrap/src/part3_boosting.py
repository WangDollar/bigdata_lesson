"""第三篇：Boosting 路线——顺序叠加能降什么？模型越多越好吗？（CTG）

对比 3-1  同样 100 个树桩，只改变组合方式：并行平均（bagging）vs 顺序叠加（AdaBoost、梯度提升）。
          参考行：深度 4 单树、随机森林（max_features=4）、梯度提升默认（深度 3）。
          机制观察：AdaBoost 的样本权重最后集中到了哪些片段 / 录音。
对比 3-2  只改变模型个数：随机森林加树 vs 梯度提升加轮（learning_rate = 0.3 / 0.1 / 0.03，深度 3）。

评价口径（与第二篇相同，已确认）
  表现   开发集按录音分组 5 折 CV：PR-AUC（与阈值无关，主指标）；balanced accuracy 以训练折病理比例为阈值（主），0.5 为参考。
  稳定性 外层 30 次按录音分层重抽样 train，固定 424 个验证片段：Brier = 偏差² + 方差，病理片段翻转率。

运行：python src/part3_boosting.py
只读 data/ctg/student/，不读测试标签。
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

from part1_single_tree import BASE, BLUE, GRID, GROUP, INK, INK2, LABEL, ORANGE, SEED, SURF, load, plt
from part2_bagging import N_OUTER, Bag, Single, cv_one, outer_one, proba1, recording_sampler, stability_one

OUT = BASE / "outputs" / "part3"
N_STUMPS = 100
AQUA, YELLOW = "#1baf7a", "#eda100"
GRID_N = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000]
LRS = (0.3, 0.1, 0.03)


class Sk:
    """把 sklearn 模型包成与第二篇相同的接口；没有内部随机性的部分固定 SEED，波动只来自训练数据。"""
    def __init__(self, factory):
        self.factory = factory

    def fit(self, X, y, groups, seed):
        self.m = self.factory().fit(X, y)
        return self

    def tree_probs(self, X):
        return proba1(self.m, X)[None, :]


def stump():
    return DecisionTreeClassifier(max_depth=1, random_state=SEED)


class CalibratedAda:
    """AdaBoost 的 predict_proba 只是压缩后的加权投票（约 0.25–0.62），不是概率。
    只在训练数据内部用按录音分组的 3 折交叉预测拟合 sigmoid（Platt）映射；ensemble=False → 最终仍是同样 100 个树桩。"""
    def fit(self, X, y, groups, seed):
        inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(X, y, groups))
        ada = AdaBoostClassifier(stump(), n_estimators=N_STUMPS, random_state=SEED)
        self.m = CalibratedClassifierCV(ada, method="sigmoid", cv=inner, ensemble=False).fit(X, y)
        return self

    def tree_probs(self, X):
        return proba1(self.m, X)[None, :]


def configs():
    rows = [
        ("3-1", "单个树桩", lambda: Single(1)),
        ("3-1", "bagging·100 树桩", lambda: Bag(1, None, "row")),
        ("3-1", "AdaBoost·100 树桩", lambda: Sk(lambda: AdaBoostClassifier(stump(), n_estimators=N_STUMPS, random_state=SEED))),
        ("3-1", "AdaBoost·100 树桩+校准", lambda: CalibratedAda()),
        ("3-1", "梯度提升·100 树桩", lambda: Sk(lambda: GradientBoostingClassifier(max_depth=1, n_estimators=N_STUMPS,
                                                                            learning_rate=0.1, random_state=SEED))),
        ("参考", "单树·深度4", lambda: Single(4)),
        ("参考", "随机森林·mf=4", lambda: Sk(lambda: RandomForestClassifier(n_estimators=100, max_features="sqrt",
                                                                       random_state=SEED, n_jobs=1))),
        ("参考", "梯度提升·默认(深度3)", lambda: Sk(lambda: GradientBoostingClassifier(random_state=SEED))),
    ]
    return pd.DataFrame(rows, columns=["block", "name", "make"])


# ---------- 机制观察：AdaBoost 的样本权重 ----------
def adaboost_weights(train, feats):
    """按 sklearn SAMME 的更新规则重算每一轮的样本权重：错分者乘 exp(α)，再归一化。"""
    X, y = train[feats].to_numpy(), train[LABEL].to_numpy()
    ada = AdaBoostClassifier(stump(), n_estimators=N_STUMPS, random_state=SEED).fit(X, y)
    w = np.full(len(y), 1 / len(y))
    trace = []
    for m, (est, alpha) in enumerate(zip(ada.estimators_, ada.estimator_weights_), 1):
        miss = est.predict(X) != y
        w = w * np.exp(alpha * miss)
        w /= w.sum()
        if m in (1, 2, 5, 10, 20, 50, 100):
            top = np.sort(w)[::-1]
            trace.append({"round": m, "weight_on_pathologic": w[y == 1].sum(),
                          "top1pct_share": top[: max(1, len(w) // 100)].sum(), "top5pct_share": top[: len(w) // 20].sum(),
                          "stump_error": (est.predict(X) != y).mean()})
    final = train[["sample_id", GROUP, LABEL]].assign(weight=w)
    by_rec = (final.groupby(GROUP).agg(weight=("weight", "sum"), n=("weight", "size"), n_path=(LABEL, "sum"))
              .sort_values("weight", ascending=False))
    return pd.DataFrame(trace), final.sort_values("weight", ascending=False), by_rec


# ---------- 对比 3-2：模型个数曲线（分组 CV） ----------
def metrics_at(y, p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    from sklearn.metrics import average_precision_score
    return {"brier": ((p - y) ** 2).mean(), "log_loss": -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean(),
            "pr_auc": average_precision_score(y, p)}


def curve_fold(kind, lr, X, y, a, b):
    rows = []
    if kind == "rf":
        rf = RandomForestClassifier(n_estimators=max(GRID_N), max_features="sqrt", random_state=SEED, n_jobs=1).fit(X[a], y[a])
        cum = {part: np.cumsum([proba1(t, X[idx]) for t in rf.estimators_], axis=0) for part, idx in (("train", a), ("valid", b))}
        stages = {part: {n: cum[part][n - 1] / n for n in GRID_N} for part in cum}
    else:
        gb = GradientBoostingClassifier(n_estimators=max(GRID_N), learning_rate=lr, max_depth=3, random_state=SEED).fit(X[a], y[a])
        stages = {}
        for part, idx in (("train", a), ("valid", b)):
            stages[part] = {n: p[:, 1] for n, p in enumerate(gb.staged_predict_proba(X[idx]), 1) if n in GRID_N}
    for part, idx in (("train", a), ("valid", b)):
        for n in GRID_N:
            rows.append({"model": "随机森林" if kind == "rf" else f"梯度提升 lr={lr}", "n": n, "part": part,
                         **metrics_at(y[idx], stages[part][n])})
    return rows


def model_count_curves(dev, feats, folds):
    X, y = dev[feats].to_numpy(), dev[LABEL].to_numpy()
    jobs = [("rf", None)] + [("gb", lr) for lr in LRS]
    res = Parallel(n_jobs=-1)(delayed(curve_fold)(kind, lr, X, y, a, b) for kind, lr in jobs for a, b in folds)
    return (pd.DataFrame([r for rows in res for r in rows])
            .groupby(["model", "n", "part"], sort=False).mean(numeric_only=True).reset_index())


# ---------- 图 ----------
def plot_parallel_vs_sequential(table, path):
    names = list(table.name)
    xs = np.array([0, 1, 2, 3, 4, 5.8, 6.8, 7.8])
    short = ["单个\n树桩", "bagging", "AdaBoost\n(未校准)", "AdaBoost\n+校准", "梯度提升", "单树\n深度4", "随机森林\nmf=4", "梯度提升\n默认"]
    raw = np.array([n == "AdaBoost·100 树桩" for n in names])   # 分数不是概率：Brier 与阈值指标不可解读
    NEUTRAL = "#c3c2b7"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.bar(xs[~raw], table.bias2[~raw], width=0.7, color=BLUE, edgecolor=SURF, linewidth=2, label="偏差²")
    ax.bar(xs[~raw], table.variance[~raw], bottom=table.bias2[~raw], width=0.7, color=ORANGE, edgecolor=SURF, linewidth=2, label="方差")
    ax.bar(xs[raw], (table.bias2 + table.variance)[raw], width=0.7, color=NEUTRAL, edgecolor=SURF, linewidth=2)
    for x, b, v, r in zip(xs, table.bias2, table.variance, raw):
        ax.annotate(f"{b + v:.3f}" + ("\n分数不是概率，不可比" if r else ""), (x, b + v), xytext=(0, 3),
                    textcoords="offset points", ha="center", color=INK2, fontsize=9)
    ax.set_ylim(0, (table.bias2 + table.variance).max() * 1.22)
    ax.set_ylabel("验证集 Brier（越低越好）")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("Brier = 偏差² + 方差（30 次换训练数据）", loc="left", fontsize=12, fontweight=600)
    ax = axes[1]
    for x, n in zip(xs, names):
        r = table.set_index("name").loc[n]
        for val, mk, lab in ((r.pr_auc, "o", "PR-AUC"), (r.bal_acc_prev, "s", "balanced acc（阈值=病理比例）")):
            ax.scatter([x], [val], marker=mk, s=46, color=INK if mk == "o" else SURF, edgecolor=INK, linewidth=1.2, zorder=3,
                       label=lab if x == xs[0] else None)
        ax.annotate(f"{r.pr_auc:.2f}", (x, r.pr_auc), xytext=(8, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
        if n == "AdaBoost·100 树桩":
            ax.annotate("全部判为病理", (x, r.bal_acc_prev), xytext=(8, 0), textcoords="offset points", va="center",
                        color=INK2, fontsize=9)
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.set_title("分组 5 折 CV", loc="left", fontsize=12, fontweight=600)
    for ax in axes:
        ax.set_xticks(xs, short, fontsize=9)
        ax.axvline(4.9, color=GRID, lw=1)
    fig.suptitle("对比 3-1　并行平均 vs 顺序叠加：同样 100 个树桩，boosting 降的是偏差（竖线右侧为参考模型）",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_curves(curves, path):
    series = [("随机森林", BLUE), ("梯度提升 lr=0.3", ORANGE), ("梯度提升 lr=0.1", AQUA), ("梯度提升 lr=0.03", YELLOW)]
    panels = [("valid", "log_loss", "验证折 log-loss（越低越好）"), ("valid", "pr_auc", "验证折 PR-AUC"),
              ("train", "log_loss", "训练折 log-loss")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (part, metric, title) in zip(axes, panels):
        for name, color in series:
            c = curves[(curves.model == name) & (curves.part == part)]
            ax.plot(c.n, c[metric], color=color, marker="o", ms=3.5, label=name)
            if part == "valid" and metric == "log_loss" and name != "随机森林":
                best = c.loc[c[metric].idxmin()]
                ax.scatter([best.n], [best[metric]], s=70, facecolor="none", edgecolor=color, linewidth=1.6, zorder=4)
        ax.set_xscale("log")
        ax.set_xlabel("树的棵数 / 提升轮数（对数刻度）")
        ax.set_title(title, loc="left", fontsize=12, fontweight=600)
    axes[0].legend(loc="upper right", fontsize=9)
    v = curves[(curves.part == "valid")]
    axes[0].set_ylim(0, min(0.5, v.log_loss.max()) * 1.05)
    axes[1].set_ylim(max(0.5, v.pr_auc.min() - 0.02), 1.0)
    fig.suptitle("对比 3-2　模型越多越好吗：随机森林进入平台，梯度提升先降后升（空心圈 = 验证损失最低点）",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    train, valid, feats = load()
    dev = pd.concat([train, valid], ignore_index=True)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(dev, dev[LABEL], dev[GROUP]))
    cfg = configs()

    # 3-1 表现
    cv_rows = Parallel(n_jobs=-1)(delayed(cv_one)(c.make, dev, feats, folds) for c in cfg.itertuples())
    cvlong = pd.concat([pd.DataFrame(r).assign(name=n) for r, n in zip(cv_rows, cfg.name)], ignore_index=True)
    cvsum = cvlong.groupby("name", sort=False).mean(numeric_only=True).drop(columns="fold")
    paired = []
    for refname in ("单树·深度4", "bagging·100 树桩", "随机森林·mf=4"):
        ref = cvlong[cvlong.name == refname].set_index("fold")
        for n in cfg.name:
            cur = cvlong[cvlong.name == n].set_index("fold")
            for m in ("pr_auc", "bal_acc_prev"):
                d = cur[m] - ref[m]
                paired.append({"ref": refname, "name": n, "metric": m, "mean_diff": d.mean(), "wins": int((d > 0).sum())})
    paired = pd.DataFrame(paired)

    # 3-1 稳定性
    Xtr, ytr, Xv, yv = train[feats].to_numpy(), train[LABEL].to_numpy(), valid[feats].to_numpy(), valid[LABEL].to_numpy()
    draw = recording_sampler(train[GROUP].to_numpy(), ytr)
    rng = np.random.default_rng(SEED)
    outer = [draw(rng) for _ in range(N_OUTER)]
    stab = []
    for c in cfg.itertuples():
        T = np.asarray(Parallel(n_jobs=-1)(delayed(outer_one)(c.make, k, idx, unit, Xtr, ytr, Xv) for k, (idx, unit) in enumerate(outer)))
        stab.append({"name": c.name, **stability_one(T, yv, ytr.mean())})
    table = cfg.drop(columns="make").merge(cvsum.reset_index(), on="name").merge(pd.DataFrame(stab), on="name")
    print(f"3-1 完成 {time.time() - t0:.0f}s")

    # 机制观察
    trace, seg_w, rec_w = adaboost_weights(train, feats)

    # 3-2
    curves = model_count_curves(dev, feats, folds)
    print(f"3-2 完成 {time.time() - t0:.0f}s")

    table.to_csv(OUT / "part3_summary.csv", index=False)
    cvlong.to_csv(OUT / "cv_folds.csv", index=False)
    paired.to_csv(OUT / "paired.csv", index=False)
    trace.to_csv(OUT / "adaboost_weight_trace.csv", index=False)
    seg_w.to_csv(OUT / "adaboost_final_weights.csv", index=False)
    curves.to_csv(OUT / "model_count_curves.csv", index=False)
    plot_parallel_vs_sequential(table, OUT / "fig1_parallel_vs_sequential.png")
    plot_curves(curves, OUT / "fig2_model_count.png")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = ["name", "pr_auc", "roc_auc", "bal_acc_prev", "sens_prev", "spec_prev", "bal_acc", "sens", "brier", "bias2",
            "variance", "flip_rate_pathologic_prev", "valid_pr_auc_mean", "valid_pr_auc_sd"]
    print(table[cols].round(4).to_string(index=False))
    print(paired.pivot_table(index="name", columns=["ref", "metric"], values=["mean_diff", "wins"], sort=False).round(3).to_string())
    print("== AdaBoost 权重轨迹 ==\n", trace.round(3).to_string(index=False))
    print("== 最终权重最高的录音 ==\n", rec_w.head(8).round(4).to_string())
    v = curves[curves.part == "valid"]
    print("== 3-2 验证折（log-loss 最低点 / 各点）==")
    for mname, c in v.groupby("model", sort=False):
        best = c.loc[c.log_loss.idxmin()]
        print(f"{mname:<16} 最低 log-loss {best.log_loss:.3f} @ n={int(best.n)}; "
              + "  ".join(f"n={int(r.n)}: ll={r.log_loss:.3f} pr={r.pr_auc:.3f} brier={r.brier:.4f}" for r in c.itertuples() if r.n in (10, 50, 100, 300, 1000)))
    t = curves[curves.part == "train"]
    print("训练折 log-loss @1000：", t[t.n == 1000].set_index("model").log_loss.round(4).to_dict())
    print(f"Wrote {OUT}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
