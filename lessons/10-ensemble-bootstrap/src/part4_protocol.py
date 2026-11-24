"""第四篇 ② 评价协议：按录音分组 vs 按行随机；OOB 的抽样单位（CTG）

唯一改变的变量：评价协议。模型设置全部固定（取第四篇 ① 嵌套选参的代表性结果）：
  单树 深度 4；逻辑回归 C=1；随机森林 max_features=4、300 棵；梯度提升 树桩、220 轮、学习率 0.1。
  A 按录音分组 5 折 × 5 次重复（不同随机种子）
  B 按行随机分层 5 折 × 5 次重复
OOB：同一套 300 棵树（max_features=4）的随机森林，分别用“按行 bootstrap”和“按录音分层 bootstrap”，
     每个片段只由没抽到它的树来预测。

运行：python src/part4_protocol.py
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from part1_single_tree import BASE, BLUE, GRID, GROUP, INK, INK2, LABEL, ORANGE, SEED, SURF, load, plt
from part2_bagging import proba1, recording_sampler

OUT = BASE / "outputs" / "part4"
N_REPEATS, N_TREES = 5, 300
MODELS = {
    "单树": lambda: DecisionTreeClassifier(max_depth=4, random_state=SEED),
    "逻辑回归": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=5000)),
    "随机森林": lambda: RandomForestClassifier(n_estimators=N_TREES, max_features=4, random_state=SEED, n_jobs=1),
    "梯度提升": lambda: GradientBoostingClassifier(n_estimators=220, max_depth=1, learning_rate=0.1, random_state=SEED),
}


def scores(y, p, prev):
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p), "bal_acc_prev": balanced_accuracy_score(y, p > prev)}


def run_fold(name, protocol, r, k, a, b, X, y):
    m = MODELS[name]().fit(X[a], y[a])
    return {"model": name, "protocol": protocol, "repeat": r, "fold": k, **scores(y[b], proba1(m, X[b]), y[a].mean())}


def oob(X, y, g, unit, seed=SEED):
    """手写 300 棵树的随机森林，记录每棵树没抽到的片段；返回 OOB 预测。"""
    rng = np.random.default_rng(seed)
    draw = recording_sampler(g, y) if unit == "recording" else None
    s, c = np.zeros(len(y)), np.zeros(len(y))
    for _ in range(N_TREES):
        idx = rng.integers(0, len(y), len(y)) if draw is None else draw(rng)[0]
        out = np.ones(len(y), bool)
        out[idx] = False
        t = DecisionTreeClassifier(max_features=4, random_state=int(rng.integers(2**31 - 1))).fit(X[idx], y[idx])
        s[out] += proba1(t, X[out])
        c[out] += 1
    return s / np.maximum(c, 1), c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, valid, feats = load()
    dev = pd.concat([train, valid], ignore_index=True)
    X, y, g = dev[feats].to_numpy(), dev[LABEL].to_numpy(), dev[GROUP].to_numpy()

    jobs = []
    for r in range(N_REPEATS):
        splits = {"按录音分组": StratifiedGroupKFold(5, shuffle=True, random_state=SEED + r).split(X, y, g),
                  "按行随机": StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(X, y)}
        for protocol, sp in splits.items():
            for k, (a, b) in enumerate(sp):
                jobs += [(name, protocol, r, k, a, b) for name in MODELS]
    res = pd.DataFrame(Parallel(n_jobs=-1)(delayed(run_fold)(*j, X, y) for j in jobs))
    res.to_csv(OUT / "protocol_folds.csv", index=False)

    summ = res.groupby(["model", "protocol"], sort=False)[["pr_auc", "roc_auc", "bal_acc_prev"]].mean().unstack("protocol")
    pd.set_option("display.width", 200)
    print("== 25 折均值（5 折 × 5 次重复）==\n", summ.round(3).to_string())
    infl = pd.DataFrame({m: summ[(m, "按行随机")] - summ[(m, "按录音分组")] for m in ("pr_auc", "roc_auc", "bal_acc_prev")})
    print("== 按行随机 − 按录音分组（虚高幅度）==\n", infl.round(3).to_string())

    # 同一位置的折，配对看排名：按行随机会不会改变“谁更好”
    rank = res.groupby(["protocol", "model"]).pr_auc.mean().unstack("model")
    print("== 两种协议下的 PR-AUC 排名 ==\n", rank.rank(axis=1, ascending=False).astype(int).to_string())

    rows = []
    for unit in ("row", "recording"):
        p, c = oob(X, y, g, unit)
        rows.append({"oob_unit": "按行 bootstrap" if unit == "row" else "按录音分层 bootstrap", "trees_per_segment": c.mean(),
                     **scores(y, p, y.mean())})
    oobt = pd.DataFrame(rows)
    rf_cv = summ.loc["随机森林"]
    print("== 随机森林 OOB vs 交叉验证 ==\n", oobt.round(3).to_string(index=False))
    print("  对照：随机森林 按录音分组 CV", rf_cv.xs("按录音分组", level="protocol").round(3).to_dict(),
          "；按行随机 CV", rf_cv.xs("按行随机", level="protocol").round(3).to_dict())
    oobt.to_csv(OUT / "oob.csv", index=False)

    # 图：哑铃图（点不需要从 0 开始的基线，柱子需要）
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    xs = np.arange(len(MODELS))
    for x, m in zip(xs, MODELS):
        a_, b_ = summ.loc[m, ("pr_auc", "按录音分组")], summ.loc[m, ("pr_auc", "按行随机")]
        ax.plot([x, x], [a_, b_], color=GRID, lw=3, zorder=1)
        ax.scatter([x], [a_], s=70, color=BLUE, zorder=3, label="按录音分组 CV" if x == 0 else None)
        ax.scatter([x], [b_], s=70, color=ORANGE, zorder=3, label="按行随机 CV" if x == 0 else None)
        ax.annotate(f"{a_:.3f}", (x, a_), xytext=(9, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
        ax.annotate(f"{b_:.3f}（+{b_ - a_:.3f}）", (x, b_), xytext=(9, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.set_xticks(xs, list(MODELS))
    ax.set_xlim(-0.4, len(MODELS) - 0.2)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.set_title("同一模型、只换评价协议：PR-AUC（25 折均值）", loc="left", fontsize=12, fontweight=600)
    ax = axes[1]
    labels = ["按录音分组\nCV", "OOB\n按录音 bootstrap", "OOB\n按行 bootstrap", "按行随机\nCV"]
    vals = [rf_cv[("pr_auc", "按录音分组")], oobt.pr_auc.iloc[1], oobt.pr_auc.iloc[0], rf_cv[("pr_auc", "按行随机")]]
    for x, (v, c) in enumerate(zip(vals, [BLUE, BLUE, ORANGE, ORANGE])):
        ax.scatter([x], [v], s=80, color=c, zorder=3)
        ax.annotate(f"{v:.3f}", (x, v), xytext=(0, 9), textcoords="offset points", ha="center", color=INK2, fontsize=9.5)
    ax.set_xticks(range(4), labels, fontsize=9)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0.9, 1.0)
    ax.set_title("随机森林：OOB 偏不偏乐观，取决于抽样单位", loc="left", fontsize=12, fontweight=600)
    fig.suptitle("② 评价协议：同一录音的片段同时出现在训练和验证中，分数虚高、排名改变", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_protocol.png", dpi=160)
    plt.close(fig)

if __name__ == "__main__":
    main()
