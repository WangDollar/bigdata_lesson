"""第四篇 ④ 特征重要性：不纯度重要性 vs 置换重要性 vs 相关簇整体置换（CTG）

模型取第四篇 ① 嵌套选参的多数结果：随机森林 max_features=10（300 棵）、梯度提升 树桩 220 轮。
在 train（1279 段）上训练，在 validation（424 段，34 个病理）上置换，评分用 PR-AUC，重复 30 次。
相关簇：train 上按 1−|r| 做层次聚类（平均连接），|r| ≥ 0.7 的特征归为一簇；簇内特征用同一个行排列一起打乱。

运行：python src/part4_importance.py
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score

from part1_single_tree import BASE, BLUE, GRID, INK, INK2, LABEL, ORANGE, SEED, SURF, load, plt

OUT = BASE / "outputs" / "part4"
N_REPEATS = 30


def permutation_drop(model, X, y, cols, rng):
    base = average_precision_score(y, model.predict_proba(X)[:, 1])
    drops = []
    for _ in range(N_REPEATS):
        Xp = X.copy()
        Xp[:, cols] = X[rng.permutation(len(X))][:, cols]      # 簇内特征用同一个排列：保留簇内相关，只切断与标签的关系
        drops.append(base - average_precision_score(y, model.predict_proba(Xp)[:, 1]))
    return np.mean(drops), np.std(drops, ddof=1)


def main():
    train, valid, feats = load()
    Xtr, ytr, Xv, yv = train[feats].to_numpy(), train[LABEL].to_numpy(), valid[feats].to_numpy(), valid[LABEL].to_numpy()

    corr = np.corrcoef(Xtr, rowvar=False)
    dist = squareform(1 - np.abs(corr), checks=False)
    cl = fcluster(linkage(dist, method="average"), t=0.3, criterion="distance")
    clusters = {c: [feats[i] for i in np.where(cl == c)[0]] for c in np.unique(cl)}
    multi = {c: f for c, f in clusters.items() if len(f) > 1}
    print("== |r| ≥ 0.7 的相关簇 ==", list(multi.values()))

    models = {"随机森林": RandomForestClassifier(n_estimators=300, max_features=10, random_state=SEED, n_jobs=-1).fit(Xtr, ytr),
              "梯度提升": GradientBoostingClassifier(n_estimators=220, max_depth=1, learning_rate=0.1, random_state=SEED).fit(Xtr, ytr)}
    rows, crow = [], []
    for name, m in models.items():
        print(f"{name} 验证集 PR-AUC = {average_precision_score(yv, m.predict_proba(Xv)[:, 1]):.3f}")
        rng = np.random.default_rng(SEED)
        for j, f in enumerate(feats):
            mu, sd = permutation_drop(m, Xv, yv, [j], rng)
            rows.append({"model": name, "feature": f, "cluster": "+".join(clusters[cl[j]]) if cl[j] in multi else "",
                         "impurity": m.feature_importances_[j], "perm_drop": mu, "perm_sd": sd})
        for c, fs in multi.items():
            cols = [feats.index(f) for f in fs]
            mu, sd = permutation_drop(m, Xv, yv, cols, rng)
            indiv = sum(r["perm_drop"] for r in rows if r["model"] == name and r["feature"] in fs)
            crow.append({"model": name, "cluster": "+".join(fs), "joint_drop": mu, "joint_sd": sd, "sum_individual_drops": indiv,
                         "sum_impurity": sum(r["impurity"] for r in rows if r["model"] == name and r["feature"] in fs)})
    imp, cimp = pd.DataFrame(rows), pd.DataFrame(crow)
    imp.to_csv(OUT / "feature_importance.csv", index=False)
    cimp.to_csv(OUT / "cluster_importance.csv", index=False)

    pd.set_option("display.width", 200)
    for name in models:
        d = imp[imp.model == name].copy()
        d["impurity_rank"] = d.impurity.rank(ascending=False).astype(int)
        d["perm_rank"] = d.perm_drop.rank(ascending=False).astype(int)
        print(f"== {name}：按不纯度重要性排序（前 10）==\n",
              d.sort_values("impurity", ascending=False).head(10)[["feature", "cluster", "impurity", "impurity_rank", "perm_drop", "perm_sd", "perm_rank"]]
              .round(4).to_string(index=False))
        print(f"   按置换重要性的前 5：", d.sort_values("perm_drop", ascending=False).head(5).feature.tolist())
    print("== 相关簇：整体打乱 vs 单个打乱之和 ==\n", cimp.round(4).to_string(index=False))

    # 图：随机森林，三个视角
    d = imp[imp.model == "随机森林"].sort_values("impurity", ascending=True).tail(12)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    labels = [f"{f}{' *' if c else ''}" for f, c in zip(d.feature, d.cluster)]
    axes[0].barh(labels, d.impurity, color=BLUE, height=0.55)
    axes[0].set_title("不纯度重要性（训练时）", loc="left", fontsize=12, fontweight=600)
    axes[1].barh(labels, d.perm_drop, xerr=d.perm_sd, color=BLUE, height=0.55, error_kw={"ecolor": INK2, "lw": 1})
    axes[1].axvline(0, color=INK2, lw=0.8)
    axes[1].set_title("单个置换：验证集 PR-AUC 下降", loc="left", fontsize=12, fontweight=600)
    c = cimp[cimp.model == "随机森林"].sort_values("joint_drop")
    yy = np.arange(len(c))
    axes[2].barh(yy + 0.18, c.sum_individual_drops, height=0.34, color=ORANGE, label="簇内特征各自打乱，下降相加")
    axes[2].barh(yy - 0.18, c.joint_drop, height=0.34, color=BLUE, label="整簇一起打乱")
    axes[2].set_yticks(yy, c.cluster.str.replace("+", "\n"), fontsize=9)
    axes[2].legend(loc="lower right", fontsize=9)
    axes[2].set_title("相关簇：整体 vs 拆开", loc="left", fontsize=12, fontweight=600)
    for ax in axes:
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
    fig.suptitle("④ 特征重要性（随机森林；* = 属于 |r|≥0.7 的相关簇）：三种算法，三种答案", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_importance.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
