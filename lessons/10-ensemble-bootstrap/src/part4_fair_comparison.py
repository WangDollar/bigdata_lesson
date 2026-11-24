"""第四篇 ① 公平比较 + ③ 阈值与校准（CTG）

五个模型在同一协议下比较：单树、逻辑回归、RBF-SVM、随机森林、梯度提升。
  外层：开发集按录音分组 5 折（与前几篇同一组折），只用来评价。
  内层：每个外层训练折里再按录音分组 4 折，用内层验证 log-loss 在事先声明的小网格中选参；选好后在整个外层训练折重训。
  网格（事先声明）：单树 max_depth∈{2..6}；LR C∈{0.01,0.1,1,10,100}；SVM C∈{0.1,1,10,100}×gamma∈{scale,0.01,0.1}；
                    RF max_features∈{2,4,10}（300 棵）；GB 深度∈{1,3}×轮数 1..1000（学习率 0.1）。
  LR、SVM 的标准化在 pipeline 内，只在训练部分拟合。SVM 没有概率：按录音分组 3 折做 Platt 校准（ensemble=False），
  不用 SVC(probability=True)——它内部按行做交叉验证，同一录音的片段会泄漏。
③ 用外层预测（每个片段都来自没见过它的模型）检查校准，并比较 0.5 与“训练折病理比例”两个阈值下的漏判 / 误报。

运行：python src/part4_fair_comparison.py
只读 data/ctg/student/，不读测试标签。
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from part1_single_tree import BASE, BLUE, GRID, GROUP, INK, INK2, LABEL, ORANGE, SEED, SURF, load, plt
from part2_bagging import proba1

OUT = BASE / "outputs" / "part4"
INNER, GB_MAX, GB_LR = 4, 1000, 0.1
MODELS = ["单树", "逻辑回归", "RBF-SVM", "随机森林", "梯度提升"]
COLOR = {"逻辑回归": BLUE, "RBF-SVM": ORANGE, "随机森林": "#1baf7a", "梯度提升": "#eda100"}


def logloss(y, p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()


def grouped_splits(X, y, g, k):
    return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED).split(X, y, g))


class Plain:
    def __init__(self, est):
        self.est = est

    def fit(self, X, y, g):
        self.est.fit(X, y)
        return self

    def proba(self, X):
        return proba1(self.est, X)


class GroupCalibratedSVM:
    def __init__(self, C, gamma):
        self.C, self.gamma = C, gamma

    def fit(self, X, y, g):
        base = make_pipeline(StandardScaler(), SVC(C=self.C, gamma=self.gamma, kernel="rbf"))
        self.m = CalibratedClassifierCV(base, method="sigmoid", cv=grouped_splits(X, y, g, 3), ensemble=False).fit(X, y)
        return self

    def proba(self, X):
        return self.m.predict_proba(X)[:, 1]


SPECS = {
    "单树": [({"max_depth": d}, lambda d=d: Plain(DecisionTreeClassifier(max_depth=d, random_state=SEED))) for d in (2, 3, 4, 5, 6)],
    "逻辑回归": [({"C": C}, lambda C=C: Plain(make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))))
             for C in (0.01, 0.1, 1, 10, 100)],
    "RBF-SVM": [({"C": C, "gamma": gm}, lambda C=C, gm=gm: GroupCalibratedSVM(C, gm))
                for C in (0.1, 1, 10, 100) for gm in ("scale", 0.01, 0.1)],
    "随机森林": [({"max_features": k}, lambda k=k: Plain(RandomForestClassifier(n_estimators=300, max_features=k,
                                                                              random_state=SEED, n_jobs=1))) for k in (2, 4, 10)],
}


def gb_make(depth, rounds):
    return Plain(GradientBoostingClassifier(n_estimators=rounds, learning_rate=GB_LR, max_depth=depth, random_state=SEED))


def select(model, X, y, g):
    """内层按录音分组 4 折，按平均验证 log-loss 选参；返回 (参数, 构造函数, 内层 log-loss)。"""
    inner = grouped_splits(X, y, g, INNER)
    if model == "梯度提升":
        best = None
        for d in (1, 3):
            curves = []
            for ia, ib in inner:
                gb = GradientBoostingClassifier(n_estimators=GB_MAX, learning_rate=GB_LR, max_depth=d, random_state=SEED).fit(X[ia], y[ia])
                curves.append([logloss(y[ib], p[:, 1]) for p in gb.staged_predict_proba(X[ib])])
            c = np.mean(curves, axis=0)
            n = int(np.argmin(c)) + 1
            if best is None or c[n - 1] < best[2]:
                best = ({"depth": d, "rounds": n}, lambda d=d, n=n: gb_make(d, n), c[n - 1])
        return best
    scores = [np.mean([logloss(y[ib], make().fit(X[ia], y[ia], g[ia]).proba(X[ib])) for ia, ib in inner]) for _, make in SPECS[model]]
    i = int(np.argmin(scores))
    return SPECS[model][i][0], SPECS[model][i][1], scores[i]


def outer_job(model, k, a, b, X, y, g):
    params, make, inner_ll = select(model, X[a], y[a], g[a])
    p = make().fit(X[a], y[a], g[a]).proba(X[b])
    return {"model": model, "fold": k, "params": params, "inner_logloss": inner_ll, "idx": b, "p": p, "prev": y[a].mean()}


def fold_metrics(y, p, prev):
    out = {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p), "log_loss": logloss(y, p),
           "brier": ((p - y) ** 2).mean()}
    for tag, thr in (("", 0.5), ("_prev", prev)):
        pred = p > thr
        out[f"bal_acc{tag}"] = balanced_accuracy_score(y, pred)
        out[f"missed{tag}"] = int((~pred & (y == 1)).sum())
        out[f"false_alarm{tag}"] = int((pred & (y == 0)).sum())
    return out


BIN_EDGES = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0 + 1e-9])   # 固定边界：中间段也看得见


def reliability(y, p, min_n=10):
    which = np.clip(np.searchsorted(BIN_EDGES, p, side="right") - 1, 0, len(BIN_EDGES) - 2)
    r = pd.DataFrame({"pred": p, "obs": y, "bin": which}).groupby("bin").agg(pred=("pred", "mean"), obs=("obs", "mean"), n=("obs", "size"))
    return r[r.n >= min_n]


# ---------- 图 ----------
def plot_comparison(res, path):
    xs = np.arange(len(MODELS))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, (m, title) in zip(axes, [("pr_auc", "PR-AUC（与阈值无关）"), ("bal_acc_prev", "Balanced accuracy（阈值 = 训练折病理比例）")]):
        for x, name in zip(xs, MODELS):
            v = res.loc[res.model == name, m]
            ax.scatter(np.full(len(v), x), v, s=20, color=GRID, edgecolor=INK2, linewidth=0.6, zorder=2)
            ax.plot([x - 0.3, x + 0.3], [v.mean()] * 2, color=INK, lw=2.2, zorder=3)
            ax.annotate(f"{v.mean():.3f}", (x + 0.33, v.mean()), va="center", color=INK2, fontsize=9.5,
                        bbox=dict(boxstyle="round,pad=0.12", fc=SURF, ec="none"), zorder=4)
        ax.set_xticks(xs, MODELS)
        ax.set_xlim(-0.5, len(MODELS) - 0.3)
        ax.set_title(title, loc="left", fontsize=12, fontweight=600)
    fig.suptitle("① 公平比较：五个模型都在训练折内部嵌套选参（点 = 外层各折，横线 = 均值）", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_calibration_threshold(oof, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, zorder=1)
    for name, color in COLOR.items():
        d = oof[oof.model == name]
        r = reliability(d.y.to_numpy(), d.p.to_numpy())
        ax.plot(r.pred, r.obs, color=color, marker="o", ms=5, label=name)
    ax.set_xlabel("预测的病理概率（分箱均值）")
    ax.set_ylabel("实际病理比例")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("校准：点越贴近对角线，概率越可信（固定分箱，n≥10）", loc="left", fontsize=12, fontweight=600)

    ax = axes[1]
    for name, color in COLOR.items():
        d = oof[oof.model == name]
        fpr, tpr, _ = roc_curve(d.y, d.p)
        ax.plot(fpr * 100, tpr * 100, color=color, lw=2, label=name)
        for col, mk in (("pred05", "o"), ("pred_prev", "s")):
            sens = d.loc[d.y == 1, col].mean() * 100
            fa = d.loc[d.y == 0, col].mean() * 100
            ax.scatter([fa], [sens], marker=mk, s=60, color=color, edgecolor=SURF, linewidth=1.5, zorder=4)
    ax.scatter([], [], marker="o", s=50, color=INK2, label="阈值 0.5")
    ax.scatter([], [], marker="s", s=50, color=INK2, label="阈值 = 病理比例")
    ax.set_xlim(0, 25)
    ax.set_ylim(50, 101)
    ax.set_xlabel("误报率：非病理片段被判为病理（%）")
    ax.set_ylabel("敏感度：病理片段被抓出（%）")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("阈值取舍：同一条曲线上换位置", loc="left", fontsize=12, fontweight=600)
    fig.suptitle("③ 阈值与校准（1703 个片段的外层预测，每个片段都来自没见过它的模型）", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    train, valid, feats = load()
    dev = pd.concat([train, valid], ignore_index=True)
    X, y, g = dev[feats].to_numpy(), dev[LABEL].to_numpy(), dev[GROUP].to_numpy()
    outer = grouped_splits(X, y, g, 5)

    jobs = Parallel(n_jobs=-1)(delayed(outer_job)(m, k, a, b, X, y, g) for m in MODELS for k, (a, b) in enumerate(outer))
    print(f"嵌套选参完成 {time.time() - t0:.0f}s")

    rows, oof = [], []
    for j in jobs:
        rows.append({"model": j["model"], "fold": j["fold"], "params": j["params"], "inner_logloss": j["inner_logloss"],
                     **fold_metrics(y[j["idx"]], j["p"], j["prev"])})
        oof.append(pd.DataFrame({"sample_id": dev.sample_id.to_numpy()[j["idx"]], "model": j["model"], "fold": j["fold"],
                                 "y": y[j["idx"]], "p": j["p"], "pred05": j["p"] > 0.5, "pred_prev": j["p"] > j["prev"]}))
    res, oof = pd.DataFrame(rows), pd.concat(oof, ignore_index=True)
    res.to_csv(OUT / "fair_comparison_folds.csv", index=False)
    oof.to_csv(OUT / "oof_predictions.csv", index=False)

    pd.set_option("display.width", 220)
    print("== 每折选出的参数 ==")
    for m in MODELS:
        print(f"  {m:<6}", [str(p) for p in res.loc[res.model == m, "params"]])
    summ = res.groupby("model", sort=False)[["pr_auc", "roc_auc", "bal_acc_prev", "bal_acc", "log_loss", "brier",
                                              "missed_prev", "false_alarm_prev", "missed", "false_alarm"]]
    tab = summ.mean()
    tab[["missed_prev", "false_alarm_prev", "missed", "false_alarm"]] = summ.sum()[["missed_prev", "false_alarm_prev", "missed", "false_alarm"]]
    tab["pr_auc_sd"] = res.groupby("model", sort=False).pr_auc.std()
    print("== 外层 5 折（漏判 / 误报为 5 折合计；共 142 个病理、1561 个非病理）==\n", tab.round(4).to_string())
    tab.to_csv(OUT / "fair_comparison_summary.csv")

    print("== 逐折配对（A − B：差值均值，A 更好的折数）==")
    pairs = [("随机森林", "逻辑回归"), ("梯度提升", "逻辑回归"), ("RBF-SVM", "逻辑回归"), ("梯度提升", "随机森林"),
             ("逻辑回归", "单树"), ("随机森林", "单树"), ("梯度提升", "单树")]
    w = {m: res.pivot(index="fold", columns="model", values=m) for m in ("pr_auc", "bal_acc_prev", "log_loss")}
    for a_, b_ in pairs:
        parts = []
        for m, sign in (("pr_auc", 1), ("bal_acc_prev", 1), ("log_loss", -1)):
            d = w[m][a_] - w[m][b_]
            parts.append(f"{m} {d.mean():+.3f}（{int((d * sign > 0).sum())}/5）")
        print(f"  {a_} − {b_}: " + " | ".join(parts))

    print("== 校准：整体平均预测 vs 实际病理比例（外层预测）==")
    for m in MODELS:
        d = oof[oof.model == m]
        r = reliability(d.y.to_numpy(), d.p.to_numpy())
        top = r.iloc[-1]
        print(f"  {m:<6} 平均预测 {d.p.mean():.3f} vs 实际 {d.y.mean():.3f}；最高分箱：预测 {top.pred:.2f} vs 实际 {top.obs:.2f}（n={int(top.n)}）")

    plot_comparison(res, OUT / "fig1_fair_comparison.png")
    plot_calibration_threshold(oof, OUT / "fig3_calibration_threshold.png")
    print(f"Wrote {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
