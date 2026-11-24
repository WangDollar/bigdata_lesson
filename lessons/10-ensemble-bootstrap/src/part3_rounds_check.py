"""第三篇补充：梯度提升的轮数（和深度）该怎么定？——默认 vs 嵌套选参 vs 偷看外层

外层：开发集按录音分组 5 折（与前几篇同一组折），只用来评价。
内层：在每个外层训练折里再按录音分组 4 折，用内层验证 log-loss 选 (深度 ∈ {1, 3}, 轮数 1..1000)，学习率固定 0.1；
      选好后在整个外层训练折上重训，到外层折评价。外层折从不参与选择。
对照：默认（深度 3, 100 轮）、树桩默认（深度 1, 100 轮）、“偷看”（直接按外层 CV 曲线挑的深度 3, 50 轮）。

运行：python src/part3_rounds_check.py
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from part1_single_tree import BASE, GROUP, LABEL, SEED, load

OUT = BASE / "outputs" / "part3"
MAX_ROUNDS, LR, DEPTHS = 1000, 0.1, (1, 3)


def logloss(y, p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()


def inner_curve(X, y, a, b, depth):
    gb = GradientBoostingClassifier(n_estimators=MAX_ROUNDS, learning_rate=LR, max_depth=depth, random_state=SEED).fit(X[a], y[a])
    return np.array([logloss(y[b], p[:, 1]) for p in gb.staged_predict_proba(X[b])])


def evaluate(X, y, g, a, b, depth, rounds):
    gb = GradientBoostingClassifier(n_estimators=rounds, learning_rate=LR, max_depth=depth, random_state=SEED).fit(X[a], y[a])
    p = gb.predict_proba(X[b])[:, 1]
    return {"pr_auc": average_precision_score(y[b], p), "bal_acc_prev": balanced_accuracy_score(y[b], p > y[a].mean()),
            "log_loss": logloss(y[b], p), "brier": ((p - y[b]) ** 2).mean()}


def main():
    train, valid, feats = load()
    dev = pd.concat([train, valid], ignore_index=True)
    X, y, g = dev[feats].to_numpy(), dev[LABEL].to_numpy(), dev[GROUP].to_numpy()
    outer = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, g))

    # 内层：每个外层训练折 × 每个深度 × 4 个内层折的验证 log-loss 曲线
    jobs = []
    for k, (a, _) in enumerate(outer):
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED).split(X[a], y[a], g[a])
        for ia, ib in inner:
            for d in DEPTHS:
                jobs.append((k, d, a[ia], a[ib]))
    curves = Parallel(n_jobs=-1)(delayed(inner_curve)(X, y, ia, ib, d) for _, d, ia, ib in jobs)
    mean_curve = {}
    for (k, d, _, _), c in zip(jobs, curves):
        mean_curve.setdefault((k, d), []).append(c)
    mean_curve = {key: np.mean(v, axis=0) for key, v in mean_curve.items()}

    rows, picks = [], []
    for k, (a, b) in enumerate(outer):
        best = {d: (int(np.argmin(mean_curve[(k, d)])) + 1, mean_curve[(k, d)].min()) for d in DEPTHS}
        d_star = min(DEPTHS, key=lambda d: best[d][1])
        picks.append({"fold": k, **{f"depth{d}_best_rounds": best[d][0] for d in DEPTHS},
                      **{f"depth{d}_inner_logloss": best[d][1] for d in DEPTHS}, "chosen_depth": d_star})
        settings = {"默认（深度3, 100 轮）": (3, 100), "树桩默认（深度1, 100 轮）": (1, 100),
                    "嵌套：深度3, 选轮数": (3, best[3][0]), "嵌套：深度+轮数都选": (d_star, best[d_star][0]),
                    "偷看外层曲线（深度3, 50 轮）": (3, 50)}
        for name, (d, n) in settings.items():
            rows.append({"fold": k, "setting": name, "depth": d, "rounds": n, **evaluate(X, y, g, a, b, d, n)})
    res, picks = pd.DataFrame(rows), pd.DataFrame(picks)
    res.to_csv(OUT / "gb_rounds_nested.csv", index=False)

    pd.set_option("display.width", 200)
    print("== 每个外层折里，内层选出的轮数 ==\n", picks.round(4).to_string(index=False))
    summ = res.groupby("setting", sort=False)[["pr_auc", "bal_acc_prev", "log_loss", "brier"]].mean()
    print("== 外层 5 折均值 ==\n", summ.round(4).to_string())
    ref = res[res.setting == "默认（深度3, 100 轮）"].set_index("fold")
    print("== 与默认逐折配对（差值均值 / 更好的折数）==")
    for name, cur in res.groupby("setting", sort=False):
        cur = cur.set_index("fold")
        parts = []
        for m, better in (("pr_auc", 1), ("bal_acc_prev", 1), ("log_loss", -1)):
            d = cur[m] - ref[m]
            parts.append(f"{m} {d.mean():+.3f}（更好 {int(((d * better) > 0).sum())}/5）")
        print(f"  {name:<22} " + " | ".join(parts))


if __name__ == "__main__":
    main()
