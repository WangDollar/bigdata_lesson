"""第四篇补充：(1) 随机森林为什么没稳定赢过逻辑回归？(2) 阈值怎么定？

(1a) 统计功效：用 ② 中“按录音分组 5 折 × 5 次重复”的 25 折（参数固定）做逐折配对。
(1b) 检验推断“信号可相加、但每个特征是阈值型”：给逻辑回归加样条特征（先分位数变换，再三次样条，
     仍是“每个特征一条曲线、再相加”，不含交互），与 ① 同一外层折、同一嵌套选参规则（C∈{0.01,0.1,1,10}，内层 log-loss）。
(2)  决策论阈值：若漏判代价 / 误报代价 = r，且概率可靠，则最优阈值 t = 1 / (1 + r)。
     用 ① 的外层预测，对 r = 1, 3, 5, 11, 20, 50 列出漏判 / 误报。阈值由 r 决定，不看标签，不存在偷看。

运行：python src/part4_followup.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, SplineTransformer

import part4_fair_comparison as fc
from part1_single_tree import BASE, GROUP, LABEL, SEED, load

OUT = BASE / "outputs" / "part4"


def paired(w, a, b, metric, higher=True):
    d = w[a] - w[b]
    return f"{metric} {d.mean():+.3f}（{a} 更好 {int(((d if higher else -d) > 0).sum())}/{len(d)} 折）"


def main():
    pd.set_option("display.width", 200)
    # (1a)
    pf = pd.read_csv(OUT / "protocol_folds.csv")
    g = pf[pf.protocol == "按录音分组"]
    g = g.assign(key=g.repeat.astype(str) + "-" + g.fold.astype(str))
    print("== (1a) 参数固定、按录音分组 25 折的逐折配对 ==")
    for m in ("pr_auc", "bal_acc_prev", "roc_auc"):
        w = g.pivot(index="key", columns="model", values=m)
        print("  " + " | ".join(paired(w, a, b, m) for a, b in (("随机森林", "逻辑回归"), ("梯度提升", "逻辑回归"), ("梯度提升", "随机森林"))))

    # (1b)
    fc.SPECS["样条逻辑回归"] = [({"C": C}, lambda C=C: fc.Plain(make_pipeline(
        QuantileTransformer(n_quantiles=200, output_distribution="uniform"),
        SplineTransformer(n_knots=5, degree=3, knots="uniform"),
        LogisticRegression(C=C, max_iter=5000)))) for C in (0.01, 0.1, 1, 10)]
    train, valid, feats = load()
    dev = pd.concat([train, valid], ignore_index=True)
    X, y, gr = dev[feats].to_numpy(), dev[LABEL].to_numpy(), dev[GROUP].to_numpy()
    rows = []
    for k, (a, b) in enumerate(fc.grouped_splits(X, y, gr, 5)):           # 与 ① 完全相同的外层折
        j = fc.outer_job("样条逻辑回归", k, a, b, X, y, gr)
        rows.append({"model": "样条逻辑回归", "fold": k, "params": j["params"], **fc.fold_metrics(y[b], j["p"], j["prev"])})
    spl = pd.DataFrame(rows)
    res = pd.concat([pd.read_csv(OUT / "fair_comparison_folds.csv"), spl], ignore_index=True)
    print("\n== (1b) 样条逻辑回归（同一外层折、同一嵌套规则）==")
    print("  选出的参数：", spl.params.tolist())
    print(res.groupby("model", sort=False)[["pr_auc", "bal_acc_prev", "log_loss"]].mean().round(3).to_string())
    for m, hi in (("pr_auc", True), ("bal_acc_prev", True), ("log_loss", False)):
        w = res.pivot(index="fold", columns="model", values=m)
        print("  " + " | ".join(paired(w, a, b, m, hi) for a, b in (("样条逻辑回归", "逻辑回归"), ("样条逻辑回归", "随机森林"), ("梯度提升", "样条逻辑回归"))))

    # (2)
    oof = pd.read_csv(OUT / "oof_predictions.csv")
    n_pos, n_neg = int(oof[oof.model == "梯度提升"].y.sum()), int((oof[oof.model == "梯度提升"].y == 0).sum())
    print(f"\n== (2) 代价比 r → 阈值 t = 1/(1+r) → 漏判 / 误报（外层预测合计：{n_pos} 个病理，{n_neg} 个非病理）==")
    rs = (1, 3, 5, 11, 20, 50)
    tab = []
    for m in ("逻辑回归", "随机森林", "梯度提升"):
        d = oof[oof.model == m]
        for r in rs:
            t = 1 / (1 + r)
            pred = d.p > t
            fn, fp = int((~pred & (d.y == 1)).sum()), int((pred & (d.y == 0)).sum())
            tab.append({"model": m, "r": r, "threshold": round(t, 3), "missed": fn, "false_alarm": fp,
                        "sensitivity": 1 - fn / n_pos, "specificity": 1 - fp / n_neg})
    tab = pd.DataFrame(tab)
    tab.to_csv(OUT / "cost_ratio_thresholds.csv", index=False)
    print(tab.pivot(index="r", columns="model", values=["threshold", "missed", "false_alarm"]).to_string())


if __name__ == "__main__":
    main()
