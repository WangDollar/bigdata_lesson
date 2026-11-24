"""第一篇补充检查：深度 2 的方差凸起从哪里来？

复用 part1_single_tree.py 的数据、种子与“按录音分层重抽样”，只看 max_depth=2（对照深度 4）。
运行：python src/part1_depth2_check.py
"""
from collections import Counter

import numpy as np
import pandas as pd

from part1_single_tree import LABEL, SEED, load, resamples, tree

train, valid, feats = load()
Xv, yv = valid[feats], valid[LABEL].to_numpy()
samples = resamples(train, "stratified", np.random.default_rng(SEED))


def split_name(t, node):
    return "叶" if t.children_left[node] == -1 else f"{feats[t.feature[node]]}≤{t.threshold[node]:g}"


def fit_all(depth):
    probs, leaf_n, rows = [], [], []
    for idx in samples:
        m = tree(depth).fit(train.iloc[idx][feats], train.iloc[idx][LABEL])
        t = m.tree_
        leaves = m.apply(Xv)
        probs.append(m.predict_proba(Xv)[:, 1])
        leaf_n.append(t.n_node_samples[leaves])
        rows.append({"root": split_name(t, 0), "left": split_name(t, t.children_left[0]),
                     "right": split_name(t, t.children_right[0])})
    return np.asarray(probs), np.asarray(leaf_n), pd.DataFrame(rows)


p2, n2, s2 = fit_all(2)
p4, _, _ = fit_all(4)
v2, v4 = p2.var(axis=0), p4.var(axis=0)
print(f"平均方差：深度 2 = {v2.mean():.4f}，深度 4 = {v4.mean():.4f}")

print("\n== 假设 1：只有几个样本的小叶节点（如 Max≤234 那片）==")
q = (valid.Mode <= 106) & (valid.Max > 234)
print(f"验证集中 Mode≤106 且 Max>234 的片段：{q.sum()} 个")
small = (n2 < 10).any(axis=0)
print(f"30 棵深度 2 树里，曾落入训练样本 <10 的叶节点的验证片段：{small.sum()} 个，"
      f"贡献深度 2 总方差的 {v2[small].sum() / v2.sum():.1%}")

print("\n== 方差集中在哪些片段 ==")
order = np.argsort(-v2)
cum = np.cumsum(v2[order]) / v2.sum()
for k in (5, 10, 20, 40):
    top = order[:k]
    print(f"方差最大的 {k:>2} 个片段：占深度 2 总方差 {cum[k - 1]:.0%}，其中病理 {int(yv[top].sum())} 个；"
          f"同样这些片段在深度 4 的方差合计占深度 4 总方差 {v4[top].sum() / v4.sum():.0%}")

print("\n== 假设 2：第二层切分在换 ==")
for col in ("root", "left", "right"):
    print(col, Counter(s2[col]).most_common(6))

# 方差最大的 20 个片段：它们在 30 棵树里的概率分布，以及按右支切分特征分组的平均概率
top = order[:20]
right_feat = s2.right.str.split("≤").str[0]
print("\n方差最大的 20 个片段（真值、特征、在 30 棵树里的概率）：")
for i in top:
    vals = p2[:, i]
    by_right = {f: round(vals[(right_feat == f).to_numpy()].mean(), 2) for f in right_feat.unique()}
    print(f"  y={yv[i]} Mode={valid.Mode.iloc[i]:>5} Mean={valid.Mean.iloc[i]:>5} ALTV={valid.ALTV.iloc[i]:>4} "
          f"ASTV={valid.ASTV.iloc[i]:>4} | 深度2 方差={v2[i]:.3f}，深度4 方差={v4[i]:.3f} | "
          f"P 取值 {sorted(Counter(np.round(vals, 2)).items())[:4]} … | 按右支特征分组的平均 P {by_right}")
