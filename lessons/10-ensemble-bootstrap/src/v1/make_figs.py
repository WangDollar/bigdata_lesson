"""重画第 10 讲提交文档的全部图（数据来自 results/，先运行 ctg_experiments.py）。运行: python -I make_figs.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

WORK = Path(__file__).resolve().parent; RES = WORK / 'results'
FIG = Path('/Users/dollar/Desktop/note/实践课作业/10. Ensemble/作业提交/figures'); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'savefig.dpi': 220,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
                     'xtick.major.width': .5, 'ytick.major.width': .5, 'axes.edgecolor': '#52514e',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b'})
# 参考调色板（dataviz 校验通过：前三槽 all-pairs；顺序蓝阶 ordinal）
BAG, RF, TREE, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'   # 实体颜色固定：bagging 蓝、随机森林 橙、单树 青
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#9a9893', '#e4e3df'
BLUES = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
J = lambda n: json.loads((RES / f'{n}.json').read_text())


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches='tight', pad_inches=.03); plt.close(fig); print('saved', name)


def ygrid(ax):
    ax.grid(axis='y', color=GRID, lw=.5); ax.set_axisbelow(True)


for old in FIG.glob('*.png'): old.unlink()

# ------------------------------------------------------------ 图1 决策树：深度 2 示例 + 深度对照
A = J('A'); nodes = A['depth2_train_nodes']
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.75), gridspec_kw=dict(width_ratios=[1.35, 1]))
a1.set_xlim(0, 10); a1.set_ylim(.4, 6.75); a1.axis('off')


def box(ax, x, yy, text, w=2.5, h=.95, leaf=False, hot=False):
    fc = '#fbe3d8' if hot else ('#eef4fc' if leaf else '#ffffff')
    ax.add_patch(FancyBboxPatch((x - w / 2, yy - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc=fc, ec=INK2, lw=.6))
    ax.text(x, yy, text, ha='center', va='center', fontsize=6.3, color=INK, linespacing=1.35)


def edge(ax, x0, y0, x1, y1, lab):
    ax.plot([x0, x1], [y0, y1], color=INK2, lw=.6)
    ax.text((x0 + x1) / 2 + (-.18 if x1 < x0 else .18), (y0 + y1) / 2 + .05, lab, fontsize=5.8, color=INK2,
            ha='right' if x1 < x0 else 'left')


r, L, R = nodes[0], nodes[1], nodes[4]
leaf = {k: nodes[k] for k in (2, 3, 5, 6)}
txt = lambda n: f"path {n['n_pos']}\nother {n['n_neg']}\nP = {n['n_pos'] / n['n']:.2f}"
box(a1, 5, 5.6, f"Mode ≤ {r['threshold']:.0f} ?\n{r['n']} segments, path {r['n_pos']}", w=3.1)
box(a1, 2.5, 3.75, f"Max ≤ {L['threshold']:.0f} ?\n{L['n']} segments")
box(a1, 7.5, 3.75, f"ALTV ≤ {R['threshold']:.1f} ?\n{R['n']} segments")
for x, k in [(1.3, 2), (3.7, 3), (6.3, 5), (8.7, 6)]:
    box(a1, x, 1.45, txt(leaf[k]), w=1.75, h=1.25, leaf=True, hot=leaf[k]['n_pos'] / leaf[k]['n'] > .5)
edge(a1, 5, 5.12, 2.5, 4.23, 'yes'); edge(a1, 5, 5.12, 7.5, 4.23, 'no')
edge(a1, 2.5, 3.27, 1.3, 2.08, 'yes'); edge(a1, 2.5, 3.27, 3.7, 2.08, 'no')
edge(a1, 7.5, 3.27, 6.3, 2.08, 'yes'); edge(a1, 7.5, 3.27, 8.7, 2.08, 'no')
a1.text(0, 6.75, '(a) depth-2 tree, 1,279 training segments', fontsize=8, va='top')
dt = pd.DataFrame(A['depth_table']); dt['x'] = range(len(dt))
a2.plot(dt.x, dt.train_ba, color=MUTED, lw=1.2, ls='--', marker='o', ms=3.2, mfc='white', label='training')
a2.plot(dt.x, dt.val_ba, color=INK, lw=1.3, marker='o', ms=3.2, label='validation')
a2.set_xticks(dt.x, [str(d) for d in dt.max_depth]); a2.set(xlabel='max_depth', ylabel='balanced accuracy', ylim=(.55, 1.02))
a2.set_title('(b) deeper trees: training keeps rising', loc='left'); ygrid(a2)
a2.text(dt.x.iloc[-1], dt.train_ba.iloc[-1] + .012, 'training', ha='right', va='bottom', fontsize=6.3, color=INK2)
a2.text(dt.x.iloc[-1], dt.val_ba.iloc[-1] - .02, 'validation', ha='right', va='top', fontsize=6.3, color=INK)
save(fig, 'fig1-tree.png')

# ------------------------------------------------------------ 图2 方差公式：平均后的方差随 B 下降，地板是 ρσ²
B = J('B'); vd = B['variance_decomposition']; cv = pd.read_csv(RES / 'B-variance-curves.csv')
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.55))
for name, col, lab in [('bagging', BAG, 'bagging (all 21 features)'), ('rf_mf4', RF, 'random forest (4 of 21)')]:
    d = cv[cv.config == name]; v = vd[name]
    a1.plot(d.B, d.formula, color=col, lw=1.2, alpha=.55)
    a1.plot(d.B, d.observed, color=col, lw=0, marker='o', ms=3.6, label=lab)
    a1.axhline(v['floor'], color=col, lw=.7, ls=':')
    a1.text(100, v['floor'] * .86, f"floor ρσ² = {v['floor']:.5f}  (ρ = {v['rho']:.3f})", fontsize=6, color=INK2,
            va='top', ha='right')
a1.set_xscale('log'); a1.set_yscale('log'); a1.set_xticks([1, 2, 5, 10, 20, 50, 100], ['1', '2', '5', '10', '20', '50', '100'])
a1.set(xlabel='number of trees averaged (B)', ylabel='variance of averaged prediction', ylim=(4e-4, 4e-2))
a1.set_title('(a) dots = measured, line = ρσ² + (1−ρ)σ²/B', loc='left'); a1.legend(frameon=False, loc='upper right')
a1.grid(axis='y', color=GRID, lw=.5, which='both'); a1.set_axisbelow(True)
mf = [1, 2, 4, 8, 13, 21]; s2 = [vd[f'rf_mf{k}']['sigma2'] for k in mf]; v100 = [vd[f'rf_mf{k}']['observed']['100'] for k in mf]
rho = [vd[f'rf_mf{k}']['rho'] for k in mf]
a2.plot(mf, s2, color=MUTED, lw=1.2, ls='--', marker='o', ms=3.4, mfc='white', label='one tree  (σ²)')
a2.plot(mf, v100, color=INK, lw=1.3, marker='o', ms=3.4, label='forest of 100 trees')
for k, a, b_ in zip(mf, v100, rho):
    a2.annotate(f'ρ={b_:.3f}', (k, a), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=5.6, color=INK2)
a2.set_yscale('log'); a2.set_xscale('log', base=2); a2.minorticks_off()
a2.set_xticks(mf, ['1', '2', '4\n(sqrt)', '8', '13', '21\n(=bagging)'])
a2.set(xlabel='max_features (candidate features per split)', ylabel='variance of prediction')
a2.set_title('(b) fewer candidate features: σ² up, ρ down', loc='left'); a2.legend(frameon=False, loc='center left')
a2.grid(axis='y', color=GRID, lw=.5, which='both'); a2.set_axisbelow(True)
save(fig, 'fig2-variance.png')

# ------------------------------------------------------------ 图3 bagging 与随机森林：树数曲线 + 根节点
D = J('D'); tc = pd.read_csv(RES / 'D-tree-count.csv')
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw=dict(width_ratios=[1.25, 1]))
for name, col, lab in [('bagging', BAG, 'bagging'), ('random_forest', RF, 'random forest')]:
    d = tc[tc.model == name]
    a1.plot(d.n_trees, d.val_auc, color=col, lw=1.3, marker='o', ms=3.2, label=lab)
a1.set_xscale('log'); a1.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500], ['1', '2', '5', '10', '20', '50', '100', '200', '500'])
a1.set(xlabel='number of trees', ylabel='validation ROC-AUC', ylim=(.82, 1.005), xlim=(.8, 650))
a1.text(30, .9985, 'random forest', fontsize=6.3, color=INK, va='bottom', ha='center')
a1.text(30, .918, 'bagging', fontsize=6.3, color=INK, va='top', ha='center')
a1.set_title('(a) validation AUC vs number of trees', loc='left'); ygrid(a1)
a1.legend(frameon=False, loc='lower right')
rb, rr = pd.Series(D['root_feature_bagging']), pd.Series(D['root_feature_rf'])
feats = list(rr.sort_values(ascending=True).index)
yy = np.arange(len(feats)); h = .38
a2.barh(yy + h / 2, [rb.get(f, 0) for f in feats], height=h, color=BAG, label='bagging', edgecolor='white', lw=.5)
a2.barh(yy - h / 2, [rr.get(f, 0) for f in feats], height=h, color=RF, label='random forest', edgecolor='white', lw=.5)
a2.set_yticks(yy, feats); a2.set(xlabel='trees (of 100) splitting on it at the root')
a2.set_title('(b) root split of each tree', loc='left'); a2.legend(frameon=False, loc='lower right')
a2.grid(axis='x', color=GRID, lw=.5); a2.set_axisbelow(True)
save(fig, 'fig3-bagging-vs-rf.png')

# ------------------------------------------------------------ 图4 AdaBoost：被跟踪样本的权重
E = J('E'); W = pd.read_csv(RES / 'E-adaboost-weights.csv', index_col=0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.4), gridspec_kw=dict(width_ratios=[1.4, 1]))
cols = [BAG, RF, TREE, C4, '#e87ba4']
tracked = E['adaboost_tracked']
for (sid, col), t in zip(zip(W.columns, cols), tracked):
    a1.plot(W.index, W[sid], color=col, lw=1.3, marker='o', ms=3)
    miss = [i + 1 for i, m in enumerate(t['missed']) if m]
    a1.plot(miss, W[sid].iloc[miss], lw=0, marker='o', ms=6.5, mfc='none', mec=INK, mew=.7)
    lab = f"{sid} ({'path' if t['y'] == 1 else 'normal'})"
    nudge = {'CTG-0021': 1.13, 'CTG-0024': .88}.get(sid, 1)
    a1.text(W.index[-1] + .15, W[sid].iloc[-1] * nudge, lab, fontsize=5.8, va='center', color=INK)
a1.set_yscale('log'); a1.set(xlabel='after round t', ylabel='weight × n  (start = 1)', xlim=(-.2, 8.4))
a1.set_xticks(range(7)); a1.set_title('(a) five tracked segments (ring = missed in round t)', loc='left')
a1.grid(axis='y', color=GRID, lw=.5, which='major'); a1.set_axisbelow(True)
share = E['adaboost_weight_share_pos']
a2.bar(range(len(share)), share, color=MUTED, width=.6, edgecolor='white')
for i, s in enumerate(share): a2.text(i, s + .015, f'{s:.2f}', ha='center', fontsize=5.8, color=INK2)
a2.axhline(108 / 1279, color=INK2, lw=.6, ls=':'); a2.text(6.45, 108 / 1279 + .01, 'class share', fontsize=5.6, color=INK2, ha='right', va='bottom')
a2.set(xlabel='after round t', ylabel='share of total weight', ylim=(0, .75)); a2.set_xticks(range(len(share)))
a2.set_title('(b) weight on pathologic', loc='left'); ygrid(a2)
save(fig, 'fig4-adaboost.png')

# ------------------------------------------------------------ 图5 梯度提升：学习率 × 轮数
cz = np.load(RES / 'E-gb-curves.npz'); rounds = np.arange(1, 501)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5))
for lr, col in zip(['0.05', '0.1', '0.3'], BLUES[:3]):
    a1.plot(rounds, cz[f'train_ll_{lr}'], color=col, lw=.9, ls='--')
    a1.plot(rounds, cz[f'val_ll_{lr}'], color=col, lw=1.3, label=f'learning_rate {lr}')
    b = int(np.argmin(cz[f'val_ll_{lr}']))
    a1.plot(b + 1, cz[f'val_ll_{lr}'][b], marker='o', ms=4, color=col, mec='white', mew=.6)
    a2.plot(rounds, cz[f'val_auc_{lr}'], color=col, lw=1.3, label=f'learning_rate {lr}')
a1.set_yscale('log'); a1.set(xlabel='boosting rounds', ylabel='log-loss', ylim=(5e-5, 1))
a1.set_title('(a) log-loss (solid validation, dashed training)', loc='left'); a1.legend(frameon=False, loc='lower left')
a1.grid(axis='y', color=GRID, lw=.5); a1.set_axisbelow(True)
a2.set(xlabel='boosting rounds', ylabel='validation ROC-AUC', ylim=(.94, 1.0))
a2.set_title('(b) validation ROC-AUC', loc='left'); a2.legend(frameon=False, loc='lower right'); ygrid(a2)
save(fig, 'fig5-gb-rounds.png')

# ------------------------------------------------------------ 图6 公平比较：25 折分组 CV
fd = pd.read_csv(RES / 'F-folds.csv')
order = ['single_tree', 'bagging', 'random_forest', 'adaboost', 'gradient_boosting', 'hist_gb', 'logistic', 'rbf_svm']
names = ['single tree', 'bagging', 'random forest', 'AdaBoost', 'gradient boosting', 'HistGB', 'logistic regression', 'RBF-SVM']
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.7), sharey=True)
rng = np.random.default_rng(0)
for ax, met, lab, lim in zip(axes, ['roc_auc', 'pr_auc', 'ba'], ['ROC-AUC', 'PR-AUC', 'balanced accuracy'],
                             [(.5, 1.0), (.4, 1.0), (.55, 1.0)]):
    for i, m in enumerate(order):
        yy = len(order) - 1 - i
        d = fd[(fd.scheme == 'grouped') & (fd.model == m)][met]
        ax.scatter(d, yy + rng.uniform(-.17, .17, len(d)), s=5, color=BAG, alpha=.45, lw=0)
        ax.plot([d.mean()], [yy], marker='|', ms=11, mew=1.6, color=INK)
        rw = fd[(fd.scheme == 'row') & (fd.model == m)][met].mean()
        ax.plot([rw], [yy], marker='D', ms=3.4, mfc='white', mec=RF, mew=.9)
    ax.set_xlim(*lim); ax.set_xlabel(lab); ax.grid(axis='x', color=GRID, lw=.5); ax.set_axisbelow(True)
axes[0].set_yticks(range(len(order)), names[::-1])
axes[0].set_title('25 grouped folds (dots), their mean (bar),\nrow-split CV mean (◇, leaks recordings)', loc='left', fontsize=7)
save(fig, 'fig6-comparison.png')

# ------------------------------------------------------------ 图7 稳定性：按片段 vs 按录音重抽样
G = J('G')
ms = ['single_tree', 'bagging', 'random_forest', 'gradient_boosting', 'logistic']
mn = ['single\ntree', 'bagging', 'random\nforest', 'gradient\nboosting', 'logistic\nregression']
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.55))
x = np.arange(len(ms)); w = .36
fr_row = [G['row_bootstrap'][m]['flip_rate'] * 100 for m in ms]; fr_rec = [G['recording_bootstrap'][m]['flip_rate'] * 100 for m in ms]
a1.bar(x - w / 2, fr_row, width=w, color='white', edgecolor=INK2, lw=.8, label='resample segments')
a1.bar(x + w / 2, fr_rec, width=w, color=INK2, edgecolor='white', lw=.5, label='resample recordings')
for i in range(len(ms)):
    a1.text(x[i] - w / 2, fr_row[i] + .15, f'{fr_row[i]:.1f}', ha='center', fontsize=5.6, color=INK2)
    a1.text(x[i] + w / 2, fr_rec[i] + .15, f'{fr_rec[i]:.1f}', ha='center', fontsize=5.6, color=INK)
a1.set_xticks(x, mn); a1.set(ylabel='% validation segments whose class flips', ylim=(0, 10.5))
a1.set_title('(a) class flips across 30 refits', loc='left'); a1.legend(frameon=False, loc='upper right'); ygrid(a1)
for i, m in enumerate(ms):
    for j, (lab, mk) in enumerate([('row_bootstrap', 'o'), ('recording_bootstrap', 's')]):
        v = np.load(RES / f'G-auc-{lab}-{m}.npy')
        a2.scatter(np.full(len(v), i + (j - .5) * .36) + rng.uniform(-.07, .07, len(v)), v, s=6, marker=mk,
                   color=(MUTED if j == 0 else INK2), alpha=.8, lw=0)
a2.set_xticks(x, mn); a2.set(ylabel='validation ROC-AUC of each refit', ylim=(.6, 1.01))
a2.scatter([], [], s=8, marker='o', color=MUTED, label='resample segments'); a2.scatter([], [], s=8, marker='s', color=INK2, label='resample recordings')
a2.legend(frameon=False, loc='lower right'); a2.set_title('(b) AUC of each of the 30 refits', loc='left'); ygrid(a2)
save(fig, 'fig7-stability.png')

# ------------------------------------------------------------ 图8 特征重要性
imp = pd.read_csv(RES / 'D-importance.csv').head(10).iloc[::-1]; jp = D['joint_permutation']
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(6.6, 2.55), gridspec_kw=dict(width_ratios=[1, 1, 1.05], wspace=.55))
hl = {'Mode', 'Mean', 'Median'}
colr = [RF if f in hl else MUTED for f in imp.feature]
a1.barh(imp.feature, imp.impurity, color=colr, height=.6, edgecolor='white')
a1.set(xlabel='MDI, training'); a1.set_title('(a) impurity importance', loc='left')
a2.barh(imp.feature, imp.perm_mean, xerr=imp.perm_sd, color=colr, height=.6, edgecolor='white', error_kw=dict(lw=.6, capsize=1.5))
a2.set(xlabel='AUC drop, validation'); a2.set_title('(b) permute one feature', loc='left')
a2.tick_params(labelleft=False)
groups = [('Mode', 'Mode'), ('Mean', 'Mean'), ('Median', 'Median'), ('Mode+Mean+Median', 'Mode+Mean+Median\n(together)'),
          ('ASTV', 'ASTV'), ('ALTV', 'ALTV'), ('ALTV+ASTV', 'ALTV+ASTV\n(together)')][::-1]
vals = [jp[k]['mean'] for k, _ in groups]; sds = [jp[k]['sd'] for k, _ in groups]
cc = [RF if ('Mode' in k or 'Mean' in k or 'Median' in k) else MUTED for k, _ in groups]
a3.barh(range(len(groups)), vals, xerr=sds, color=cc, height=.6, edgecolor='white', error_kw=dict(lw=.6, capsize=1.5))
a3.set_yticks(range(len(groups)), [l for _, l in groups]); a3.set(xlabel='AUC drop, validation')
a3.set_title('(c) alone vs together', loc='left')
for ax in (a1, a2, a3): ax.grid(axis='x', color=GRID, lw=.5); ax.set_axisbelow(True)
save(fig, 'fig8-importance.png')

# ============================================================ 开头的两张示意图（中文）
from matplotlib.patches import FancyArrowPatch
ZH = {'font.family': ['PingFang SC', 'Arial Unicode MS', 'DejaVu Sans'], 'font.size': 7}
HOT, HOT_EC = '#fbe3d8', RF


def rbox(ax, x, y, w, h, text, fc='white', ec=INK2, fs=6.6, bold=False, color=INK, lw=.6, z=3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=fc, ec=ec, lw=lw, zorder=z))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=color, fontweight='bold' if bold else 'normal',
            linespacing=1.35, zorder=z + 1)


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=.7, ms=7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=2))


def label(ax, x, y, text, bg, fs=5.8):
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=INK2, linespacing=1.35, zorder=6,
            bbox=dict(fc=bg, ec='none', pad=1.2))


# ---------------- 图 1 课程位置
with plt.rc_context(ZH):
    fig, ax = plt.subplots(figsize=(6.7, 2.6)); ax.set_xlim(0, 13.4); ax.set_ylim(.3, 5.5); ax.axis('off')
    phases = [
        ('理解数据', 1.25, 2.3, ['变量类型与数据类型', '高维数据与降维'], '一行是什么、独立单位\n是什么（片段还是录音）；\n特征之间的冗余'),
        ('统计推断', 3.85, 2.3, ['统计检验与假设设计', '多重检验校正'], '差距是否大于波动；\n配对比较；比较多了\n会挑出“幸运的赢家”'),
        ('预测模型', 6.75, 2.9, ['机器学习基础与朴素贝叶斯', '线性模型、LDA 与 LR', 'SVM 与 Margin', '集成学习与 Bootstrapping'],
         '前几讲各学一种模型；\n本讲把模型组合起来，\nLR、SVM 是对照，决策树是积木'),
        ('可信的模型', 9.85, 2.3, ['欠拟合与过拟合的识别', '正则化与验证策略'], 'bagging 压方差、boosting\n压偏差；OOB、分组 CV、\n早停已经在本讲出现'),
        ('课程项目', 12.35, 1.9, ['生物医学问题建模\n与项目汇报'], '表现、稳定性、解释\n成本和验证范围\n一起写进结论'),
    ]
    for title, cx, w, items, note in phases:
        ax.text(cx, 5.25, title, ha='center', va='center', fontsize=7.6, fontweight='bold', color=INK)
        ax.add_patch(FancyBboxPatch((cx - w / 2, 2.0), w, 2.95, boxstyle='round,pad=0.02,rounding_size=0.12', fc='#f6f6f4', ec='none'))
        n = len(items)
        for i, it in enumerate(items):
            hot = '集成' in it
            if n == 1: yy, hh = 3.5, .8
            else: hh = .5; yy = 4.6 - hh / 2 - i * (hh + (.1 if n > 2 else .14)) - (0 if n > 2 else .15)
            if hot: hh = .74; yy -= .1
            rbox(ax, cx, yy, w - .25, hh, (it + '\n（本讲）') if hot else it,
                 fc=HOT if hot else 'white', ec=HOT_EC if hot else '#c9c8c3', fs=6.3, bold=hot)
        ax.text(cx, 1.05, note, ha='center', va='center', fontsize=5.9, color=INK2, linespacing=1.4)
    for x0, x1 in [(2.4, 2.7), (5.0, 5.3), (8.2, 8.7), (11.0, 11.4)]:
        arrow(ax, x0, 3.5, x1, 3.5)
    ax.text(.05, 1.78, '与本讲的关系', fontsize=5.8, color=MUTED, va='center')
    save(fig, 'fig0a-course-map.png')

# ---------------- 图 2 集成学习框架
with plt.rc_context(ZH):
    fig, ax = plt.subplots(figsize=(6.7, 4.2)); ax.set_xlim(0, 13.4); ax.set_ylim(0, 8.35); ax.axis('off')
    rbox(ax, 6.7, 7.9, 6.6, .55, '训练数据  →  多个基学习器（大多是决策树）  →  组合成一个预测', fc='#f6f6f4', ec='none', fs=7.2, bold=True)
    cols = [(2.3, '#eef4fc', '① Bagging → 随机森林'), (6.7, '#fdf3e1', '② Boosting → AdaBoost'), (11.1, '#f1f1ef', '③ Stacking')]
    for cx, fc, title in cols:
        ax.add_patch(FancyBboxPatch((cx - 2.05, 2.25), 4.1, 5.1, boxstyle='round,pad=0.02,rounding_size=0.15', fc=fc, ec='none', zorder=0))
        ax.text(cx, 7.0, title, ha='center', va='center', fontsize=7.8, fontweight='bold', color=INK)
    # ① bagging：三份样本 → 三棵深树（互不相连）→ 平均
    cx = 2.3
    for dx, lab in zip([-1.3, 0, 1.3], ['样本 1', '样本 2', '样本 B']):
        rbox(ax, cx + dx, 6.3, 1.1, .5, lab); rbox(ax, cx + dx, 4.75, 1.1, .5, '深树')
        arrow(ax, cx + dx, 6.05, cx + dx, 5.0); arrow(ax, cx + dx, 4.5, cx + dx * .35, 3.98)
    label(ax, cx, 5.52, '有放回重抽样（bootstrap）\n随机森林：每个节点再随机抽几个特征', '#eef4fc')
    rbox(ax, cx, 3.7, 2.4, .55, '平均 / 投票', bold=True)
    ax.text(cx, 2.85, '并行：树和树互不依赖\n主要降低方差', ha='center', va='center', fontsize=6.4, color=INK, linespacing=1.4)
    # ② boosting：树桩 1 → 树桩 2 → 树桩 T（横向箭头表示顺序依赖）→ 加权求和
    cx = 6.7
    for dx, lab in zip([-1.3, 0, 1.3], ['树桩 1', '树桩 2', '树桩 T']):
        rbox(ax, cx + dx, 6.3, 1.1, .5, lab); arrow(ax, cx + dx, 6.05, cx + dx * .35, 3.98)
    arrow(ax, cx - .75, 6.3, cx - .55, 6.3); arrow(ax, cx + .55, 6.3, cx + .75, 6.3)
    label(ax, cx, 5.2, '每一轮调高错分样本的权重\n（梯度提升：下一棵树拟合残差 y − p）', '#fdf3e1')
    rbox(ax, cx, 3.7, 2.4, .55, '按话语权加权求和', bold=True)
    ax.text(cx, 2.85, '串行：后一棵专改前面的错\n主要降低偏差（GBDT、XGBoost 同类）', ha='center', va='center', fontsize=6.4, color=INK, linespacing=1.4)
    # ③ stacking：不同算法 → 袋外预测 → 元学习器
    cx = 11.1
    for dx, lab in zip([-1.3, 0, 1.3], ['LR', 'SVM', '随机森林']):
        rbox(ax, cx + dx, 6.3, 1.1, .5, lab); arrow(ax, cx + dx, 6.05, cx + dx * .35, 3.98)
    label(ax, cx, 5.2, '第一层：不同算法\n各自给出袋外预测', '#f1f1ef')
    rbox(ax, cx, 3.7, 2.4, .55, '元学习器（常用 LR）', bold=True)
    ax.text(cx, 2.85, '分两层：第二层学出怎么组合\n投票可看成权重固定的特例', ha='center', va='center', fontsize=6.4, color=INK, linespacing=1.4)
    # 共同原理
    ax.add_patch(FancyBboxPatch((.25, .15), 12.9, 1.75, boxstyle='round,pad=0.02,rounding_size=0.12', fc='white', ec=INK2, lw=.6))
    ax.text(.55, 1.55, '共同原理：好而不同', fontsize=7.2, fontweight='bold', color=INK, va='center')
    ax.text(.55, 1.0, '平均后的方差 = ρσ² + (1 − ρ)σ²/B：模型越多第二项越小，但地板是 ρσ²；想再降，就要让模型之间更“不同”（ρ 更小）',
            fontsize=6.2, color=INK, va='center')
    ax.text(.55, .5, 'Bootstrapping 的两种用途：① 造出多份训练集（bagging）；② 估计模型的稳定性与指标的不确定性', fontsize=6.2, color=INK, va='center')
    save(fig, 'fig0b-framework.png')
