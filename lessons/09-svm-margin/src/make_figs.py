"""重画提交文档用的全部图。运行: python -I make_figs.py"""
import sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
LESSON = Path(__file__).resolve().parents[1]  # lessons/09-svm-margin
FIG = LESSON/'report/figures'
sys.path.insert(0, str(LESSON/'src/course'))
from lab_utils import *
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'savefig.dpi': 220,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6, 'xtick.major.width': .5, 'ytick.major.width': .5})
NO_C, AB_C, DH_C, INK = '#087c77', '#b75d24', '#c9a227', '#193638'
train, validation = load_development()
geo = geometry_experiment(train)
grid, models = parameter_experiment(train, validation)
XL, YL = 'Pelvic incidence (source scale)', 'Degree of spondylolisthesis (source scale)'

def boundary(ax, model, title, standardized=False, scaler=None, subset=None, legend=False, ids=False):
    data = train if subset is None else subset
    X = scaler.transform(data[FEATURES]) if standardized else data[FEATURES].to_numpy()
    low, high = (np.percentile(X, [0, 98], axis=0) if len(X) > 20 else (X.min(0), X.max(0)))
    pad = .12*np.maximum(high-low, 1e-3); low = low-pad; high = high+pad
    xx, yy = np.meshgrid(np.linspace(low[0], high[0], 220), np.linspace(low[1], high[1], 220))
    pts = np.c_[xx.ravel(), yy.ravel()]
    sc = model.decision_function(pts if standardized else pd.DataFrame(pts, columns=FEATURES)).reshape(xx.shape)
    ax.contourf(xx, yy, sc >= 0, levels=[-.5, .5, 1.5], colors=['#e3efee', '#f9e9dc'], alpha=.7)
    ax.contour(xx, yy, sc, levels=[-1, 0, 1], colors=['#52686a', INK, '#52686a'], linestyles=['--', '-', '--'], linewidths=[.7, 1.1, .7])
    y = data.abnormal.to_numpy()
    for cls, col, mk in [(0, NO_C, 'o'), (1, AB_C, '^')]:
        m = y == cls
        ax.scatter(X[m, 0], X[m, 1], color=col, marker=mk, s=10 if subset is None else 22, alpha=.85, linewidths=0, label=['NO', 'AB'][cls])
    sv = svc_of(model).support_
    ax.scatter(X[sv, 0], X[sv, 1], s=34 if subset is None else 70, facecolors='none', edgecolors=INK, linewidths=.55, label='support vectors')
    if ids:
        for i, name in enumerate(data.sample_id):
            ax.annotate(name, X[i], xytext=(3, 3), textcoords='offset points', fontsize=6)
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]))
    ax.set_title(title, fontsize=7.5, loc='left', linespacing=1.3)
    if legend: ax.legend(loc='upper left', frameon=False, handletextpad=.3, borderaxespad=.2)
    return int(np.any((X < low) | (X > high), axis=1).sum())

def save(fig, name):
    fig.savefig(FIG/name, bbox_inches='tight', pad_inches=.03); plt.close(fig); print('saved', name)

for old in FIG.glob('*.png'): old.unlink()

# 图1 全部训练记录
fig, ax = plt.subplots(figsize=(5.9, 3.2))
for cls, col, mk in [(0, NO_C, 'o'), (1, AB_C, '^')]:
    d = train[train.abnormal == cls]
    ax.scatter(d[FEATURES[0]], d[FEATURES[1]], color=col, marker=mk, s=14, alpha=.75, linewidths=0, label=['NO (normal)', 'AB (abnormal)'][cls])
ax.set(xlabel=XL, ylabel=YL); ax.legend(frameon=False, loc='upper left')
ax.annotate('one AB record at 418', xy=(train.loc[train.degree_spondylolisthesis.idxmax(), FEATURES[0]], 418), xytext=(-95, -12), textcoords='offset points', fontsize=6.5, arrowprops=dict(arrowstyle='-', lw=.5, color='.4'))
save(fig, 'fig1-train-scatter.png')

# 图2 (a) 三条候选线 (b) 最大间隔解 + 两个支持向量的连线
z, y = geo['z'], geo['subset'].abnormal.to_numpy(); w, b = geo['model'].coef_[0], geo['model'].intercept_[0]
ids = geo['subset'].sample_id.tolist(); zp, zn = z[ids.index('V061')], z[ids.index('V217')]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.9, 2.95))
for cls, col, mk in [(0, NO_C, 'o'), (1, AB_C, '^')]:
    m = y == cls; a1.scatter(z[m, 0], z[m, 1], c=col, marker=mk, s=22, label=['NO', 'AB'][cls])
for i, name in enumerate(ids): a1.annotate(name, z[i], xytext=(3, 3), textcoords='offset points', fontsize=6)
xx = np.linspace(-2.2, 1.9, 100)
for off, col, lab in [(-.6, '#9c562b', 'b − 0.6'), (0, INK, 'b (SVM)'), (.6, '#5c7b9c', 'b + 0.6')]:
    a1.plot(xx, -(w[0]*xx+b+off)/w[1], color=col, lw=1, label=lab)
a1.set(xlabel='z1 (standardized pelvic incidence)', ylabel='z2 (standardized spondylolisthesis)', xlim=(-2.2, 1.9), ylim=(-1.3, 2.2), aspect='equal')
a1.set_title('(a) three lines, same normal vector,\nall with zero training errors', loc='left', fontsize=7.5, linespacing=1.3)
a1.legend(frameon=False, loc='lower right', fontsize=6)
boundary(a2, geo['model'], '(b) max-margin line bisects the two support vectors\nw = 0.546 (z$_+$ − z$_-$),   band width = |z$_+$ − z$_-$| = 1.914', True, geo['scaler'], geo['subset'], ids=True)
a2.annotate('', xy=zp, xytext=zn, arrowprops=dict(arrowstyle='<->', color='#a3262e', lw=1.1, shrinkA=3, shrinkB=3))
a2.set(xlabel='z1', ylabel='z2', aspect='equal')
save(fig, 'fig2-geometry.png')

# 图3 linear C
fig, axes = plt.subplots(1, 3, figsize=(5.9, 2.4), sharey=True)
for ax, C in zip(axes, [.01, 1., 100.]):
    row = grid[(grid.kernel == 'linear') & (grid.C == C)].iloc[0]
    out3 = boundary(ax, models[('linear', C, .1)], f'linear, C = {C:g}\nval BA {row.val_balanced_accuracy:.3f}', legend=(C == .01)); ax.set_xlabel('pelvic incidence')
axes[0].set_ylabel('spondylolisthesis'); fig.subplots_adjust(wspace=.08)
save(fig, 'fig3-linear-C.png'); print('  outside view per panel:', out3)

# 图4 RBF 两个切片
fig, axes = plt.subplots(2, 3, figsize=(5.9, 4.6), sharex=True, sharey=True)
for r, cfgs in enumerate([[(1., .01), (1., 1.), (1., 10.)], [(1., 10.), (10., 10.), (100., 10.)]]):
    for ax, (C, g) in zip(axes[r], cfgs):
        row = grid[(grid.kernel == 'rbf') & (grid.C == C) & (grid.gamma == g)].iloc[0]
        boundary(ax, models[('rbf', C, g)], f'C = {C:g}, γ = {g:g}\ntrain {row.train_balanced_accuracy:.2f} · val {row.val_balanced_accuracy:.2f}')
        if r == 1: ax.set_xlabel('pelvic incidence')
for ax in axes[:, 0]: ax.set_ylabel('spondylolisthesis')
fig.text(.01, .975, 'Row 1 — fix C = 1, change γ', fontsize=8, style='italic'); fig.text(.01, .487, 'Row 2 — fix γ = 10, change C', fontsize=8, style='italic')
fig.subplots_adjust(wspace=.08, hspace=.62, top=.90, bottom=.09)
save(fig, 'fig4-rbf-slices.png')

# 图5 RBF 网格热图：训练 / 验证
rbf = grid[grid.kernel == 'rbf']
fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.35))
for ax, col, ttl in zip(axes, ['train_balanced_accuracy', 'val_balanced_accuracy'], ['(a) training balanced accuracy', '(b) validation balanced accuracy']):
    pv = rbf.pivot(index='C', columns='gamma', values=col)
    im = ax.imshow(pv.values, cmap='YlGnBu', vmin=.5, vmax=.92, aspect='auto')
    for i in range(4):
        for j in range(4):
            v = pv.values[i, j]; ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=6.8, color='white' if v > .78 else INK)
    ax.set_xticks(range(4)); ax.set_xticklabels([f'{g:g}' for g in pv.columns]); ax.set_yticks(range(4)); ax.set_yticklabels([f'{c:g}' for c in pv.index])
    ax.set_xlabel('gamma'); ax.set_title(ttl, loc='left', fontsize=7.5); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
axes[0].set_ylabel('C')
axes[1].add_patch(Rectangle((2.5, 1.5), 1, 1, fill=False, ec='#a3262e', lw=1.4))
fig.subplots_adjust(wspace=.18)
save(fig, 'fig5-rbf-grid.png')

# 图6 三种核
fig, axes = plt.subplots(1, 3, figsize=(5.9, 2.4), sharey=True)
for ax, (key, lab) in zip(axes, [(('linear', 1., .1), 'linear'), (('poly', 1., .1), 'polynomial, degree 2'), (('rbf', 1., .1), 'RBF, γ = 0.1')]):
    row = grid[(grid.kernel == key[0]) & (grid.C == key[1]) & (grid.gamma == key[2])].iloc[0]
    boundary(ax, models[key], f'{lab}\nval BA {row.val_balanced_accuracy:.3f}'); ax.set_xlabel('pelvic incidence')
axes[0].set_ylabel('spondylolisthesis'); fig.subplots_adjust(wspace=.08)
save(fig, 'fig6-kernels.png')

# 图7 验证集按亚型 + 两个冻结模型的错误
zf = zipfile.ZipFile(LESSON/'data/raw/vertebral-column.zip')
lines = [l.split() for l in zf.read('column_3C.dat').decode('utf8', 'replace').strip().splitlines() if l.strip()]
d3 = pd.DataFrame({'sample_id': [f'V{i+1:03d}' for i in range(len(lines))], 'class3': [l[6] for l in lines]})
va = validation.merge(d3, on='sample_id')
xlim, ylim = (22, 100), (-15, 47)
fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.9), sharey=True)
for ax, (key, lab) in zip(axes, [(('linear', 1., .1), '(a) linear, C = 1'), (('rbf', 10., 10.), '(b) RBF, C = 10, γ = 10')]):
    gx, gy = np.meshgrid(np.linspace(*xlim, 260), np.linspace(*ylim, 260))
    sc = models[key].decision_function(pd.DataFrame(np.c_[gx.ravel(), gy.ravel()], columns=FEATURES)).reshape(gx.shape)
    ax.contourf(gx, gy, sc >= 0, levels=[-.5, .5, 1.5], colors=['#e3efee', '#f9e9dc'], alpha=.7)
    ax.contour(gx, gy, sc, levels=[0], colors=[INK], linewidths=[1.0])
    pred = (models[key].decision_function(va[FEATURES]) >= 0).astype(int); wrong = pred != va.abnormal.to_numpy()
    for c3, col, mk, lg in [('NO', NO_C, 'o', 'NO'), ('DH', DH_C, 's', 'DH (labelled AB)'), ('SL', AB_C, '^', 'SL (labelled AB)')]:
        d = va[va.class3 == c3]; ax.scatter(d[FEATURES[0]], d[FEATURES[1]], color=col, marker=mk, s=15, alpha=.9, linewidths=0, label=lg, zorder=3)
    ax.scatter(va[FEATURES[0]][wrong], va[FEATURES[1]][wrong], s=62, facecolors='none', edgecolors='#a3262e', linewidths=.9, label='misclassified', zorder=4)
    inview = (va[FEATURES[0]].between(*xlim)) & (va[FEATURES[1]].between(*ylim))
    ax.set(xlim=xlim, ylim=ylim, xlabel='pelvic incidence'); ax.set_title(f'{lab}: {int(wrong.sum())} errors', loc='left', fontsize=7.5)
    print(f'  {lab}: errors {int(wrong.sum())}, errors in view {int((wrong & inview).sum())}, points outside view {int((~inview).sum())}', va[~inview].class3.value_counts().to_dict())
axes[0].set_ylabel('spondylolisthesis'); axes[0].legend(frameon=False, loc='upper left', fontsize=6, handletextpad=.3)
fig.subplots_adjust(wspace=.06)
save(fig, 'fig7-validation-subtypes.png')
