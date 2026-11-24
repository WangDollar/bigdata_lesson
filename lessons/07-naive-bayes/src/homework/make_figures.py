"""第7讲作业配图（3 张单栏图）。

图1  交叉熵 -ln p：标出任务 5.1 的四个手算点（说明"都判对"时损失仍能区分模型）。
图2  验证集上灵敏度/特异度随阈值变化：说明阈值规则为什么选 0.5。
图3  腰痛特征额外复制 0/1/5 份时，24 条验证记录的后验概率：说明违反条件独立会让概率被重复计票。

只用 train 和 validation，不碰 test。
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / 'src/course'))
sys.path.insert(0, str(Path.home() / '.claude/skills/nature-figure/scripts'))
from lab_utils import FEATURES, fit_model, threshold_table, choose_threshold  # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

OUT = BASE / 'report/figures'
OUT.mkdir(parents=True, exist_ok=True)
T = 'bladder_inflammation'

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['PingFang SC', 'Heiti SC', 'Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.8,
    'legend.frameon': False,
})

POS = '#3a6ea5'    # 真实为膀胱炎 / 灵敏度
NEG = '#c8793a'    # 真实为非膀胱炎 / 特异度
GREY = '#8c8c8c'
W = 3.5            # 单栏 89 mm


def save(fig, name):
    require_matplotlib_panel_alignment(fig, json_out=OUT / f'{name}.alignment.json',
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', dpi=600, bbox_inches='tight')
    plt.close(fig)


train = pd.read_csv(BASE / 'data/student/train.csv')
val = pd.read_csv(BASE / 'data/student/validation.csv')
model = fit_model(train)
vp = model.predict_proba(val[FEATURES])[:, 1]

# ---------- 图1 交叉熵曲线 ----------
p = np.linspace(0.005, 0.995, 400)
fig, ax = plt.subplots(figsize=(W, 2.4))
ax.plot(p, -np.log(p), color=POS, lw=1.4)
ax.plot(p, -np.log(1 - p), color=NEG, lw=1.4)
ax.text(0.06, 4.45, '真实 y=1 时：损失 = −ln p', color=POS, fontsize=7, va='center')
ax.text(0.93, 3.2, '真实 y=0 时：损失 = −ln(1−p)', color=NEG, fontsize=7, va='center', ha='right')
points = [  # (p, 损失, 颜色, 文字, 文字位置)
    (0.6, -np.log(0.6), POS, '模型A p=0.6\n0.511', (0.40, 1.60)),
    (0.9, -np.log(0.9), POS, '模型B p=0.9\n0.105', (0.80, 0.62)),
    (0.6, -np.log(0.4), NEG, 'y=0 时 p=0.6\n0.916', (0.66, 2.15)),
    (0.99, -np.log(0.01), NEG, 'y=0 时 p=0.99\n4.605', (0.70, 4.25)),
]
for px, py, c, label, (tx, ty) in points:
    ax.plot(px, min(py, 4.6), 'o', ms=4, color=c, zorder=3)
    ax.annotate(label, (px, min(py, 4.6)), xytext=(tx, ty), fontsize=6.5, color='#333333',
                arrowprops=dict(arrowstyle='-', lw=0.6, color=GREY), ha='center', va='center')
ax.set_xlim(0, 1)
ax.set_ylim(0, 4.9)
ax.set_xlabel('模型给出的 p = P(Y=1|x)')
ax.set_ylabel('单条记录的交叉熵损失')
save(fig, 'fig1_logloss')

# ---------- 图2 阈值与灵敏度/特异度 ----------
th = threshold_table(val[T], vp)
chosen = choose_threshold(th)
fig, ax = plt.subplots(figsize=(W, 2.3))
ax.axvspan(0.2, 0.5, color='#e8eef5', lw=0, zorder=0)
ax.axhline(0.9, color=GREY, lw=0.7, ls=':', zorder=1)
ax.plot(th.threshold, th.sensitivity, '-o', color=POS, lw=1.3, ms=3.5, zorder=3)
ax.plot(th.threshold, th.specificity, '--s', color=NEG, lw=1.3, ms=3.2, zorder=2)
ax.axvline(chosen, color='#333333', lw=0.8, zorder=1)
ax.text(0.91, 0.60, '灵敏度', color=POS, ha='right', fontsize=7)
ax.text(0.07, 0.25, '特异度', color=NEG, ha='left', fontsize=7)
ax.text(0.35, 1.035, '两者都为 1', ha='center', fontsize=6.5, color='#555555')
ax.text(chosen + 0.015, 0.70, f'规则选中 t={chosen:g}', fontsize=6.5, color='#333333')
ax.text(0.52, 0.885, '灵敏度下限 0.9', fontsize=6, color=GREY, ha='left', va='top')
ax.set_xlim(0, 1)
ax.set_ylim(0.2, 1.08)
ax.set_xticks([0.05, 0.2, 0.3, 0.5, 0.7, 0.9])
ax.set_xticklabels(['0.05', '0.2', '0.3', '0.5', '0.7', '0.9'])
ax.set_xlabel('分类阈值 t（p ≥ t 判为阳性）')
ax.set_ylabel('验证集比例（n=24）')
save(fig, 'fig2_threshold')

# ---------- 图3 复制腰痛后概率被"重复计票" ----------
copies_list = [0, 1, 5]
probs = []
for copies in copies_list:
    tx, vx = train[FEATURES].copy(), val[FEATURES].copy()
    for i in range(copies):
        tx[f'c{i}'] = tx.lumbar_pain
        vx[f'c{i}'] = vx.lumbar_pain
    m = BernoulliNB(alpha=1, binarize=None).fit(tx, train[T])
    probs.append(m.predict_proba(vx)[:, 1])
probs = np.column_stack(probs)
xs = np.arange(len(copies_list))
fig, ax = plt.subplots(figsize=(W, 2.6))
ax.axhline(0.5, color=GREY, lw=0.8, ls='--', zorder=1)
for row, (sid, y) in enumerate(zip(val.sample_id, val[T])):
    hi = sid in ('A094', 'A081')
    ax.plot(xs, probs[row], '-o', color=POS if y == 1 else NEG, ms=2.8 if not hi else 3.6,
            lw=0.7 if not hi else 1.6, alpha=0.45 if not hi else 1, zorder=3 if hi else 2)
r094 = int(np.flatnonzero(val.sample_id == 'A094')[0])
r081 = int(np.flatnonzero(val.sample_id == 'A081')[0])
ax.text(2.09, probs[r094, 2], f'A094 真实阳性\n{probs[r094, 0]:.3f}→{probs[r094, 2]:.3f}',
        fontsize=6.5, color=POS, va='center')
ax.text(2.09, probs[r081, 2], f'A081 真实阴性\n{probs[r081, 0]:.3f}→{probs[r081, 2]:.3f}',
        fontsize=6.5, color=NEG, va='center')
ax.text(2.09, 0.535, '阈值 0.5', fontsize=6.5, color=GREY, va='bottom')
ax.plot([], [], '-o', color=POS, ms=3, lw=1, label='真实为膀胱炎（12 条）')
ax.plot([], [], '-o', color=NEG, ms=3, lw=1, label='真实不是（12 条）')
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.16), ncol=2, handlelength=1.6, columnspacing=1.2)
ax.set_xticks(xs)
ax.set_xticklabels(['不复制\n（原模型）', '多复制 1 份\n（共 2 列腰痛）', '多复制 5 份\n（共 6 列腰痛）'])
ax.set_xlim(-0.25, 2.85)
ax.set_ylim(-0.02, 1.02)
ax.set_ylabel('验证集记录的 P(Y=1|x)')
save(fig, 'fig3_duplicate')

print('chosen threshold', chosen)
print('probs A094', probs[r094].round(3), 'A081', probs[r081].round(3))
print('below 0.5 at 5 copies among positives:', int(((probs[:, 2] < .5) & (val[T] == 1)).sum()),
      '| above 0.5 among negatives:', int(((probs[:, 2] >= .5) & (val[T] == 0)).sum()))
