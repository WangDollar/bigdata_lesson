import sys, zipfile, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.spatial.distance import pdist
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
LESSON = Path(__file__).resolve().parents[1]  # lessons/09-svm-margin
sys.path.insert(0, str(LESSON/'src/course'))
from lab_utils import *
from sklearn.svm import SVC
train, validation = load_development()
geo = geometry_experiment(train); sub = geo['subset']; z = geo['z']; m8 = geo['model']
ids = sub.sample_id.tolist(); y8 = sub.abnormal.to_numpy()

print('===== 1. 八点：对偶系数 =====')
print('support', [ids[i] for i in m8.support_], 'dual_coef (y*alpha)', m8.dual_coef_)
zp, zn = z[ids.index('V061')], z[ids.index('V217')]
d = zp - zn; print('d = z(V061)-z(V217) =', d, ' ||d|| =', np.linalg.norm(d), ' ||d||^2 =', d@d)
alpha = 2/(d@d); print('alpha = 2/||d||^2 =', alpha, ' alpha*d =', alpha*d, ' sklearn w =', m8.coef_[0])
mid = (zp+zn)/2; print('midpoint', mid, ' f(mid) =', m8.decision_function([mid])[0], ' band width 2/||w|| =', 2/np.linalg.norm(m8.coef_[0]))
print('b from SV: 1 - w.z(V061) =', 1 - m8.coef_[0]@zp, ' sklearn b =', m8.intercept_[0])
keep = np.array([i != ids.index('V061') for i in range(8)])
r = SVC(kernel='linear', C=10000., tol=1e-9, max_iter=200000).fit(z[keep], y8[keep])
kid = [ids[i] for i in range(8) if keep[i]]
print('删 V061 后: SV =', [kid[i] for i in r.support_], ' alpha =', np.abs(r.dual_coef_), ' w =', r.coef_[0], ' b =', r.intercept_[0], ' width =', 2/np.linalg.norm(r.coef_[0]))
print('  新解下各点 y*f:', dict(zip(kid, np.round((2*y8[keep]-1)*r.decision_function(z[keep]), 4))))

print('\n===== 2. linear C=1 全训练集：alpha 分类 =====')
grid, models = parameter_experiment(train, validation)
ml = models[('linear', 1., .1)]; svc = svc_of(ml); sc = ml.named_steps['standardscaler']
Z = sc.transform(train[FEATURES]); ym = 2*train.abnormal.to_numpy()-1; f = ml.decision_function(train[FEATURES])
a = np.abs(svc.dual_coef_[0]); C = svc.C
print('n_sv', len(a), ' alpha==C:', int(np.sum(a > C-1e-8)), ' 0<alpha<C:', int(np.sum(a <= C-1e-8)), ' free alphas:', np.round(a[a <= C-1e-8], 4))
free = svc.support_[a <= C-1e-8]; bnd = svc.support_[a > C-1e-8]
print('free SV 的 y*f:', np.round(ym[free]*f[free], 6), ' ids:', train.sample_id.to_numpy()[free])
print('bounded SV 的 y*f 范围:', round(float((ym[bnd]*f[bnd]).min()), 4), '~', round(float((ym[bnd]*f[bnd]).max()), 6), ' 其中 y*f<0:', int(np.sum(ym[bnd]*f[bnd] < 0)))
print('w from dual:', svc.dual_coef_[0] @ svc.support_vectors_, ' coef_:', svc.coef_[0], ' sum(y*alpha) =', float(svc.dual_coef_.sum()))
lin = grid[grid.kernel == 'linear']
print(pd.DataFrame({'C': lin.C, 'total_hinge': (lin.mean_hinge*186).round(3), 'norm_w': (2/lin.linear_width).round(4), 'half_w2': lin.regularizer.round(4), 'train_BA': lin.train_balanced_accuracy.round(4)}).to_string(index=False))

print('\n===== 3. RBF 冻结配置：用 alpha 与核手工重算分数 =====')
mr = models[('rbf', 10., 10.)]; sr = svc_of(mr); scr = mr.named_steps['standardscaler']
Zv = scr.transform(validation[FEATURES]); fv = mr.decision_function(validation[FEATURES])
def manual(zq): return float(np.sum(sr.dual_coef_[0]*np.exp(-sr.gamma*np.sum((sr.support_vectors_-zq)**2, axis=1))) + sr.intercept_[0])
man = np.array([manual(q) for q in Zv]); print('max |manual - decision_function| =', np.abs(man-fv).max(), ' n_sv =', len(sr.support_))
q = Zv[0]; K = np.exp(-sr.gamma*np.sum((sr.support_vectors_-q)**2, axis=1))
print('验证集第一条', validation.sample_id.iloc[0], 'z =', np.round(q, 4), ' f =', round(fv[0], 4), ' K>0.01 的 SV 数:', int(np.sum(K > .01)), '/', len(K))
nn = [int(np.sum(np.exp(-sr.gamma*np.sum((sr.support_vectors_-qq)**2, axis=1)) > .01)) for qq in Zv]
print('每个验证点 K>0.01 的 SV 数: median', np.median(nn), ' min', min(nn), ' max', max(nn))

print('\n===== 4. 课程二次核 (0.1*z.u+1)^2 的显式六维映射 =====')
g = .1
def phi(v): return np.array([1, math.sqrt(2*g)*v[0], math.sqrt(2*g)*v[1], g*v[0]**2, math.sqrt(2)*g*v[0]*v[1], g*v[1]**2])
zz, uu = np.array([.7108, .6700]), np.array([-.6233, -.7018])
print('z.u =', zz@uu, ' K =', (g*(zz@uu)+1)**2, ' phi(z) =', np.round(phi(zz), 5), ' phi(u) =', np.round(phi(uu), 5), ' phi.phi =', phi(zz)@phi(uu))
print('逐项乘积:', np.round(phi(zz)*phi(uu), 6))

print('\n===== 5. 标准化空间的两两距离与核矩阵 =====')
D = pdist(Z); print('pairwise distance: median', round(np.median(D), 3), ' 25%/75%', np.round(np.percentile(D, [25, 75]), 3), ' 95%', round(np.percentile(D, 95), 3), ' max', round(D.max(), 3))
for gm in [.01, .1, 1, 10]:
    Kd = np.exp(-gm*D**2); print(f'gamma={gm:<5} 1/sqrt(gamma)={1/math.sqrt(gm):.3f}  K median={np.median(Kd):.4f}  K 5%~95%={np.percentile(Kd,5):.4f}~{np.percentile(Kd,95):.4f}  K>0.5 的点对比例={np.mean(Kd>.5):.3f}')

print('\n===== 6. 小 gamma 的 RBF ≈ linear(C\'=2*gamma*C) =====')
ba = lambda s: metrics(validation.abnormal, s)['balanced_accuracy']
for Cr in [1., 10., 100.]:
    rb = models[('rbf', Cr, .01)]; Ceq = 2*.01*Cr
    le = make_model('linear', Ceq).fit(train[FEATURES], train.abnormal)
    fr, fl_ = rb.decision_function(validation[FEATURES]), le.decision_function(validation[FEATURES])
    print(f'RBF(C={Cr:g},g=.01) vs linear(C={Ceq:g}): 预测一致 {np.mean((fr>=0)==(fl_>=0)):.3f}  corr={np.corrcoef(fr,fl_)[0,1]:.4f}  max|df|={np.abs(fr-fl_).max():.3f}  BA rbf={ba(fr):.4f} lin={ba(fl_):.4f}')

print('\n===== 7. 验证集按亚型看错误 =====')
zf = zipfile.ZipFile(LESSON/'data/raw/vertebral-column.zip')
lines = [l.split() for l in zf.read('column_3C.dat').decode('utf8', 'replace').strip().splitlines() if l.strip()]
d3 = pd.DataFrame({'sample_id': [f'V{i+1:03d}' for i in range(len(lines))], 'class3': [l[6] for l in lines]})
va = validation.merge(d3, on='sample_id')
for fam, key in [('linear', ('linear', 1., .1)), ('rbf', ('rbf', 10., 10.))]:
    s = models[key].decision_function(va[FEATURES]); p = (s >= 0).astype(int); t = va.assign(pred=p)
    print(fam, 'FN by subtype:', t[(t.abnormal == 1) & (t.pred == 0)].class3.value_counts().to_dict(), ' FP:', t[(t.abnormal == 0) & (t.pred == 1)][['sample_id', 'degree_spondylolisthesis']].values.tolist())
    print('   召回 DH', f"{int(((t.class3=='DH')&(t.pred==1)).sum())}/{int((t.class3=='DH').sum())}", ' SL', f"{int(((t.class3=='SL')&(t.pred==1)).sum())}/{int((t.class3=='SL').sum())}", ' NO 判对', f"{int(((t.class3=='NO')&(t.pred==0)).sum())}/{int((t.class3=='NO').sum())}")
    if fam == 'linear':
        for lo, hi in [(-9, -1), (-1, 0), (0, 1), (1, 99)]:
            mk = (s >= lo) & (s < hi); print(f'     f∈[{lo},{hi}): n={int(mk.sum())}  AB比例={va.abnormal[mk].mean():.2f}  DH={int((va.class3[mk]=="DH").sum())} SL={int((va.class3[mk]=="SL").sum())} NO={int((va.class3[mk]=="NO").sum())}')
print('验证集 DH 的滑脱程度:', va[va.class3 == 'DH'].degree_spondylolisthesis.describe()[['min', '50%', 'max']].round(2).to_dict(), ' SL 被漏的那条:', va[(va.class3=='SL')].sort_values('degree_spondylolisthesis').head(2)[['sample_id','pelvic_incidence','degree_spondylolisthesis']].values.tolist())

print('\n===== 8. 线性模型：由到边界的距离预测翻转率 =====')
fl = ml.decision_function(validation[FEATURES]); dist = np.abs(fl)/np.linalg.norm(svc.coef_[0])
print('验证点到边界距离(标准化单位): median', round(np.median(dist), 3), ' min', round(dist.min(), 4), ' <0.1:', int(np.sum(dist < .1)), ' <0.2:', int(np.sum(dist < .2)), ' <0.4:', int(np.sum(dist < .4)), ' 带半宽 1/||w|| =', round(1/np.linalg.norm(svc.coef_[0]), 4))
obs = noise_experiment(ml, train, validation)
for lv, o in zip([.05, .1, .2], obs.mean_flip_rate[1:]):
    print(f'noise {lv:.2f}: 理论 mean Phi(-d/sigma) = {np.mean(norm.cdf(-dist/lv)):.4f}   实验 = {o:.4f}')
