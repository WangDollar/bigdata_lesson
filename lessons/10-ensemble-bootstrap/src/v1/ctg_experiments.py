"""第 10 讲 CTG 实验：工作单 A–H 用到的全部数字。只读 student/ 下的 train 和 validation，不读测试标签。
用法: python -I ctg_experiments.py [protocol A B C D E F G ...]   不带参数则全部运行
结果: results/<节>.json，作图用的数组在 results/*.npz / *.csv"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.inspection import permutation_importance

DATA = Path('/Users/dollar/Desktop/note/实践课作业/10. Ensemble/data-ctg/student')
WORK = Path(__file__).resolve().parent
OUT = WORK / 'results'; OUT.mkdir(exist_ok=True)
SEED = 1010
LABEL, GROUP = 'pathologic', 'recording_id'
train = pd.read_csv(DATA / 'train.csv'); valid = pd.read_csv(DATA / 'validation.csv')
FEATS = [c for c in train.columns if c not in {'sample_id', GROUP, LABEL}]
X, y, g = train[FEATS], train[LABEL].to_numpy(), train[GROUP].to_numpy()
Xv, yv, gv = valid[FEATS], valid[LABEL].to_numpy(), valid[GROUP].to_numpy()
dev = pd.concat([train, valid], ignore_index=True)
Xd, yd, gd = dev[FEATS], dev[LABEL].to_numpy(), dev[GROUP].to_numpy()
XN, XvN = X.to_numpy(), Xv.to_numpy()


def models():
    # 与课程 demo_ensemble.py 相同的预设参数；n_jobs 只影响速度，不影响结果
    return {
        'single_tree': DecisionTreeClassifier(max_depth=4, random_state=SEED),
        'bagging': BaggingClassifier(estimator=DecisionTreeClassifier(random_state=SEED), n_estimators=100,
                                     max_features=1.0, random_state=SEED, n_jobs=-1),
        'random_forest': RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=SEED, n_jobs=-1),
        'adaboost': AdaBoostClassifier(n_estimators=50, random_state=SEED),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        'logistic': make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED)),
        'rbf_svm': make_pipeline(StandardScaler(), SVC(C=1, kernel='rbf', random_state=SEED)),
    }


PROBA_MODELS = ['single_tree', 'bagging', 'random_forest', 'adaboost', 'gradient_boosting', 'logistic']


def score(model, A):
    return model.decision_function(A) if hasattr(model, 'decision_function') else model.predict_proba(A)[:, 1]


def metrics(yt, s, pred):
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    return dict(roc_auc=roc_auc_score(yt, s), pr_auc=average_precision_score(yt, s),
                balanced_accuracy=balanced_accuracy_score(yt, pred), accuracy=accuracy_score(yt, pred),
                sensitivity=tp / (tp + fn), specificity=tn / (tn + fp), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def bootstrap_resamples(n_refits=30):
    # 与 demo_ensemble.py 的 prediction_stability 完全相同的 30 份训练集
    rng = np.random.default_rng(SEED)
    return [rng.choice(len(y), size=len(y), replace=True) for _ in range(n_refits)]


def flip_rate(classes):
    share = classes.mean(axis=0)
    return float(((share > 0) & (share < 1)).mean())


def to_jsonable(o):
    if isinstance(o, dict): return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray): return to_jsonable(o.tolist())
    if isinstance(o, float) and np.isnan(o): return None
    return o


def save(name, obj):
    (OUT / f'{name}.json').write_text(json.dumps(to_jsonable(obj), ensure_ascii=False, indent=1), encoding='utf8')


# ---------------------------------------------------------------- 数据与协议
def sec_protocol():
    r = {}
    for name, d in [('train', train), ('valid', valid), ('dev', dev)]:
        gg = d.groupby(GROUP)[LABEL].agg(['size', 'sum'])
        r[name] = dict(n=len(d), recordings=len(gg), pos=int(d[LABEL].sum()), pos_rate=float(d[LABEL].mean()),
                       seg_per_rec_median=float(gg['size'].median()), seg_per_rec_max=int(gg['size'].max()),
                       rec_with_pos=int((gg['sum'] > 0).sum()),
                       rec_mixed=int(((gg['sum'] > 0) & (gg['sum'] < gg['size'])).sum()),
                       pos_per_pos_rec=sorted(gg.loc[gg['sum'] > 0, 'sum'].astype(int).tolist(), reverse=True))
    r['valid_all_normal_baseline'] = dict(accuracy=float((yv == 0).mean()), balanced_accuracy=0.5)
    # 同一录音的片段有多像：标准化后每个片段的最近邻是否来自同一录音
    Z = ((Xd - Xd.mean()) / Xd.std()).to_numpy()
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(Z); np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    has_sibling = pd.Series(gd).map(pd.Series(gd).value_counts()).to_numpy() > 1
    same = gd[nn] == gd
    r['nearest_neighbour_same_recording'] = dict(all=float(same.mean()), among_with_sibling=float(same[has_sibling].mean()),
                                                 n_with_sibling=int(has_sibling.sum()))
    r['nearest_neighbour_same_label'] = float((yd[nn] == yd).mean())
    corr = Xd.corr()
    r['high_corr_pairs'] = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(FEATS) for b in FEATS[i + 1:]
                            if abs(corr.loc[a, b]) > 0.8]
    corr.to_csv(OUT / 'feature-correlation.csv')
    save('protocol', r); return r


# ---------------------------------------------------------------- A 决策树
def node_table(t, A, ya):
    dp = t.decision_path(A).toarray().astype(bool); tr = t.tree_; rows = []
    for k in range(tr.node_count):
        m = dp[:, k]; leaf = tr.children_left[k] == -1
        rows.append(dict(node=k, feature=None if leaf else FEATS[tr.feature[k]],
                         threshold=None if leaf else float(tr.threshold[k]),
                         left=int(tr.children_left[k]), right=int(tr.children_right[k]),
                         n=int(m.sum()), n_pos=int(ya[m].sum()), n_neg=int((ya[m] == 0).sum())))
    return rows


def gini(pos, n):
    p = pos / n; return 1 - p ** 2 - (1 - p) ** 2


def sec_A():
    r = {}
    t2 = DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(X, y)
    r['depth2_train_nodes'] = node_table(t2, X, y)
    r['depth2_valid_nodes'] = node_table(t2, Xv, yv)
    nodes = r['depth2_train_nodes']
    root, L, R = nodes[0], nodes[nodes[0]['left']], nodes[nodes[0]['right']]
    r['root_gini'] = gini(root['n_pos'], root['n'])
    r['root_split_weighted_child_gini'] = (L['n'] * gini(L['n_pos'], L['n']) + R['n'] * gini(R['n_pos'], R['n'])) / root['n']
    # 验证集里每个叶节点挑几个样本，便于手推
    leaf = t2.apply(Xv); prob = t2.predict_proba(Xv)[:, 1]
    ex = valid[['sample_id', GROUP] + ['Mode', 'Max', 'ALTV', 'Mean', 'ASTV', 'DP'] + [LABEL]].copy()
    ex['leaf'] = leaf; ex['p'] = prob
    r['valid_examples'] = ex.groupby('leaf').head(4).to_dict('records')
    # 深度对照
    rows = []
    for d in [1, 2, 3, 4, 5, 6, 8, 10, None]:
        t = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(X, y)
        pt, pv = t.predict(X), t.predict(Xv); sv = t.predict_proba(Xv)[:, 1]
        rows.append(dict(max_depth='none' if d is None else d, leaves=int(t.get_n_leaves()), actual_depth=int(t.get_depth()),
                         train_acc=accuracy_score(y, pt), train_ba=balanced_accuracy_score(y, pt),
                         val_acc=accuracy_score(yv, pv), val_ba=balanced_accuracy_score(yv, pv),
                         val_auc=roc_auc_score(yv, sv), val_pr_auc=average_precision_score(yv, sv)))
    r['depth_table'] = rows
    # bootstrap 后根节点是否变化（深度 2）
    rng = np.random.default_rng(SEED); roots = []; classes = []
    for b in range(30):
        idx = rng.choice(len(y), len(y), replace=True)
        t = DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(X.iloc[idx], y[idx]); tr = t.tree_
        L_, R_ = tr.children_left[0], tr.children_right[0]
        roots.append(dict(b=b, root=FEATS[tr.feature[0]], root_thr=float(tr.threshold[0]),
                          left=FEATS[tr.feature[L_]] if tr.children_left[L_] != -1 else None,
                          right=FEATS[tr.feature[R_]] if tr.children_left[R_] != -1 else None))
        classes.append(t.predict(Xv))
    rt = pd.DataFrame(roots); rt.to_csv(OUT / 'A-bootstrap-roots.csv', index=False)
    r['bootstrap_root_counts_first10'] = rt.head(10)['root'].value_counts().to_dict()
    r['bootstrap_root_counts_30'] = rt['root'].value_counts().to_dict()
    r['bootstrap_root_threshold_range'] = {f: [float(s.min()), float(s.max())] for f, s in rt.groupby('root')['root_thr']}
    r['bootstrap_structures_30'] = rt.assign(s=rt.root + '|' + rt.left.astype(str) + '|' + rt.right.astype(str))['s'].value_counts().to_dict()
    r['bootstrap_depth2_flip_rate'] = flip_rate(np.array(classes))
    # 换单位：六个“每秒”计数改成“每小时”（×3600），比较树、未标准化 LR/SVM、标准化 pipeline
    rate = ['AC', 'FM', 'UC', 'DL', 'DS', 'DP']
    X2, Xv2 = X.copy(), Xv.copy(); X2[rate] = X2[rate] * 3600; Xv2[rate] = Xv2[rate] * 3600
    cmp = {}
    for name, m in [('single_tree', DecisionTreeClassifier(max_depth=4, random_state=SEED)),
                    ('random_forest', RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)),
                    ('gradient_boosting', GradientBoostingClassifier(n_estimators=100, random_state=SEED)),
                    ('logistic_unscaled', LogisticRegression(max_iter=20000, random_state=SEED)),
                    ('svm_unscaled', SVC(C=1, kernel='rbf', random_state=SEED)),
                    ('logistic_scaled', make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED))),
                    ('svm_scaled', make_pipeline(StandardScaler(), SVC(C=1, kernel='rbf', random_state=SEED)))]:
        a, b2 = clone(m).fit(X, y), clone(m).fit(X2, y)
        sa, sb = score(a, Xv), score(b2, Xv2); pa, pb = a.predict(Xv), b2.predict(Xv2)
        cmp[name] = dict(max_score_diff=float(np.abs(sa - sb).max()), pred_changed=int((pa != pb).sum()),
                         auc_per_second=roc_auc_score(yv, sa), auc_per_hour=roc_auc_score(yv, sb),
                         ba_per_second=balanced_accuracy_score(yv, pa), ba_per_hour=balanced_accuracy_score(yv, pb),
                         sens_per_second=float(pa[yv == 1].mean()), sens_per_hour=float(pb[yv == 1].mean()))
    r['unit_change'] = cmp
    save('A', r); return r


# ---------------------------------------------------------------- B 投票与方差公式
def per_tree_probs(model, A):
    if isinstance(model, BaggingClassifier):
        return np.array([e.predict_proba(A[:, f])[:, 1] for e, f in zip(model.estimators_, model.estimators_features_)])
    return np.array([e.predict_proba(A)[:, 1] for e in model.estimators_])


def sec_B():
    r = {}
    # B1 工作单表格
    tab = pd.DataFrame(dict(s=list('ABCD'), y=[1, 0, 1, 0], v1=[1, 0, 0, 0], v2=[1, 1, 1, 0], v3=[0, 1, 1, 1],
                            p1=[.8, .2, .35, .15], p2=[.7, .6, .75, .25], p3=[.45, .65, .8, .7]))
    tab['hard'] = (tab[['v1', 'v2', 'v3']].sum(1) >= 2).astype(int)
    tab['soft_mean'] = tab[['p1', 'p2', 'p3']].mean(1); tab['soft'] = (tab.soft_mean >= .5).astype(int)
    r['worksheet_vote'] = tab.to_dict('records')
    # B2 自己数据上的五棵小树
    # 深度 4、叶子至少 10 个片段的五棵小树（叶子不那么纯，概率才不全是 0/1）
    rng = np.random.default_rng(SEED + 8); P = []; C = []
    for b in range(5):
        idx = rng.choice(len(y), len(y), replace=True)
        t = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=SEED).fit(X.iloc[idx], y[idx])
        P.append(t.predict_proba(Xv)[:, 1]); C.append(t.predict(Xv))
    P, C = np.array(P), np.array(C)
    hard = (C.sum(0) >= 3).astype(int); soft_p = P.mean(0); soft = (soft_p >= .5).astype(int)
    r['five_trees_unanimous'] = float(((C.sum(0) == 0) | (C.sum(0) == 5)).mean())
    dis = np.where(hard != soft)[0]
    r['five_trees_disagree_n'] = int(len(dis))
    r['five_trees_disagree'] = [dict(sample_id=valid.sample_id.iloc[i], y=int(yv[i]), probs=P[:, i].round(3).tolist(),
                                     classes=C[:, i].tolist(), hard=int(hard[i]), soft_mean=float(soft_p[i]))
                                for i in dis]
    r['five_trees_single'] = [dict(ba=balanced_accuracy_score(yv, C[b]), sens=float(C[b][yv == 1].mean()),
                                   auc=roc_auc_score(yv, P[b])) for b in range(5)]
    r['five_trees_hard'] = dict(ba=balanced_accuracy_score(yv, hard), sens=float(hard[yv == 1].mean()),
                                auc_vote_share=roc_auc_score(yv, C.mean(0)))
    r['five_trees_soft'] = dict(ba=balanced_accuracy_score(yv, soft), sens=float(soft[yv == 1].mean()),
                                auc=roc_auc_score(yv, soft_p))
    r['five_trees_n_distinct_prob_values'] = dict(single=[int(len(np.unique(p))) for p in P],
                                                  soft=int(len(np.unique(soft_p.round(6)))))
    # B3 方差公式：同一批 30 份 bootstrap 训练集，每份训练 100 棵树
    resamples = bootstrap_resamples(30)
    # 每份训练集用不同的算法种子：公式里的 σ² 和 ρ 同时包含“换训练集”和“树自己的随机性”（ESL 15.2 的设定）
    configs = {'bagging': lambda rs=SEED: BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100,
                                                             max_features=1.0, random_state=rs, n_jobs=-1)}
    for k in [1, 2, 4, 8, 13, 21]:
        configs[f'rf_mf{k}'] = (lambda rs=SEED, k=k: RandomForestClassifier(n_estimators=100, max_features=k, random_state=rs, n_jobs=-1))
    Bs = [1, 2, 3, 5, 10, 20, 50, 100]; dec = {}; curves = []
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, make in configs.items():
        t0 = time.time()
        P = np.array([per_tree_probs(make(SEED + d).fit(X.iloc[idx], y[idx]), XvN) for d, idx in enumerate(resamples)])  # (30,100,424)
        sigma2 = P.reshape(-1, P.shape[-1]).var(0)
        within = P.var(1, ddof=1).mean(0)
        between = P.mean(1).var(0, ddof=1)
        rho_s2 = between - within / P.shape[1]
        s2, rs2 = float(sigma2.mean()), float(rho_s2.mean())
        # 实测：把 100 棵树切成 100/B 组互不重叠的 B 棵树小森林，各自算跨训练集的方差再平均
        obs = {B: float(np.mean([P[:, j * B:(j + 1) * B, :].mean(1).var(0, ddof=1).mean() for j in range(100 // B)])) for B in Bs}
        pred = {B: rs2 + (s2 - rs2) / B for B in Bs}
        # 原训练集上拟合一次：验证表现、树间“错误”相关（各树残差在验证样本上的平均两两相关）
        m = make().fit(X, y); Pt = per_tree_probs(m, XvN); resid = Pt - yv
        cc = np.corrcoef(resid); err_corr = float(cc[~np.eye(len(cc), dtype=bool)].mean())
        pm = Pt.mean(0)
        cvs = []
        for tr_i, te_i in sgk.split(Xd, yd, gd):
            mm = make().fit(Xd.iloc[tr_i], yd[tr_i]); s = mm.predict_proba(Xd.iloc[te_i])[:, 1]
            cvs.append((roc_auc_score(yd[te_i], s), average_precision_score(yd[te_i], s)))
        cvs = np.array(cvs)
        dec[name] = dict(sigma2=s2, rho_sigma2=rs2, rho=rs2 / s2, observed=obs, formula=pred,
                         floor=rs2, tree_error_corr=err_corr, val_auc=roc_auc_score(yv, pm),
                         val_pr_auc=average_precision_score(yv, pm), val_ba=balanced_accuracy_score(yv, (pm >= .5).astype(int)),
                         cv_auc_mean=float(cvs[:, 0].mean()), cv_auc_sd=float(cvs[:, 0].std(ddof=1)),
                         cv_pr_mean=float(cvs[:, 1].mean()), seconds=time.time() - t0)
        for B in Bs: curves.append(dict(config=name, B=B, observed=obs[B], formula=pred[B]))
        print(f'  B3 {name}: rho={rs2 / s2:.3f} sigma2={s2:.4f} var100={obs[100]:.5f} ({time.time() - t0:.0f}s)', flush=True)
    r['variance_decomposition'] = dec
    pd.DataFrame(curves).to_csv(OUT / 'B-variance-curves.csv', index=False)
    save('B', r); return r


# ---------------------------------------------------------------- C bootstrap 与 OOB
def sec_C():
    r = {}
    r['slide_example'] = dict(draw=[2, 2, 5, 1, 2], counts={i: [2, 2, 5, 1, 2].count(i) for i in range(1, 6)})
    sim = {}
    for n in [5, len(y)]:
        rng = np.random.default_rng(SEED)
        fr = np.array([len(np.unique(rng.integers(0, n, n))) / n for _ in range(1000)])
        sim[n] = dict(mean=fr.mean(), sd=fr.std(ddof=1), min=fr.min(), max=fr.max(), theory=1 - (1 - 1 / n) ** n,
                      oob_mean=1 - fr.mean())
    r['unique_fraction_sim'] = sim
    # 行级 bootstrap 下袋外片段的“兄弟片段”是否在袋内
    rng = np.random.default_rng(SEED); sib = []; rec_oob = []
    recs = pd.Series(g)
    for _ in range(1000):
        inbag = np.zeros(len(y), bool); inbag[rng.integers(0, len(y), len(y))] = True
        inbag_recs = set(g[inbag]); oob = ~inbag
        sib.append(np.mean([gi in inbag_recs for gi in g[oob]]))
        rec_oob.append(1 - len(inbag_recs) / recs.nunique())
    r['row_bootstrap_oob_rows_with_inbag_sibling'] = dict(mean=float(np.mean(sib)), min=float(np.min(sib)))
    r['row_bootstrap_recordings_fully_oob'] = float(np.mean(rec_oob))
    # OOB 与验证、分组 CV（都只用 train）
    rf = RandomForestClassifier(n_estimators=300, max_features='sqrt', oob_score=True, random_state=SEED, n_jobs=-1).fit(X, y)
    op = rf.oob_decision_function_[:, 1]
    r['rf300_row_oob'] = dict(accuracy=rf.oob_score_, auc=roc_auc_score(y, op), pr_auc=average_precision_score(y, op),
                              ba=balanced_accuracy_score(y, (op >= .5).astype(int)),
                              sensitivity=float((op >= .5)[y == 1].mean()))
    vp = rf.predict_proba(Xv)[:, 1]
    r['rf300_validation'] = dict(accuracy=accuracy_score(yv, vp >= .5), auc=roc_auc_score(yv, vp),
                                 pr_auc=average_precision_score(yv, vp), ba=balanced_accuracy_score(yv, (vp >= .5).astype(int)),
                                 sensitivity=float((vp >= .5)[yv == 1].mean()))
    # 按录音做 bootstrap 的 OOB（手写 300 棵 sqrt 特征树）
    urec = np.unique(g); rec_index = {k: np.where(g == k)[0] for k in urec}
    rng = np.random.default_rng(SEED); acc = np.zeros(len(y)); cnt = np.zeros(len(y))
    for b in range(300):
        draw = rng.choice(urec, len(urec), replace=True)
        w = np.zeros(len(y));
        for k in draw: w[rec_index[k]] += 1
        t = DecisionTreeClassifier(max_features='sqrt', random_state=SEED + b).fit(XN, y, sample_weight=w)
        oob = w == 0
        acc[oob] += t.predict_proba(XN[oob])[:, 1]; cnt[oob] += 1
    ok = cnt > 0; gp = acc[ok] / cnt[ok]
    r['rf300_recording_oob'] = dict(coverage=float(ok.mean()), accuracy=accuracy_score(y[ok], gp >= .5),
                                    auc=roc_auc_score(y[ok], gp), pr_auc=average_precision_score(y[ok], gp),
                                    ba=balanced_accuracy_score(y[ok], (gp >= .5).astype(int)),
                                    sensitivity=float((gp >= .5)[y[ok] == 1].mean()))
    # 训练集内分组 5 折 CV
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED); s = np.zeros(len(y))
    for a, b in sgk.split(X, y, g):
        s[b] = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X.iloc[a], y[a]).predict_proba(X.iloc[b])[:, 1]
    r['rf300_train_grouped_cv_pooled'] = dict(accuracy=accuracy_score(y, s >= .5), auc=roc_auc_score(y, s),
                                              pr_auc=average_precision_score(y, s), ba=balanced_accuracy_score(y, (s >= .5).astype(int)),
                                              sensitivity=float((s >= .5)[y == 1].mean()))
    save('C', r); return r


# ---------------------------------------------------------------- D 单树 / bagging / RF
def sec_D():
    r = {}
    rows = []
    for n in [1, 2, 5, 10, 20, 50, 100, 200, 300, 500]:
        for name, m in (('bagging', BaggingClassifier(estimator=DecisionTreeClassifier(random_state=SEED), n_estimators=n,
                                                      oob_score=n >= 20, random_state=SEED, n_jobs=-1)),
                        ('random_forest', RandomForestClassifier(n_estimators=n, max_features='sqrt', oob_score=n >= 20,
                                                                 random_state=SEED, n_jobs=-1))):
            m.fit(X, y); p = m.predict_proba(Xv)[:, 1]
            row = dict(model=name, n_trees=n, val_auc=roc_auc_score(yv, p), val_pr_auc=average_precision_score(yv, p),
                       val_ba=balanced_accuracy_score(yv, (p >= .5).astype(int)), oob_acc=np.nan, oob_auc=np.nan)
            if n >= 20:
                od = m.oob_decision_function_[:, 1]; ok = ~np.isnan(od)
                row.update(oob_acc=m.oob_score_, oob_auc=roc_auc_score(y[ok], od[ok]))
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / 'D-tree-count.csv', index=False)
    r['tree_count'] = rows
    # bagging 与 RF 的差别：根节点用了什么、验证集上概率为 0 的并列
    bag = models()['bagging'].fit(X, y); rf = models()['random_forest'].fit(X, y)
    bag_root = pd.Series([FEATS[f[e.tree_.feature[0]]] for e, f in zip(bag.estimators_, bag.estimators_features_)]).value_counts()
    rf_root = pd.Series([FEATS[e.tree_.feature[0]] for e in rf.estimators_]).value_counts()
    r['root_feature_bagging'] = bag_root.to_dict(); r['root_feature_rf'] = rf_root.to_dict()
    pb, pr = bag.predict_proba(Xv)[:, 1], rf.predict_proba(Xv)[:, 1]
    for name, p in [('bagging', pb), ('random_forest', pr)]:
        r[f'{name}_val_prob'] = dict(n_zero=int((p == 0).sum()), n_zero_pos=int(((p == 0) & (yv == 1)).sum()),
                                     n_le_0_05_pos=int(((p <= .05) & (yv == 1)).sum()),
                                     n_unique=int(len(np.unique(p))), pos_probs=np.sort(p[yv == 1]).round(2).tolist())
    # 各录音里的病理片段：bagging / RF 概率
    vv = valid[['sample_id', GROUP, LABEL, 'DP', 'ALTV', 'ASTV', 'Mode', 'Mean']].copy(); vv['p_bag'] = pb; vv['p_rf'] = pr
    vv[vv[LABEL] == 1].sort_values('p_bag').to_csv(OUT / 'D-valid-positives.csv', index=False)
    r['positives_bag_low_rf_high'] = vv[(vv[LABEL] == 1) & (vv.p_bag <= .1) & (vv.p_rf >= .3)].to_dict('records')
    # max_features=None 的 RF（等价于 bagging，用同一实现隔离特征随机化）
    rf_all = RandomForestClassifier(n_estimators=100, max_features=None, random_state=SEED, n_jobs=-1).fit(X, y)
    pa = rf_all.predict_proba(Xv)[:, 1]
    r['rf_max_features_all'] = dict(val_auc=roc_auc_score(yv, pa), val_pr_auc=average_precision_score(yv, pa))
    # 特征重要性
    forest = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X, y)
    perm = permutation_importance(forest, Xv, yv, scoring='roc_auc', n_repeats=20, random_state=SEED, n_jobs=1)
    imp = pd.DataFrame(dict(feature=FEATS, impurity=forest.feature_importances_, perm_mean=perm.importances_mean,
                            perm_sd=perm.importances_std)).sort_values('impurity', ascending=False)
    imp['impurity_rank'] = np.arange(1, len(imp) + 1)
    imp['perm_rank'] = imp.perm_mean.rank(ascending=False, method='first').astype(int)
    imp.to_csv(OUT / 'D-importance.csv', index=False)
    r['importance'] = imp.to_dict('records')
    base = roc_auc_score(yv, forest.predict_proba(Xv)[:, 1]); rng = np.random.default_rng(SEED); joint = {}
    for cols in [['Mode'], ['Mean'], ['Median'], ['Mode', 'Mean', 'Median'], ['Width'], ['Min'], ['Width', 'Min'],
                 ['ALTV'], ['ASTV'], ['ALTV', 'ASTV']]:
        drops = []
        for _ in range(20):
            Xp = Xv.copy(); Xp[cols] = Xv[cols].to_numpy()[rng.permutation(len(Xv))]
            drops.append(base - roc_auc_score(yv, forest.predict_proba(Xp)[:, 1]))
        joint['+'.join(cols)] = dict(mean=float(np.mean(drops)), sd=float(np.std(drops)))
    r['joint_permutation'] = joint; r['importance_base_auc'] = base
    save('D', r); return r


# ---------------------------------------------------------------- E boosting
def sec_E():
    r = {}
    # E1 手写 AdaBoost（SAMME, K=2）跟踪样本权重，并与 sklearn 对照
    n = len(y); w = np.full(n, 1 / n); rounds = []; W = [w.copy()]; miss_hist = []
    for t in range(6):
        st = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(X, y, sample_weight=w)
        miss = st.predict(X) != y; err = w[miss].sum() / w.sum(); alpha = np.log((1 - err) / err)
        w_new = w * np.exp(alpha * miss); w_new /= w_new.sum()
        rounds.append(dict(t=t + 1, feature=FEATS[st.tree_.feature[0]], threshold=float(st.tree_.threshold[0]),
                           left_pred=int(np.argmax(st.tree_.value[1][0])), right_pred=int(np.argmax(st.tree_.value[2][0])),
                           n_miss=int(miss.sum()), n_miss_pos=int((miss & (y == 1)).sum()), n_miss_neg=int((miss & (y == 0)).sum()),
                           err=float(err), alpha=float(alpha), alpha_classic=float(.5 * np.log((1 - err) / err)),
                           up=float(np.exp(alpha)), err_under_new_weights=float(w_new[miss].sum())))
        miss_hist.append(miss); w = w_new; W.append(w.copy())
    W = np.array(W)
    sk = AdaBoostClassifier(n_estimators=6, random_state=SEED).fit(X, y)
    r['adaboost_rounds'] = rounds
    r['adaboost_sklearn'] = dict(errors=sk.estimator_errors_.tolist(), weights=sk.estimator_weights_.tolist(),
                                 features=[FEATS[e.tree_.feature[0]] for e in sk.estimators_])
    # 跟踪样本：第 1 轮被错分的病理片段前两个、正常片段第一个，第 1 轮判对的病理和正常片段各一个（按 sample_id 顺序）
    m1 = miss_hist[0]; order = np.argsort(train.sample_id.to_numpy())
    pick = []
    for cond, k in [((y == 1) & m1, 2), ((y == 0) & m1, 1), ((y == 1) & ~m1, 1), ((y == 0) & ~m1, 1)]:
        pick += [i for i in order if cond[i]][:k]
    r['adaboost_tracked'] = [dict(sample_id=train.sample_id.iloc[i], y=int(y[i]), weight_x_n=(W[:, i] * n).round(3).tolist(),
                                  missed=[bool(mh[i]) for mh in miss_hist]) for i in pick]
    pd.DataFrame(W[:, pick] * n, columns=[train.sample_id.iloc[i] for i in pick]).to_csv(OUT / 'E-adaboost-weights.csv', index_label='round')
    r['adaboost_weight_share_pos'] = [float(Wt[y == 1].sum()) for Wt in W]
    # E2 梯度提升：初值、负梯度 y-p、第一棵树叶值（牛顿步）、跟踪样本
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=.1, max_depth=3, random_state=SEED).fit(X, y)
    p0 = y.mean(); F0 = np.log(p0 / (1 - p0))
    r['gb_init'] = dict(p0=p0, F0=F0, sklearn_p0=float(gb.init_.predict_proba(X.iloc[:1])[0, 1]),
                        resid_pos=1 - p0, resid_neg=-p0)
    t1 = gb.estimators_[0, 0]; leaf = t1.apply(XN.astype(np.float32)); resid = y - p0; leaves = []
    for lf in np.unique(leaf):
        m = leaf == lf
        newton = resid[m].sum() / (p0 * (1 - p0) * m.sum())
        leaves.append(dict(leaf=int(lf), n=int(m.sum()), n_pos=int(y[m].sum()), mean_resid=float(resid[m].mean()),
                           newton=float(newton), sklearn_value=float(t1.tree_.value[lf][0][0])))
    r['gb_first_tree_leaves'] = leaves
    r['gb_first_tree_root'] = dict(feature=FEATS[t1.tree_.feature[0]], threshold=float(t1.tree_.threshold[0]))
    stages = [0, 1, 2, 5, 10, 20, 50, 100]
    sp = np.array([np.full(len(yv), p0)] + [p[:, 1] for p in gb.staged_predict_proba(Xv)])
    pick_v = []
    pv = sp[-1]
    for cond in [(yv == 1) & (pv > .9), (yv == 1) & (pv > .3) & (pv < .7), (yv == 1) & (pv < .1), (yv == 0) & (pv > .3)]:
        idx = [i for i in np.argsort(valid.sample_id.to_numpy()) if cond[i]]
        if idx: pick_v.append(idx[0])
    r['gb_tracked'] = [dict(sample_id=valid.sample_id.iloc[i], y=int(yv[i]), p=[float(sp[s, i]) for s in stages],
                            resid=[float(yv[i] - sp[s, i]) for s in stages]) for i in pick_v]
    r['gb_tracked_stages'] = stages
    # E3 learning_rate × 轮数
    curves = {}; summ = {}
    for lr in [.05, .1, .3, 1.0]:
        m = GradientBoostingClassifier(n_estimators=500, learning_rate=lr, max_depth=3, random_state=SEED).fit(X, y)
        tr_ll = np.array([log_loss(y, p[:, 1], labels=[0, 1]) for p in m.staged_predict_proba(X)])
        va_p = [p[:, 1] for p in m.staged_predict_proba(Xv)]
        va_ll = np.array([log_loss(yv, p, labels=[0, 1]) for p in va_p])
        va_auc = np.array([roc_auc_score(yv, p) for p in va_p]); va_pr = np.array([average_precision_score(yv, p) for p in va_p])
        va_ba = np.array([balanced_accuracy_score(yv, (p >= .5).astype(int)) for p in va_p])
        curves[lr] = dict(train_ll=tr_ll, val_ll=va_ll, val_auc=va_auc, val_pr=va_pr, val_ba=va_ba)
        b = int(va_ll.argmin())
        summ[lr] = dict(best_round=b + 1, best_val_ll=float(va_ll[b]), val_auc_at_best=float(va_auc[b]),
                        val_ll_100=float(va_ll[99]), val_ll_500=float(va_ll[-1]), val_auc_100=float(va_auc[99]),
                        val_auc_500=float(va_auc[-1]), val_pr_best=float(va_pr[b]), val_pr_500=float(va_pr[-1]),
                        val_ba_best=float(va_ba[b]), val_ba_500=float(va_ba[-1]),
                        train_ll_100=float(tr_ll[99]), train_ll_500=float(tr_ll[-1]),
                        max_auc=float(va_auc.max()), max_auc_round=int(va_auc.argmax()) + 1)
    np.savez(OUT / 'E-gb-curves.npz', **{f'{k}_{lr}': v for lr, d in curves.items() for k, v in d.items()})
    r['gb_lr_summary'] = summ
    # sklearn 自带 early stopping（按行随机留出 20%）停在哪一轮
    es = GradientBoostingClassifier(n_estimators=500, learning_rate=.1, n_iter_no_change=10, validation_fraction=.2,
                                    random_state=SEED).fit(X, y)
    r['gb_sklearn_early_stopping_rounds'] = int(es.n_estimators_)
    save('E', r); return r


# ---------------------------------------------------------------- F 公平比较
def sec_F():
    r = {}
    ms = models(); ms['hist_gb'] = HistGradientBoostingClassifier(random_state=SEED)
    val = {}
    for name, m in ms.items():
        m.fit(X, y); val[name] = metrics(yv, score(m, Xv), m.predict(Xv))
    r['validation'] = val
    folds = []
    for seed in range(SEED, SEED + 5):
        for scheme, cv in [('grouped', StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)),
                           ('row', StratifiedKFold(n_splits=5, shuffle=True, random_state=seed))]:
            if scheme == 'row' and seed != SEED: continue
            for k, (a, b) in enumerate(cv.split(Xd, yd, gd)):
                for name, m in ms.items():
                    mm = clone(m).fit(Xd.iloc[a], yd[a]); s = score(mm, Xd.iloc[b]); pr = mm.predict(Xd.iloc[b])
                    folds.append(dict(scheme=scheme, seed=seed, fold=k, model=name, roc_auc=roc_auc_score(yd[b], s),
                                      pr_auc=average_precision_score(yd[b], s), ba=balanced_accuracy_score(yd[b], pr),
                                      sensitivity=float(pr[yd[b] == 1].mean()), n_pos=int(yd[b].sum()),
                                      n_pos_rec=int(len(np.unique(gd[b][yd[b] == 1])))))
        print(f'  F seed {seed} done', flush=True)
    fd = pd.DataFrame(folds); fd.to_csv(OUT / 'F-folds.csv', index=False)
    agg = lambda d: d.groupby('model')[['roc_auc', 'pr_auc', 'ba', 'sensitivity']].agg(['mean', 'std'])
    first = fd[(fd.scheme == 'grouped') & (fd.seed == SEED)]
    r['grouped_cv_seed1010'] = {m: {f'{a}_{b}': v for (a, b), v in row.items()} for m, row in agg(first).iterrows()}
    rep = fd[fd.scheme == 'grouped']
    r['grouped_cv_5x5'] = {m: {f'{a}_{b}': v for (a, b), v in row.items()} for m, row in agg(rep).iterrows()}
    rowcv = fd[fd.scheme == 'row']
    r['row_cv_seed1010'] = {m: {f'{a}_{b}': v for (a, b), v in row.items()} for m, row in agg(rowcv).iterrows()}
    r['fold_pos_recordings'] = first.drop_duplicates('fold')[['fold', 'n_pos', 'n_pos_rec']].to_dict('records')
    pv = rep.pivot_table(index=['seed', 'fold'], columns='model', values=['roc_auc', 'pr_auc', 'ba'])
    paired = {}
    for a, b in [('random_forest', 'logistic'), ('gradient_boosting', 'logistic'), ('random_forest', 'rbf_svm'),
                 ('random_forest', 'bagging'), ('random_forest', 'gradient_boosting'), ('random_forest', 'single_tree')]:
        paired[f'{a}-{b}'] = {met: dict(mean=float((pv[met][a] - pv[met][b]).mean()), sd=float((pv[met][a] - pv[met][b]).std(ddof=1)),
                                        n_pos=int(((pv[met][a] - pv[met][b]) > 0).sum()), n=int(len(pv)))
                              for met in ['roc_auc', 'pr_auc', 'ba']}
    r['paired_5x5'] = paired
    save('F', r); return r


# ---------------------------------------------------------------- G 稳定性与不确定性
def sec_G():
    r = {}
    resamples = bootstrap_resamples(30)
    rec_rng = np.random.default_rng(SEED); urec = np.unique(g); rec_index = {k: np.where(g == k)[0] for k in urec}
    rec_resamples = [np.concatenate([rec_index[k] for k in rec_rng.choice(urec, len(urec), replace=True)]) for _ in range(30)]
    for label, rs in [('row_bootstrap', resamples), ('recording_bootstrap', rec_resamples)]:
        out = {}
        names = list(models()) if label == 'row_bootstrap' else ['single_tree', 'bagging', 'random_forest', 'gradient_boosting', 'logistic']
        for name in names:
            P, C, A = [], [], []
            for idx in rs:
                m = clone(models()[name]).fit(X.iloc[idx], y[idx]); C.append(m.predict(Xv)); A.append(roc_auc_score(yv, score(m, Xv)))
                if name in PROBA_MODELS: P.append(m.predict_proba(Xv)[:, 1])
            C = np.array(C); A = np.array(A); np.save(OUT / f'G-auc-{label}-{name}.npy', A)
            d = dict(flip_rate=flip_rate(C), val_auc_mean=A.mean(), val_auc_sd=A.std(ddof=1), val_auc_min=A.min(),
                     flip_rate_pos=flip_rate(C[:, yv == 1]))
            if P:
                P = np.array(P); sd = P.std(0)
                d.update(sd_mean=sd.mean(), sd_max=sd.max(), sd_mean_pos=sd[yv == 1].mean())
                if label == 'row_bootstrap': np.save(OUT / f'G-sd-{name}.npy', sd)
            out[name] = d
            print(f'  G {label} {name} flip={d["flip_rate"]:.3f}', flush=True)
        r[label] = out
    # 算法随机性：训练数据固定，只换 random_state
    seeds = {}
    for name in ['single_tree', 'bagging', 'random_forest', 'adaboost', 'gradient_boosting']:
        P, C = [], []
        for s in range(10):
            m = clone(models()[name]).set_params(random_state=s)
            m.fit(X, y); P.append(m.predict_proba(Xv)[:, 1]); C.append(m.predict(Xv))
        P, C = np.array(P), np.array(C)
        seeds[name] = dict(sd_mean=P.std(0).mean(), flip_rate=flip_rate(C))
    r['algorithm_seed_only'] = seeds
    # 固定已拟合模型：在验证集上按录音（分层：含病理 / 不含病理）重抽样 2000 次，与按片段重抽样对比
    fitted = {k: models()[k].fit(X, y) for k in ['random_forest', 'gradient_boosting', 'logistic', 'rbf_svm', 'bagging', 'single_tree']}
    S = {k: score(m, Xv) for k, m in fitted.items()}; Pd = {k: m.predict(Xv) for k, m in fitted.items()}
    pos_rec = np.unique(gv[yv == 1]); neg_rec = np.setdiff1d(np.unique(gv), pos_rec)
    vidx = {k: np.where(gv == k)[0] for k in np.unique(gv)}
    rng = np.random.default_rng(SEED); res = {'recording': [], 'segment': []}
    pos_i, neg_i = np.where(yv == 1)[0], np.where(yv == 0)[0]
    for _ in range(2000):
        idx_r = np.concatenate([vidx[k] for k in np.concatenate([rng.choice(pos_rec, len(pos_rec)), rng.choice(neg_rec, len(neg_rec))])])
        idx_s = np.concatenate([rng.choice(pos_i, len(pos_i)), rng.choice(neg_i, len(neg_i))])
        for kind, idx in [('recording', idx_r), ('segment', idx_s)]:
            yy = yv[idx]; row = {}
            for k in S:
                row[f'{k}_auc'] = roc_auc_score(yy, S[k][idx]); row[f'{k}_pr'] = average_precision_score(yy, S[k][idx])
                row[f'{k}_ba'] = balanced_accuracy_score(yy, Pd[k][idx])
            res[kind].append(row)
    ci = {}
    for kind in res:
        d = pd.DataFrame(res[kind])
        for a, b in [('random_forest', 'logistic'), ('gradient_boosting', 'logistic'), ('random_forest', 'bagging')]:
            for met in ['auc', 'pr', 'ba']:
                d[f'{a}-{b}_{met}'] = d[f'{a}_{met}'] - d[f'{b}_{met}']
        ci[kind] = {c: dict(point=None, lo=float(d[c].quantile(.025)), hi=float(d[c].quantile(.975)),
                            share_gt0=float((d[c] > 0).mean())) for c in d.columns}
    for k in S:
        for met, v in [('auc', roc_auc_score(yv, S[k])), ('pr', average_precision_score(yv, S[k])),
                       ('ba', balanced_accuracy_score(yv, Pd[k]))]:
            for kind in ci: ci[kind][f'{k}_{met}']['point'] = v
    r['validation_bootstrap_ci'] = ci
    r['validation_pos_recordings'] = int(len(pos_rec)); r['validation_neg_only_recordings'] = int(len(neg_rec))
    save('G', r); return r


SECTIONS = dict(protocol=sec_protocol, A=sec_A, B=sec_B, C=sec_C, D=sec_D, E=sec_E, F=sec_F, G=sec_G)
if __name__ == '__main__':
    import sklearn
    todo = sys.argv[1:] or list(SECTIONS)
    print('sklearn', sklearn.__version__, 'pandas', pd.__version__, flush=True)
    for s in todo:
        t0 = time.time(); SECTIONS[s](); print(f'[{s}] done in {time.time() - t0:.0f}s', flush=True)
