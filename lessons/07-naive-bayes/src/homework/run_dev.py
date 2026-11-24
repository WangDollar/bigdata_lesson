"""Run notebook parts A, A2, B, C (no test data) and dump every number needed for the write-up."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.naive_bayes import BernoulliNB
BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / 'src/course'))
from lab_utils import *

pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
train = pd.read_csv(BASE / 'data/student/train.csv')
val = pd.read_csv(BASE / 'data/student/validation.csv')
T = 'bladder_inflammation'
out = {}

print('== A')
for name, f in [('train', train), ('val', val)]:
    print(name, len(f), f[T].value_counts().to_dict(), 'profiles', len(f[FEATURES].drop_duplicates()),
          'extra dup', int(f.drop(columns='sample_id').duplicated().sum()))
print('id overlap', len(set(train.sample_id) & set(val.sample_id)))
seen = set(map(tuple, train[FEATURES].to_numpy()))
print('val seen', sum(tuple(x) in seen for x in val[FEATURES].to_numpy()))
print(train.head())
prof = train.groupby(FEATURES)[T].agg(['count', 'sum']).reset_index()
print(prof)
x = (0, 0, 0, 1, 1, 1)
print('x in train:', train[(train[FEATURES] == list(x)).all(axis=1)][T].value_counts().to_dict())

print('== A2')
model = fit_model(train)
hand, w, post = hand_calculation(train, list(x))
print(hand)
print('weights', w, 'posterior', post)
print('sklearn', model.predict_proba(pd.DataFrame([x], columns=FEATURES))[0, 1])
print('class_log_prior', np.exp(model.class_log_prior_))

print('== B')
vp = model.predict_proba(val[FEATURES])[:, 1]
tp_ = model.predict_proba(train[FEATURES])[:, 1]
bp = np.full(len(val), train[T].mean())
print(pd.DataFrame({'NB_train': scores(train[T], tp_), 'NB_val': scores(val[T], vp),
                    'prior_val': scores(val[T], bp)}).T)
th = threshold_table(val[T], vp)
print(th[['threshold', 'tn', 'fp', 'fn', 'tp', 'sensitivity', 'specificity']])
print('chosen', choose_threshold(th))
print('val probs by profile:')
v = val.copy(); v['p'] = vp
print(v.groupby(FEATURES + [T])['p'].agg(['count', 'mean']).reset_index())

print('== C')
dep = dependence(train).sort_values('difference', key=abs, ascending=False)
print(dep.head(8))
for copies in [0, 1, 5]:
    tx, vx = train[FEATURES].copy(), val[FEATURES].copy()
    for i in range(copies):
        tx[f'c{i}'] = tx.lumbar_pain; vx[f'c{i}'] = vx.lumbar_pain
    dm = BernoulliNB(alpha=1, binarize=None).fit(tx, train[T])
    dp = dm.predict_proba(vx)[:, 1]
    s = scores(val[T], dp)
    print('copies', copies, {k: round(s[k], 4) for k in ['accuracy', 'log_loss', 'tn', 'fp', 'fn', 'tp']},
          'mean max p', round(np.maximum(dp, 1 - dp).mean(), 4))
    ch = pd.DataFrame({'id': val.sample_id, 'y': val[T], 'orig': vp, 'dup': dp, 'chg': abs(dp - vp)})
    print(ch.sort_values('chg', ascending=False).head(3))
print(reliability(val[T], vp))
