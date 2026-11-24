"""Reproduce the student's primary and sensitivity analyses.

Usage: python analyze.py
Dependencies: numpy, pandas, scipy, statsmodels, matplotlib (see results.json for versions).
All outputs are written beside this script; the input CSV is never changed.
"""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.power import TTestIndPower
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

LESSON = Path(__file__).resolve().parents[2]  # lessons/05-hypothesis-testing
OUT = LESSON / 'outputs' / 'homework'
SOURCE = LESSON / 'data' / 'heart-disease-hypothesis.csv'
df = pd.read_csv(SOURCE)
assert df.record_id.is_unique, 'Repeated record IDs need investigation.'
assert df.disease.notna().all() and df.num.notna().all()
assert set(df.disease.unique()) == {0, 1}
assert (df.disease == (df.num > 0).astype(int)).all()
assert set(df.sex.dropna().unique()) <= {0, 1}
cc = df.dropna(subset=['disease', 'thalach']).copy()

def welch(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n0 = len(a), len(b)
    v1, v0 = a.var(ddof=1), b.var(ddof=1)
    delta = a.mean() - b.mean()
    se2 = v1/n1 + v0/n0
    dof = se2**2 / ((v1/n1)**2/(n1-1) + (v0/n0)**2/(n0-1))
    se = np.sqrt(se2)
    crit = stats.t.ppf(.975, dof)
    test = stats.ttest_ind(a, b, equal_var=False, alternative='two-sided')
    lo, hi = test.confidence_interval(.95)
    # Independent formula checks protect the sign and reported confidence interval.
    assert np.isclose(test.statistic, delta/se)
    assert np.allclose([lo, hi], [delta-crit*se, delta+crit*se])
    assert np.isclose(test.pvalue, 2*stats.t.sf(abs(delta/se), dof))
    pooled_sd = np.sqrt(((n1-1)*v1+(n0-1)*v0)/(n1+n0-2))
    d = delta/pooled_sd
    g = (1-3/(4*(n1+n0-2)-1))*d
    return dict(n1=n1, n0=n0, mean_difference=float(delta), se=float(se),
                t=float(test.statistic), df=float(dof), p=float(test.pvalue),
                ci_low=float(lo), ci_high=float(hi),
                cohen_d=float(d), hedges_g=float(g))

summary = []
for group, gdf in df.groupby('disease', sort=True):
    vals = gdf.thalach.dropna()
    summary.append(dict(disease=int(group), total_n=len(gdf), analysis_n=len(vals),
                        missing_n=int(gdf.thalach.isna().sum()),
                        missing_percent=100*gdf.thalach.isna().mean(),
                        mean=vals.mean(), sd=vals.std(ddof=1), median=vals.median(),
                        q1=vals.quantile(.25), q3=vals.quantile(.75),
                        minimum=vals.min(), maximum=vals.max(),
                        skew=float(stats.skew(vals, bias=False))))
summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT/'group-summary.csv', index=False)
primary = welch(cc.loc[cc.disease == 1, 'thalach'], cc.loc[cc.disease == 0, 'thalach'])

missing_site = df.groupby('site', sort=True).thalach.agg(total_n='size', observed_n='count')
missing_site['missing_n'] = missing_site.total_n - missing_site.observed_n
missing_site['missing_percent'] = missing_site.missing_n/missing_site.total_n*100
missing_site.to_csv(OUT/'missingness-by-site.csv')
missing_cells = df.groupby(['site','disease'], sort=True).thalach.agg(total_n='size', observed_n='count')
missing_cells['missing_n'] = missing_cells.total_n-missing_cells.observed_n
missing_cells.to_csv(OUT/'missingness-by-site-and-disease.csv')

by_site = []
for site, sdf in cc.groupby('site', sort=True):
    by_site.append(dict(site=site, **welch(sdf.loc[sdf.disease==1,'thalach'],
                                        sdf.loc[sdf.disease==0,'thalach'])))
pd.DataFrame(by_site).to_csv(OUT/'site-sensitivity.csv', index=False)

model_data = cc.dropna(subset=['age', 'sex', 'site']).copy()
ols = smf.ols('thalach ~ disease + age + sex + C(site)', data=model_data).fit(cov_type='HC3', use_t=True)
ci = ols.conf_int(alpha=.05)
coefficients = pd.DataFrame(dict(coefficient=ols.params, hc3_se=ols.bse,
                                t=ols.tvalues, p=ols.pvalues,
                                ci_low=ci[0], ci_high=ci[1]))
coefficients.to_csv(OUT/'adjusted-model.csv')
adjusted = dict(n=int(ols.nobs), formula='thalach ~ disease + age + sex + C(site)',
                covariance='HC3', interval_distribution='t', df_resid=float(ols.df_resid),
                coefficient=float(ols.params['disease']), se=float(ols.bse['disease']),
                ci_low=float(ci.loc['disease',0]), ci_high=float(ci.loc['disease',1]),
                p=float(ols.pvalues['disease']), r_squared=float(ols.rsquared),
                covariates={k: dict(coefficient=float(ols.params[k]), p=float(ols.pvalues[k]))
                            for k in ols.params.index if k not in ('Intercept', 'disease')})

a = cc.loc[cc.disease == 1, 'thalach'].to_numpy()
b = cc.loc[cc.disease == 0, 'thalach'].to_numpy()
rank = stats.mannwhitneyu(a,b,alternative='two-sided',method='asymptotic',use_continuity=True)
rank_result = dict(U=float(rank.statistic), p=float(rank.pvalue), method='asymptotic, tie and continuity corrected')
rank_result['prob_disease_higher'] = float(rank.statistic/(len(a)*len(b)))  # ties count one half
numeric = df.select_dtypes(include='number')
numeric.agg(['count','min','max']).T.to_csv(OUT/'field-audit.csv')
clinical = ['site','age','sex','trestbps','chol','thalach','oldpeak','num']
duplicate_profiles = int(df.duplicated(subset=clinical).sum())
duplicate_groups = [sorted(g.record_id) for _, g in
                    df[df.duplicated(subset=clinical, keep=False)].groupby(clinical, dropna=False)]
# Zero values are physiologically impossible, so they are reported as disguised missing values.
implausible_zeros = dict(chol_zero_by_site={k: int(v) for k, v in df[df.chol == 0].site.value_counts().items()},
                         trestbps_zero=int((df.trestbps == 0).sum()))

outliers = {}
for group, vals in cc.groupby('disease').thalach:
    q1, q3 = vals.quantile([.25, .75])
    fence = 1.5*(q3-q1)
    outliers[int(group)] = sorted(float(v) for v in vals[(vals < q1-fence) | (vals > q3+fence)])

balance = {int(g): dict(age_mean=float(s.age.mean()), male_percent=100*float(s.sex.mean()))
           for g, s in cc.groupby('disease')}
site_disease = {k: dict(n=int(s.size), disease_percent=100*float(s.mean()))
                for k, s in cc.groupby('site').disease}

# Additions after the first results: they describe missingness and robustness, not a new primary test.
miss = df.thalach.isna()
def profile(frame):
    return dict(n=len(frame), age_mean=float(frame.age.mean()),
                male_percent=100*float(frame.sex.mean()), disease_percent=100*float(frame.disease.mean()))
va = df.site == 'VA Long Beach'
missing_profile = dict(observed=profile(df[~miss]), missing=profile(df[miss]),
                       va_observed=profile(df[va & ~miss]), va_missing=profile(df[va & miss]),
                       trestbps_also_missing=int(df.loc[miss, 'trestbps'].isna().sum()),
                       oldpeak_also_missing=int(df.loc[miss, 'oldpeak'].isna().sum()))
no_va = cc[cc.site != 'VA Long Beach']
exclude_va = welch(no_va.loc[no_va.disease == 1, 'thalach'], no_va.loc[no_va.disease == 0, 'thalach'])
inter = smf.ols('thalach ~ disease * C(site) + age + sex', data=model_data).fit(cov_type='HC3', use_t=True)
terms = [name for name in inter.params.index if name.startswith('disease:')]
joint = inter.wald_test(', '.join(f'{name} = 0' for name in terms), scalar=True)
interaction = dict(formula='thalach ~ disease * C(site) + age + sex', covariance='HC3',
                   F=float(joint.statistic), df_num=int(joint.df_num), df_denom=float(joint.df_denom),
                   p=float(joint.pvalue))

# Design-stage examples: this sample's pooled SD is used only as a planning value for a future study.
n1, n0, v1, v0 = len(a), len(b), a.var(ddof=1), b.var(ddof=1)
pooled_sd = float(np.sqrt(((n1-1)*v1+(n0-1)*v0)/(n1+n0-2)))
solver = TTestIndPower()
def n_per_group(delta, sd):
    return int(np.ceil(solver.solve_power(effect_size=delta/sd, alpha=.05, power=.8, alternative='two-sided')))
power_examples = dict(planning_sd=pooled_sd, alpha=.05, power=.8,
                      n_per_group_delta10=n_per_group(10, pooled_sd),
                      n_per_group_delta5=n_per_group(5, pooled_sd),
                      n_per_group_delta10_double_sd=n_per_group(10, 2*pooled_sd))

# Word for Mac ships DengXian, the same Chinese font used by the answer document.
word_fonts = Path('/Applications/Microsoft Word.app/Contents/Resources/DFonts')
for font_file in word_fonts.glob('Deng*.ttf'):
    font_manager.fontManager.addfont(str(font_file))
for family in ('DengXian', 'Microsoft YaHei', 'PingFang SC', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS'):
    try:
        font_manager.findfont(family, fallback_to_default=False); break
    except ValueError:
        continue
else:
    family = 'sans-serif'
plt.rcParams.update({'font.family': family, 'axes.unicode_minus': False, 'font.size': 8.5})
ink, muted, grid = '#0b0b0b', '#52514e', '#e4e3df'
fig, axes = plt.subplots(2, 1, figsize=(6.3, 3.3), sharex=True, sharey=True, dpi=220)
# 10-beat bins each hold one multiple of 5 and one of 10, so heaping at round values does not create a comb.
bins = np.arange(55, 215, 10)
for ax, group, color, label in ((axes[0], 0, '#2a78d6', '无疾病证据组 disease=0'),
                                (axes[1], 1, '#eb6834', '疾病组 disease=1')):
    vals = cc.loc[cc.disease == group, 'thalach']
    ax.hist(vals, bins=bins, color=color, edgecolor='white', linewidth=.8)
    ax.axvline(vals.mean(), color=ink, linewidth=1, linestyle=(0, (3, 2)))
    ax.text(.01, .92, f'{label}，n={len(vals)}，均值 {vals.mean():.1f}（虚线）', transform=ax.transAxes, color=ink, va='top')
    ax.set_ylabel('人数', color=muted)
    ax.grid(axis='y', color=grid, linewidth=.6); ax.set_axisbelow(True)
    for side in ('top', 'right'): ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'): ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted, length=0)
tallest = max(np.histogram(cc.loc[cc.disease == g, 'thalach'], bins)[0].max() for g in (0, 1))
axes[0].set_ylim(0, tallest*1.3)  # headroom keeps the group labels clear of the bars
axes[1].set_xlabel('最大运动心率 thalach（次/分）', color=muted)
fig.tight_layout(h_pad=.6)
fig.savefig(LESSON/'report'/'thalach-distribution.png', facecolor='white')
plt.close(fig)

result = dict(source=str(SOURCE), sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                            scipy=scipy.__version__, statsmodels=statsmodels.__version__,
                            matplotlib=matplotlib.__version__),
              total_n=len(df), included_n=len(cc), excluded_n=len(df)-len(cc),
              duplicate_record_ids=int(df.record_id.duplicated().sum()),
              duplicate_selected_profiles_beyond_first=duplicate_profiles,
              duplicate_profile_groups=duplicate_groups, implausible_zeros=implausible_zeros,
              tukey_outliers=outliers, balance=balance, site_disease=site_disease,
              group_summary=summary, primary=primary, adjusted=adjusted,
              mann_whitney=rank_result, by_site=by_site,
              missingness_by_site=missing_site.reset_index().to_dict(orient='records'),
              missing_profile=missing_profile, exclude_va=exclude_va, interaction=interaction,
              power_examples=power_examples,
              independent_20_tests_fwer=1-.95**20)
(OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
