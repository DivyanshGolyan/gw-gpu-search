#!/usr/bin/env python
"""Whole-search FAR across chunks (R7): pool the time-slide backgrounds only where the ranking statistic means
the same thing in every chunk.

For a grid of statistic thresholds s, each chunk c has n_c(s) background events louder than s in T_c years
of background. If one rate r(s) fits all chunks, n_c ~ Poisson(r T_c); the pooled estimate is
r(s) = sum n_c / sum T_c (sum of counts over sum of exposure), and its floor is 1 / sum T_c.
Comparability test per threshold: Poisson likelihood-ratio (deviance) of "one rate" against "a rate per
chunk", chi-squared with C - 1 dof. Thresholds where it fails (p < 0.01 / number of thresholds tested) mean
the statistic is not comparable there, and FARs stay per chunk. Slide reuse makes counts clumpier than
Poisson (slide_reuse.py: N_eff/n ~ 0.4-0.9 in the tail on week 1), so a pass here is necessary, not sufficient.

Also prints, for each candidate in --candidates (candidates.json), the per-chunk FAR, the pooled FAR at its
statistic when allowed, the layer-combined FAR (n_layers x min over layers, fixed in advance) and the
false-alarm probability over the whole search, FAP = 1 - exp(-FAR x T_live_total).
Usage: pool_far.py OUT.json CHUNK1/search.hdf CHUNK2/search.hdf ... [--candidates candidates.json --n-layers 2]
"""
import argparse, json
import numpy as np
import h5py
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('chunks', nargs='+')
ap.add_argument('--candidates'); ap.add_argument('--n-layers', type=int, default=2)
a = ap.parse_args()
YEAR = 365.25 * 86400

S, Tb, Tl, names = [], [], [], []
for p in a.chunks:
    with h5py.File(p) as f:
        S.append(np.sort(f['background/stat'][:])); Tb.append(float(f.attrs['background_time_yr']))
        Tl.append(float(f.attrs['live_time_s']) / YEAR); names.append(p)
Tb = np.array(Tb); C = len(S)
smin = max(float(s.min()) for s in S); smax = max(float(s.max()) for s in S)
grid = np.linspace(smin, smax, 30)
rows = []
for s in grid:
    n = np.array([len(x) - np.searchsorted(x, s, 'left') for x in S], float)
    r = n.sum() / Tb.sum()
    mu = r * Tb
    dev = 2 * np.sum(np.where(n > 0, n * np.log(np.where(n > 0, n, 1) / np.where(mu > 0, mu, 1)), 0) - (n - mu))
    p = float(stats.chi2.sf(dev, C - 1)) if C > 1 and n.sum() > 0 else 1.0
    rows.append(dict(stat=float(s), counts=n.astype(int).tolist(), pooled_far_per_yr=float(r),
                     per_chunk_far_per_yr=(n / Tb).tolist(), deviance=float(dev), p_comparable=p))
alpha = 0.01 / len(grid)
ok = [r['stat'] for r in rows if r['p_comparable'] >= alpha]
res = dict(chunks=names, background_time_yr=Tb.tolist(), live_time_yr=float(sum(Tl)), pooled_floor_per_yr=float(1 / Tb.sum()),
           alpha_per_threshold=alpha, comparable_all=all(r['p_comparable'] >= alpha for r in rows),
           first_failing_stat=next((r['stat'] for r in rows if r['p_comparable'] < alpha), None), grid=rows,
           n_layers=a.n_layers)
for r in rows[::3]:
    print(f"stat {r['stat']:.2f}  pooled FAR {r['pooled_far_per_yr']:.4g}/yr  counts {r['counts']}  p {r['p_comparable']:.3g}")
print('comparable at every threshold:', res['comparable_all'], ' pooled floor', res['pooled_floor_per_yr'], '/yr')

if a.candidates:
    cands = json.load(open(a.candidates))['candidates']; T = sum(Tl)
    allbg = np.sort(np.concatenate(S)); out = []
    for c in cands:
        pooled = (len(allbg) - np.searchsorted(allbg, c['stat'], 'left')) / Tb.sum() if res['comparable_all'] else None
        far = pooled if pooled is not None else c['far_per_yr']
        comb = a.n_layers * far
        out.append(dict(gps=c['gps'], status=c['status'], far_chunk=c['far_per_yr'], far_pooled=pooled,
                        far_layers=comb, fap_whole_search=float(1 - np.exp(-comb * T))))
        print(out[-1])
    res['candidates'] = out
json.dump(res, open(a.out, 'w'), indent=1)
