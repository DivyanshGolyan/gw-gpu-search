#!/usr/bin/env python
"""R7 calibration: are production FARs honest on noise-only data?

Input: coinc_rank outputs of noise-only pseudo-experiments (--pseudo-shift: one detector moved far outside
the slides, so the zero lag has no real coincidences), all run with the exact production settings.
Two checks, for the exclusive FAR (what candidates are quoted with) and the inclusive FAR:
 1. Counting: over all experiments, the number of zero-lag events with FAR <= F should be Poisson with mean
    N_exp * F * T_live. Plotted as observed / expected against F with the Poisson 90% band.
 2. Loudest event: per experiment, p = 1 - exp(-FAR_min * T_live) should be uniform on [0, 1] (KS test).
    An experiment with no zero-lag event above --min-stat has FAR_min above the FAR at min-stat; its p is
    drawn uniformly above that bound (censored), which is noted in the output.
Usage: r7_calibration.py OUT_PREFIX ps_*.hdf
"""
import json, sys
import numpy as np
import h5py
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out, files = sys.argv[1], sys.argv[2:]
YEAR = 365.25 * 86400
rng = np.random.default_rng(1)
fars = {'excl': [], 'incl': []}; loud = {'excl': [], 'incl': []}; T = None; censored = 0; fmax = []
for p in files:
    with h5py.File(p) as f:
        T = float(f.attrs['live_time_s']) / YEAR
        g = f['foreground']; b = f['background']
        Tb = float(f.attrs['background_time_yr'])
        f_minstat = len(b['stat']) / Tb          # FAR at the min-stat cut: the largest FAR the output can show
        fmax.append(f_minstat)
        for k in fars:
            x = g[f'far_{k}_per_yr'][:] if len(g['stat']) else np.zeros(0)
            x = np.maximum(x, 0.0)
            fars[k].append(x)
            if len(x):
                loud[k].append(1 - np.exp(-x.min() * T))
            else:
                lo = 1 - np.exp(-f_minstat * T); loud[k].append(rng.uniform(lo, 1.0))
        censored += int(len(g['stat']) == 0)
n = len(files); fcap = min(fmax)
grid = np.logspace(-2, np.log10(fcap), 40)
res = dict(n_experiments=n, live_time_yr=T, far_cap_per_yr=fcap, censored_loudest=censored, checks={})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for k, ls in (('excl', '-'), ('incl', '--')):
    allf = np.concatenate(fars[k])
    obs = np.array([(allf <= F).sum() for F in grid]); exp = n * grid * T
    lo, hi = stats.poisson.ppf(0.05, exp), stats.poisson.ppf(0.95, exp)
    axes[0].plot(grid, np.where(exp > 0, obs / exp, np.nan), ls, label=f'{k}: observed / expected')
    if k == 'excl':
        axes[0].fill_between(grid, lo / exp, hi / exp, color='0.85', label='Poisson 90% band')
    pv = np.array(loud[k]); ks = stats.kstest(pv, 'uniform')
    q = np.sort(pv); axes[1].plot(np.arange(1, n + 1) / n, q, ls + 'o', ms=3, label=f'{k}: KS p = {ks.pvalue:.2f}')
    res['checks'][k] = dict(far_grid=grid.tolist(), observed=obs.tolist(), expected=exp.tolist(),
                            outside_band=[float(F) for F, o, a, b in zip(grid, obs, lo, hi) if o < a or o > b],
                            loudest_p=pv.tolist(), ks_stat=float(ks.statistic), ks_pvalue=float(ks.pvalue))
axes[0].set_xscale('log'); axes[0].axhline(1, color='k', lw=0.5); axes[0].set_xlabel('FAR threshold (1/yr)')
axes[0].set_ylabel('zero-lag count, observed / expected'); axes[0].legend(fontsize=8)
axes[0].set_title(f'{n} noise-only pseudo-experiments x {T * 365.25:.1f} d')
axes[1].plot([0, 1], [0, 1], 'k', lw=0.5); axes[1].set_xlabel('uniform quantile'); axes[1].set_ylabel('loudest-event p')
axes[1].legend(fontsize=8); axes[1].set_title('loudest event per experiment')
fig.savefig(out + '.png', dpi=100)
json.dump(res, open(out + '.json', 'w'), indent=1)
for k in res['checks']:
    c = res['checks'][k]
    print(k, 'KS p', round(c['ks_pvalue'], 3), 'grid points outside the 90% band:', len(c['outside_band']), 'of', len(grid))
    for F in (0.1, 1, 10, 100):
        if F <= fcap:
            allf = np.concatenate(fars[k]); print(f'  FAR<={F:g}/yr: observed {(allf <= F).sum()} expected {n * F * T:.2f}')
