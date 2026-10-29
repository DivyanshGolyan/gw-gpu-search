#!/usr/bin/env python
"""Slide-reuse check (R7): time-slide background events share single-detector triggers, so background counts
are clumped, not Poisson. For the loudest n background events, group them by the H1 trigger and by the L1
trigger they use; with clump sizes c_i the effective number of independent counts is (sum c)^2 / sum c^2.
The relative FAR uncertainty is then ~ 1/sqrt(N_eff) rather than 1/sqrt(n).
Usage: slide_reuse.py SEARCH.hdf [OUT.json]"""
import json, sys
import numpy as np
import h5py
f = h5py.File(sys.argv[1]); b = f['background']; Tb = float(f.attrs['background_time_yr'])
s = b['stat'][:]; o = np.argsort(-s)
res = dict(background_time_yr=Tb, rows=[])
for n in (10, 30, 100, 300, 1000, 3000, 10000):
    if n > len(o):
        break
    i = o[:n]; row = dict(n=n, stat_min=float(s[i[-1]]), far_per_yr=n / Tb)
    for ifo in ('H1', 'L1'):
        _, c = np.unique(np.round(b[f'{ifo}_time'][:][i], 3), return_counts=True)
        row[f'n_eff_{ifo}'] = float(c.sum() ** 2 / (c ** 2).sum()); row[f'max_reuse_{ifo}'] = int(c.max())
    row['n_eff'] = min(row['n_eff_H1'], row['n_eff_L1'])
    row['rel_err_poisson'] = 1 / np.sqrt(n); row['rel_err_clumped'] = 1 / np.sqrt(row['n_eff'])
    res['rows'].append(row)
    print(' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()))
if len(sys.argv) > 2:
    json.dump(res, open(sys.argv[2], 'w'), indent=1)
