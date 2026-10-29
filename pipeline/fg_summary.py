#!/usr/bin/env python
"""Small per-week summary of a coinc_rank search output, for the project share: the foreground group and
attrs (no background), plus data-quality context for every foreground row with FAR below --far-thr:
  gate_dt  distance to the nearest autogate in either detector (s), from the per-piece 'gates' groups
  env_H1/env_L1  single-detector peaks with SNR >= 8 within +-8 s, excluding +-0.2 s around the candidate
candidates.py reads these columns when present.
Usage: fg_summary.py RANK.hdf OUT.hdf --weekdir WEEKDIR --peaks H1=merged_H1.hdf --peaks L1=merged_L1.hdf
(a --peaks value may also be a comma-separated list of piece files, merged in memory)
"""
import argparse, glob
import numpy as np
import h5py
from merge_peaks import merge

ap = argparse.ArgumentParser()
ap.add_argument('rank'); ap.add_argument('out')
ap.add_argument('--weekdir', action='append', default=[], help='week dir; gates are read from WEEKDIR/search/grp_*/peaks/*.hdf')
ap.add_argument('--peaks', action='append', default=[])
ap.add_argument('--far-thr', type=float, default=10.0)
a = ap.parse_args()

gt = []
for w in a.weekdir:
    for p in glob.glob(w.rstrip('/') + '/search/grp_*/peaks/*.hdf'):
        with h5py.File(p) as f:
            if 'gates' in f and 'time' in f['gates']:
                gt.append(f['gates/time'][:])
gt = np.sort(np.concatenate(gt)) if gt else np.zeros(0)

with h5py.File(a.rank) as f, h5py.File(a.out, 'w') as o:
    for k, v in f.attrs.items():
        o.attrs[k] = v
    g = f['foreground']; og = o.create_group('foreground')
    for k in g:
        og[k] = g[k][:]
    t = g['H1_time'][:]; m = g['far_excl_per_yr'][:] < a.far_thr
    gate_dt = np.full(len(t), np.nan)
    if len(gt):
        j = np.clip(np.searchsorted(gt, t), 1, len(gt) - 1)
        gate_dt = np.minimum(np.abs(gt[j - 1] - t), np.abs(gt[j] - t))
    og['gate_dt'] = gate_dt
    for spec in a.peaks:
        ifo, p = spec.split('=', 1)
        env = np.full(len(t), -1)
        pf = merge(p.split(','), ('end_time', 'snr'))[0] if ',' in p else h5py.File(p)
        et = pf['end_time']
        for i in np.nonzero(m)[0]:
            lo, hi = np.searchsorted(et, t[i] - 8), np.searchsorted(et, t[i] + 8)
            tt, ss = et[lo:hi], pf['snr'][lo:hi]
            env[i] = int(((ss >= 8) & (np.abs(tt - t[i]) > 0.2)).sum())
        og[f'env_{ifo}'] = env
    o.attrs['n_gates'] = len(gt)
print(a.out, 'foreground', len(t), 'with FAR <', a.far_thr, int(m.sum()), 'gates', len(gt))
